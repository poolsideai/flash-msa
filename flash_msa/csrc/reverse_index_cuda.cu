#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT32(x)

namespace {

constexpr int kThreads = 256;

__global__ void count_edges_kernel(
    int const* __restrict__ block_indices,
    int* __restrict__ counts,
    int B,
    int Hp,
    int S,
    int Kb,
    int NB)
{
    int64_t E = (int64_t)B * Hp * S * Kb;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int slot = (int)(e % Kb);
        int64_t tmp = e / Kb;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);

        int key_block = block_indices[((int64_t)(b * Hp + p) * S + q) * Kb + slot];
        // Top-k selection may fill early queries with future blocks scored at
        // -inf.  They have no causal contribution, so do not schedule them.
        if ((unsigned)key_block >= (unsigned)NB || key_block > q / 128) {
            continue;
        }
        int bucket = (b * Hp + p) * NB + key_block;
        atomicAdd(counts + bucket, 1);
    }
}

__global__ void scan_fill_meta_kernel(
    int const* __restrict__ counts,
    int* __restrict__ bucket_offsets,
    int* __restrict__ task_meta,
    int B,
    int Hp,
    int NB,
    int query_chunk,
    int padded_tasks)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int task = 0;
    int buckets = B * Hp * NB;
    for (int bucket = 0; bucket < buckets; ++bucket) {
        bucket_offsets[bucket] = task;
        int count = counts[bucket];
        int chunks = (count + query_chunk - 1) / query_chunk;
        int key_block = bucket % NB;
        int tmp = bucket / NB;
        int proxy_head = tmp % Hp;
        int batch = tmp / Hp;

        for (int c = 0; c < chunks && task < padded_tasks; ++c) {
            int valid = count - c * query_chunk;
            valid = valid > query_chunk ? query_chunk : valid;
            int64_t row = (int64_t)task * 4;
            task_meta[row + 0] = batch;
            task_meta[row + 1] = proxy_head;
            task_meta[row + 2] = key_block;
            task_meta[row + 3] = valid;
            ++task;
        }
    }
}

__global__ void scatter_edges_kernel(
    int const* __restrict__ block_indices,
    int const* __restrict__ bucket_offsets,
    int* __restrict__ write_counts,
    int* __restrict__ task_qids,
    int B,
    int Hp,
    int S,
    int Kb,
    int NB,
    int query_chunk,
    int padded_tasks)
{
    int64_t E = (int64_t)B * Hp * S * Kb;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int slot = (int)(e % Kb);
        int64_t tmp = e / Kb;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);

        int key_block = block_indices[((int64_t)(b * Hp + p) * S + q) * Kb + slot];
        if ((unsigned)key_block >= (unsigned)NB || key_block > q / 128) {
            continue;
        }

        int bucket = (b * Hp + p) * NB + key_block;
        int local = atomicAdd(write_counts + bucket, 1);
        int task = bucket_offsets[bucket] + local / query_chunk;
        if (task >= padded_tasks) {
            continue;
        }
        int lane = local - (local / query_chunk) * query_chunk;
        task_qids[(int64_t)task * query_chunk + lane] = q;
    }
}

__global__ void count_remote_slots_kernel(
    int const* __restrict__ block_indices,
    int* __restrict__ counts,
    int B,
    int Hp,
    int S,
    int Kb,
    int NB)
{
    int remote_slots = Kb - 1;
    int64_t E = (int64_t)B * Hp * S * remote_slots;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int slot = (int)(e % remote_slots) + 1;
        int64_t tmp = e / remote_slots;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);
        int key_block = block_indices[((int64_t)(b * Hp + p) * S + q) * Kb + slot];
        if ((unsigned)key_block >= (unsigned)NB || key_block >= q / 128) {
            continue;
        }
        int bucket = (b * Hp + p) * NB + key_block;
        atomicAdd(counts + bucket, 1);
    }
}

__global__ void scan_remote_offsets_kernel(
    int const* __restrict__ counts,
    int* __restrict__ bucket_offsets,
    int buckets)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int edge = 0;
    for (int bucket = 0; bucket < buckets; ++bucket) {
        bucket_offsets[bucket] = edge;
        edge += counts[bucket];
    }
    bucket_offsets[buckets] = edge;
}

__global__ void scatter_remote_slots_kernel(
    int const* __restrict__ block_indices,
    int const* __restrict__ bucket_offsets,
    int* __restrict__ write_counts,
    int64_t* __restrict__ destinations,
    int* __restrict__ positions,
    uint8_t* __restrict__ valid,
    int B,
    int Hp,
    int S,
    int Kb,
    int NB)
{
    int remote_slots = Kb - 1;
    int64_t E = (int64_t)B * Hp * S * remote_slots;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; e < E; e += stride) {
        int slot = (int)(e % remote_slots) + 1;
        int64_t tmp = e / remote_slots;
        int q = (int)(tmp % S);
        tmp /= S;
        int p = (int)(tmp % Hp);
        int b = (int)(tmp / Hp);
        int key_block = block_indices[((int64_t)(b * Hp + p) * S + q) * Kb + slot];
        if ((unsigned)key_block >= (unsigned)NB || key_block >= q / 128) {
            continue;
        }

        int bucket = (b * Hp + p) * NB + key_block;
        int pos = bucket_offsets[bucket] + atomicAdd(write_counts + bucket, 1);
        destinations[pos] = (int64_t)(b * Hp + p) * S + q;
        positions[e] = pos;
        valid[pos] = true;
    }
}

__global__ void fill_dense_schedule_kernel(
    int64_t const* __restrict__ bucket_offsets,
    int* __restrict__ task_meta,
    int* __restrict__ task_qids,
    int64_t num_tasks,
    int buckets,
    int Hp,
    int NB,
    int S,
    int query_chunk)
{
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t task = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         task < num_tasks;
         task += stride) {
        int low = 0;
        int high = buckets;
        while (low + 1 < high) {
            int middle = (low + high) / 2;
            if (bucket_offsets[middle] <= task) {
                low = middle;
            } else {
                high = middle;
            }
        }

        int bucket = low;
        int key_block = bucket % NB;
        int tmp = bucket / NB;
        int proxy_head = tmp % Hp;
        int batch = tmp / Hp;
        int64_t local_task = task - bucket_offsets[bucket];
        int query_start = key_block * 128 + (int)local_task * query_chunk;
        int valid = min(query_chunk, S - query_start);

        int64_t meta_row = task * 4;
        task_meta[meta_row + 0] = batch;
        task_meta[meta_row + 1] = proxy_head;
        task_meta[meta_row + 2] = key_block;
        task_meta[meta_row + 3] = valid;
        for (int lane = 0; lane < query_chunk; ++lane) {
            task_qids[task * query_chunk + lane] = lane < valid ? query_start + lane : -1;
        }
    }
}

} // namespace

void run_build_reverse_index(
    torch::Tensor block_indices,
    torch::Tensor counts,
    torch::Tensor write_counts,
    torch::Tensor bucket_offsets,
    torch::Tensor task_meta,
    torch::Tensor task_qids,
    int64_t block_size,
    int64_t query_chunk)
{
    CHECK_INPUT(block_indices);
    CHECK_INPUT(counts);
    CHECK_INPUT(write_counts);
    CHECK_INPUT(bucket_offsets);
    CHECK_INPUT(task_meta);
    CHECK_INPUT(task_qids);

    TORCH_CHECK(block_indices.dim() == 4, "block_indices must have shape [B, Hp, S, Kb]");
    TORCH_CHECK(task_meta.dim() == 2 && task_meta.size(1) == 4, "task_meta must have shape [T, 4]");
    TORCH_CHECK(task_qids.dim() == 2 && task_qids.size(1) == query_chunk, "task_qids must have shape [T, query_chunk]");
    TORCH_CHECK(block_size == 128, "backward metadata currently requires block_size=128");
    TORCH_CHECK(query_chunk > 0, "query_chunk must be positive");

    int B = (int)block_indices.size(0);
    int Hp = (int)block_indices.size(1);
    int S = (int)block_indices.size(2);
    int Kb = (int)block_indices.size(3);
    int NB = S / (int)block_size;
    int buckets = B * Hp * NB;
    int padded_tasks = (int)task_meta.size(0);

    TORCH_CHECK(S % block_size == 0, "S must be divisible by block_size");
    TORCH_CHECK(counts.numel() == buckets, "counts has wrong size");
    TORCH_CHECK(write_counts.numel() == buckets, "write_counts has wrong size");
    TORCH_CHECK(bucket_offsets.numel() == buckets, "bucket_offsets has wrong size");
    TORCH_CHECK(task_qids.size(0) == padded_tasks, "task_meta/task_qids row mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int>(), 0, counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(write_counts.data_ptr<int>(), 0, write_counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_meta.data_ptr<int>(), 0, task_meta.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(task_qids.data_ptr<int>(), 0xff, task_qids.numel() * sizeof(int), stream));

    int64_t edges = (int64_t)B * Hp * S * Kb;
    int blocks = (int)std::min<int64_t>((edges + kThreads - 1) / kThreads, 65535);
    blocks = std::max(blocks, 1);

    count_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        block_indices.data_ptr<int>(),
        counts.data_ptr<int>(),
        B,
        Hp,
        S,
        Kb,
        NB);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scan_fill_meta_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int>(),
        bucket_offsets.data_ptr<int>(),
        task_meta.data_ptr<int>(),
        B,
        Hp,
        NB,
        (int)query_chunk,
        padded_tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scatter_edges_kernel<<<blocks, kThreads, 0, stream>>>(
        block_indices.data_ptr<int>(),
        bucket_offsets.data_ptr<int>(),
        write_counts.data_ptr<int>(),
        task_qids.data_ptr<int>(),
        B,
        Hp,
        S,
        Kb,
        NB,
        (int)query_chunk,
        padded_tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_build_remote_layout(
    torch::Tensor block_indices,
    torch::Tensor counts,
    torch::Tensor write_counts,
    torch::Tensor bucket_offsets,
    torch::Tensor destinations,
    torch::Tensor positions,
    torch::Tensor valid,
    int64_t block_size)
{
    CHECK_INPUT(block_indices);
    CHECK_INPUT(counts);
    CHECK_INPUT(write_counts);
    CHECK_INPUT(bucket_offsets);
    CHECK_CUDA(destinations);
    CHECK_CONTIGUOUS(destinations);
    CHECK_INPUT(positions);
    CHECK_CUDA(valid);
    CHECK_CONTIGUOUS(valid);

    TORCH_CHECK(block_size == 128, "remote layout currently requires block_size=128");
    TORCH_CHECK(block_indices.dim() == 4, "block_indices must have shape [B, Hp, S, Kb]");
    TORCH_CHECK(destinations.scalar_type() == at::kLong, "destinations must be int64");
    TORCH_CHECK(valid.scalar_type() == at::kByte, "valid must be uint8");

    int B = (int)block_indices.size(0);
    int Hp = (int)block_indices.size(1);
    int S = (int)block_indices.size(2);
    int Kb = (int)block_indices.size(3);
    int NB = S / (int)block_size;
    int buckets = B * Hp * NB;
    int64_t remote_edges = (int64_t)B * Hp * S * (Kb - 1);

    TORCH_CHECK(S % block_size == 0, "S must be divisible by block_size");
    TORCH_CHECK(Kb > 1, "remote layout requires at least two selected blocks");
    TORCH_CHECK(counts.numel() == buckets && write_counts.numel() == buckets, "remote counts have wrong size");
    TORCH_CHECK(bucket_offsets.numel() == buckets + 1, "bucket_offsets has wrong size");
    TORCH_CHECK(destinations.numel() == remote_edges && positions.numel() == remote_edges && valid.numel() == remote_edges, "remote edge buffers have wrong size");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr<int>(), 0, counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(write_counts.data_ptr<int>(), 0, write_counts.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(destinations.data_ptr<int64_t>(), 0, destinations.numel() * sizeof(int64_t), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(positions.data_ptr<int>(), 0xff, positions.numel() * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(valid.data_ptr<uint8_t>(), 0, valid.numel() * sizeof(uint8_t), stream));

    int blocks = (int)std::min<int64_t>((remote_edges + kThreads - 1) / kThreads, 65535);
    blocks = std::max(blocks, 1);
    count_remote_slots_kernel<<<blocks, kThreads, 0, stream>>>(
        block_indices.data_ptr<int>(), counts.data_ptr<int>(), B, Hp, S, Kb, NB);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scan_remote_offsets_kernel<<<1, 1, 0, stream>>>(
        counts.data_ptr<int>(), bucket_offsets.data_ptr<int>(), buckets);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    scatter_remote_slots_kernel<<<blocks, kThreads, 0, stream>>>(
        block_indices.data_ptr<int>(), bucket_offsets.data_ptr<int>(), write_counts.data_ptr<int>(),
        destinations.data_ptr<int64_t>(), positions.data_ptr<int>(), valid.data_ptr<uint8_t>(),
        B, Hp, S, Kb, NB);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_build_dense_schedule(
    torch::Tensor bucket_offsets,
    torch::Tensor task_meta,
    torch::Tensor task_qids,
    int64_t proxy_heads,
    int64_t num_blocks,
    int64_t seq_len,
    int64_t query_chunk)
{
    CHECK_CUDA(bucket_offsets);
    CHECK_CONTIGUOUS(bucket_offsets);
    CHECK_INPUT(task_meta);
    CHECK_INPUT(task_qids);
    TORCH_CHECK(bucket_offsets.scalar_type() == at::kLong, "bucket_offsets must be int64");
    TORCH_CHECK(task_meta.dim() == 2 && task_meta.size(1) == 4, "task_meta must have shape [T, 4]");
    TORCH_CHECK(task_qids.dim() == 2 && task_qids.size(1) == query_chunk, "task_qids must have shape [T, query_chunk]");
    TORCH_CHECK(task_meta.size(0) == task_qids.size(0), "dense schedule task rows must match");
    TORCH_CHECK(bucket_offsets.numel() > 1, "dense schedule needs at least one bucket");

    int buckets = (int)bucket_offsets.numel() - 1;
    TORCH_CHECK(buckets % (proxy_heads * num_blocks) == 0, "dense schedule bucket count is invalid");
    int64_t num_tasks = task_meta.size(0);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int blocks = (int)std::min<int64_t>((num_tasks + kThreads - 1) / kThreads, 65535);
    blocks = std::max(blocks, 1);
    fill_dense_schedule_kernel<<<blocks, kThreads, 0, stream>>>(
        bucket_offsets.data_ptr<int64_t>(),
        task_meta.data_ptr<int>(),
        task_qids.data_ptr<int>(),
        num_tasks,
        buckets,
        (int)proxy_heads,
        (int)num_blocks,
        (int)seq_len,
        (int)query_chunk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_build_reverse_index", &run_build_reverse_index, "Build MSA reverse index on CUDA");
    m.def("run_build_remote_layout", &run_build_remote_layout, "Build the MSA remote varlen layout on CUDA");
    m.def("run_build_dense_schedule", &run_build_dense_schedule, "Build the dense MSA backward schedule on CUDA");
}
