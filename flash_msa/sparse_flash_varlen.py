"""Selected-block attention using the MoBA varlen reparameterization."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from flash_attn.cute import utils

from flash_msa._flash_attn_compat import (
    flash_attn_supports_narrow_value_dim,
    flash_attn_varlen_forward,
    flash_attn_varlen_paged_forward,
)
from flash_msa.reverse_index_cuda import SparseAttentionMetadata


BLOCK_SIZE = 128

# Had to write this section to set the narrow V head for Proxy LSE varlen flash call
# to the minimum dim supported on each backend. FA4 on B200 fails with V_dim=8
LSE_VALUE_DIM_H100 = 8
LSE_VALUE_DIM_B200 = 16


@cute.jit
def _causal_document_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    document_ids = aux_tensors[0]
    q_global = q_idx + seqlen_info.offset_q
    kv_global = kv_idx + seqlen_info.offset_k
    q_pos = cute.make_rmem_tensor(1, cutlass.Int32)
    kv_pos = cute.make_rmem_tensor(1, cutlass.Int32)
    q_pos.store(q_global)
    kv_pos.store(kv_global)
    q_document = utils.scalar_to_ssa(document_ids[q_pos[0]], cutlass.Int32)
    kv_document = utils.scalar_to_ssa(document_ids[kv_pos[0]], cutlass.Int32)
    return (kv_idx <= q_idx) & (q_document == kv_document)


@cute.jit
def _remote_document_mask(
    batch: cute.TensorSSA,
    head: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list,
) -> cute.TensorSSA:
    q_document_ids, k_document_ids = aux_tensors
    q_global = q_idx + seqlen_info.offset_q
    q_pos = cute.make_rmem_tensor(1, cutlass.Int32)
    q_pos.store(q_global)
    q_document = utils.scalar_to_ssa(q_document_ids[q_pos[0]], cutlass.Int32)
    block_size = utils.scalar_to_ssa(BLOCK_SIZE, cutlass.Int32)
    k_global = batch[0] * block_size[0] + kv_idx[0]
    k_pos = cute.make_rmem_tensor(1, cutlass.Int32)
    k_pos.store(utils.scalar_to_ssa(k_global, cutlass.Int32))
    k_document = utils.scalar_to_ssa(k_document_ids[k_pos[0]], cutlass.Int32)
    return q_document == k_document


def _lse_value_dim(device: torch.device) -> int:
    instruction_set = torch.cuda.get_device_capability(device)
    if instruction_set[0] == 9:
        return LSE_VALUE_DIM_H100
    if instruction_set[0] == 10:
        return LSE_VALUE_DIM_B200
    raise NotImplementedError(
        f"Proxy LSE dummy V is not configured for SM{instruction_set[0]}{instruction_set[1]}"
    )


def _local_block_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    document_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run causal attention independently inside every local 128-token block."""

    batch, n_heads, seq_len, head_dim = map(int, q.shape)
    n_kv_heads = int(k.shape[1])
    value_dim = int(v.shape[-1])
    total_tokens = batch * seq_len
    num_sequences = total_tokens // BLOCK_SIZE
    cu_seqlens = (
        torch.arange(
            num_sequences + 1,
            device=q.device,
            dtype=torch.int32,
        )
        * BLOCK_SIZE
    )

    q_tokens = q.transpose(1, 2).contiguous().view(total_tokens, n_heads, head_dim)
    k_tokens = k.transpose(1, 2).contiguous().view(total_tokens, n_kv_heads, head_dim)
    v_tokens = v.transpose(1, 2).contiguous().view(total_tokens, n_kv_heads, value_dim)
    return flash_attn_varlen_forward(
        q=q_tokens,
        k=k_tokens,
        v=v_tokens,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=BLOCK_SIZE,
        max_seqlen_k=BLOCK_SIZE,
        softmax_scale=float(scale),
        causal=not document_ids.numel(),
        mask_mod=_causal_document_mask if document_ids.numel() else None,
        aux_tensors=[document_ids.reshape(-1)] if document_ids.numel() else None,
    )


@torch.compile(fullgraph=True, dynamic=False)
def _merge_remote_attention(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    remote_output: torch.Tensor,
    remote_lse: torch.Tensor,
    remote_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = remote_positions >= 0
    positions = remote_positions.clamp_min(0)
    gathered_lse = remote_lse[positions].masked_fill(
        ~valid[..., None],
        float("-inf"),
    )
    max_lse = torch.maximum(local_lse, gathered_lse.amax(dim=1))
    local_weight = (local_lse - max_lse).exp()
    remote_weight = (gathered_lse - max_lse[:, None]).exp()
    denominator = local_weight + remote_weight.sum(dim=1)
    output = (
        local_output * local_weight[..., None]
        + (
            remote_output[positions].float()
            * remote_weight[..., None]
            * valid[..., None, None]
        ).sum(dim=1)
    ) / denominator[..., None]
    return output, max_lse + denominator.log()


@torch.compile(fullgraph=True, dynamic=False)
def _merge_remote_lse(
    local_lse: torch.Tensor,
    remote_lse: torch.Tensor,
    remote_positions: torch.Tensor,
) -> torch.Tensor:
    valid = remote_positions >= 0
    gathered_lse = remote_lse[remote_positions.clamp_min(0)].masked_fill(
        ~valid[..., None],
        float("-inf"),
    )
    max_lse = torch.maximum(local_lse, gathered_lse.amax(dim=1))
    return (
        max_lse
        + (
            (local_lse - max_lse).exp()
            + (gathered_lse - max_lse[:, None]).exp().sum(dim=1)
        ).log()
    )


def sparse_flash_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    metadata: SparseAttentionMetadata,
    scale: float,
    return_output: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Return selected-block output and LSE without host-side scheduling."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors")
    if q.device.type != "cuda":
        raise ValueError("sparse varlen FlashAttention requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"sparse varlen FlashAttention supports fp16/bf16, got {q.dtype}"
        )

    batch, n_heads, seq_len, head_dim = map(int, q.shape)
    n_kv_heads = int(k.shape[1])
    n_proxy_heads = metadata.n_proxy_heads
    num_blocks = seq_len // BLOCK_SIZE
    if seq_len % BLOCK_SIZE:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if n_heads % n_proxy_heads:
        raise ValueError("query heads must be divisible by proxy heads")
    if n_proxy_heads < n_kv_heads or n_proxy_heads % n_kv_heads:
        raise ValueError("proxy heads must be >= and divisible by KV heads")
    if (metadata.batch, metadata.seq_len) != (batch, seq_len):
        raise ValueError("metadata has an incompatible batch or sequence length")
    if metadata.task_meta.device != q.device:
        raise ValueError("metadata must be on the same device as q")

    main_per_proxy = n_heads // n_proxy_heads

    # FA4 accepts a narrow dummy V for LSE-only proxy attention. SM100 requires
    # 16 elements because its packed-GQA epilogue rejects an 8-element V.
    if return_output or not flash_attn_supports_narrow_value_dim():
        attention_v = v
    else:
        attention_v = torch.zeros(
            (*v.shape[:-1], _lse_value_dim(v.device)),
            device=v.device,
            dtype=v.dtype,
        )
    local_out, local_lse_hs = _local_block_attention(
        q,
        k,
        attention_v,
        scale=float(scale),
        document_ids=metadata.document_ids,
    )
    local_lse = (
        local_lse_hs.transpose(0, 1)
        .reshape(batch, seq_len, n_proxy_heads, main_per_proxy)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(batch * n_proxy_heads * seq_len, main_per_proxy)
    )
    if return_output:
        local_output = (
            local_out.reshape(batch, seq_len, n_proxy_heads, main_per_proxy, -1)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(batch * n_proxy_heads * seq_len, main_per_proxy, -1)
            .float()
        )
    else:
        local_output = None

    if metadata.top_k_blocks == 1:
        lse = local_lse
        output = local_output
    else:
        destination = metadata.remote_destinations
        cu_seqlens_q = metadata.remote_cu_seqlens
        q_grouped = (
            q.reshape(batch, n_proxy_heads, main_per_proxy, seq_len, head_dim)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(batch * n_proxy_heads * seq_len, main_per_proxy, head_dim)
        )
        packed_q = q_grouped[destination]

        proxy_per_kv = n_proxy_heads // n_kv_heads
        sequence = torch.arange(
            batch * n_proxy_heads * num_blocks,
            device=q.device,
        )
        block = sequence % num_blocks
        group = sequence.div(num_blocks, rounding_mode="floor")
        batch_index = group.div(n_proxy_heads, rounding_mode="floor")
        proxy_head = group % n_proxy_heads
        kv_head = proxy_head.div(proxy_per_kv, rounding_mode="floor")
        physical_page = (
            ((batch_index * n_kv_heads + kv_head) * num_blocks + block)
            .to(torch.int32)
            .view(-1, 1)
        )

        value_dim = int(attention_v.shape[-1])
        k_pages = k.view(-1, BLOCK_SIZE, 1, head_dim)
        v_pages = attention_v.view(-1, BLOCK_SIZE, 1, value_dim)
        paged_result = flash_attn_varlen_paged_forward(
            q=packed_q,
            k_pages=k_pages,
            v_pages=v_pages,
            cu_seqlens_q=cu_seqlens_q,
            page_table=physical_page,
            max_seqlen_q=seq_len,
            max_seqlen_k=BLOCK_SIZE,
            softmax_scale=float(scale),
            causal=False,
            mask_mod=(_remote_document_mask if metadata.document_ids.numel() else None),
            aux_tensors=(
                [metadata.remote_q_document_ids, metadata.remote_k_document_ids]
                if metadata.document_ids.numel()
                else None
            ),
        )
        if paged_result is None:
            packed_k = k_pages[physical_page[:, 0].long()].reshape(-1, 1, head_dim)
            packed_v = v_pages[physical_page[:, 0].long()].reshape(-1, 1, value_dim)
            cu_seqlens_k = (
                torch.arange(
                    physical_page.shape[0] + 1,
                    device=q.device,
                    dtype=torch.int32,
                )
                * BLOCK_SIZE
            )
            paged_result = flash_attn_varlen_forward(
                q=packed_q,
                k=packed_k,
                v=packed_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=seq_len,
                max_seqlen_k=BLOCK_SIZE,
                softmax_scale=float(scale),
                causal=False,
                mask_mod=(
                    _remote_document_mask if metadata.document_ids.numel() else None
                ),
                aux_tensors=(
                    [metadata.remote_q_document_ids, metadata.remote_k_document_ids]
                    if metadata.document_ids.numel()
                    else None
                ),
            )
        remote_output, remote_lse_hs = paged_result
        remote_lse = remote_lse_hs.transpose(0, 1).contiguous()
        remote_positions = metadata.remote_positions.view(
            batch * n_proxy_heads * seq_len,
            metadata.top_k_blocks - 1,
        )

        if local_output is None:
            output = None
            lse = _merge_remote_lse(local_lse, remote_lse, remote_positions)
        else:
            output, lse = _merge_remote_attention(
                local_output,
                local_lse,
                remote_output,
                remote_lse,
                remote_positions,
            )

    lse = (
        lse.view(batch, n_proxy_heads, seq_len, main_per_proxy)
        .permute(0, 1, 3, 2)
        .contiguous()
        .view(batch, n_heads, seq_len)
    )
    if output is None:
        return None, lse
    output = (
        output.to(q.dtype)
        .view(batch, n_proxy_heads, seq_len, main_per_proxy, -1)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch, n_heads, seq_len, -1)
    )
    return output, lse


__all__ = ["sparse_flash_varlen_forward"]
