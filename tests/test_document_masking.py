import pytest
import torch

from flash_msa import (
    flash_msa_func,
    flash_msa_warmup_func,
    prepare_sparse_attention,
    sparse_main_attention,
)
from flash_msa.reverse_index_cuda import (
    build_block_sparse_selection,
    build_dense_causal_schedule_cuda,
)


def test_dense_warmup_schedule_is_built_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    task_meta, task_qids = build_dense_causal_schedule_cuda(
        batch=2,
        n_proxy_heads=2,
        seq_len=256,
        query_chunk=16,
        device=torch.device("cuda"),
    )

    assert task_meta.shape == (96, 4)
    for batch in range(2):
        for head in range(2):
            for block in range(2):
                matches = task_meta[:, :3] == torch.tensor(
                    [batch, head, block],
                    device="cuda",
                )
                rows = matches.all(dim=1).nonzero().flatten()
                actual = task_qids[rows].flatten()
                actual = actual[actual >= 0]
                expected = torch.arange(
                    block * 128,
                    256,
                    device="cuda",
                    dtype=torch.int32,
                )
                torch.testing.assert_close(actual, expected)


def test_blackwell_schedule_unions_256_token_query_blocks() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    positions = torch.arange(512, device="cuda", dtype=torch.int32)
    local_blocks = (positions // 128).view(1, 1, 512, 1).expand(1, 2, -1, -1)
    sentinels = torch.full_like(local_blocks, 4)
    selection = build_block_sparse_selection(
        torch.cat((local_blocks, sentinels), dim=-1),
        n_main_heads=8,
        query_block_size=256,
    )

    assert selection.proxy_block_counts.shape == (1, 2, 2)
    assert selection.main_block_counts.shape == (1, 8, 2)
    torch.testing.assert_close(
        selection.proxy_block_counts,
        torch.full((1, 2, 2), 2, device="cuda", dtype=torch.int32),
    )


def test_sparse_gqa_document_mask_matches_eager_oracle() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FA4 sparse oracle is an SM90 regression")

    torch.manual_seed(19)
    batch, seq_len, head_dim = 1, 512, 128
    n_heads, n_kv_heads = 8, 2
    n_proxy_heads, n_proxy_kv_heads = 4, 2
    top_k = 256
    scale = head_dim**-0.5
    documents = torch.empty(batch, seq_len, device="cuda", dtype=torch.int32)
    documents[:, :173] = 0
    documents[:, 173:381] = 1
    documents[:, 381:] = 2

    def rand(heads: int) -> torch.Tensor:
        return torch.randn(
            batch,
            heads,
            seq_len,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )

    q_proxy, k_proxy = rand(n_proxy_heads), rand(n_proxy_kv_heads)
    q, k, v = (
        torch.randn(
            batch,
            seq_len,
            heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).transpose(1, 2)
        for heads in (n_heads, n_kv_heads, n_kv_heads)
    )
    metadata = prepare_sparse_attention(
        q_proxy,
        k_proxy,
        q,
        k,
        v,
        top_k,
        scale,
        documents,
    )
    actual, actual_lse = sparse_main_attention(q, k, v, metadata, scale)

    num_blocks = seq_len // 128
    top_k_blocks = top_k // 128
    proxy_k = k_proxy.repeat_interleave(
        n_proxy_heads // n_proxy_kv_heads,
        dim=1,
    )
    proxy_scores = (q_proxy @ proxy_k.transpose(-2, -1)) * scale
    positions = torch.arange(seq_len, device="cuda")
    valid = positions[None, :] <= positions[:, None]
    valid = valid[None, None] & (
        documents[:, None, :, None] == documents[:, None, None, :]
    )
    proxy_scores = proxy_scores.masked_fill(~valid, float("-inf"))
    block_scores = proxy_scores.view(
        batch,
        n_proxy_heads,
        seq_len,
        num_blocks,
        128,
    ).amax(dim=-1)
    local_blocks = (positions // 128).view(1, 1, seq_len, 1)
    block_scores.scatter_(
        3, local_blocks.expand(batch, n_proxy_heads, -1, -1), torch.inf
    )
    top_values, expected_blocks = block_scores.topk(top_k_blocks, dim=-1)
    expected_blocks = expected_blocks.masked_fill(top_values.isneginf(), num_blocks)
    block_mask = torch.zeros_like(block_scores, dtype=torch.bool)
    valid_blocks = expected_blocks < num_blocks
    block_mask.scatter_(3, expected_blocks.clamp_max(num_blocks - 1), valid_blocks)
    membership_bits = torch.zeros_like(metadata.selection.membership_bits)
    expected_blocks_i32 = expected_blocks.to(torch.int32)
    membership_bits.scatter_add_(
        3,
        expected_blocks.clamp_max(num_blocks - 1).div(32, rounding_mode="floor"),
        torch.where(
            valid_blocks,
            torch.bitwise_left_shift(
                torch.ones_like(expected_blocks_i32),
                expected_blocks_i32.remainder(32),
            ),
            0,
        ),
    )
    torch.testing.assert_close(metadata.selection.membership_bits, membership_bits)
    token_mask = (
        block_mask[..., None]
        .expand(batch, n_proxy_heads, seq_len, num_blocks, 128)
        .reshape(batch, n_proxy_heads, seq_len, seq_len)
        .repeat_interleave(n_heads // n_proxy_heads, dim=1)
    )
    attention_mask = token_mask & valid
    main_k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
    main_v = v.repeat_interleave(n_heads // n_kv_heads, dim=1)
    scores = (q.float() @ main_k.float().transpose(-2, -1)) * scale
    scores.masked_fill_(~attention_mask, float("-inf"))
    expected_lse = scores.logsumexp(dim=-1)
    expected = (
        (scores.softmax(dim=-1) @ main_v.float())
        .to(torch.bfloat16)
        .transpose(1, 2)
        .reshape(batch, seq_len, -1)
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-5, rtol=2e-5)


def test_membership_bitset_preserves_bit_31() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    selected = torch.full(
        (1, 1, 4096, 1),
        31,
        device="cuda",
        dtype=torch.int32,
    )
    selection = build_block_sparse_selection(
        selected,
        n_main_heads=1,
        query_block_size=128,
    )

    assert selection.membership_bits.shape == (1, 1, 4096, 1)
    torch.testing.assert_close(
        selection.membership_bits,
        torch.full_like(selection.membership_bits, torch.iinfo(torch.int32).min),
    )


@pytest.mark.parametrize("kernel", [flash_msa_func, flash_msa_warmup_func])
def test_packed_documents_isolate_outputs_and_gradients(kernel) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Flash-MSA requires Hopper or newer")

    torch.manual_seed(7)
    batch, seq_len, head_dim = 1, 512, 128
    documents = torch.empty(batch, seq_len, device="cuda", dtype=torch.int32)
    documents[:, :190] = 0
    documents[:, 190:350] = 1
    documents[:, 350:] = 2
    cu_seqlens = torch.tensor([0, 190, 350, 512], device="cuda", dtype=torch.int32)
    shapes = {
        "q_proxy": (batch, 1, seq_len, head_dim),
        "k_proxy": (batch, 1, seq_len, head_dim),
        "q": (batch, 6, seq_len, head_dim),
        "k": (batch, 1, seq_len, head_dim),
        "v": (batch, 1, seq_len, head_dim),
    }
    inputs = {
        name: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for name, shape in shapes.items()
    }

    def run(values, *, use_cu_seqlens: bool = False):
        tensors = {
            name: value.detach().clone().requires_grad_(True)
            for name, value in values.items()
        }
        out, aux = kernel(
            tensors["q_proxy"],
            tensors["k_proxy"],
            tensors["q"],
            tensors["k"],
            tensors["v"],
            256,
            head_dim**-0.5,
            None if use_cu_seqlens else documents,
            **({"cu_seqlens": cu_seqlens} if use_cu_seqlens else {}),
        )
        (out[:, 190:350].float().square().sum() + aux).backward()
        return out.detach(), tensors

    perturbed = {name: value.clone() for name, value in inputs.items()}
    for value in perturbed.values():
        value[:, :, :190] += torch.randn_like(value[:, :, :190]) * 10
        value[:, :, 350:] += torch.randn_like(value[:, :, 350:]) * 10

    output, tensors = run(inputs)
    cu_output, cu_tensors = run(inputs, use_cu_seqlens=True)
    perturbed_output, perturbed_tensors = run(perturbed)
    torch.testing.assert_close(cu_output, output)
    for name in inputs:
        assert tensors[name].grad is not None
        assert cu_tensors[name].grad is not None
        torch.testing.assert_close(cu_tensors[name].grad, tensors[name].grad)
    torch.testing.assert_close(output[:, 190:350], perturbed_output[:, 190:350])
    for name in ("q", "k", "v"):
        tensor = tensors[name]
        assert tensor.grad is not None
        outside = torch.cat((tensor.grad[:, :, :190], tensor.grad[:, :, 350:]), dim=2)
        assert outside.abs().max() == 0
    for name in ("q_proxy", "k_proxy"):
        grad = tensors[name].grad
        perturbed_grad = perturbed_tensors[name].grad
        assert grad is not None and perturbed_grad is not None
        assert grad[:, :, 190:350].abs().max() > 0
        torch.testing.assert_close(
            grad[:, :, 190:350],
            perturbed_grad[:, :, 190:350],
        )
