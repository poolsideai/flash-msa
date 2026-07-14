"""Varlen FlashAttention selected-block forward path for MSA main attention."""

from __future__ import annotations

import torch

from flash_msa._flash_attn_compat import flash_attn_block_sparse_forward
from flash_msa.reverse_index_cuda import SparseAttentionMetadata

BLOCK_SIZE = 128


def _validate_head_tiling(
    n_heads: int,
    n_kv_heads: int,
    n_proxy_heads: int,
) -> None:
    """Validate that selected attention supports this head configuration."""

    if n_heads % n_proxy_heads != 0:
        raise NotImplementedError("n_heads must be divisible by n_proxy_heads")
    if n_heads % n_kv_heads != 0:
        raise NotImplementedError("n_heads must be divisible by n_kv_heads")
    if n_proxy_heads < n_kv_heads or n_proxy_heads % n_kv_heads != 0:
        raise NotImplementedError(
            "MSA forward requires n_proxy_heads >= n_kv_heads and divisibility"
        )


def run_main_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    metadata: SparseAttentionMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run varlen selected attention and return its forward outputs."""

    if q.device.type != "cuda":
        raise ValueError("MSA forward requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"MSA forward supports fp16/bf16, got {q.dtype}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, S, D]")

    _, n_heads, seq_len, head_dim = map(int, q.shape)
    n_kv_heads = int(k.shape[1])
    n_proxy_heads = metadata.n_proxy_heads
    if seq_len % BLOCK_SIZE:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if head_dim != 128:
        raise NotImplementedError(f"MSA forward requires D=128, got {head_dim}")
    _validate_head_tiling(n_heads, n_kv_heads, n_proxy_heads)

    output_tokens, lse_main = flash_attn_block_sparse_forward(
        q=q.detach().transpose(1, 2),
        k=k.detach().transpose(1, 2),
        v=v.detach().transpose(1, 2),
        selection=metadata.selection,
        use_main_schedule=True,
        group_size=n_heads // n_proxy_heads,
        query_block_size=metadata.query_block_size,
        key_block_size=BLOCK_SIZE,
        softmax_scale=float(scale),
        document_ids=metadata.document_ids,
    )
    o_main = output_tokens.transpose(1, 2)
    kl_loss = torch.zeros((), dtype=torch.float32, device=q.device)
    return o_main, lse_main, kl_loss
