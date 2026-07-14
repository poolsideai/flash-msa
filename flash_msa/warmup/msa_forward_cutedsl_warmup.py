"""Warmup MSA forward path.

This forward intentionally runs only dense causal main attention.  Proxy
attention is recomputed in backward for the dense causal KL objective.
"""

import torch

from flash_msa._flash_attn_compat import (
    flash_attn_causal_document_mask,
    flash_attn_varlen_forward,
)


def _lse_from_flash(
    lse: torch.Tensor, *, batch: int, n_heads: int, seq_len: int
) -> torch.Tensor:
    if lse.shape == (n_heads, batch * seq_len):
        return (
            lse.transpose(0, 1)
            .contiguous()
            .view(batch, seq_len, n_heads)
            .permute(0, 2, 1)
        )
    if lse.shape == (batch, n_heads, seq_len):
        return lse.contiguous()
    if lse.shape == (n_heads, batch, seq_len):
        return lse.permute(1, 0, 2).contiguous()
    raise RuntimeError(f"unexpected FlashAttention LSE shape: {tuple(lse.shape)}")


def run_main_forward_token_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    document_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run dense causal main attention without changing the token-major layout."""

    if q.device.type != "cuda":
        raise ValueError("warmup MSA forward requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"warmup MSA forward supports fp16/bf16, got {q.dtype}")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (B, S, H, D)")

    batch, seq_len, n_heads, head_dim = q.shape
    if k.shape[:2] != (batch, seq_len) or v.shape[:2] != (batch, seq_len):
        raise ValueError(
            "q, k, and v must agree on batch, sequence, and head dimension"
        )
    if v.shape != k.shape:
        raise ValueError("k and v must have the same shape")
    if k.shape[-1] != head_dim:
        raise ValueError(
            "q, k, and v must agree on batch, sequence, and head dimension"
        )
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("token-major q, k, and v must be contiguous")
    n_kv_heads = k.shape[2]
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")

    q_pack = q.view(batch * seq_len, n_heads, head_dim)
    k_pack = k.view(batch * seq_len, n_kv_heads, head_dim)
    v_pack = v.view(batch * seq_len, n_kv_heads, head_dim)
    cu_seqlens = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * int(
        seq_len
    )

    out, lse = flash_attn_varlen_forward(
        q=q_pack,
        k=k_pack,
        v=v_pack,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=int(seq_len),
        max_seqlen_k=int(seq_len),
        softmax_scale=float(scale),
        causal=False,
        mask_mod=flash_attn_causal_document_mask(),
        aux_tensors=[document_ids.reshape(-1)],
    )

    o_main = out.view(batch, seq_len, n_heads, head_dim)
    lse_main = _lse_from_flash(lse, batch=batch, n_heads=n_heads, seq_len=seq_len)
    kl_loss = torch.zeros((), dtype=torch.float32, device=q.device)
    return o_main, lse_main, kl_loss
