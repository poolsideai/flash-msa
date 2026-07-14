"""Warmup MSA tiled CuTeDSL backward.

Warmup attention uses the full causal block mask.  This module builds that
schedule and reuses the fused selected-edge CuTeDSL backward over every causal
block edge.  Proxy forward is not run during forward; proxy LSE is computed in
backward before launching the tiled gradient kernel.
"""

import torch

BLOCK_SIZE = 128
_SCHEDULE_CACHE: dict[
    tuple[int, int, int, int, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}


def _device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _dense_causal_schedule(
    *,
    batch: int,
    n_proxy_heads: int,
    seq_len: int,
    query_chunk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reverse-index tasks for all causal block edges."""

    if seq_len % BLOCK_SIZE != 0:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    key = (
        _device_index(device),
        int(batch),
        int(n_proxy_heads),
        int(seq_len),
        int(query_chunk),
        int(BLOCK_SIZE),
    )
    if key in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[key]

    from flash_msa.reverse_index_cuda import build_dense_causal_schedule_cuda

    task_meta, task_qids = build_dense_causal_schedule_cuda(
        batch=int(batch),
        n_proxy_heads=int(n_proxy_heads),
        seq_len=int(seq_len),
        query_chunk=int(query_chunk),
        device=device,
    )
    _SCHEDULE_CACHE[key] = (task_meta, task_qids)
    return task_meta, task_qids


def _lse_from_flash(
    lse: torch.Tensor,
    *,
    batch: int,
    n_heads: int,
    seq_len: int,
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


def _run_proxy_lse_flash(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    *,
    scale: float,
    document_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute dense causal proxy LSE for the KL-gradient branch."""

    from flash_msa._flash_attn_compat import (
        flash_attn_causal_document_mask,
        flash_attn_lse_value_dim,
        flash_attn_varlen_forward,
    )

    batch, n_proxy_heads, seq_len, head_dim = q_proxy.shape
    n_proxy_kv_heads = k_proxy.shape[1]
    q_pack = (
        q_proxy.transpose(1, 2)
        .contiguous()
        .view(batch * seq_len, n_proxy_heads, head_dim)
    )
    k_pack = (
        k_proxy.transpose(1, 2)
        .contiguous()
        .view(batch * seq_len, n_proxy_kv_heads, head_dim)
    )
    cu_seqlens = torch.arange(
        batch + 1, device=q_proxy.device, dtype=torch.int32
    ) * int(seq_len)

    value = torch.zeros(
        (*k_pack.shape[:-1], flash_attn_lse_value_dim(k_proxy.device)),
        device=k_proxy.device,
        dtype=k_proxy.dtype,
    )
    _out, lse = flash_attn_varlen_forward(
        q=q_pack,
        k=k_pack,
        v=value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=int(seq_len),
        max_seqlen_k=int(seq_len),
        softmax_scale=float(scale),
        causal=False,
        mask_mod=flash_attn_causal_document_mask(),
        aux_tensors=[document_ids.reshape(-1)],
    )
    return _lse_from_flash(lse, batch=batch, n_heads=n_proxy_heads, seq_len=seq_len)
