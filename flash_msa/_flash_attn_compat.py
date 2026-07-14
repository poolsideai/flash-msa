"""Compatibility hooks for FlashAttention 3 and FlashAttention 4."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from flash_msa.reverse_index_cuda import BlockSparseSelection


@lru_cache(maxsize=1)
def _fa4_fixed_func():
    try:
        from flash_attn.cute.interface import flash_attn_func
    except ModuleNotFoundError as exc:
        if exc.name == "flash_attn" or (
            exc.name is not None and exc.name.startswith("flash_attn.cute")
        ):
            return None
        raise
    return flash_attn_func


@lru_cache(maxsize=1)
def _fa4_varlen_func():
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func
    except ModuleNotFoundError as exc:
        if exc.name == "flash_attn" or (
            exc.name is not None and exc.name.startswith("flash_attn.cute")
        ):
            return None
        raise
    return flash_attn_varlen_func


@lru_cache(maxsize=1)
def _fa3_varlen_func():
    try:
        from flash_attn_interface import flash_attn_varlen_func
    except ModuleNotFoundError as exc:
        if exc.name == "flash_attn_interface":
            return None
        raise
    return flash_attn_varlen_func


def flash_attn_supports_narrow_value_dim() -> bool:
    """Whether the active backend accepts the 8-wide dummy V used for LSE-only calls."""

    return _fa4_varlen_func() is not None


def flash_attn_lse_value_dim(device: torch.device) -> int:
    """Return the smallest FA4 value head accepted for an LSE-only call."""

    major, minor = torch.cuda.get_device_capability(device)
    if major == 9:
        return 8
    if major in (10, 11):
        return 16
    raise NotImplementedError(
        f"Proxy LSE dummy V is not configured for SM{major}{minor}"
    )


@lru_cache(maxsize=1)
def flash_attn_causal_document_mask() -> Callable:
    """Build the flattened same-document causal mask used by dense warmup."""

    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils as cute_utils

    @cute.jit
    def _mask(_batch, _head, q_idx, kv_idx, seqlen_info, aux_tensors):
        document_ids = aux_tensors[0]
        q_global = q_idx + seqlen_info.offset_q
        kv_global = kv_idx + seqlen_info.offset_k
        q_pos = cute.make_rmem_tensor(1, cutlass.Int32)
        kv_pos = cute.make_rmem_tensor(1, cutlass.Int32)
        q_pos.store(q_global)
        kv_pos.store(kv_global)
        q_document = cute_utils.scalar_to_ssa(
            document_ids[q_pos[0]],
            cutlass.Int32,
        )
        kv_document = cute_utils.scalar_to_ssa(
            document_ids[kv_pos[0]],
            cutlass.Int32,
        )
        return (kv_idx <= q_idx) & (q_document == kv_document)

    return _mask


_block_sparse_mask_cache: dict[tuple[int, int, int, bool], Callable] = {}


def _block_sparse_mask(
    group_size: int,
    block_size: int,
    top_k_blocks: int,
    with_documents: bool,
) -> Callable:
    """Build the exact token-level predicate for scheduled FA4 blocks."""

    key = (group_size, block_size, top_k_blocks, with_documents)
    if key not in _block_sparse_mask_cache:
        import cutlass
        import cutlass.cute as cute
        from flash_attn.cute import utils as cute_utils
        from flash_attn.cute.block_sparsity import fast_sampling

        def _member(batch, head, q_idx, kv_idx, aux_tensors):
            selected, counts = aux_tensors[:2]
            proxy_head = head[0] // group_size
            kv_block = kv_idx[0] // block_size
            target = cute_utils.scalar_to_ssa(kv_block, cutlass.Int32)
            count = cute_utils.scalar_to_ssa(
                counts[batch[0], proxy_head, q_idx[0]],
                cutlass.Int32,
            )
            picked = cute_utils.scalar_to_ssa(
                selected[batch[0], proxy_head, q_idx[0], 0],
                cutlass.Int32,
            )
            member = (cute_utils.scalar_to_ssa(0, cutlass.Int32) < count) & (
                picked == target
            )
            for slot in range(1, top_k_blocks):
                picked = cute_utils.scalar_to_ssa(
                    selected[batch[0], proxy_head, q_idx[0], slot],
                    cutlass.Int32,
                )
                valid = cute_utils.scalar_to_ssa(slot, cutlass.Int32) < count
                member = member | (valid & (picked == target))
            return member

        if with_documents:

            @fast_sampling
            @cute.jit
            def _mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
                offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
                causal = kv_idx <= (
                    q_idx + cute_utils.scalar_to_ssa(offset, cutlass.Int32)
                )
                document_ids = aux_tensors[2]
                query_document = cute_utils.scalar_to_ssa(
                    document_ids[batch[0], q_idx[0]],
                    cutlass.Int32,
                )
                key_document = cute_utils.scalar_to_ssa(
                    document_ids[batch[0], kv_idx[0]],
                    cutlass.Int32,
                )
                return (
                    causal
                    & _member(batch, head, q_idx, kv_idx, aux_tensors)
                    & (query_document == key_document)
                )

        else:

            @fast_sampling
            @cute.jit
            def _mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
                offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
                causal = kv_idx <= (
                    q_idx + cute_utils.scalar_to_ssa(offset, cutlass.Int32)
                )
                return causal & _member(
                    batch,
                    head,
                    q_idx,
                    kv_idx,
                    aux_tensors,
                )

        _block_sparse_mask_cache[key] = _mask
    return _block_sparse_mask_cache[key]


def flash_attn_block_sparse_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selection: BlockSparseSelection,
    use_main_schedule: bool,
    group_size: int,
    query_block_size: int,
    key_block_size: int,
    softmax_scale: float,
    document_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact block-sparse attention with FA4."""

    flash_attn_func = _fa4_fixed_func()
    if flash_attn_func is None:
        raise ModuleNotFoundError(
            "Flash-MSA sparse attention requires FlashAttention 4"
        )
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    if use_main_schedule:
        block_counts = selection.main_block_counts
        block_indices = selection.main_block_indices
        empty_block_counts = selection.main_empty_block_counts
    else:
        block_counts = selection.proxy_block_counts
        block_indices = selection.proxy_block_indices
        empty_block_counts = selection.proxy_empty_block_counts

    selected_aux = selection.indices.to(torch.int32).contiguous()
    selected_aux.__leading_dim__ = 3
    count_aux = selection.counts.to(torch.int32).contiguous()
    count_aux.__leading_dim__ = 2
    aux_tensors = [selected_aux, count_aux]
    if document_ids.numel():
        document_aux = document_ids.to(torch.int32).contiguous()
        document_aux.__leading_dim__ = 1
        aux_tensors.append(document_aux)

    block_sparse_tensors = BlockSparseTensorsTorch(
        mask_block_cnt=block_counts,
        mask_block_idx=block_indices,
        full_block_cnt=empty_block_counts,
        full_block_idx=block_indices[:1, :1, :1, :1],
        block_size=(query_block_size, key_block_size),
    )
    out, lse = flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
        pack_gqa=False,
        mask_mod=_block_sparse_mask(
            group_size,
            key_block_size,
            int(selection.indices.shape[-1]),
            bool(document_ids.numel()),
        ),
        aux_tensors=aux_tensors,
        block_sparse_tensors=block_sparse_tensors,
        return_lse=True,
    )
    return out, lse


def flash_attn_varlen_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    mask_mod: Callable | None = None,
    aux_tensors: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(out, lse)`` for FlashAttention 3 or 4."""

    flash_attn_varlen_func = _fa3_varlen_func()
    if flash_attn_varlen_func is not None:
        if mask_mod is not None:
            raise NotImplementedError("document masking requires FlashAttention 4")
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            return_attn_probs=True,
        )
        return out, lse

    flash_attn_varlen_func = _fa4_varlen_func()
    if flash_attn_varlen_func is None:
        raise ModuleNotFoundError(
            "Flash-MSA requires FlashAttention 3 or 4 varlen support"
        )

    out, lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        return_lse=True,
    )
    return out, lse
