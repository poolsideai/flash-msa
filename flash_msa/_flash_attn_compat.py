"""FlashAttention 4 adapters for document-packed MSA training."""

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


_block_sparse_bitset_mask_cache: dict[tuple[int, int], Callable] = {}


def _block_sparse_bitset_mask(
    group_size: int,
    block_size: int,
) -> Callable:
    """Build an exact predicate backed by packed token-block membership bits."""

    key = (group_size, block_size)
    if key not in _block_sparse_bitset_mask_cache:
        import cutlass
        import cutlass.cute as cute
        from flash_attn.cute import utils as cute_utils
        from flash_attn.cute.block_sparsity import fast_sampling

        def _member(batch, head, q_idx, kv_idx, aux_tensors):
            membership = aux_tensors[0]
            proxy_head = head[0] // group_size
            kv_block = kv_idx[0] // block_size
            word_idx = kv_block // 32
            bit_idx = kv_block - word_idx * 32
            word = cute_utils.scalar_to_ssa(
                membership[batch[0], proxy_head, q_idx[0], word_idx],
                cutlass.Uint32,
            )
            bit = cute_utils.shl_u32(
                cutlass.Uint32(1),
                cutlass.Uint32(bit_idx),
            )
            return cutlass.Boolean(cute_utils.ssa_to_scalar(word) & bit)

        @fast_sampling
        @cute.jit
        def _mask(batch, head, q_idx, kv_idx, seqlen_info, aux_tensors):
            offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
            causal = kv_idx <= (q_idx + cute_utils.scalar_to_ssa(offset, cutlass.Int32))
            document_ids = aux_tensors[1]
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

        _block_sparse_bitset_mask_cache[key] = _mask
    return _block_sparse_bitset_mask_cache[key]


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

    membership_aux = selection.membership_bits
    membership_aux.__leading_dim__ = 3
    document_aux = document_ids.to(torch.int32).contiguous()
    document_aux.__leading_dim__ = 1
    aux_tensors = [membership_aux, document_aux]

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
        mask_mod=_block_sparse_bitset_mask(
            group_size,
            key_block_size,
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
    """Return `(out, lse)` from FA4 varlen attention."""

    flash_attn_varlen_func = _fa4_varlen_func()
    if flash_attn_varlen_func is None:
        raise ModuleNotFoundError("Flash-MSA requires FlashAttention 4")

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
