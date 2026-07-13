"""Block-sparse top-k GQA attention training entrypoint.

This module owns the Python boundary for the native fused MSA kernels.
"""

import torch

from flash_msa.msa_select_cutedsl import compute_proxy_lse, select_blocks
from flash_msa.msa_backward_cutedsl import (
    _derive_head_tiling,
    run_fused_backward,
    run_main_backward,
    run_proxy_backward,
)
from flash_msa.msa_forward_cutedsl import run_main_forward
from flash_msa.reverse_index_cuda import (
    SparseAttentionMetadata,
    build_sparse_attention_metadata_cuda,
    resolve_document_ids,
)

BLOCK_SIZE = 128


def _validate_inputs(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
) -> tuple[int, int]:
    if (
        q_proxy.ndim != 4
        or k_proxy.ndim != 4
        or q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
    ):
        raise ValueError("all attention tensors must have shape (B, H, S, D)")

    b, n_proxy_heads, s, head_dim = q_proxy.shape
    bk, n_proxy_kv_heads, sk, dk = k_proxy.shape
    bq, n_heads, sq, dq = q.shape
    bm, n_kv_heads, sm, dm = k.shape
    bv, n_v_heads, sv, dv = v.shape

    if not (
        b == bk == bq == bm == bv
        and s == sk == sq == sm == sv
        and head_dim == dk == dq == dm == dv
        and n_kv_heads == n_v_heads
    ):
        raise ValueError(
            "q_proxy, k_proxy, q, k, and v must agree on batch, sequence, head dimension, and KV head count"
        )

    if s % BLOCK_SIZE != 0:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}, got {s}")
    if top_k % BLOCK_SIZE != 0:
        raise ValueError(f"top_k must be divisible by {BLOCK_SIZE}, got {top_k}")
    if n_proxy_heads % n_proxy_kv_heads != 0:
        raise ValueError("n_proxy_heads must be divisible by n_proxy_kv_heads")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if n_heads % n_proxy_heads != 0:
        raise ValueError("n_heads must be divisible by n_proxy_heads")
    if q_proxy.shape[-1] != 128:
        raise NotImplementedError("Headdim must be 128")

    num_blocks = s // BLOCK_SIZE
    top_k_blocks = int(top_k) // BLOCK_SIZE
    if not 1 <= top_k_blocks <= num_blocks:
        raise ValueError(
            f"top_k selects {top_k_blocks} blocks, but sequence has {num_blocks} blocks"
        )
    if top_k_blocks > 32:
        raise NotImplementedError("No more than 32 blocks / topk=4096 supported")

    return num_blocks, top_k_blocks


def prepare_sparse_attention(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    scale: float,
    document_list: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
) -> SparseAttentionMetadata:
    """Select blocks and pack the immutable schedule used by forward and backward."""

    document_ids = resolve_document_ids(q, document_list, cu_seqlens)
    num_blocks, top_k_blocks = _validate_inputs(
        q_proxy,
        k_proxy,
        q,
        k,
        v,
        int(top_k),
    )
    block_indices = select_blocks(
        q_proxy,
        k_proxy,
        document_ids=document_ids,
        scale=float(scale),
        num_blocks=num_blocks,
        top_k_blocks=top_k_blocks,
    )
    _main_per_proxy, query_chunk, _rows_per_task, _proxy_rows = _derive_head_tiling(
        q.shape[1],
        k.shape[1],
        q_proxy.shape[1],
    )
    return build_sparse_attention_metadata_cuda(
        block_indices,
        backward_query_chunk=2 * query_chunk,
        document_ids=document_ids,
    )


def _restore_metadata(
    tensors: tuple[torch.Tensor, ...],
    shape: tuple[int, int, int, int],
) -> SparseAttentionMetadata:
    return SparseAttentionMetadata(*tensors, *shape)


class _SparseMainAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        metadata: SparseAttentionMetadata,
    ):
        o_main, lse_main, _kl_loss = run_main_forward(
            q,
            k,
            v,
            scale=float(scale),
            metadata=metadata,
        )
        metadata_tensors = (
            metadata.task_meta,
            metadata.task_qids,
            metadata.remote_destinations,
            metadata.remote_positions,
            metadata.remote_cu_seqlens,
            metadata.document_ids,
            metadata.remote_q_document_ids,
            metadata.remote_k_document_ids,
        )
        ctx.save_for_backward(q, k, v, lse_main, o_main, *metadata_tensors)
        ctx.metadata_shape = (
            metadata.batch,
            metadata.n_proxy_heads,
            metadata.seq_len,
            metadata.top_k_blocks,
        )
        ctx.scale = float(scale)
        ctx.mark_non_differentiable(lse_main)
        ctx.set_materialize_grads(False)
        out = o_main.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        return out, lse_main

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None, _grad_lse: torch.Tensor | None):
        q, k, v, lse_main, o_main, *metadata_tensors = ctx.saved_tensors
        if grad_out is None:
            grad_o_main = torch.zeros_like(o_main)
        else:
            grad_o_main = (
                grad_out.reshape(q.shape[0], q.shape[2], q.shape[1], q.shape[3])
                .transpose(1, 2)
                .contiguous()
            )
        metadata = _restore_metadata(tuple(metadata_tensors), ctx.metadata_shape)
        dq, dk, dv = run_main_backward(
            q,
            k,
            v,
            grad_o_main,
            lse_main,
            o_main,
            metadata.task_meta,
            metadata.task_qids,
            metadata.document_ids,
            n_proxy_heads=metadata.n_proxy_heads,
            scale=ctx.scale,
        )
        return dq, dk, dv, None, None


def sparse_main_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: SparseAttentionMetadata,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run main sparse attention with a main-gradient-only autograd backward."""

    return _SparseMainAttentionFunction.apply(q, k, v, float(scale), metadata)


def sparse_proxy_vjp(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    lse_main: torch.Tensor,
    metadata: SparseAttentionMetadata,
    scale: float,
    *,
    kl_metric: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return proxy-output cotangents for an immediate Indexer projection VJP."""

    record_kl_metric = kl_metric is not None
    if kl_metric is None:
        kl_metric = torch.empty((), device=q.device, dtype=torch.float32)
    else:
        assert kl_metric.shape == () and kl_metric.dtype == torch.float32
        kl_metric.zero_()
    lse_proxy = compute_proxy_lse(
        q_proxy,
        k_proxy,
        scale=float(scale),
        metadata=metadata,
    )
    normalization = 1.0 / float(q.shape[0] * q_proxy.shape[1] * q.shape[2])
    return run_proxy_backward(
        q_proxy,
        k_proxy,
        q,
        k,
        lse_main,
        lse_proxy,
        metadata.task_meta,
        metadata.task_qids,
        metadata.document_ids,
        kl_metric,
        scale=float(scale),
        normalization=normalization,
        record_kl_metric=record_kl_metric,
    )


def _run_fused_selected_edge_backward(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse_main: torch.Tensor,
    o_main: torch.Tensor,
    grad_o_main: torch.Tensor,
    grad_kl: torch.Tensor | None,
    metadata: SparseAttentionMetadata,
    kl_metric: torch.Tensor,
    *,
    scale: float,
    record_kl_metric: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the native reverse-index fused selected-edge backward."""

    bsz, n_heads, seq_len, _ = q.shape
    n_proxy_heads = q_proxy.shape[1]

    lse_proxy = (
        compute_proxy_lse(
            q_proxy,
            k_proxy,
            scale=float(scale),
            metadata=metadata,
        )
        if grad_kl is not None or record_kl_metric
        else torch.empty(
            (bsz, n_proxy_heads, seq_len),
            device=q.device,
            dtype=torch.float32,
        )
    )

    delta_main = (o_main.float() * grad_o_main.float()).sum(dim=-1)
    # Proxy KL gradients are linear in the upstream scalar.  Run the native
    # kernel at the static normalization scale, then apply the CUDA scalar to
    # only the proxy gradients.  This avoids the synchronizing ``.item()`` that
    # would otherwise be needed for a by-value CuTeDSL kernel argument.
    normalization = 1.0 / float(bsz * n_proxy_heads * seq_len)
    proxy_grad_scale = 0.0 if grad_kl is None else normalization
    dq_proxy, dk_proxy, dq, dk, dv = run_fused_backward(
        q_proxy,
        k_proxy,
        q,
        k,
        v,
        grad_o_main,
        lse_main,
        lse_proxy,
        delta_main,
        metadata.task_meta,
        metadata.task_qids,
        metadata.document_ids,
        kl_metric,
        scale=float(scale),
        grad_kl_scale=proxy_grad_scale,
        kl_metric_scale=normalization if record_kl_metric else 0.0,
        record_kl_metric=record_kl_metric,
    )
    if grad_kl is not None:
        proxy_multiplier = grad_kl.detach().to(device=q.device, dtype=dq_proxy.dtype)
        dq_proxy = dq_proxy * proxy_multiplier
        dk_proxy = dk_proxy * proxy_multiplier.to(dtype=dk_proxy.dtype)
    return dq_proxy, dk_proxy, dq, dk, dv


class _SparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q_proxy: torch.Tensor,
        k_proxy: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        top_k: int,
        scale: float,
        document_ids: torch.Tensor,
        kl_metric: torch.Tensor,
        record_kl_metric: bool,
    ):
        b, n_proxy_heads, s, head_dim = q_proxy.shape
        n_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        num_blocks, top_k_blocks = _validate_inputs(
            q_proxy, k_proxy, q, k, v, int(top_k)
        )
        block_indices = select_blocks(
            q_proxy,
            k_proxy,
            document_ids=document_ids,
            scale=float(scale),
            num_blocks=num_blocks,
            top_k_blocks=top_k_blocks,
        )

        _main_per_proxy, query_chunk, _rows_per_task, _proxy_rows = _derive_head_tiling(
            n_heads, n_kv_heads, n_proxy_heads
        )
        metadata = build_sparse_attention_metadata_cuda(
            block_indices,
            backward_query_chunk=2 * query_chunk,
            document_ids=document_ids,
        )

        o_main, lse_main, kl_loss = run_main_forward(
            q,
            k,
            v,
            scale=float(scale),
            metadata=metadata,
        )

        out = o_main.transpose(1, 2).reshape(b, s, -1)

        save_tensors = (
            q_proxy,
            k_proxy,
            q,
            k,
            v,
            lse_main,
            o_main,
            metadata.task_meta,
            metadata.task_qids,
            metadata.remote_destinations,
            metadata.remote_positions,
            metadata.remote_cu_seqlens,
            metadata.document_ids,
            metadata.remote_q_document_ids,
            metadata.remote_k_document_ids,
        )
        ctx.save_for_backward(*save_tensors)
        ctx.scale = float(scale)
        ctx.kl_metric = kl_metric
        ctx.record_kl_metric = bool(record_kl_metric)
        if ctx.record_kl_metric:
            kl_metric.zero_()
        ctx.metadata_shape = (
            metadata.batch,
            metadata.n_proxy_heads,
            metadata.seq_len,
            metadata.top_k_blocks,
        )
        ctx.set_materialize_grads(False)
        return out, kl_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None, grad_kl: torch.Tensor | None):
        (
            q_proxy,
            k_proxy,
            q,
            k,
            v,
            lse_main,
            o_main,
            task_meta,
            task_qids,
            remote_destinations,
            remote_positions,
            remote_cu_seqlens,
            document_ids,
            remote_q_document_ids,
            remote_k_document_ids,
        ) = ctx.saved_tensors
        batch, n_proxy_heads, seq_len, top_k_blocks = ctx.metadata_shape
        metadata = SparseAttentionMetadata(
            task_meta=task_meta,
            task_qids=task_qids,
            remote_destinations=remote_destinations,
            remote_positions=remote_positions,
            remote_cu_seqlens=remote_cu_seqlens,
            document_ids=document_ids,
            remote_q_document_ids=remote_q_document_ids,
            remote_k_document_ids=remote_k_document_ids,
            batch=batch,
            n_proxy_heads=n_proxy_heads,
            seq_len=seq_len,
            top_k_blocks=top_k_blocks,
        )

        if grad_out is None:
            grad_o_main = torch.zeros_like(o_main)
        else:
            b, s, _ = grad_out.shape
            grad_o_main = (
                grad_out.reshape(b, s, q.shape[1], q.shape[3])
                .transpose(1, 2)
                .contiguous()
            )

        dq_proxy, dk_proxy, dq, dk, dv = _run_fused_selected_edge_backward(
            q_proxy,
            k_proxy,
            q,
            k,
            v,
            lse_main,
            o_main,
            grad_o_main,
            grad_kl,
            metadata,
            ctx.kl_metric,
            scale=ctx.scale,
            record_kl_metric=ctx.record_kl_metric,
        )
        return dq_proxy, dk_proxy, dq, dk, dv, None, None, None, None, None


def sparse_attention(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    scale: float,
    document_list: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    kl_metric: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the attention output and the proxy-KL autograd placeholder.

    ``cu_seqlens`` contains flattened document offsets and must include every
    batch-row boundary. It is expanded entirely on CUDA so packed attention
    does not add a device-to-host synchronization.
    """
    document_list = resolve_document_ids(q, document_list, cu_seqlens)
    record_kl_metric = kl_metric is not None
    if kl_metric is None:
        kl_metric = torch.empty((), device=q.device, dtype=torch.float32)
    else:
        assert kl_metric.shape == () and kl_metric.dtype == torch.float32
    return _SparseAttentionFunction.apply(
        q_proxy,
        k_proxy,
        q,
        k,
        v,
        int(top_k),
        float(scale),
        document_list,
        kl_metric,
        record_kl_metric,
    )
