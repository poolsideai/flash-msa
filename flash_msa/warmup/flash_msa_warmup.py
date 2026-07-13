"""Python autograd boundary for warmup MSA attention."""

import torch

from flash_msa.reverse_index_cuda import resolve_document_ids


def _validate_inputs(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
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
            "q_proxy, k_proxy, q, k, and v must agree on batch, sequence, "
            "head dimension, and KV head count"
        )
    if head_dim != 128:
        raise NotImplementedError("warmup MSA kernel requires D=128")
    if n_proxy_heads % n_proxy_kv_heads != 0:
        raise ValueError("n_proxy_heads must be divisible by n_proxy_kv_heads")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if n_heads % n_proxy_heads != 0:
        raise ValueError("n_heads must be divisible by n_proxy_heads")
    if n_proxy_heads < n_kv_heads or n_proxy_heads % n_kv_heads != 0:
        raise NotImplementedError(
            "warmup MSA kernel requires n_proxy_heads >= n_kv_heads and divisibility"
        )


class _DenseMainAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_proxy_heads: int,
        scale: float,
        document_ids: torch.Tensor,
    ):
        from flash_msa.warmup.msa_forward_cutedsl_warmup import run_main_forward

        o_main, lse_main, _kl_loss = run_main_forward(
            q,
            k,
            v,
            scale=float(scale),
            document_ids=document_ids,
        )
        ctx.save_for_backward(q, k, v, lse_main, o_main, document_ids)
        ctx.n_proxy_heads = int(n_proxy_heads)
        ctx.scale = float(scale)
        ctx.mark_non_differentiable(lse_main)
        ctx.set_materialize_grads(False)
        out = o_main.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        return out, lse_main

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None, _grad_lse: torch.Tensor | None):
        q, k, v, lse_main, o_main, document_ids = ctx.saved_tensors
        if grad_out is None:
            grad_o_main = torch.zeros_like(o_main)
        else:
            grad_o_main = (
                grad_out.reshape(q.shape[0], q.shape[2], q.shape[1], q.shape[3])
                .transpose(1, 2)
                .contiguous()
            )
        from flash_msa.msa_backward_cutedsl import (
            _derive_head_tiling,
            run_main_backward,
        )
        from flash_msa.warmup.msa_backward_cutedsl_warmup import _dense_causal_schedule

        _main_per_proxy, query_chunk, _rows_per_task, _proxy_query_rows = (
            _derive_head_tiling(q.shape[1], k.shape[1], ctx.n_proxy_heads)
        )
        task_meta, task_qids = _dense_causal_schedule(
            batch=q.shape[0],
            n_proxy_heads=ctx.n_proxy_heads,
            seq_len=q.shape[2],
            query_chunk=2 * query_chunk,
            device=q.device,
        )
        dq, dk, dv = run_main_backward(
            q,
            k,
            v,
            grad_o_main,
            lse_main,
            o_main,
            task_meta,
            task_qids,
            document_ids,
            n_proxy_heads=ctx.n_proxy_heads,
            scale=ctx.scale,
        )
        return dq, dk, dv, None, None, None


def dense_main_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_proxy_heads: int,
    scale: float,
    document_list: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run dense warmup attention with a main-gradient-only backward."""

    document_ids = resolve_document_ids(q, document_list, cu_seqlens)
    return _DenseMainAttentionFunction.apply(
        q,
        k,
        v,
        int(n_proxy_heads),
        float(scale),
        document_ids,
    )


def dense_proxy_vjp(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    lse_main: torch.Tensor,
    scale: float,
    document_list: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    kl_metric: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense proxy cotangents for an immediate Indexer projection VJP."""

    from flash_msa.msa_backward_cutedsl import _derive_head_tiling, run_proxy_backward
    from flash_msa.warmup.msa_backward_cutedsl_warmup import (
        _dense_causal_schedule,
        _run_proxy_lse_flash,
    )

    document_ids = resolve_document_ids(q, document_list, cu_seqlens)
    _validate_inputs(q_proxy, k_proxy, q, k, k)
    record_kl_metric = kl_metric is not None
    if kl_metric is None:
        kl_metric = torch.empty((), device=q.device, dtype=torch.float32)
    else:
        assert kl_metric.shape == () and kl_metric.dtype == torch.float32
        kl_metric.zero_()
    _main_per_proxy, query_chunk, _rows_per_task, _proxy_query_rows = (
        _derive_head_tiling(q.shape[1], k.shape[1], q_proxy.shape[1])
    )
    task_meta, task_qids = _dense_causal_schedule(
        batch=q.shape[0],
        n_proxy_heads=q_proxy.shape[1],
        seq_len=q.shape[2],
        query_chunk=2 * query_chunk,
        device=q.device,
    )
    lse_proxy = _run_proxy_lse_flash(
        q_proxy,
        k_proxy,
        scale=float(scale),
        document_ids=document_ids,
    )
    normalization = 1.0 / float(q.shape[0] * q_proxy.shape[1] * q.shape[2])
    return run_proxy_backward(
        q_proxy,
        k_proxy,
        q,
        k,
        lse_main,
        lse_proxy,
        task_meta,
        task_qids,
        document_ids,
        kl_metric,
        scale=float(scale),
        normalization=normalization,
        record_kl_metric=record_kl_metric,
    )


class _WarmupSparseAttentionFunction(torch.autograd.Function):
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
        _ = int(top_k)
        _validate_inputs(q_proxy, k_proxy, q, k, v)
        if q.device.type != "cuda":
            raise RuntimeError("warmup MSA forward requires CUDA tensors")

        from flash_msa.warmup.msa_forward_cutedsl_warmup import run_main_forward

        o_main, lse_main, kl_loss = run_main_forward(
            q, k, v, scale=float(scale), document_ids=document_ids
        )
        out = o_main.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

        ctx.save_for_backward(q_proxy, k_proxy, q, k, v, lse_main, o_main, document_ids)
        ctx.scale = float(scale)
        ctx.kl_metric = kl_metric
        ctx.record_kl_metric = bool(record_kl_metric)
        if ctx.record_kl_metric:
            kl_metric.zero_()
        ctx.set_materialize_grads(False)
        return out, kl_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor | None, grad_kl: torch.Tensor | None):
        q_proxy, k_proxy, q, k, v, lse_main, o_main, document_ids = ctx.saved_tensors

        from flash_msa.warmup.msa_backward_cutedsl_warmup import run_warmup_backward

        dq_proxy, dk_proxy, dq, dk, dv = run_warmup_backward(
            q_proxy,
            k_proxy,
            q,
            k,
            v,
            lse_main,
            o_main,
            grad_out,
            grad_kl,
            document_ids,
            ctx.kl_metric,
            scale=ctx.scale,
            record_kl_metric=ctx.record_kl_metric,
        )
        return dq_proxy, dk_proxy, dq, dk, dv, None, None, None, None, None


def sparse_attention_warmup(
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
    """Compute dense causal warmup attention and proxy-KL gradients."""

    document_list = resolve_document_ids(q, document_list, cu_seqlens)
    record_kl_metric = kl_metric is not None
    if kl_metric is None:
        kl_metric = torch.empty((), device=q.device, dtype=torch.float32)
    else:
        assert kl_metric.shape == () and kl_metric.dtype == torch.float32
    return _WarmupSparseAttentionFunction.apply(
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
