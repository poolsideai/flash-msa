"""Specialized Triton VJP for the paper-shaped sparse Indexer."""

import torch
import triton
import triton.language as tl

from flash_msa.reverse_index_cuda import SparseAttentionMetadata


@triton.jit
def _proxy_vjp_kernel(  # type: ignore[no-untyped-def]
    Q,
    K,
    Q_PROXY,
    K_PROXY,
    LSE_MAIN,
    DQ_PROXY,
    DK_PROXY,
    KL_OUT,
    UNION_IDX,
    UNION_CNT,
    MEMBERSHIP,
    DOCUMENT_IDS,
    main_scale,
    proxy_scale,
    normalization,
    sqb,
    sqh,
    sqn,
    sqd,
    skb,
    skh,
    skn,
    skd,
    spqb,
    spqh,
    spqn,
    spqd,
    spkb,
    spkn,
    spkd,
    slb,
    slh,
    sln,
    sdqb,
    sdqh,
    sdqn,
    sdqd,
    sdkb,
    sdkn,
    sdkd,
    suib,
    suih,
    suiq,
    suik,
    sucb,
    such,
    sucq,
    smemb,
    smemh,
    smemn,
    smemw,
    sdb,
    sdn,
    sequence_length,
    MAIN_PER_PROXY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RECORD_KL: tl.constexpr,
):
    query_block, proxy_head, batch = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    query_offsets = query_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    key_offsets = tl.arange(0, BLOCK_SIZE)
    dim_offsets = tl.arange(0, HEAD_DIM)
    valid_query = query_offsets < sequence_length

    q_proxy = tl.load(
        Q_PROXY
        + batch * spqb
        + proxy_head * spqh
        + query_offsets[:, None] * spqn
        + dim_offsets[None, :] * spqd,
        mask=valid_query[:, None],
        other=0.0,
    )
    query_documents = tl.load(
        DOCUMENT_IDS + batch * sdb + query_offsets * sdn,
        mask=valid_query,
        other=-1,
    )
    union_count = tl.load(
        UNION_CNT + batch * sucb + proxy_head * such + query_block * sucq,
    )

    proxy_max = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    proxy_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for union_offset in range(union_count):
        key_block = tl.load(
            UNION_IDX
            + batch * suib
            + proxy_head * suih
            + query_block * suiq
            + union_offset * suik,
        )
        key_positions = key_block * BLOCK_SIZE + key_offsets
        valid_key = key_positions < sequence_length
        k_proxy = tl.load(
            K_PROXY
            + batch * spkb
            + key_positions[:, None] * spkn
            + dim_offsets[None, :] * spkd,
            mask=valid_key[:, None],
            other=0.0,
        )
        proxy_scores = (
            tl.dot(q_proxy, tl.trans(k_proxy), input_precision="ieee") * proxy_scale
        ).to(tl.float32)
        membership_word = tl.load(
            MEMBERSHIP
            + batch * smemb
            + proxy_head * smemh
            + query_offsets * smemn
            + (key_block // 32) * smemw,
            mask=valid_query,
            other=0,
        )
        member = (membership_word & (1 << (key_block % 32))) != 0
        key_documents = tl.load(
            DOCUMENT_IDS + batch * sdb + key_positions * sdn,
            mask=valid_key,
            other=-2,
        )
        valid = (
            valid_query[:, None]
            & valid_key[None, :]
            & (query_offsets[:, None] >= key_positions[None, :])
            & member[:, None]
            & (query_documents[:, None] == key_documents[None, :])
        )
        proxy_scores = tl.where(valid, proxy_scores, -1.0e9)
        next_max = tl.maximum(proxy_max, tl.max(proxy_scores, axis=1))
        probability = tl.where(
            valid,
            tl.exp(proxy_scores - next_max[:, None]),
            0.0,
        )
        proxy_sum = proxy_sum * tl.exp(proxy_max - next_max) + tl.sum(
            probability,
            axis=1,
        )
        proxy_max = next_max
    proxy_lse = tl.where(
        proxy_sum > 0,
        proxy_max + tl.log(proxy_sum),
        float("inf"),
    )

    dq_proxy = tl.zeros((BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)
    kl_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for union_offset in range(union_count):
        key_block = tl.load(
            UNION_IDX
            + batch * suib
            + proxy_head * suih
            + query_block * suiq
            + union_offset * suik,
        )
        key_positions = key_block * BLOCK_SIZE + key_offsets
        valid_key = key_positions < sequence_length
        k_proxy = tl.load(
            K_PROXY
            + batch * spkb
            + key_positions[:, None] * spkn
            + dim_offsets[None, :] * spkd,
            mask=valid_key[:, None],
            other=0.0,
        )
        proxy_scores = (
            tl.dot(q_proxy, tl.trans(k_proxy), input_precision="ieee") * proxy_scale
        ).to(tl.float32)
        membership_word = tl.load(
            MEMBERSHIP
            + batch * smemb
            + proxy_head * smemh
            + query_offsets * smemn
            + (key_block // 32) * smemw,
            mask=valid_query,
            other=0,
        )
        member = (membership_word & (1 << (key_block % 32))) != 0
        key_documents = tl.load(
            DOCUMENT_IDS + batch * sdb + key_positions * sdn,
            mask=valid_key,
            other=-2,
        )
        valid = (
            valid_query[:, None]
            & valid_key[None, :]
            & (query_offsets[:, None] >= key_positions[None, :])
            & member[:, None]
            & (query_documents[:, None] == key_documents[None, :])
        )
        student = tl.where(
            valid,
            tl.exp(proxy_scores - proxy_lse[:, None]),
            0.0,
        )
        k_main = tl.load(
            K
            + batch * skb
            + proxy_head * skh
            + key_positions[:, None] * skn
            + dim_offsets[None, :] * skd,
            mask=valid_key[:, None],
            other=0.0,
        )
        teacher = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        for head_offset in range(MAIN_PER_PROXY):
            main_head = proxy_head * MAIN_PER_PROXY + head_offset
            q_main = tl.load(
                Q
                + batch * sqb
                + main_head * sqh
                + query_offsets[:, None] * sqn
                + dim_offsets[None, :] * sqd,
                mask=valid_query[:, None],
                other=0.0,
            )
            main_scores = (
                tl.dot(q_main, tl.trans(k_main), input_precision="ieee") * main_scale
            ).to(tl.float32)
            main_lse = tl.load(
                LSE_MAIN + batch * slb + main_head * slh + query_offsets * sln,
                mask=valid_query,
                other=0.0,
            )
            teacher += tl.where(
                valid,
                tl.exp(main_scores - main_lse[:, None]),
                0.0,
            )
        teacher /= MAIN_PER_PROXY
        gradient = tl.where(
            valid,
            (student - teacher) * normalization,
            0.0,
        ).to(tl.float32)
        dq_proxy += (
            tl.dot(gradient.to(k_proxy.dtype), k_proxy, input_precision="ieee")
            * proxy_scale
        ).to(tl.float32)
        dk_proxy = (
            tl.dot(
                tl.trans(gradient).to(q_proxy.dtype),
                q_proxy,
                input_precision="ieee",
            )
            * proxy_scale
        ).to(tl.float32)
        tl.atomic_add(
            DK_PROXY
            + batch * sdkb
            + key_positions[:, None] * sdkn
            + dim_offsets[None, :] * sdkd,
            dk_proxy,
            mask=valid_key[:, None],
        )
        if RECORD_KL:
            kl_term = tl.where(
                valid & (teacher > 0),
                teacher
                * (
                    tl.log(tl.maximum(teacher, 1e-30))
                    - (proxy_scores - proxy_lse[:, None])
                ),
                0.0,
            )
            kl_acc += tl.sum(kl_term, axis=1)
    tl.store(
        DQ_PROXY
        + batch * sdqb
        + proxy_head * sdqh
        + query_offsets[:, None] * sdqn
        + dim_offsets[None, :] * sdqd,
        dq_proxy,
        mask=valid_query[:, None],
    )
    if RECORD_KL:
        tl.atomic_add(KL_OUT, tl.sum(kl_acc) * normalization)


def run_triton_proxy_vjp(
    q_proxy: torch.Tensor,
    k_proxy: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    lse_main: torch.Tensor,
    metadata: SparseAttentionMetadata,
    *,
    scale: float,
    kl_metric: torch.Tensor,
    record_kl_metric: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the immediate sparse Indexer VJP without a proxy FA4 pass."""

    if not (
        q_proxy.shape[1] == k.shape[1]
        and k_proxy.shape[1] == 1
        and q_proxy.shape[-1] == q.shape[-1] == 128
        and metadata.query_block_size == 128
    ):
        raise NotImplementedError(
            "sparse Indexer VJP requires one proxy head per KV head, one shared "
            "proxy-K head, D=128, and 128-token query blocks"
        )
    batch, n_main_heads, sequence_length, head_dim = q.shape
    n_proxy_heads = q_proxy.shape[1]
    main_per_proxy = n_main_heads // n_proxy_heads
    selection = metadata.selection
    dq_proxy = torch.zeros_like(q_proxy, dtype=torch.float32)
    dk_proxy = torch.zeros_like(k_proxy, dtype=torch.float32)
    kl_metric.zero_()
    normalization = 1.0 / float(batch * n_proxy_heads * sequence_length)
    _proxy_vjp_kernel[(sequence_length // 128, n_proxy_heads, batch)](
        q,
        k,
        q_proxy,
        k_proxy,
        lse_main,
        dq_proxy,
        dk_proxy,
        kl_metric,
        selection.proxy_block_indices,
        selection.proxy_block_counts,
        selection.membership_bits,
        metadata.document_ids,
        float(scale),
        head_dim**-0.5,
        normalization,
        *q.stride(),
        *k.stride(),
        *q_proxy.stride(),
        k_proxy.stride(0),
        k_proxy.stride(2),
        k_proxy.stride(3),
        *lse_main.stride(),
        *dq_proxy.stride(),
        dk_proxy.stride(0),
        dk_proxy.stride(2),
        dk_proxy.stride(3),
        *selection.proxy_block_indices.stride(),
        *selection.proxy_block_counts.stride(),
        *selection.membership_bits.stride(),
        *metadata.document_ids.stride(),
        sequence_length,
        MAIN_PER_PROXY=main_per_proxy,
        BLOCK_SIZE=128,
        HEAD_DIM=head_dim,
        RECORD_KL=record_kl_metric,
        num_warps=8,
        num_stages=2,
    )
    return dq_proxy.to(q_proxy.dtype), dk_proxy.to(k_proxy.dtype)
