"""CUDA reverse-index builder for selected-block MSA backward."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch.utils.cpp_extension import load


BLOCK_SIZE = 128
QUERY_CHUNK = 32
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "csrc", "reverse_index_cuda.cu")
_EXT = None


def _cuda_arch_flag() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"-arch=sm_{major}{minor}"


def _load_ext():
    global _EXT
    if _EXT is None:
        python_bin = os.path.dirname(sys.executable)
        ninja_path = os.path.join(python_bin, "ninja")
        if os.path.exists(ninja_path):
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if python_bin not in path_entries:
                os.environ["PATH"] = os.pathsep.join(
                    [python_bin, os.environ.get("PATH", "")]
                )
        try:
            import nvidia.cu13
        except ModuleNotFoundError:
            extra_include_paths = None
        else:
            include_dir = Path(next(iter(nvidia.cu13.__path__))) / "include"
            extra_include_paths = [str(include_dir)] if include_dir.is_dir() else None
        _EXT = load(
            name="msa_reverse_index_ext",
            sources=[_SRC],
            extra_include_paths=extra_include_paths,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                _cuda_arch_flag(),
            ],
            verbose=False,
        )
    return _EXT


@dataclass
class ReverseIndexWorkspace:
    cache: dict[tuple[int, int, int, int, int, int, int], dict[str, torch.Tensor]] = (
        field(default_factory=dict)
    )

    def get(
        self,
        batch: int,
        n_proxy_heads: int,
        seq_len: int,
        top_k_blocks: int,
        query_chunk: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        num_blocks = seq_len // BLOCK_SIZE
        padded_tasks = (
            batch
            * n_proxy_heads
            * (((seq_len * top_k_blocks + query_chunk - 1) // query_chunk) + num_blocks)
        )
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (
            int(device_index),
            int(batch),
            int(n_proxy_heads),
            int(seq_len),
            int(top_k_blocks),
            int(query_chunk),
            int(padded_tasks),
        )
        if key not in self.cache:
            self.cache[key] = {
                "counts": torch.empty(
                    (batch * n_proxy_heads * num_blocks,),
                    device=device,
                    dtype=torch.int32,
                ),
                "write_counts": torch.empty(
                    (batch * n_proxy_heads * num_blocks,),
                    device=device,
                    dtype=torch.int32,
                ),
                "bucket_offsets": torch.empty(
                    (batch * n_proxy_heads * num_blocks,),
                    device=device,
                    dtype=torch.int32,
                ),
                "task_meta": torch.empty(
                    (padded_tasks, 4),
                    device=device,
                    dtype=torch.int32,
                ),
                "task_qids": torch.empty(
                    (padded_tasks, query_chunk),
                    device=device,
                    dtype=torch.int32,
                ),
            }
        return self.cache[key]


_DEFAULT_WORKSPACE = ReverseIndexWorkspace()


class BlockSparseSelection(NamedTuple):
    """Exact token selections and reusable FA4 forward schedules."""

    counts: torch.Tensor
    membership_bits: torch.Tensor
    proxy_block_counts: torch.Tensor
    proxy_block_indices: torch.Tensor
    proxy_empty_block_counts: torch.Tensor
    main_block_counts: torch.Tensor
    main_block_indices: torch.Tensor
    main_empty_block_counts: torch.Tensor


@dataclass(frozen=True)
class SparseAttentionMetadata:
    """Persistent exact selection and backward schedule for one forward."""

    selection: BlockSparseSelection
    task_meta: torch.Tensor
    task_qids: torch.Tensor
    document_ids: torch.Tensor
    query_block_size: int
    top_k_blocks: int

    @property
    def batch(self) -> int:
        return int(self.selection.counts.shape[0])

    @property
    def n_proxy_heads(self) -> int:
        return int(self.selection.counts.shape[1])

    @property
    def seq_len(self) -> int:
        return int(self.selection.counts.shape[2])

    @property
    def selected_block_counts(self) -> torch.Tensor:
        return self.selection.counts

    @property
    def scheduled_block_counts(self) -> torch.Tensor:
        return self.selection.main_block_counts


def build_block_sparse_selection(
    block_indices: torch.Tensor,
    *,
    n_main_heads: int,
    query_block_size: int,
) -> BlockSparseSelection:
    """Build exact token selections and reusable FA4 forward schedules."""

    batch, n_proxy_heads, seq_len, top_k_blocks = map(int, block_indices.shape)
    if n_main_heads % n_proxy_heads:
        raise ValueError("n_main_heads must be divisible by n_proxy_heads")
    if seq_len % query_block_size:
        raise ValueError(
            "sequence length must be divisible by the FA4 query block size"
        )
    num_blocks = seq_len // BLOCK_SIZE
    num_query_blocks = seq_len // query_block_size
    valid = block_indices < num_blocks
    counts = valid.sum(dim=-1, dtype=torch.int32).contiguous()
    membership_words = (num_blocks + 31) // 32
    membership_bits = torch.zeros(
        (batch, n_proxy_heads, seq_len, membership_words),
        dtype=torch.int32,
        device=block_indices.device,
    )
    word_indices = block_indices.clamp_max(num_blocks - 1).div(
        32,
        rounding_mode="floor",
    )
    bits = torch.bitwise_left_shift(
        torch.ones_like(block_indices),
        block_indices.remainder(32),
    )
    membership_bits.scatter_add_(
        3,
        word_indices.to(torch.int64),
        torch.where(valid, bits, 0),
    )

    union_slots = torch.where(valid, block_indices, num_blocks).view(
        batch,
        n_proxy_heads,
        num_query_blocks,
        query_block_size * top_k_blocks,
    )
    union = torch.zeros(
        (batch, n_proxy_heads, num_query_blocks, num_blocks + 1),
        dtype=torch.bool,
        device=block_indices.device,
    )
    union.scatter_(3, union_slots.to(torch.int64), True)
    union = union[..., :num_blocks]
    proxy_counts = union.sum(dim=-1, dtype=torch.int32).contiguous()
    max_union = min(num_blocks, query_block_size * top_k_blocks)
    priority = num_blocks - torch.arange(
        num_blocks,
        dtype=torch.int32,
        device=block_indices.device,
    )
    proxy_indices = (
        torch.topk(
            union.to(torch.int32) * priority,
            max_union,
            dim=-1,
            sorted=True,
        )
        .indices.to(torch.int32)
        .contiguous()
    )
    main_per_proxy = n_main_heads // n_proxy_heads
    main_counts = proxy_counts.repeat_interleave(main_per_proxy, dim=1).contiguous()
    return BlockSparseSelection(
        counts=counts,
        membership_bits=membership_bits,
        proxy_block_counts=proxy_counts,
        proxy_block_indices=proxy_indices,
        proxy_empty_block_counts=torch.zeros_like(proxy_counts),
        main_block_counts=main_counts,
        main_block_indices=proxy_indices.repeat_interleave(
            main_per_proxy,
            dim=1,
        ).contiguous(),
        main_empty_block_counts=torch.zeros(
            (batch, n_main_heads, num_query_blocks),
            dtype=torch.int32,
            device=block_indices.device,
        ),
    )


def document_ids_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Expand flattened document offsets to the device-side mask representation."""

    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be a one-dimensional int32 tensor")
    if cu_seqlens.device.type != "cuda":
        raise ValueError("cu_seqlens must be a CUDA tensor")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least two offsets")

    total_tokens = batch_size * seq_len
    document_starts = torch.zeros(
        total_tokens + 1,
        device=cu_seqlens.device,
        dtype=torch.bool,
    )
    document_starts.scatter_(0, cu_seqlens.to(torch.int64), True)
    return (
        document_starts[:-1]
        .cumsum(dim=0, dtype=torch.int32)
        .sub_(1)
        .reshape(batch_size, seq_len)
    )


def resolve_document_ids(
    q: torch.Tensor,
    document_list: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    *,
    seq_dim: int = 2,
) -> torch.Tensor:
    """Return the device document IDs accepted by the native kernels."""

    if document_list is not None and cu_seqlens is not None:
        raise ValueError("document_list and cu_seqlens are mutually exclusive")
    if cu_seqlens is not None:
        document_list = document_ids_from_cu_seqlens(
            cu_seqlens,
            batch_size=q.shape[0],
            seq_len=q.shape[seq_dim],
        )
    if document_list is None:
        raise ValueError("Flash-MSA requires packed-document metadata")
    return document_list


def build_reverse_index_cuda(
    block_indices: torch.Tensor,
    *,
    query_chunk: int = QUERY_CHUNK,
    workspace: ReverseIndexWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded ``(task_meta, task_qids)`` tensors fully on CUDA.

    ``block_indices`` has shape ``[B, Hp, S, top_k_blocks]``. The returned
    tensors have fixed padded task count:
    ``B * Hp * (ceil(S * top_k_blocks / query_chunk) + S / 128)``.
    """

    if block_indices.device.type != "cuda":
        raise ValueError("CUDA reverse-index builder requires a CUDA tensor")
    if block_indices.ndim != 4:
        raise ValueError("block_indices must have shape [B, Hp, S, top_k_blocks]")

    block_indices_c = block_indices.detach().to(torch.int32).contiguous()
    batch, n_proxy_heads, seq_len, top_k_blocks = map(int, block_indices_c.shape)
    if seq_len % BLOCK_SIZE != 0:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if top_k_blocks < 1:
        raise ValueError("top_k_blocks must be positive")
    query_chunk = int(query_chunk)
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")

    ws = (workspace or _DEFAULT_WORKSPACE).get(
        batch,
        n_proxy_heads,
        seq_len,
        top_k_blocks,
        query_chunk,
        block_indices_c.device,
    )
    _load_ext().run_build_reverse_index(
        block_indices_c,
        ws["counts"],
        ws["write_counts"],
        ws["bucket_offsets"],
        ws["task_meta"],
        ws["task_qids"],
        int(BLOCK_SIZE),
        int(query_chunk),
    )
    return ws["task_meta"], ws["task_qids"]


def build_dense_causal_schedule_cuda(
    *,
    batch: int,
    n_proxy_heads: int,
    seq_len: int,
    query_chunk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense causal backward tasks without per-task host work."""

    if device.type != "cuda":
        raise ValueError("dense schedule builder requires a CUDA device")
    if seq_len % BLOCK_SIZE:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if batch < 1 or n_proxy_heads < 1 or query_chunk < 1:
        raise ValueError("batch, n_proxy_heads, and query_chunk must be positive")

    num_blocks = seq_len // BLOCK_SIZE
    offsets = [0]
    for _ in range(batch * n_proxy_heads):
        for key_block in range(num_blocks):
            queries = seq_len - key_block * BLOCK_SIZE
            offsets.append(offsets[-1] + (queries + query_chunk - 1) // query_chunk)

    bucket_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
    num_tasks = offsets[-1]
    task_meta = torch.empty((num_tasks, 4), dtype=torch.int32, device=device)
    task_qids = torch.empty((num_tasks, query_chunk), dtype=torch.int32, device=device)
    _load_ext().run_build_dense_schedule(
        bucket_offsets,
        task_meta,
        task_qids,
        int(n_proxy_heads),
        int(num_blocks),
        int(seq_len),
        int(query_chunk),
    )
    return task_meta, task_qids


def build_sparse_attention_metadata_cuda(
    block_indices: torch.Tensor,
    *,
    n_main_heads: int,
    backward_query_chunk: int,
    document_ids: torch.Tensor,
) -> SparseAttentionMetadata:
    """Build persistent exact forward and backward schedules on CUDA."""

    if block_indices.device.type != "cuda" or block_indices.ndim != 4:
        raise ValueError("block_indices must be a CUDA tensor shaped [B, Hp, S, Kb]")
    block_indices_c = block_indices.detach().to(torch.int32).contiguous()
    batch, _n_proxy_heads, seq_len, _top_k_blocks = map(int, block_indices_c.shape)
    if seq_len % BLOCK_SIZE:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if document_ids.shape != (batch, seq_len):
        raise ValueError("document_ids must have shape [B, S]")
    document_ids_c = document_ids.detach().to(torch.int32).contiguous()
    major, minor = torch.cuda.get_device_capability(block_indices_c.device)
    match major:
        case 9:
            query_block_size = BLOCK_SIZE
        case 10 | 11:
            query_block_size = 2 * BLOCK_SIZE
        case _:
            raise NotImplementedError(
                f"FA4 block-sparse attention is not supported on SM{major}{minor}"
            )
    selection = build_block_sparse_selection(
        block_indices_c,
        n_main_heads=int(n_main_heads),
        query_block_size=query_block_size,
    )

    # Use a per-forward workspace: these tensors are saved by autograd and must
    # not be overwritten by a later forward before its corresponding backward.
    backward_workspace = ReverseIndexWorkspace()
    task_meta, task_qids = build_reverse_index_cuda(
        block_indices_c,
        query_chunk=int(backward_query_chunk),
        workspace=backward_workspace,
    )

    return SparseAttentionMetadata(
        selection=selection,
        task_meta=task_meta,
        task_qids=task_qids,
        document_ids=document_ids_c,
        query_block_size=query_block_size,
        top_k_blocks=int(block_indices_c.shape[3]),
    )


__all__ = [
    "BlockSparseSelection",
    "ReverseIndexWorkspace",
    "SparseAttentionMetadata",
    "build_block_sparse_selection",
    "build_dense_causal_schedule_cuda",
    "build_reverse_index_cuda",
    "build_sparse_attention_metadata_cuda",
    "document_ids_from_cu_seqlens",
    "resolve_document_ids",
]
