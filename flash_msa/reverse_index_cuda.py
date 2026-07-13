"""CUDA reverse-index builder for selected-block MSA backward."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

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
        _EXT = load(
            name="msa_reverse_index_ext",
            sources=[_SRC],
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


@dataclass
class SparseAttentionMetadata:
    """Persistent selection and reverse-index metadata for one forward."""

    task_meta: torch.Tensor
    task_qids: torch.Tensor
    remote_destinations: torch.Tensor
    remote_valid: torch.Tensor
    remote_cu_seqlens: torch.Tensor
    document_ids: torch.Tensor
    remote_q_document_ids: torch.Tensor
    remote_k_document_ids: torch.Tensor
    batch: int
    n_proxy_heads: int
    seq_len: int
    top_k_blocks: int


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


def build_sparse_attention_metadata_cuda(
    block_indices: torch.Tensor,
    *,
    backward_query_chunk: int,
    document_ids: torch.Tensor,
) -> SparseAttentionMetadata:
    """Build persistent padded backward tasks without host synchronization."""

    if block_indices.device.type != "cuda" or block_indices.ndim != 4:
        raise ValueError("block_indices must be a CUDA tensor shaped [B, Hp, S, Kb]")
    block_indices_c = block_indices.detach().to(torch.int32).contiguous()
    batch, n_proxy_heads, seq_len, top_k_blocks = map(int, block_indices_c.shape)
    if seq_len % BLOCK_SIZE:
        raise ValueError(f"sequence length must be divisible by {BLOCK_SIZE}")
    if document_ids.numel():
        if document_ids.shape != (batch, seq_len):
            raise ValueError("document_ids must have shape [B, S]")
        document_ids_c = document_ids.detach().to(torch.int32).contiguous()
    else:
        document_ids_c = document_ids.detach().to(torch.int32).contiguous()
    # Use a per-forward workspace: these tensors are saved by autograd and must
    # not be overwritten by a later forward before its corresponding backward.
    backward_workspace = ReverseIndexWorkspace()
    task_meta, task_qids = build_reverse_index_cuda(
        block_indices_c,
        query_chunk=int(backward_query_chunk),
        workspace=backward_workspace,
    )

    num_blocks = seq_len // BLOCK_SIZE
    num_buckets = batch * n_proxy_heads * num_blocks
    num_remote_edges = batch * n_proxy_heads * seq_len * (top_k_blocks - 1)
    remote_counts = torch.empty(
        num_buckets, device=block_indices_c.device, dtype=torch.int32
    )
    remote_write_counts = torch.empty_like(remote_counts)
    remote_cu_seqlens = torch.empty(
        num_buckets + 1, device=block_indices_c.device, dtype=torch.int32
    )
    remote_destinations = torch.empty(
        num_remote_edges, device=block_indices_c.device, dtype=torch.int64
    )
    remote_valid = torch.empty(
        num_remote_edges, device=block_indices_c.device, dtype=torch.uint8
    )
    if num_remote_edges:
        _load_ext().run_build_remote_layout(
            block_indices_c,
            remote_counts,
            remote_write_counts,
            remote_cu_seqlens,
            remote_destinations,
            remote_valid,
            int(BLOCK_SIZE),
        )
    if document_ids_c.numel():
        document_ids_by_proxy = (
            document_ids_c[:, None, :].expand(batch, n_proxy_heads, seq_len).reshape(-1)
        )
        remote_q_document_ids = document_ids_by_proxy[remote_destinations]
        remote_k_document_ids = document_ids_by_proxy.contiguous()
    else:
        remote_q_document_ids = document_ids_c
        remote_k_document_ids = document_ids_c

    return SparseAttentionMetadata(
        task_meta=task_meta,
        task_qids=task_qids,
        remote_destinations=remote_destinations,
        remote_valid=remote_valid,
        remote_cu_seqlens=remote_cu_seqlens,
        document_ids=document_ids_c,
        remote_q_document_ids=remote_q_document_ids,
        remote_k_document_ids=remote_k_document_ids,
        batch=batch,
        n_proxy_heads=n_proxy_heads,
        seq_len=seq_len,
        top_k_blocks=top_k_blocks,
    )


__all__ = [
    "ReverseIndexWorkspace",
    "SparseAttentionMetadata",
    "build_reverse_index_cuda",
    "build_sparse_attention_metadata_cuda",
]
