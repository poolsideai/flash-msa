"""Public API for Flash-MSA."""

from flash_msa.flash_msa import (
    prepare_sparse_attention,
    sparse_attention as flash_msa_func,
    sparse_main_attention,
    sparse_proxy_vjp,
)
from flash_msa.warmup.flash_msa_warmup import (
    dense_main_attention,
    dense_proxy_vjp,
    sparse_attention_warmup,
)

flash_msa_func_warmup = sparse_attention_warmup
flash_msa_warmup_func = sparse_attention_warmup

__all__ = [
    "flash_msa_func",
    "flash_msa_func_warmup",
    "flash_msa_warmup_func",
    "prepare_sparse_attention",
    "sparse_main_attention",
    "sparse_proxy_vjp",
    "dense_main_attention",
    "dense_proxy_vjp",
    "sparse_attention_warmup",
]
