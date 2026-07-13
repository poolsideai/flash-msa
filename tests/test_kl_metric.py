import pytest
import torch

from flash_msa import flash_msa_func, flash_msa_warmup_func
from flash_msa.tests.testing_model import Model
from flash_msa.tests.testing_model_warmup import WarmupModel


@pytest.mark.parametrize(
    ("kernel", "reference_cls", "top_k"),
    [
        (flash_msa_func, Model, 256),
        (flash_msa_warmup_func, WarmupModel, 512),
    ],
)
def test_kl_metric_matches_eager(kernel, reference_cls, top_k) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Flash-MSA requires Hopper or newer")

    torch.manual_seed(7)
    batch, seq_len, head_dim = 1, 512, 128
    n_heads, n_kv_heads, n_proxy_heads, n_proxy_kv_heads = 6, 1, 1, 1
    shapes = (
        (batch, n_proxy_heads, seq_len, head_dim),
        (batch, n_proxy_kv_heads, seq_len, head_dim),
        (batch, n_heads, seq_len, head_dim),
        (batch, n_kv_heads, seq_len, head_dim),
        (batch, n_kv_heads, seq_len, head_dim),
    )
    q_proxy, k_proxy, q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        for shape in shapes
    ]
    reference = reference_cls(
        n_heads,
        n_kv_heads,
        head_dim,
        n_proxy_heads,
        n_proxy_kv_heads,
        top_k,
        False,
    )
    expected = reference._attention_eager(q_proxy, k_proxy, q, k, v)[1].detach()
    metric = torch.full((), -1.0, device="cuda")

    output, _ = kernel(
        q_proxy,
        k_proxy,
        q,
        k,
        v,
        top_k,
        head_dim**-0.5,
        kl_metric=metric,
    )
    assert metric == 0.0
    output.float().square().mean().backward()

    torch.testing.assert_close(metric, expected.float(), atol=1e-3, rtol=1e-2)
