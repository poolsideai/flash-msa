import pytest
import torch

from flash_msa import (
    dense_main_attention,
    dense_proxy_vjp,
    flash_msa_func,
    flash_msa_warmup_func,
    prepare_sparse_attention,
    sparse_main_attention,
    sparse_proxy_vjp,
)


def _inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch, seq_len, head_dim = 1, 512, 128
    shapes = {
        "q_proxy": (batch, 1, seq_len, head_dim),
        "k_proxy": (batch, 1, seq_len, head_dim),
        "q": (batch, 6, seq_len, head_dim),
        "k": (batch, 1, seq_len, head_dim),
        "v": (batch, 1, seq_len, head_dim),
    }
    tensors = {
        name: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for name, shape in shapes.items()
    }
    document_ids = torch.empty(batch, seq_len, device="cuda", dtype=torch.int32)
    document_ids[:, :190] = 0
    document_ids[:, 190:350] = 1
    document_ids[:, 350:] = 2
    return tensors, document_ids


def _clone(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in inputs.items()
    }


@pytest.mark.parametrize("dense", [False, True], ids=["sparse", "warmup"])
def test_decomposed_main_and_indexer_gradients_match_combined(dense: bool) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Flash-MSA requires Hopper or newer")

    torch.manual_seed(11)
    inputs, document_ids = _inputs()
    reference = _clone(inputs)
    candidate = _clone(inputs)
    top_k = 512 if dense else 256
    scale = 128**-0.5
    grad_out = torch.randn(
        1,
        512,
        6 * 128,
        device="cuda",
        dtype=torch.bfloat16,
    )

    combined = flash_msa_warmup_func if dense else flash_msa_func
    reference_out, reference_aux = combined(
        reference["q_proxy"],
        reference["k_proxy"],
        reference["q"],
        reference["k"],
        reference["v"],
        top_k,
        scale,
        document_ids,
    )
    (reference_out * grad_out).sum().add(reference_aux).backward()

    if dense:
        candidate_out, lse = dense_main_attention(
            candidate["q"],
            candidate["k"],
            candidate["v"],
            scale,
            document_ids,
        )
        dq_proxy, dk_proxy = dense_proxy_vjp(
            candidate["q_proxy"],
            candidate["k_proxy"],
            candidate["q"],
            candidate["k"],
            lse,
            scale,
            document_ids,
        )
    else:
        metadata = prepare_sparse_attention(
            candidate["q_proxy"],
            candidate["k_proxy"],
            candidate["q"],
            candidate["k"],
            candidate["v"],
            top_k,
            scale,
            document_ids,
        )
        candidate_out, lse = sparse_main_attention(
            candidate["q"],
            candidate["k"],
            candidate["v"],
            metadata,
            scale,
        )
        dq_proxy, dk_proxy = sparse_proxy_vjp(
            candidate["q_proxy"],
            candidate["k_proxy"],
            candidate["q"],
            candidate["k"],
            lse,
            metadata,
            scale,
        )
    (candidate_out * grad_out).sum().backward()

    torch.testing.assert_close(candidate_out, reference_out)
    for name in ("q", "k", "v"):
        assert candidate[name].grad is not None
        assert reference[name].grad is not None
        torch.testing.assert_close(candidate[name].grad, reference[name].grad)
    assert reference["q_proxy"].grad is not None
    assert reference["k_proxy"].grad is not None
    torch.testing.assert_close(dq_proxy, reference["q_proxy"].grad)
    torch.testing.assert_close(dk_proxy, reference["k_proxy"].grad)
