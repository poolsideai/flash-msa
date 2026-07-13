import pytest
import torch

from flash_msa import flash_msa_func, flash_msa_warmup_func
from flash_msa.reverse_index_cuda import build_dense_causal_schedule_cuda


def test_dense_warmup_schedule_is_built_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    task_meta, task_qids = build_dense_causal_schedule_cuda(
        batch=2,
        n_proxy_heads=2,
        seq_len=256,
        query_chunk=16,
        device=torch.device("cuda"),
    )

    assert task_meta.shape == (96, 4)
    for batch in range(2):
        for head in range(2):
            for block in range(2):
                matches = task_meta[:, :3] == torch.tensor(
                    [batch, head, block],
                    device="cuda",
                )
                rows = matches.all(dim=1).nonzero().flatten()
                actual = task_qids[rows].flatten()
                actual = actual[actual >= 0]
                expected = torch.arange(
                    block * 128,
                    256,
                    device="cuda",
                    dtype=torch.int32,
                )
                torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kernel", [flash_msa_func, flash_msa_warmup_func])
def test_packed_documents_isolate_outputs_and_gradients(kernel) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Flash-MSA requires Hopper or newer")

    torch.manual_seed(7)
    batch, seq_len, head_dim = 1, 512, 128
    documents = torch.empty(batch, seq_len, device="cuda", dtype=torch.int32)
    documents[:, :190] = 0
    documents[:, 190:350] = 1
    documents[:, 350:] = 2
    cu_seqlens = torch.tensor([0, 190, 350, 512], device="cuda", dtype=torch.int32)
    shapes = {
        "q_proxy": (batch, 1, seq_len, head_dim),
        "k_proxy": (batch, 1, seq_len, head_dim),
        "q": (batch, 6, seq_len, head_dim),
        "k": (batch, 1, seq_len, head_dim),
        "v": (batch, 1, seq_len, head_dim),
    }
    inputs = {
        name: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for name, shape in shapes.items()
    }

    def run(values, *, use_cu_seqlens: bool = False):
        tensors = {
            name: value.detach().clone().requires_grad_(True)
            for name, value in values.items()
        }
        out, aux = kernel(
            tensors["q_proxy"],
            tensors["k_proxy"],
            tensors["q"],
            tensors["k"],
            tensors["v"],
            256,
            head_dim**-0.5,
            None if use_cu_seqlens else documents,
            **({"cu_seqlens": cu_seqlens} if use_cu_seqlens else {}),
        )
        (out[:, 190:350].float().square().sum() + aux).backward()
        return out.detach(), tensors

    perturbed = {name: value.clone() for name, value in inputs.items()}
    for value in perturbed.values():
        value[:, :, :190] += torch.randn_like(value[:, :, :190]) * 10
        value[:, :, 350:] += torch.randn_like(value[:, :, 350:]) * 10

    output, tensors = run(inputs)
    cu_output, cu_tensors = run(inputs, use_cu_seqlens=True)
    perturbed_output, perturbed_tensors = run(perturbed)
    torch.testing.assert_close(cu_output, output)
    for name in inputs:
        assert tensors[name].grad is not None
        assert cu_tensors[name].grad is not None
        torch.testing.assert_close(cu_tensors[name].grad, tensors[name].grad)
    torch.testing.assert_close(output[:, 190:350], perturbed_output[:, 190:350])
    for name in ("q", "k", "v"):
        tensor = tensors[name]
        assert tensor.grad is not None
        outside = torch.cat((tensor.grad[:, :, :190], tensor.grad[:, :, 350:]), dim=2)
        assert outside.abs().max() == 0
    for name in ("q_proxy", "k_proxy"):
        grad = tensors[name].grad
        perturbed_grad = perturbed_tensors[name].grad
        assert grad is not None and perturbed_grad is not None
        assert grad[:, :, 190:350].abs().max() > 0
        torch.testing.assert_close(
            grad[:, :, 190:350],
            perturbed_grad[:, :, 190:350],
        )
