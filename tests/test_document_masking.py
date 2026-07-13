import pytest
import torch

from flash_msa import flash_msa_func, flash_msa_warmup_func


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

    def run(values):
        tensors = {
            name: value.detach().clone().requires_grad_(True)
            for name, value in values.items()
        }
        out, _aux = kernel(
            tensors["q_proxy"],
            tensors["k_proxy"],
            tensors["q"],
            tensors["k"],
            tensors["v"],
            256,
            head_dim**-0.5,
            documents,
        )
        out[:, 190:350].float().square().sum().backward()
        return out.detach(), tensors

    perturbed = {name: value.clone() for name, value in inputs.items()}
    for value in perturbed.values():
        value[:, :, :190] += torch.randn_like(value[:, :, :190]) * 10
        value[:, :, 350:] += torch.randn_like(value[:, :, 350:]) * 10

    output, tensors = run(inputs)
    perturbed_output, _ = run(perturbed)
    torch.testing.assert_close(output[:, 190:350], perturbed_output[:, 190:350])
    for tensor in tensors.values():
        assert tensor.grad is not None
        outside = torch.cat((tensor.grad[:, :, :190], tensor.grad[:, :, 350:]), dim=2)
        assert outside.abs().max() == 0
