import pytest
import torch

from slime_plugins.models.vision_rotary import FP32VisionRotaryEmbedding


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_outer_dtype_conversions_preserve_original_frequencies(dtype):
    rotary = FP32VisionRotaryEmbedding(36)
    expected = rotary.inv_freq.clone()
    outer = torch.nn.Sequential(rotary)
    before_keys = list(outer.state_dict())
    for conversion in (lambda: outer.to(dtype=dtype), outer.half, outer.bfloat16, outer.float):
        conversion()
        assert rotary.inv_freq.dtype == torch.float32
        assert torch.equal(rotary.inv_freq, expected)
        assert list(outer.state_dict()) == before_keys == []
    assert not torch.equal(expected, expected.to(dtype).float())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_device_conversion_and_rotated_activation_backward(device):
    rotary = FP32VisionRotaryEmbedding(36).bfloat16().to(device)
    positions = torch.arange(50, device=device).reshape(-1, 1)
    angles = rotary(positions)
    reference = FP32VisionRotaryEmbedding(36, device=device)(positions)
    assert torch.equal(angles, reference)
    activation = torch.randn(50, 18, device=device, requires_grad=True)
    rotated = activation * angles.cos() + activation.flip(-1) * angles.sin()
    rotated.square().mean().backward()
    assert activation.grad is not None
    assert torch.isfinite(activation.grad).all()
    assert activation.grad.abs().sum() > 0
    expected_input = activation.detach().clone().requires_grad_()
    expected_rotated = expected_input * reference.cos() + expected_input.flip(-1) * reference.sin()
    expected_rotated.square().mean().backward()
    assert torch.equal(activation.grad, expected_input.grad)
    rotary.cpu().half()
    assert rotary.inv_freq.device.type == "cpu"
    assert torch.equal(rotary.inv_freq, FP32VisionRotaryEmbedding(36).inv_freq)
