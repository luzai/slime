"""Vision rotary frequencies that survive model-wide dtype conversion."""

import torch
from torch import nn


class FP32VisionRotaryEmbedding(nn.Module):
    """Match Qwen3.5's HF rotary formula while retaining FP32 frequencies."""

    def __init__(self, dim: int, theta: float = 10000.0, *, device=None):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("inv_freq", self._frequencies(device), persistent=False)

    def _frequencies(self, device):
        # Rebuild from the definition, never widen frequencies already rounded
        # by an outer Float16Module / .bfloat16() / .half() conversion.
        indices = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.theta ** (indices / self.dim))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.inv_freq = self._frequencies(self.inv_freq.device)
        return self

    def forward(self, position_ids):
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)
