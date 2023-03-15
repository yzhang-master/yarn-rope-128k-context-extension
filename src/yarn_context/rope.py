import torch
from torch import Tensor, nn


def rotate_half(tensor: Tensor) -> Tensor:
    """Rotate adjacent half-dimensional complex planes: (x1, x2) -> (-x2, x1)."""
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def base_inv_freq(head_dim: int, base: float = 10_000.0, dtype=torch.float32) -> Tensor:
    if head_dim < 2 or head_dim % 2:
        raise ValueError("RoPE head_dim must be a positive even number")
    if base <= 1:
        raise ValueError("RoPE base must exceed one")
    dimensions = torch.arange(0, head_dim, 2, dtype=dtype)
    return base ** (-dimensions / head_dim)


def rope_cos_sin(
    inv_freq: Tensor, positions: Tensor, mscale: float | Tensor = 1.0
) -> tuple[Tensor, Tensor]:
    """Compute duplicated-half cos/sin tensors for arbitrary integer positions."""
    if positions.ndim != 1 or not torch.all(positions >= 0):
        raise ValueError("positions must be a nonnegative one-dimensional tensor")
    frequencies = torch.outer(positions.to(dtype=inv_freq.dtype), inv_freq)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return mscale * angles.cos(), mscale * angles.sin()


def apply_rope(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply [sequence, head_dim] rotations to [batch, heads, sequence, head_dim]."""
    if tensor.ndim != 4 or cos.shape != sin.shape or cos.shape[-1] != tensor.shape[-1]:
        raise ValueError("expected BHSd activations and matching Sd cos/sin tensors")
    return tensor * cos[None, None, :, :] + rotate_half(tensor) * sin[None, None, :, :]


class RoPE(nn.Module):
    """Standard RoPE over explicit positions; no maximum-length lookup table."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        self.register_buffer("inv_freq", base_inv_freq(head_dim, base), persistent=False)

    def forward(
        self, query: Tensor, key: Tensor, positions: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if positions is None:
            positions = torch.arange(query.shape[-2], device=query.device)
        positions = positions.to(device=query.device)
        cos, sin = rope_cos_sin(self.inv_freq.to(dtype=query.dtype), positions)
        return apply_rope(query, cos, sin), apply_rope(key, cos, sin)
