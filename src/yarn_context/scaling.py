import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from yarn_context.rope import apply_rope, base_inv_freq, rope_cos_sin


@dataclass(frozen=True)
class YaRNConfig:
    head_dim: int = 64
    base: float = 10_000.0
    original_max_position: int = 4_096
    extended_max_position: int = 131_072
    scaling_factor: float = 32.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale_coefficient: float = 0.1

    def __post_init__(self) -> None:
        if self.head_dim < 4 or self.head_dim % 2:
            raise ValueError("YaRN head_dim must be even and at least four")
        if self.base <= 1 or self.scaling_factor < 1:
            raise ValueError("base must exceed one and scaling_factor must be at least one")
        if self.original_max_position <= 0:
            raise ValueError("original_max_position must be positive")
        expected = self.original_max_position * self.scaling_factor
        if not math.isclose(self.extended_max_position, expected):
            raise ValueError("extended length must equal original length times scaling_factor")
        if not 0 < self.beta_slow < self.beta_fast:
            raise ValueError("require 0 < beta_slow < beta_fast")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def ntk_aware_inv_freq(
    head_dim: int, base: float = 10_000.0, scaling_factor: float = 32.0, dtype=torch.float32
) -> Tensor:
    """Fixed NTK-aware scaling implemented as an equivalent RoPE-base change."""
    if scaling_factor < 1:
        raise ValueError("scaling_factor must be at least one")
    scaled_base = base * scaling_factor ** (head_dim / (head_dim - 2))
    return base_inv_freq(head_dim, scaled_base, dtype=dtype)


def correction_dimension(rotations: float, config: YaRNConfig) -> float:
    """Dimension whose wavelength completes the requested rotations at original length."""
    return (
        config.head_dim
        * math.log(config.original_max_position / (rotations * 2 * math.pi))
        / (2 * math.log(config.base))
    )


def correction_bounds(config: YaRNConfig) -> tuple[int, int]:
    low = max(0, math.floor(correction_dimension(config.beta_fast, config)))
    high = min(config.head_dim // 2 - 1, math.ceil(correction_dimension(config.beta_slow, config)))
    if low >= high:
        raise ValueError("invalid YaRN correction range")
    return low, high


def yarn_ramp(config: YaRNConfig, dtype=torch.float32) -> Tensor:
    """Zero for extrapolated high frequencies, one for interpolated low frequencies."""
    low, high = correction_bounds(config)
    dimensions = torch.arange(config.head_dim // 2, dtype=dtype)
    return ((dimensions - low) / (high - low)).clamp(0.0, 1.0)


def yarn_inv_freq(config: YaRNConfig, dtype=torch.float32) -> Tensor:
    """Blend NTK-by-parts extrapolation and linear position interpolation."""
    original = base_inv_freq(config.head_dim, config.base, dtype=dtype)
    ramp = yarn_ramp(config, dtype=dtype)
    return original * (1.0 - ramp) + (original / config.scaling_factor) * ramp


def yarn_mscale(scaling_factor: float, coefficient: float = 0.1) -> float:
    """YaRN attention correction applied to both query and key rotations."""
    if scaling_factor < 1 or coefficient < 0:
        raise ValueError("scaling_factor must be at least one and coefficient nonnegative")
    if scaling_factor == 1:
        return 1.0
    return 1.0 + coefficient * math.log(scaling_factor)


class ScaledRoPE(nn.Module):
    """Select standard, fixed NTK-aware, or YaRN rotary frequencies."""

    def __init__(self, config: YaRNConfig, method: str = "yarn") -> None:
        super().__init__()
        if method not in {"original", "ntk", "yarn"}:
            raise ValueError(f"unknown RoPE scaling method: {method}")
        self.config = config
        self.method = method
        if method == "original":
            inv_freq = base_inv_freq(config.head_dim, config.base)
        elif method == "ntk":
            inv_freq = ntk_aware_inv_freq(
                config.head_dim, config.base, config.scaling_factor
            )
        else:
            inv_freq = yarn_inv_freq(config)
        self.mscale = (
            yarn_mscale(config.scaling_factor, config.mscale_coefficient)
            if method == "yarn"
            else 1.0
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, query: Tensor, key: Tensor, positions: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if positions is None:
            positions = torch.arange(query.shape[-2], device=query.device)
        positions = positions.to(device=query.device)
        cos, sin = rope_cos_sin(
            self.inv_freq.to(dtype=query.dtype), positions, mscale=self.mscale
        )
        return apply_rope(query, cos, sin), apply_rope(key, cos, sin)
