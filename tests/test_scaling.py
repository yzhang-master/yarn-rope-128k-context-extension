import json
import math
from pathlib import Path

import torch

from yarn_context.rope import base_inv_freq, rope_cos_sin
from yarn_context.scaling import (
    ScaledRoPE,
    YaRNConfig,
    correction_bounds,
    correction_dimension,
    ntk_aware_inv_freq,
    yarn_inv_freq,
    yarn_mscale,
    yarn_ramp,
)


def test_ntk_scaling_matches_scaled_base_reference() -> None:
    dim = 16
    factor = 8.0
    base = 10_000.0
    scaled_base = base * factor ** (dim / (dim - 2))
    dimensions = torch.arange(0, dim, 2, dtype=torch.float64)
    expected = scaled_base ** (-dimensions / dim)
    actual = ntk_aware_inv_freq(dim, base, factor, dtype=torch.float64)
    assert torch.allclose(actual, expected)
    assert torch.equal(ntk_aware_inv_freq(dim, base, 1.0), base_inv_freq(dim, base))


def test_yarn_correction_bounds_and_ramp_match_reference_math() -> None:
    config = YaRNConfig()
    expected_low = math.floor(
        64 * math.log(4096 / (32 * 2 * math.pi)) / (2 * math.log(10_000))
    )
    expected_high = math.ceil(
        64 * math.log(4096 / (1 * 2 * math.pi)) / (2 * math.log(10_000))
    )
    raw_dimension = 64 * math.log(4096 / (32 * 2 * math.pi)) / (2 * math.log(10_000))
    assert math.isclose(correction_dimension(32, config), raw_dimension)
    assert correction_bounds(config) == (expected_low, expected_high)

    ramp = yarn_ramp(config, dtype=torch.float64)
    original = base_inv_freq(64, dtype=torch.float64)
    actual = yarn_inv_freq(config, dtype=torch.float64)
    expected = original * (1 - ramp) + original / 32 * ramp
    assert torch.allclose(actual, expected)
    assert torch.equal(actual[0], original[0])
    assert torch.allclose(actual[-1], original[-1] / 32)
    assert torch.all(ramp[1:] >= ramp[:-1])
    assert ramp[0].item() == 0.0
    assert ramp[-1].item() == 1.0


def test_yarn_mscale_corrects_query_key_and_attention_logits() -> None:
    config = YaRNConfig(head_dim=8)
    rope = ScaledRoPE(config, method="yarn")
    torch.manual_seed(12)
    query = torch.randn(1, 1, 2, 8)
    key = torch.randn(1, 1, 2, 8)
    positions = torch.tensor([3, 17])
    actual_q, actual_k = rope(query, key, positions)

    scale = 1 + 0.1 * math.log(32)
    assert math.isclose(yarn_mscale(32), scale)
    assert rope.mscale == scale
    cos, sin = rope_cos_sin(yarn_inv_freq(config), positions, mscale=scale)
    unscaled_cos, unscaled_sin = rope_cos_sin(yarn_inv_freq(config), positions)
    expected_q = query * cos + torch.cat((-query[..., 4:], query[..., :4]), dim=-1) * sin
    unscaled_q = query * unscaled_cos + torch.cat(
        (-query[..., 4:], query[..., :4]), dim=-1
    ) * unscaled_sin
    unscaled_k = key * unscaled_cos + torch.cat(
        (-key[..., 4:], key[..., :4]), dim=-1
    ) * unscaled_sin
    assert torch.allclose(actual_q, expected_q)
    assert torch.allclose(actual_q.norm(dim=-1), scale * unscaled_q.norm(dim=-1))
    actual_logits = actual_q @ actual_k.transpose(-1, -2)
    unscaled_logits = unscaled_q @ unscaled_k.transpose(-1, -2)
    assert torch.allclose(actual_logits, scale**2 * unscaled_logits)


def test_shipped_128k_configs_compute_last_position_without_lookup_overflow() -> None:
    for filename, head_dim in (("yarn_128k.json", 64), ("tiny_yarn_128k.json", 8)):
        payload = json.loads((Path(__file__).parents[1] / "configs" / filename).read_text())
        config = YaRNConfig(**payload)
        assert config.extended_max_position == 131_072
        assert config.scaling_factor == 32
        rope = ScaledRoPE(config)
        query = torch.randn(1, 1, 1, head_dim)
        key = torch.randn(1, 1, 1, head_dim)
        rotated_q, rotated_k = rope(query, key, torch.tensor([131_071]))
        assert rotated_q.shape == query.shape
        assert rotated_k.shape == key.shape
        assert torch.isfinite(rotated_q).all()
        assert torch.isfinite(rotated_k).all()
