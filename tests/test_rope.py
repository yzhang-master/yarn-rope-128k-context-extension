import torch

from yarn_context.rope import RoPE, apply_rope, rope_cos_sin, rotate_half


def test_rotate_half_and_single_position_match_complex_reference() -> None:
    vector = torch.tensor([[1.0, 2.0, 3.0, 5.0]], dtype=torch.float64)
    rotated = torch.tensor([[-3.0, -5.0, 1.0, 2.0]], dtype=torch.float64)
    assert torch.equal(rotate_half(vector), rotated)

    inv_freq = torch.tensor([1.0, 0.1], dtype=torch.float64)
    cos, sin = rope_cos_sin(inv_freq, torch.tensor([2]))
    actual = apply_rope(vector[None, None, :, :], cos, sin)[0, 0, 0]
    expected = torch.tensor(
        [
            1.0 * torch.cos(torch.tensor(2.0)) - 3.0 * torch.sin(torch.tensor(2.0)),
            2.0 * torch.cos(torch.tensor(0.2)) - 5.0 * torch.sin(torch.tensor(0.2)),
            3.0 * torch.cos(torch.tensor(2.0)) + 1.0 * torch.sin(torch.tensor(2.0)),
            5.0 * torch.cos(torch.tensor(0.2)) + 2.0 * torch.sin(torch.tensor(0.2)),
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(actual, expected)


def test_rotation_preserves_norm_and_relative_dot_product() -> None:
    torch.manual_seed(3)
    rope = RoPE(8).to(dtype=torch.float64)
    query = torch.randn(1, 2, 1, 8, dtype=torch.float64)
    key = torch.randn(1, 2, 1, 8, dtype=torch.float64)

    rotated_q_zero, rotated_k_zero = rope(query, key, torch.tensor([0]))
    rotated_q, rotated_k = rope(query, key, torch.tensor([4]))
    _, rotated_k_relative = rope(query, key, torch.tensor([7]))

    assert torch.allclose(rotated_q.norm(dim=-1), query.norm(dim=-1))
    assert torch.allclose(rotated_k.norm(dim=-1), key.norm(dim=-1))
    shifted_key = rope(query, key, torch.tensor([11]))[1]
    dot = (rotated_q * shifted_key).sum(dim=-1)
    relative_dot = (rotated_q_zero * rotated_k_relative).sum(dim=-1)
    assert torch.allclose(dot, relative_dot)
    assert torch.allclose(rotated_k, rope(query, key, torch.tensor([4]))[1])
