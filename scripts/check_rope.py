import torch

from yarn_context.rope import RoPE


def main() -> None:
    torch.manual_seed(0)
    rope = RoPE(8)
    query = torch.randn(1, 1, 3, 8)
    key = torch.randn(1, 1, 3, 8)
    rotated_q, rotated_k = rope(query, key)
    norm_error = (rotated_q.norm(dim=-1) - query.norm(dim=-1)).abs().max()
    print(f"RoPE shapes: {tuple(rotated_q.shape)}, {tuple(rotated_k.shape)}")
    print(f"Maximum query norm error: {norm_error.item():.8f}")


if __name__ == "__main__":
    main()
