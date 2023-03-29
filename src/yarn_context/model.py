import math
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from yarn_context.scaling import ScaledRoPE, YaRNConfig


@dataclass(frozen=True)
class TinyPasskeyConfig:
    vocab_size: int = 16
    model_dim: int = 32
    num_heads: int = 4
    rope: YaRNConfig = field(
        default_factory=lambda: YaRNConfig(
            head_dim=8,
            original_max_position=16,
            extended_max_position=64,
            scaling_factor=4.0,
        )
    )

    def __post_init__(self) -> None:
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.model_dim // self.num_heads != self.rope.head_dim:
            raise ValueError("RoPE head_dim must match the attention heads")


class TinyPasskeyModel(nn.Module):
    """One real causal self-attention layer with selectable rotary scaling."""

    def __init__(
        self, config: TinyPasskeyConfig = TinyPasskeyConfig(), method: str = "original"
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.qkv = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.output = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size)
        self.rope = ScaledRoPE(config.rope, method=method)

    def set_rope_method(self, method: str) -> None:
        self.rope = ScaledRoPE(self.config.rope, method=method)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have shape [batch, nonempty sequence]")
        batch, length = token_ids.shape
        hidden = self.embedding(token_ids)
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.config.num_heads, -1)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query, key = self.rope(query, key)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.config.rope.head_dim)
        causal = torch.ones(length, length, dtype=torch.bool, device=token_ids.device).tril()
        scores = scores.masked_fill(~causal, -torch.inf)
        attended = torch.softmax(scores, dim=-1) @ value
        attended = attended.transpose(1, 2).reshape(batch, length, self.config.model_dim)
        return self.lm_head(self.output(attended))
