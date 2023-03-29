import torch
from torch import Tensor

QUERY_TOKEN = 11
FIRST_FILLER_TOKEN = 6
LAST_FILLER_TOKEN = 10


def make_passkey_batch(
    batch_size: int,
    length: int,
    generator: torch.Generator,
    needle_positions: list[int] | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate one numeric needle in filler and ask for it at the final query."""
    if batch_size <= 0 or length < 4:
        raise ValueError("batch_size must be positive and length at least four")
    tokens = torch.randint(
        FIRST_FILLER_TOKEN,
        LAST_FILLER_TOKEN + 1,
        (batch_size, length),
        generator=generator,
    )
    labels = torch.randint(1, 6, (batch_size,), generator=generator)
    if needle_positions is None:
        positions = torch.randint(0, length - 1, (batch_size,), generator=generator)
    else:
        if len(needle_positions) != batch_size:
            raise ValueError("needle_positions must match batch_size")
        positions = torch.tensor(needle_positions)
        if torch.any(positions < 0) or torch.any(positions >= length - 1):
            raise ValueError("needle positions must precede the final query")
    tokens[torch.arange(batch_size), positions] = labels
    tokens[:, -1] = QUERY_TOKEN
    return tokens, labels


def make_passkey_perplexity_batch(
    batch_size: int, length: int, generator: torch.Generator
) -> Tensor:
    """Deterministic local token sequences for model-specific causal NLL evaluation."""
    tokens, _ = make_passkey_batch(batch_size, length + 1, generator)
    return tokens
