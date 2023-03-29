import time

import torch
from torch.nn import functional as F

from yarn_context.model import TinyPasskeyModel
from yarn_context.passkey import make_passkey_batch


def train_passkey(
    model: TinyPasskeyModel,
    steps: int,
    length: int,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    seed: int = 19,
) -> dict[str, object]:
    """Short full-gradient fine-tuning on the local synthetic passkey task."""
    if steps <= 0 or learning_rate <= 0 or batch_size <= 0:
        raise ValueError("steps, learning_rate, and batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    history = []
    started = time.perf_counter()
    for step in range(steps):
        tokens, labels = make_passkey_batch(batch_size, length, generator)
        logits = model(tokens)[:, -1]
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.detach())})
    return {
        "steps": steps,
        "length": length,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }


@torch.no_grad()
def evaluate_passkey(
    model: TinyPasskeyModel,
    method: str,
    length: int,
    samples: int = 128,
    batch_size: int = 32,
    seed: int = 101,
) -> dict[str, float | int | str]:
    """Evaluate exact numeric passkey retrieval at arbitrary explicit lengths."""
    if samples <= 0 or batch_size <= 0:
        raise ValueError("samples and batch_size must be positive")
    model.set_rope_method(method)
    model.eval()
    generator = torch.Generator().manual_seed(seed + length)
    correct = 0
    loss_sum = 0.0
    seen = 0
    while seen < samples:
        current_batch = min(batch_size, samples - seen)
        tokens, labels = make_passkey_batch(current_batch, length, generator)
        logits = model(tokens)[:, -1]
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == labels).sum())
        seen += current_batch
    return {
        "method": method,
        "length": length,
        "samples": samples,
        "correct": correct,
        "accuracy": correct / samples,
        "loss": loss_sum / samples,
    }
