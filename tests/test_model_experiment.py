import torch

from yarn_context.experiment import evaluate_passkey, train_passkey
from yarn_context.model import TinyPasskeyModel
from yarn_context.passkey import QUERY_TOKEN, make_passkey_batch


def test_passkey_data_has_exactly_one_needle_and_final_query() -> None:
    generator = torch.Generator().manual_seed(5)
    tokens, labels = make_passkey_batch(8, 12, generator)
    assert torch.all(tokens[:, -1] == QUERY_TOKEN)
    assert torch.all((tokens[:, :-1] >= 1).sum(dim=1) >= 12 - 1)
    for row, label in zip(tokens, labels, strict=True):
        assert int((row[:-1] == label).sum()) >= 1
        assert int((row[:-1] <= 5).sum()) == 1


def test_causal_attention_blocks_future_tokens() -> None:
    torch.manual_seed(4)
    model = TinyPasskeyModel().eval()
    generator = torch.Generator().manual_seed(3)
    tokens, _ = make_passkey_batch(2, 12, generator)
    changed = tokens.clone()
    changed[:, 8] = (changed[:, 8] % 4) + 6
    with torch.no_grad():
        original_logits = model(tokens)
        changed_logits = model(changed)
    assert torch.allclose(original_logits[:, :8], changed_logits[:, :8], atol=1e-6)
    assert not torch.allclose(original_logits[:, 8], changed_logits[:, 8])


def test_short_real_finetune_reduces_loss_and_evaluates_longer_sequences() -> None:
    torch.manual_seed(8)
    model = TinyPasskeyModel()
    result = train_passkey(model, steps=30, length=12, batch_size=16, learning_rate=0.02)
    assert result["final_loss"] < result["initial_loss"]
    assert result["training_seconds"] > 0
    evaluation = evaluate_passkey(model, "yarn", length=24, samples=8, batch_size=4)
    assert evaluation["length"] == 24
    assert evaluation["samples"] == 8
    assert 0 <= evaluation["accuracy"] <= 1
    assert evaluation["loss"] > 0
