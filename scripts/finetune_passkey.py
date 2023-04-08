import argparse
import copy
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from yarn_context.experiment import evaluate_passkey, train_passkey
from yarn_context.model import TinyPasskeyModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--adapt-steps", type=int, default=80)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/passkey_experiment.json"))
    return parser.parse_args()


def evaluate_grid(
    model: TinyPasskeyModel, samples: int, batch_size: int
) -> list[dict[str, float | int | str]]:
    return [
        evaluate_passkey(model, method, length, samples=samples, batch_size=batch_size)
        for length in (16, 32, 64)
        for method in ("original", "ntk", "yarn")
    ]


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.smoke:
        args.pretrain_steps = 10
        args.adapt_steps = 2
        args.samples = 8
        args.batch_size = 4

    model = TinyPasskeyModel(method="original")
    initial = evaluate_grid(model, args.samples, args.batch_size)
    pretraining = train_passkey(
        model,
        steps=args.pretrain_steps,
        length=16,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    pretrained = copy.deepcopy(model)
    before_adaptation = evaluate_grid(pretrained, args.samples, args.batch_size)

    model.set_rope_method("yarn")
    adaptation = train_passkey(
        model,
        steps=args.adapt_steps,
        length=32,
        learning_rate=args.learning_rate / 2,
        batch_size=args.batch_size,
        seed=args.seed + 1,
    )
    after_adaptation = evaluate_grid(model, args.samples, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_name("tiny_passkey_yarn.pt")
    torch.save(
        {"config": asdict(model.config), "state_dict": model.state_dict()},
        checkpoint_path,
    )
    torch.save(
        {"config": asdict(pretrained.config), "state_dict": pretrained.state_dict()},
        args.output.with_name("tiny_passkey_pretrained.pt"),
    )
    report = {
        "scope": (
            "Tiny synthetic numeric-passkey proxy at 16/32/64 tokens; "
            "not full 128k or language-model evaluation"
        ),
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial": initial,
        "pretraining": pretraining,
        "before_yarn_adaptation": before_adaptation,
        "yarn_adaptation": adaptation,
        "after_yarn_adaptation": after_adaptation,
        "checkpoint": str(checkpoint_path),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    concise = {
        "scope": report["scope"],
        "parameters": report["parameters"],
        "pretraining": {key: value for key, value in pretraining.items() if key != "history"},
        "adaptation": {key: value for key, value in adaptation.items() if key != "history"},
        "before_yarn_adaptation": before_adaptation,
        "after_yarn_adaptation": after_adaptation,
        "checkpoint": str(checkpoint_path),
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"Wall time: {time.perf_counter() - started:.3f} seconds")
