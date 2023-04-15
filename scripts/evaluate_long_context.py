import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from yarn_context.experiment import evaluate_passkey
from yarn_context.model import TinyPasskeyConfig, TinyPasskeyModel
from yarn_context.passkey import make_passkey_perplexity_batch
from yarn_context.scaling import YaRNConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["passkey", "perplexity"], required=True)
    parser.add_argument("--method", choices=["original", "ntk", "yarn"], default="yarn")
    parser.add_argument("--length", type=int, default=131_072)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--rope-config",
        type=Path,
        help="Optional JSON override; head_dim must match the checkpoint",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.no_grad()
def synthetic_perplexity(
    model: TinyPasskeyModel, length: int, samples: int, batch_size: int
) -> dict[str, float | int | str]:
    """Causal NLL on local synthetic sequences, not natural-language perplexity."""
    generator = torch.Generator().manual_seed(500 + length)
    device = next(model.parameters()).device
    total_nll = 0.0
    token_count = 0
    seen = 0
    model.eval()
    while seen < samples:
        current_batch = min(batch_size, samples - seen)
        tokens = make_passkey_perplexity_batch(current_batch, length, generator).to(device)
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        sequence_nll = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="sum"
        )
        total_nll += float(sequence_nll)
        token_count += targets.numel()
        seen += current_batch
    nll = total_nll / token_count
    return {
        "scope": "synthetic causal perplexity, not natural-language model quality",
        "length": length,
        "sequences": samples,
        "target_tokens": token_count,
        "nll": nll,
        "perplexity": math.exp(nll),
    }


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config_values = payload["config"]
    rope_values = (
        json.loads(args.rope_config.read_text())
        if args.rope_config
        else config_values["rope"]
    )
    config_values["rope"] = YaRNConfig(**rope_values)
    config = TinyPasskeyConfig(**config_values)
    model = TinyPasskeyModel(config, method=args.method)
    model.load_state_dict(payload["state_dict"])
    model.to(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    if args.mode == "passkey":
        report = evaluate_passkey(
            model,
            args.method,
            args.length,
            samples=args.samples,
            batch_size=args.batch_size,
        )
    else:
        report = synthetic_perplexity(model, args.length, args.samples, args.batch_size)
    report["method"] = args.method
    report["device"] = args.device
    if args.device.startswith("cuda"):
        report["peak_memory_bytes"] = torch.cuda.max_memory_allocated(args.device)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
