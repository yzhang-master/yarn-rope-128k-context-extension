# yarn-rope-128k-context-extension

> **Era:** 2023–2024 · Senior AI Engineer focus area (long-context model training). Techniques from 2023, reimplemented in 2026 with current PyTorch.

Actual rotary position embeddings, fixed NTK-aware scaling, YaRN NTK-by-parts interpolation and attention-temperature correction, validated 128k-position configurations, and a real tiny causal-attention fine-tune/passkey experiment. The measured proxy reaches only 64 tokens; **full 128k results and GPU memory measurements are pending**.

## Problem

Extending a model beyond its training positions changes every rotary angle. Naive extrapolation can destroy attention behavior at longer distances, while scaling every frequency equally discards useful high-frequency information. A correct implementation must preserve RoPE geometry, apply NTK/YaRN frequency corrections in the right dimensions, account for attention-logit scale, and separate short proxy evidence from actual long-context evaluation.

## Method

### RoPE

For each even/odd half pair `(x1, x2)` at position `p`, RoPE applies:

```text
x1' = x1 cos(p θj) - x2 sin(p θj)
x2' = x2 cos(p θj) + x1 sin(p θj)
θj = base^(-2j / head_dim)
```

The implementation uses the duplicated-half layout `(x1, x2) -> (-x2, x1)` and arbitrary explicit positions rather than a fixed-size embedding table. Tests compare rotations with an independent complex-plane calculation, verify norm preservation, and check relative dot products.

### NTK-aware and YaRN scaling

Fixed NTK-aware scaling changes the base to:

```text
base' = base × factor^(head_dim / (head_dim - 2))
```

YaRN instead blends unscaled high frequencies and linearly interpolated low frequencies. For original context `L`, rotation thresholds `βfast=32`, `βslow=1`:

```text
d(r) = head_dim × log(L / (2πr)) / (2 log base)
low = floor(d(βfast)); high = ceil(d(βslow))
γj = clamp((j - low) / (high - low), 0, 1)
θ'j = θj [(1 - γj) + γj / factor]
```

Thus `γ=0` preserves high-frequency extrapolation and `γ=1` applies full interpolation. The YaRN attention correction is `mscale = 1 + 0.1 log(factor)`. It multiplies both query and key rotations, so attention logits gain `mscale²`; tests explicitly verify this rather than only checking cosine values.

`configs/yarn_128k.json` describes 4,096 → 131,072 positions (`factor=32`) for 64-dimensional heads. `configs/tiny_yarn_128k.json` is the matching 8-dimensional-head override for the tiny checkpoint. Tests evaluate finite rotations at position **131,071**. This validates configuration arithmetic, not a 131,072-token attention run.

### Tiny fine-tune and passkey proxy

The 5,136-parameter model has token embeddings, one four-head causal self-attention layer, QKV/output projections, and a token classifier. Each synthetic sequence contains one numeric token in positions 1–5 amid filler tokens 6–10; token 11 at the end queries the number. Exact retrieval is measured at 16, 32, and 64 tokens with 128 fixed-seed samples per cell.

Training first runs 300 steps at length 16 with original RoPE. The same weights are evaluated with original, NTK, and YaRN rotations. A separate 80-step YaRN adaptation then runs at length 32. This is real gradient-based training on local generated data, not a scripted retrieval score.

## Results

Measured locally with Python 3.12.3, Torch 2.15.0+cpu, seed 19, one Torch thread, batch size 32, and initial learning rate 0.01 (0.005 for adaptation).

- Pre-training: loss **2.781836 → 0.000008173**, 300 steps, **1.991 s**.
- YaRN adaptation: initial loss **0.000010036**, final loss **0.0** at float32 print precision, 80 steps, **0.814 s**.
- Entire experiment wall time: **6.543 s**, including 27 evaluation cells and checkpoint writes.

Accuracy before YaRN adaptation:

| Length | Original RoPE | NTK-aware | YaRN |
|---:|---:|---:|---:|
| 16 | 128/128 = 100.00% | 128/128 = 100.00% | 128/128 = 100.00% |
| 32 | 95/128 = 74.22% | 124/128 = 96.88% | 128/128 = 100.00% |
| 64 | 77/128 = 60.16% | 87/128 = 67.97% | 128/128 = 100.00% |

Accuracy after 80-step YaRN adaptation:

| Length | Original RoPE | NTK-aware | YaRN |
|---:|---:|---:|---:|
| 16 | 128/128 = 100.00% | 128/128 = 100.00% | 128/128 = 100.00% |
| 32 | 121/128 = 94.53% | 128/128 = 100.00% | 128/128 = 100.00% |
| 64 | 118/128 = 92.19% | 116/128 = 90.63% | 128/128 = 100.00% |

These are deterministic single-run results on an easy synthetic task. **They do not demonstrate 128k retrieval, long-context language quality, or a guaranteed advantage on pretrained LLMs.**

### Full 128k and GPU memory: pending

No full-length attention evaluation was run, and no GPU was available. The following real commands load the saved tiny checkpoint, override it with the compatible 128k YaRN configuration, and write exact metrics. The perplexity mode reports causal perplexity on local synthetic sequences, **not natural-language perplexity**.

```bash
python scripts/finetune_passkey.py
python scripts/evaluate_long_context.py \
  --checkpoint results/tiny_passkey_yarn.pt --mode passkey \
  --length 131072 --samples 128 --batch-size 1 \
  --rope-config configs/tiny_yarn_128k.json --device cpu \
  --output results/passkey_128k.json
python scripts/evaluate_long_context.py \
  --checkpoint results/tiny_passkey_yarn.pt --mode perplexity \
  --length 131072 --samples 8 --batch-size 1 \
  --rope-config configs/tiny_yarn_128k.json --device cpu \
  --output results/perplexity_128k.json
```

On a suitable GPU, changing `--device cpu` to `--device cuda` also writes `peak_memory_bytes` from `torch.cuda.max_memory_allocated`:

```bash
python scripts/evaluate_long_context.py \
  --checkpoint results/tiny_passkey_yarn.pt --mode passkey \
  --length 131072 --samples 128 --batch-size 1 \
  --rope-config configs/tiny_yarn_128k.json --device cuda \
  --output results/passkey_128k_cuda_memory.json
```

| Pending measurement | Command output |
|---|---|
| 131,072-token tiny passkey accuracy | `results/passkey_128k.json` |
| 131,072-token synthetic causal perplexity | `results/perplexity_128k.json` |
| 131,072-token CUDA peak allocated bytes | `results/passkey_128k_cuda_memory.json` |

The evaluator materializes full causal attention scores and therefore has quadratic sequence-memory cost. These commands may require substantial RAM/VRAM; no successful full-length execution or memory footprint is claimed.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest
python scripts/check_rope.py
python scripts/finetune_passkey.py --smoke --output results/smoke_passkey.json
python scripts/finetune_passkey.py
```

Without installing, prefix Python commands with `PYTHONPATH=src`. The JSON report includes all 300 + 80 loss values and every initial/before/after accuracy and loss. Two checkpoints are saved. A real checkpoint-load path can be smoke-tested at short length with:

```bash
python scripts/evaluate_long_context.py \
  --checkpoint results/tiny_passkey_yarn.pt --mode passkey \
  --length 64 --samples 8 --batch-size 4 \
  --output results/checkpoint_passkey_64.json
```

Tests cover manual RoPE and scaling formulas, ramp direction/bounds, NTK base equivalence, query/key/logit temperature scaling, both 128k configurations, causality, generated-data invariants, genuine loss reduction, and longer-sequence evaluation.

## Limitations

- The proxy model is randomly initialized and task-specific. It is not a pretrained transformer or a general language model.
- Numeric passkey retrieval at 16–64 tokens is much easier than natural-language passkey benchmarks and does not measure long-range semantics.
- Full attention is deliberately explicit and quadratic. There is no FlashAttention, KV cache, sliding window, ring attention, or memory-efficient kernel.
- Fixed NTK scaling is implemented, not dynamic per-forward NTK interpolation. YaRN uses the standard fixed NTK-by-parts ramp.
- Only 128k position arithmetic is tested locally. Full 128k passkey/perplexity, natural-language evaluation, pretrained-model integration, and all GPU measurements remain pending.
- No API, hosted model, external dataset, GPU result, or full-context claim is used.

## References

- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021. arXiv:2104.09864
- B. Peng et al. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR 2024. arXiv:2309.00071

## License

MIT
