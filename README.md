# yarn-rope-128k-context-extension

> **Era:** 2023–2024 · Senior AI Engineer focus area (long-context model training). Techniques from 2023, reimplemented in 2026 with current PyTorch.

An explicit rotary-position-embedding implementation. NTK-aware scaling, YaRN corrections, a 128k configuration, and tiny-model experiments are added in subsequent stages. No full 128k evaluation has been run.

## Reproduce

```bash
pip install -e ".[dev]"
ruff check .
pytest
python scripts/check_rope.py
```

## References

- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021. arXiv:2104.09864
- B. Peng et al. *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR 2024. arXiv:2309.00071

## License

MIT
