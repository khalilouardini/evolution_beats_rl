# ES vs GRPO+RLVR

Compute-matched comparison of **LoRA-space Evolution Strategies** and **GRPO with verifiable rewards**, on math reasoning tasks (GSM8K, MATH-500).

**Research question.** Given equal FLOPs and equal LoRA parameter budget, does ES match or exceed GRPO+RLVR on (1) sample efficiency, (2) seed-to-seed stability, (3) generalization (pass@k)?

This work fills a gap in Cognizant AI Lab's *Evolution Strategies at Scale* ([arXiv:2509.24372](https://arxiv.org/abs/2509.24372)), which benchmarks ES vs **vanilla PPO** but not GRPO+RLVR — the regime where RL's known weaknesses (high gradient variance, reward hacking) are already partially mitigated.

## Quickstart

```bash
# Python 3.10–3.12, single GPU with ≥24 GB VRAM (RTX 4090 / 3090 / A5000 class)
uv venv --python 3.11 .venv
source .venv/bin/activate

make install-dev

# Phase 0 smoke: Qwen2.5-0.5B-Instruct, 50 GSM8K prompts, pass@1 ≥ 0.30
make smoke
```

## Status

Phase 0 (Bootstrap) — scaffolding complete; data loader, eval harness, FLOP counter next.

## Plan and conventions

See [`CLAUDE.md`](./CLAUDE.md) for the full phased plan, exit gates, hyperparameter defaults, decision rules, and progress log.
