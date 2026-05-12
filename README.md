# ES vs GRPO+RLVR

Compute-matched comparison of **LoRA-space Evolution Strategies** and **GRPO with verifiable rewards**, on math reasoning tasks (GSM8K, MATH-500).

**Research question.** Given equal FLOPs and equal LoRA parameter budget, does ES match or exceed GRPO+RLVR on (1) sample efficiency, (2) seed-to-seed stability, (3) generalization (pass@k)?

This work fills a gap in Cognizant AI Lab's *Evolution Strategies at Scale* ([arXiv:2509.24372](https://arxiv.org/abs/2509.24372)), which benchmarks ES vs **vanilla PPO** but not GRPO+RLVR — the regime where RL's known weaknesses (high gradient variance, reward hacking) are already partially mitigated.

## Execution strategy

Toy-scale validation on Apple Silicon (M1 Pro, MPS) first → cloud-scale headline runs on Modal once toy results pass the Phase 3.5 decision gate. Phases 0–3 use Qwen2.5-0.5B-Instruct on a GSM8K subset with no vLLM (rollouts via `transformers.generate()`). Phase 4+ scales to both Qwen2.5-{0.5B,1.5B}-Instruct, full GSM8K + MATH-500, 5 seeds, vLLM colocate. See [`CLAUDE.md`](./CLAUDE.md) §6 for details.

## Quickstart (toy phase, Apple Silicon)

```bash
# Python 3.10–3.12. Tested on MacBook Pro M1 Pro, 16 GB unified memory.
uv venv --python 3.11 .venv
source .venv/bin/activate

make install-dev   # toy install — no vLLM

# Phase 0 smoke: Qwen2.5-0.5B-Instruct on MPS, 50 GSM8K prompts, pass@1 ≥ 0.30
make smoke
```

## Cloud install (Phase 4+, CUDA Linux)

```bash
make install-cloud   # adds vllm 0.6.x on top of toy deps
```

## Status

Phase 0 (Bootstrap) — scaffold + plan-update complete; data loader, eval harness, FLOP counter, smoke script next.

## Plan and conventions

See [`CLAUDE.md`](./CLAUDE.md) for the full phased plan, exit gates, hyperparameter defaults, decision rules, and progress log.
