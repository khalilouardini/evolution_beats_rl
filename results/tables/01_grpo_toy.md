# Phase 1 — Toy GRPO results (3 seeds, Qwen2.5-0.5B-Instruct, GSM8K-500)

**Gate status: NOT MET** (requires mean eval pass@1 improvement ≥ +2 points; measured +0.0).

Config: `configs/grpo/toy_0.5b.yaml` @ commit 88d78a7 — LoRA r=16/α=32, LR 5e-6,
num_generations=4, max_completion_length=512, β=0.04, ε=0.2, loss_type=grpo,
100 steps, 500 train prompts (data_seed=0), eval = 50 GSM8K test items, greedy.

| Seed | Run | pass@1 step 0 | pass@1 step 100 | Δ | Train FLOPs | Eval FLOPs | Wall (train) |
|---|---|---|---|---|---|---|---|
| 0 | `p1_grpo_qwen0.5b_s0_2026-06-11` | 0.380 | 0.420 | +0.040 | 2.973e15 | 7.56e13 | 3.02 h |
| 1 | `p1_grpo_qwen0.5b_s1_2026-06-12` | 0.380 | 0.340 | −0.040 | 2.939e15 | 7.34e13 | 2.98 h |
| 2 | `p1_grpo_qwen0.5b_s2_2026-06-12` | 0.380 | 0.380 | ±0.000 | 3.010e15 | 7.33e13 | 3.14 h |
| **mean ± std** | | 0.380 | **0.380 ± 0.040** | **+0.000 ± 0.040** | median **2.973e15** | | 3.05 h |

Full eval trajectories (steps 0/25/50/75/100):
- s0: 0.38, 0.42, 0.38, 0.38, 0.42
- s1: 0.38, 0.42, 0.30, 0.36, 0.34
- s2: 0.38, 0.32, 0.40, 0.40, 0.38

## What the runs show

- **Training-distribution improvement is real and consistent**: train accuracy reward
  rose in all 3 seeds (+2.5 / +5.0 / +8.4 points, first→last 20-step band, sampled
  at temperature 1.0). Format reward adopted in all 3 (0.26–0.38 → 0.59–0.64).
  No reward collapse, no NaNs, KL bounded (max 0.011 nats).
- **The policy displacement is tiny**: final KL to reference 0.003–0.008 nats.
  LR 5e-6 was inherited from full-parameter GRPO configs; on LoRA-only training
  with β=0.04 the effective step is very small.
- **The eval instrument cannot resolve effects of this size**: 50 items → SE ≈ 6.9
  points at p≈0.4. All per-seed eval deltas (±4 pts) are within one SE.
  Seed-to-seed std (4.0 pts) satisfies the <5 pt detectability bar, but the
  mean improvement (+0.0) misses the ≥+2 requirement.

## Sanity checks (CLAUDE.md §6 Phase 1)

| Check | Result |
|---|---|
| Reward curve up, no collapse | PASS (all 3 seeds) |
| Eval pass@1 +2 pts | **FAIL** (+0.0 mean) |
| Seed std < 5 pts | PASS (4.0 pts) |
| KL bounded | PASS (max 0.011) |

## Compute anchor

Median train FLOPs across seeds = **2.973e15** — candidate C_toy for the Phase 3
matched comparison, pending resolution of the gate failure.

Prior artifacts: the discarded 256-token-cap seed-0 run (truncation-noise
forensics) is preserved at `runs/p1_grpo_qwen0.5b_s0_2026-06-11_maxlen256`;
see Progress Log 2026-06-11.
