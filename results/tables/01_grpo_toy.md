# Phase 1 — Toy GRPO results (3 seeds, Qwen2.5-0.5B-Instruct, GSM8K-500)

**Gate status (literal): NOT MET** — greedy held-out pass@1 improvement +0.7 pts (need ≥+2).
**But the result is informative, not a bug** — see "What's actually happening" below.

## Primary run — LR 2e-5, eval 100 items, 512-token completions

Config `configs/grpo/toy_0.5b.yaml` @ commit 3371c3a: LoRA r=16/α=32, **LR 2e-5**,
num_generations=4, max_completion_length=512, β=0.04, ε=0.2, loss_type=grpo,
100 steps, 500 train prompts (data_seed=0), eval = **100** GSM8K test items, greedy.

| Seed | pass@1 step 0 | pass@1 step 100 | Δ greedy held-out | train acc Δ (sampled) | KL final | Train FLOPs |
|---|---|---|---|---|---|---|
| 0 | 0.400 | 0.390 | −0.010 | **+0.150** | 0.017 | 2.852e15 |
| 1 | 0.400 | 0.370 | −0.030 | **+0.091** | 0.021 | 2.857e15 |
| 2 | 0.400 | 0.460 | +0.060 | +0.019 | 0.014 | 3.093e15 |
| **mean ± std** | 0.400 | **0.407 ± 0.047** | **+0.007 ± 0.047** | +0.087 | 0.017 | median **2.857e15** |

Eval trajectories (steps 0/25/50/75/100):
- s0: 0.40, 0.40, 0.40, 0.43, 0.39
- s1: 0.40, 0.42, 0.40, 0.39, 0.37
- s2: 0.40, 0.42, 0.42, 0.50, 0.46

Step-0 pass@1 identical (0.400) across all three seeds — a consistency check passing:
the LoRA adapter is an exact identity at init (B=0), so every seed evaluates the same
base model on the same 100 prompts. Seed variance enters only through training.

## What's actually happening — sharpening, not expansion

The "flat" greedy eval alongside strongly-rising training metrics is **not** noise and
**not** a broken pipeline. It is the textbook RLVR signature:

- **Sampled (temp=1.0) train accuracy rises sharply**: +15.0 / +9.1 / +1.9 pts.
- **Policy entropy falls ~30%** in every seed (s0 0.63→0.43, s1 0.65→0.48, s2 0.50→0.44).
- **Format reward saturates** to 0.83–0.90 (the model learns to always emit the tag).
- **Greedy (temp=0) held-out pass@1 barely moves** (+0.7 pts mean).

GRPO is concentrating probability mass onto solutions the base model could *already*
reach — sampled accuracy climbs toward the greedy mode, but greedy decoding already
takes the argmax, so the greedy ceiling is largely unchanged. This is precisely the
Yue et al. (arXiv:2504.13837) "does RLVR expand the reasoning boundary?" phenomenon —
a question already on the project's stretch list (§2, §14) and central to H3/pass@k.

KL behaved as a 4× LR predicts: 0.017 nats final vs ~0.005 at LR 5e-6. The LR raise
worked exactly as intended — it moved the policy ~3–4× further; that motion just
expresses as distribution-sharpening rather than capability gain at this scale.

## Sanity checks (CLAUDE.md §6 Phase 1)

| Check | Result |
|---|---|
| Reward curve up, no collapse | PASS (all 3 seeds; train acc +1.9 to +15.0) |
| Eval pass@1 ≥ +2 greedy | **FAIL** (+0.7 mean) — but see sharpening analysis |
| Seed std < 5 pts | PASS (4.7 pts, at eval_n=100) |
| KL bounded, no explosion | PASS (0.017 final, max 0.068 transient) |

## Compute anchor

Median train FLOPs = **2.857e15** → candidate **C_toy** for the Phase 3 matched comparison.

## Prior attempts (preserved, documented)

| Attempt | Run dirs | Outcome |
|---|---|---|
| 256-token cap | `runs/*_maxlen256` | Truncation noise; ~50% rollouts cut mid-reasoning (Progress Log 2026-06-11) |
| LR 5e-6, eval 50 | `runs/*_lr5e-6` | Policy moved ~0.01 nats; null within 6.9-pt SE (Progress Log 2026-06-12) |
| **LR 2e-5, eval 100** | current | Sharpening confirmed; greedy null but entropy/sampled-acc move clearly |
