# CLAUDE.md — ES vs GRPO+RLVR on Verifiable Rewards
You are working on an ML research project autonomously. Read this file completely on every session start. Update the **Progress Log** at the bottom every time you finish a task or hit a decision point. Commit after every successful experiment.
---
## 1. Mission
Compare **zeroth-order optimization (Evolution Strategies in LoRA-space)** against **first-order policy gradient (GRPO with verifiable rewards)** on math reasoning tasks, **matched by compute budget**.
Fill the gap identified in Cognizant AI Lab, arXiv:2509.24372, which benchmarks ES vs vanilla PPO but not GRPO+RLVR — the regime where RL's known weaknesses (high gradient variance under long horizons, reward hacking) are already partially mitigated.
### Execution strategy: toy-scale first, scale on validation
Headline numbers require CUDA hardware (Phase 4 on Modal/RunPod). But before spending cloud GPU-hours, we validate the full pipeline (loaders, verifier, FLOP counter, LoRA flatten/unflatten, ES update rule, fairness protocol) on a small-scale comparison runnable locally on Apple Silicon. Phases 1–3 are toy-scale on M1 Pro; Phase 3.5 is a go/no-go decision gate; Phase 4+ is cloud-scale. **Toy results are pipeline validation and hypothesis screening, not headline.**
### Primary research question
Given equal FLOPs and equal LoRA parameter budget, does ES match or exceed GRPO+RLVR on:
1. **Sample efficiency** — prompts to reach target accuracy
2. **Stability** — seed-to-seed variance of final accuracy
3. **Generalization** — pass@k for k ∈ {1, 8, 32}, OOD test sets
### Hypothesis (state for falsification)
- H1: ES wins on (2) — lower seed variance due to Gaussian smoothing
- H2: GRPO wins on (1) — variance reduction from group-normalized advantages + verifiable rewards closes the gap ES exploited vs vanilla PPO
- H3: ES is competitive on (3) at small ranks; degrades faster than GRPO as rank grows
If the empirical result contradicts all three, that itself is the paper. **Do not discard surprising results — interrogate them.**
---
## 2. Success Criteria
**Minimum bar for writeup (workshop paper, ~4 pages):**
- [ ] 5-seed runs for every headline configuration
- [ ] Compute-matched comparison (FLOPs counted, not just wall time)
- [ ] ≥2 model sizes: Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct
- [ ] ≥2 datasets: GSM8K (primary) + MATH-500 (secondary)
- [ ] One Pareto frontier plot: test accuracy vs cumulative compute, with 95% CI bands
- [ ] One stability plot: distribution of final accuracy across seeds, per method
- [ ] Honest failure-mode section
- [ ] Reproducible: single command `make reproduce` re-runs all headline numbers
**Stretch:**
- [ ] Code/HumanEval as third domain
- [ ] CMA-ES or sNES variant vs simple Gaussian ES
- [ ] Pass@k boundary analysis (Yue et al. capability-shrinkage debate)
- [ ] Qwen2.5-3B size scaling point (cloud burst)
---
## 3. Hard Constraints
- **Hardware (toy phase, Phases 0–3)**: MacBook Pro M1 Pro, 16 GB unified memory, Metal 3 (MPS). No CUDA, no `nvidia-smi`. Realistic working set: ~10 GB after macOS overhead. Close browser tabs before runs.
- **Hardware (cloud phase, Phase 4+)**: Modal or RunPod CUDA box (target A10G or A100 spot). Project-level cloud budget approved ≈ **$400–1200** total. Single-run cap **$50** still applies — confirm before any one job projects over that. Track spend in `results/spend.md`.
- **Wall-clock (toy phase)**: each single-seed toy run ≤ 4h. If a toy experiment will take >8h, downsize first (fewer steps, smaller train subset) before launching.
- **Wall-clock (cloud phase)**: each single-seed cloud run ≤ 6h. If projected >36h, stop and report.
- **Determinism**: every run must set `seed` from config and write `seed` to wandb. No silent seed defaults. (Note: no vLLM seed plumbing needed in toy phase — only the transformers/torch seed.)
- **No fabricated numbers**. Ever. If a run failed, the table cell is empty and a note explains why.
---
## 4. Stack
Pinned in `pyproject.toml`. Toy-phase install is `uv pip install -e ".[dev]"`. Cloud-phase install adds the `cloud` extra: `uv pip install -e ".[dev,cloud]"`.

**Always-on:**
- **GRPO**: `trl` (≥0.12, <1). Toy phase: `use_vllm=False`, generations via `model.generate()`. Cloud phase: `use_vllm=True, vllm_mode="colocate"`. Fallback to `verl` if TRL becomes unstable at cloud scale.
- **PEFT**: `peft` (≥0.13) for LoRA adapters.
- **ES**: custom implementation in `src/es/`. Reference: github.com/VsonicV/es-fine-tuning-paper (read it before writing; do not copy verbatim).
- **Eval**: `lm-eval-harness` for GSM8K / MATH-500 standard metrics; custom thin wrapper in `src/eval/`.
- **Logging**: `wandb` (project = `es-vs-grpo`). Group runs by `phase` tag. Tag toy runs with `scale=toy`, cloud runs with `scale=cloud`.
- **Configs**: plain yaml + `OmegaConf`. (Decision: plain yaml over hydra — hydra's CLI composition isn't worth the import-time cost for this project size.)

**Cloud-only (Phase 4+):**
- **vLLM** (≥0.6.3, <0.7) in colocate mode (`vllm_mode="colocate"`, `gpu_memory_utilization=0.3–0.5`). Not installed in toy phase — has no usable MPS backend.

**Forbidden:**
- `deepspeed`, `accelerate` multi-GPU plumbing, FSDP — single GPU/single-process only; these add complexity that buys nothing here.
- `bitsandbytes`, 4-bit/8-bit quantization in toy phase — CUDA-only; will crash on M1 Pro.
- `flash-attn` in toy phase — CUDA-only.
---
## 5. Repo Layout
```
es-vs-grpo/
├── CLAUDE.md                   # this file — keep current
├── README.md                   # public-facing summary
├── pyproject.toml
├── Makefile                    # `make smoke`, `make repro`, `make figs`
├── configs/
│   ├── grpo/                   # one yaml per (model, dataset) pair
│   ├── es/
│   └── ablations/
├── src/
│   ├── data/
│   │   ├── gsm8k.py            # loader + verifier
│   │   ├── math500.py
│   │   └── rewards.py          # accuracy + format reward fns
│   ├── grpo/
│   │   └── train.py            # TRL GRPOTrainer wrapper
│   ├── es/
│   │   ├── salimans.py         # antithetic NES
│   │   ├── snes.py             # separable NES (stretch)
│   │   └── train.py            # main ES loop
│   ├── eval/
│   │   ├── harness.py          # shared pass@k evaluator
│   │   └── compute.py          # FLOP accounting
│   └── utils/
│       ├── lora.py             # LoRA param flatten/unflatten for ES
│       ├── checkpoint.py
│       └── seed.py
├── scripts/
│   ├── smoke_test.sh
│   ├── grpo_baseline.sh
│   ├── es_baseline.sh
│   └── matched_comparison.sh
├── tests/
│   ├── test_rewards.py
│   ├── test_lora_flatten.py    # round-trip param vector ↔ LoRA dict
│   └── test_compute_counter.py
├── runs/                       # gitignored
└── results/
    ├── spend.md
    ├── tables/
    └── figures/
```
---
## 6. Phased Plan
Phases 0–3 are **toy-scale on M1 Pro** (pipeline validation + hypothesis screening). Phase 3.5 is the go/no-go decision gate. Phase 4+ are **cloud-scale on Modal** (publishable numbers). Each phase has an exit gate; do not start phase N+1 until the gate for N is met. Update Progress Log when a gate passes.

### Phase 0 — Bootstrap (target: half a day on M1 Pro)
1. Init repo, write `pyproject.toml`, install M1 deps (`uv pip install -e ".[dev]"` — no `cloud` extra).
2. Implement `src/data/gsm8k.py`:
   - Load `openai/gsm8k` (main split).
   - Verifier extracts the final numeric answer (regex on `#### N` or last number) and compares to ground truth.
   - Unit test: 100 reference completions should give ≥95% verifier agreement with the gold labels.
3. Implement `src/eval/harness.py`:
   - Take a model + LoRA adapter, run greedy decode (via `transformers.generate()`) on GSM8K test slice, return pass@1.
   - Also support `n_samples` for pass@k via temperature 0.7 sampling.
   - MPS device auto-select; CPU fallback for ops that need it (set `PYTORCH_ENABLE_MPS_FALLBACK=1`).
4. Implement `src/eval/compute.py`:
   - FLOP counter: forward ≈ 2 · params · tokens; backward ≈ 4 · params · tokens.
   - For ES, only forward. For GRPO, forward+backward on policy, forward on ref.
   - Returns dict `{forward_flops, backward_flops, total_flops, generated_tokens}` per run.
5. Write `scripts/smoke_test.sh`:
   - Loads Qwen2.5-0.5B-Instruct on MPS, runs eval on 50 GSM8K test items, asserts pass@1 ≥ 0.30.
   - Wall-clock budget: <20 min on M1 Pro (was <5 min on 4090).
**Exit gate 0**: `make smoke` passes; baseline Qwen2.5-0.5B-Instruct pass@1 on GSM8K test recorded in `results/tables/00_baseline.md`. Expect ~0.40–0.45.

### Phase 1 — TOY GRPO Baseline on M1 Pro (target: 1 weekend)
Goal: a GRPO+LoRA pipeline (no vLLM, generations via `transformers.generate()`) that demonstrably improves Qwen2.5-0.5B-Instruct on a 500-prompt GSM8K subset, with full FLOP accounting.
1. Implement `src/grpo/train.py` using TRL `GRPOTrainer`:
   - Model: **Qwen2.5-0.5B-Instruct only**. 1.5B deferred to Phase 4 (won't fit M1 Pro comfortably during training).
   - LoRA: r=16, α=32, target=all-linear, dropout=0.
   - Reward: accuracy (binary) + format (small bonus for `<answer>...</answer>`). Log both separately.
   - `num_generations=4` (toy, was 8), `max_completion_length=512`, `max_prompt_length=256`. (512 was originally 256; raised after seed-0 forensics showed ~50% of rollouts truncated mid-reasoning at 256, polluting reward and eval — see Progress Log 2026-06-11.)
   - `use_vllm=False`. Rollouts via `model.generate()` on MPS.
   - `learning_rate=5e-6`, `gradient_accumulation_steps=4`, `beta=0.04` (KL).
   - Train subset: 500 prompts of GSM8K `main/train`. Eval slice: 50 prompts of `main/test` every 25 steps.
   - Log to wandb: reward mean/std per step, gen length, KL, eval pass@1, accumulated FLOPs.
2. Run, **3 seeds, 100 steps each**. Target wall-clock: ≤4h/seed.

**Sanity checks before declaring this phase done:**
- Reward curve goes up and plateaus, doesn't collapse.
- Eval pass@1 improves by ≥2 absolute points on the eval slice (lower bar than scaled plan — we're on a subset).
- Seed-to-seed final-accuracy std small enough to detect ES differences at toy scale (<5 abs points).
- KL to ref grows but doesn't explode.

**Exit gate 1 (toy)**: 3-seed mean ± std logged in `results/tables/01_grpo_toy.md`. Total FLOPs and wall-clock per run recorded.

### Phase 2 — TOY ES Baseline on M1 Pro (target: 1 weekend)
Goal: a working LoRA-space ES that demonstrably improves Qwen2.5-0.5B-Instruct on the same 500-prompt GSM8K subset.
1. Implement `src/utils/lora.py`:
   - `flatten_lora(model) -> torch.Tensor` returning a 1-D vector of all LoRA A and B params.
   - `unflatten_lora(model, vec)` writing the vector back in place.
   - `lora_shape(model) -> dict` with per-layer slice info.
   - **Test**: round-trip preserves model output bit-exactly. Block this phase on the test passing.
2. Implement `src/es/salimans.py`:
   ```python
   def es_step(theta, eval_fn, sigma, N, antithetic=True):
       # Sample epsilons, evaluate f(theta ± sigma*eps) via eval_fn
       # Return estimated gradient (Eq. 2 from Salimans 2017)
   ```
   - Antithetic sampling on by default.
   - Fitness shaping: centered-rank normalize before update (Wierstra et al.). Critical for stability — without it ES is brittle.
   - Adam on the ES gradient estimate. LR=0.01.
   - σ constant 0.02 to start. Add cosine decay only if unstable.
3. `src/es/train.py` main loop:
   - Per generation: sample **N=10 perturbations** (toy, was 40) → 20 forward rollouts with antithetic.
   - Each perturbation evaluated on a batch of **B=8** prompts (toy, was 16); fitness = mean accuracy + 0.1·format bonus.
   - Use `transformers.generate()` (no vLLM in toy). **Batch all prompts per perturbation in a single generate() call** — without this, ES will be dramatically slower than necessary.
   - LoRA hot-swap between perturbations via the §11 flatten/unflatten path; no model rebuild per perturbation.
   - Log to wandb: best/mean/std fitness per gen, σ, LR, accumulated FLOPs.
4. Smoke run: Qwen2.5-0.5B-Instruct, LoRA r=8, N=10, 5 generations. Should show *any* upward fitness trend within 30–60 min. If not, debug before scaling.
5. Real toy run: Qwen2.5-0.5B-Instruct, LoRA **r=8** (smaller than GRPO toy — ES degrades faster with rank, this gives ES a fair shot at toy scale), N=10, 30 generations, 3 seeds. Target ~5–10h/seed.

**Exit gate 2 (toy)**: ES improves Qwen2.5-0.5B-Instruct GSM8K pass@1 by ≥1 absolute point over the SFT baseline on the toy subset; results in `results/tables/02_es_toy.md`. Improvement does NOT need to match GRPO at toy scale — just demonstrate the method moves the model.

### Phase 3 — TOY Matched Comparison on M1 Pro (target: 1 weekend)
This is the local headline — proof-of-concept, not publishable.
1. Define toy compute budget $C_{toy}$ = median total FLOPs of a successful Phase-1 GRPO run.
2. Configure ES to terminate when accumulated FLOPs reach $C_{toy}$.
3. Run, Qwen2.5-0.5B-Instruct only, GSM8K only:
   - GRPO: 3 seeds at $C_{toy}$.
   - ES: 3 seeds at $C_{toy}$.
4. Hyperparameter fairness: do NOT tune ES hyperparameters on test. Lock σ, N, LR from Phase 2. Same for GRPO LR, β from Phase 1.
5. Evaluate every checkpoint on the 100-prompt held-out eval slice. Plot accuracy-vs-cumulative-FLOPs curves with seed envelopes.

**Exit gate 3 (toy)**: `results/figures/pareto_toy.pdf` exists with both methods, 3-seed bands. Numerical summary in `results/tables/03_toy_matched.md`.

### Phase 3.5 — Decision Gate (target: 1 hour, then surface to user)
Look at Phase 3 toy results and answer **three questions in writing** in `results/tables/03_5_decision.md`:
1. **Pipeline correctness check**: does the FLOP counter agree with `wall_clock × measured_tok_per_sec × ~6 · params` within ±20%? If not, the comparison is on broken accounting — fix before promoting.
2. **Signal check**: is there *any* method-vs-method difference (≥1 absolute point gap between ES and GRPO at compute-matched, in either direction)? If literally identical curves, suspect a bug.
3. **Direction check**: does the toy signal align with H1/H2/H3 from §1, or contradict all three? Either is fine — write it down.

Then propose to user one of: **promote** to Phase 4 cloud / **debug** Phase 3 and re-run / **rescope** (different research question). Do not promote silently.

**Exit gate 3.5**: User signs off on Phase 4 launch (or directs an alternative).

### Phase 4 — CLOUD Headline Experiments on Modal (target: 2 weekends + scattered)
The publishable comparison. Prerequisites:
- Modal account + image with `uv pip install -e ".[dev,cloud]"`.
- Per-experiment cost estimated in `results/spend.md` *before* launch.
- Spot pricing assumed (A10G ≈ $0.6/hr spot, A100 ≈ $1.5/hr spot). Project budget cap ~$1200.

1. Define cloud compute budget $C$ = median total FLOPs of a successful Phase-4 GRPO run on Qwen2.5-0.5B-Instruct full GSM8K.
2. Configure ES to terminate at $C$.
3. Run, for **both Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct**, **both datasets (GSM8K, MATH-500)**:
   - GRPO (with vLLM colocate): 5 seeds at $C$.
   - ES: 5 seeds at $C$.
   Total estimated ~200 GPU-hours, ~$400–700 spot.
4. Hyperparameter fairness: Phase 3 toy-locked ES hyperparameters. Phase 1 toy-locked GRPO LR + β. **No tuning on test set.**
5. Evaluate every checkpoint on held-out test set. Plot accuracy-vs-cumulative-FLOPs with 95% CI bands across 5 seeds.

**Exit gate 4**: `results/figures/pareto.pdf` shows both methods with 95% CI bands across 5 seeds, both models, both datasets. Numerical summary in `results/tables/04_cloud_matched.md`.

### Phase 5 — Ablations (target: 1 weekend cloud)
Pick **at most 3** of the following based on what's most surprising in Phase 4:
- **Rank sweep**: r ∈ {4, 8, 16, 32, 64} — does ES degrade faster with rank?
- **Horizon sweep**: max_completion_length ∈ {256, 512, 1024, 2048} — does ES's horizon-independence show up?
- **Reward density**: GSM8K (sparse) vs a process-reward variant (dense). Hypothesis: ES gap shrinks with denser reward.
- **Population size**: ES N ∈ {20, 40, 80, 160} — sample efficiency Pareto.
- **Base model family**: Llama-3.2-1B and SmolLM2-1.7B as cross-family check.
- **ES variant**: simple Gaussian ES vs sNES vs CMA-ES on small rank.

Do not try to do all of these. Pick the 3 that best support the Phase 4 narrative or expose its limits.

**Exit gate 5**: Each chosen ablation has a figure or table in `results/`.

### Phase 6 — Writeup (target: half a week)
1. Draft sections in `paper/` (use ICLR or NeurIPS workshop template):
   - Abstract
   - Introduction (lead with the gap from arXiv:2509.24372)
   - Background (GRPO equations, ES equations, LoRA)
   - Method (LoRA-space ES, fairness protocol, compute accounting, toy→cloud staging)
   - Experiments (toy validation results in appendix; headline cloud results in main body)
   - Limitations (single GPU per run, two model sizes, English math only)
   - Related work
2. Generate all figures from `results/` via `make figs`. No hand-tweaked plots without script.
3. Push code + README + paper draft. Stop and surface to user for review.

**Exit gate 6**: User reviews paper draft.
---
## 7. Conventions
### Naming
- Run names: `{phase}_{method}_{model}_{seed}_{date}` — e.g. `p3_es_qwen1.5b_s42_2026-05-18`.
- Wandb groups: by phase. Wandb tags: `{method}`, `{model}`, `{dataset}`.
- Checkpoints: `runs/{run_name}/ckpt_step{N}.pt`.
### Git
- Branch per phase: `phase-0-bootstrap`, `phase-1-grpo`, etc. Squash-merge to main on gate pass.
- Commit message format: `[phase-N] short description`.
- **Never** commit `runs/`, `wandb/`, or any `.pt` file. `.gitignore` is your friend.
### Logging
- Every run logs to wandb. Run dies → check wandb for last logged step.
- Local mirror: every wandb-logged scalar also dumped to `runs/{name}/metrics.jsonl` line-by-line, so a wandb outage doesn't kill the experiment.
- Compute accumulator in `runs/{name}/flops.json` updated every 10 steps.
### Reproducibility
- Every config yaml includes `seed`, `git_sha`, `model_revision` (HF commit hash).
- `make repro` re-runs all phase-1/2/3 headline configs from yaml. It should produce numbers within ±0.5 abs points of recorded values.
---
## 8. Decision Rules
These tell you when to **proceed**, **retry**, or **stop and ask**.
| Situation | Action |
|---|---|
| Smoke test fails | Debug. Do not advance phase. |
| Single training run OOMs on M1 Pro (toy) | Lower per-device batch by half; if still OOM, lower `num_generations` by 2; if still OOM, drop `max_completion_length` by half. M1 Pro unified RAM is the budget — close other apps. |
| Single training run OOMs on Modal (cloud) | Lower `vllm_gpu_memory_utilization` by 0.05, then per-device batch by half. If still OOM, drop model size and note. |
| Training run NaNs after >100 steps | Lower LR by 2×, restart from last good checkpoint. After 2 NaN retries on same config, stop and report. |
| GRPO reward collapses (mean drops by >50% over 100 steps) | Raise KL coef β, restart. After 2 retries, stop and report. |
| ES fitness flat for >20 generations | Check σ (too small → no signal; too large → noise drowns signal). Try σ × 2 and σ / 2 each for 10 gens. If still flat, escalate. |
| Toy result shows no method-vs-method signal at Phase 3.5 | Debug FLOP counter and verifier first; do NOT promote to Phase 4. Re-run Phase 3 toy with fresh seeds. |
| Toy result shows clear gap (>5 abs points) at Phase 3.5 | Document, then propose Phase 4 launch. Do NOT assume scale will preserve the gap — write down the prediction. |
| Cloud cost projected >$50 for the next single experiment | **Stop. Ask user before spending.** |
| Surprising result (e.g. ES > GRPO by >5 points at cloud scale) | Re-run with 2 fresh seeds. Verify with a different eval slice. Do **not** publish without replication. |
| You think the plan needs to change | Edit this CLAUDE.md and commit the change with `[plan-update]` prefix. Surface the change in the next progress update. |
| You're blocked >2 hours on infra | Stop, write up the blocker in `Progress Log`, surface to user. |
---
## 9. Fast Eval Loop (use this constantly)
Before any "real" run, verify the change on this 5-minute loop:
```bash
# 5 min: tiny model + tiny data + 20 steps
python -m src.grpo.train \
  --config configs/grpo/smoke.yaml \
  --max_steps 20 \
  --eval_subset 50
```
`configs/grpo/smoke.yaml` should use Qwen2.5-0.5B-Instruct, 50 train prompts, eval on 50 test prompts. If pass@1 doesn't move at all in 20 steps **and** the smoke config has worked before, your change broke something. Bisect.
Same idea for ES: `configs/es/smoke.yaml` uses N=10, 5 generations, eval on 50 prompts.
---
## 10. Hyperparameter Defaults
Start here, deviate only with logged justification.

### TOY phase (Phases 1–3, M1 Pro)
**GRPO toy:**
- Model: Qwen2.5-0.5B-Instruct
- LoRA: r=16, α=32, dropout=0, target=all-linear
- LR=5e-6, gradient_accumulation_steps=4, **num_generations=4**
- **max_completion_length=512** (raised from 256 — truncation noise, see Progress Log 2026-06-11), **max_prompt_length=256**
- β (KL coef)=0.04, ε (clip)=0.2
- Reward = accuracy + 0.1 · format_bonus
- `use_vllm=False` (vLLM has no MPS backend)
- Train subset: 500 prompts; eval slice: 50 prompts every 25 steps
- max_steps=100; 3 seeds

**ES toy:**
- Model: Qwen2.5-0.5B-Instruct
- LoRA: **r=8**, α=16, dropout=0, target=all-linear
- **N=10** (population), antithetic=True (→ 20 forward rollouts/gen)
- σ=0.02 constant
- Optimizer: Adam, LR=0.01
- Fitness shaping: centered rank
- **Batch B=8** prompts per perturbation evaluation
- max_completion_length=512 for rollouts (matches GRPO toy — fairness)
- 30 generations; 3 seeds
- Same reward as GRPO for fairness

### CLOUD phase (Phase 4+, Modal)
**GRPO cloud:**
- Models: Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct
- LoRA: r=16, α=32, dropout=0, target=all-linear
- LR=5e-6, gradient_accumulation_steps=4, num_generations=8
- max_completion_length=512, max_prompt_length=512
- β=0.04, ε=0.2
- Reward = accuracy + 0.1 · format_bonus
- `use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.35, sleep_level=1`
- Full GSM8K train + MATH-500; max_steps=500 (0.5B), 800 (1.5B); 5 seeds

**ES cloud:**
- Models: Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct
- LoRA: r=16, α=32, dropout=0, target=all-linear
- N=40 (population), antithetic=True
- σ=0.02 constant (may adjust based on Phase 3 outcome)
- Optimizer: Adam, LR=0.01 (locked from Phase 3 toy)
- Fitness shaping: centered rank
- Batch B=16 prompts per perturbation evaluation
- 100+ generations; 5 seeds
- Same reward as GRPO for fairness

### Eval (both phases)
- pass@1: greedy (temperature=0)
- pass@k for k>1: temperature=0.7, top_p=0.95
- Always use the test split — never the train split — for reported numbers
---
## 11. Known Gotchas

### Always
- **LoRA flatten/unflatten** is the #1 source of silent ES bugs. Test bit-exact round-trip on a real model, not a toy.
- **Qwen2.5 Instruct on GSM8K is already strong** (~50% pass@1 for 0.5B, ~73% for 1.5B). Don't expect huge headline gains from either method — the interesting signal is in stability, sample efficiency curves, and the Pareto frontier.
- **GSM8K format reward** can be hacked: model emits `<answer>X</answer>` for any X to grab the bonus. Keep format bonus small (≤0.1 of accuracy reward) and audit completions early.
- **Seed isolation**: HF `set_seed` does not seed vLLM's sampler (Phase 4+). Pass `seed=` to `SamplingParams` explicitly.
- **MATH-500 verifier** (Phase 4+) is harder than GSM8K's — fractions, surds, equivalent forms. Use the `latex2sympy` route from `lm-eval-harness` rather than rolling your own.

### Toy phase (M1 Pro / MPS)
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** is required for some attention/normalization ops that don't have native MPS kernels in older torch. Set in the smoke script and training entrypoints.
- **Unified memory pressure**: M1 Pro 16 GB is shared with macOS, browsers, IDE. Realistic working budget ≈ 10 GB. Close browser tabs before launching training; watch for `Memory pressure: critical` in Activity Monitor.
- **No `bitsandbytes`, no `flash-attn`, no `xformers`** on MPS. If a TRL/peft config silently tries to load 8-bit/4-bit quantization, it'll crash. Force `torch_dtype=torch.bfloat16` (MPS supports bf16) or `torch.float16`.
- **MPS bf16 has subtle dtype-promotion bugs in some torch 2.4.x releases.** If you see NaNs immediately on bf16, try fp16 or fp32 and file a bug.
- **No `device_map="auto"` on MPS** for small models — it'll uselessly try to shard. Use `device_map="mps"` or `model.to("mps")` directly.

### Cloud phase (CUDA / Modal)
- **TRL + vLLM colocate** can hang silently if `sleep_level` and `gpu_memory_utilization` aren't tuned for the GPU. If a run is stuck with idle GPU util, kill and lower memory util by 0.05.
- **vLLM LoRA hot-swap** has a small overhead per swap; batch all evaluations per perturbation together.
---
## 12. What to Ask the User Before Doing
- Any cloud run with projected cost >$50.
- Any change that violates a numbered constraint in §3.
- Any deviation from §6 phase structure (skipping phases, reordering).
- Surprising results before re-running for replication.
- When phase 5 draft is ready.
You do **not** need to ask before:
- Editing this CLAUDE.md to fix typos, clarify, or add lessons learned.
- Choosing among the ablations in phase 4 — pick whichever 3 best support the narrative.
- Routine debugging, retries within the decision-rule budget.
- Adding tests.
---
## 13. Progress Log
Append entries below. Most recent first. Each entry: ISO date, phase, what happened, next action.
```
2026-06-11  plan-update  Seed-0 toy GRPO (100 steps, max_completion_length=256) completed cleanly in 1.44h but MISSED the +2-point eval gate: train reward 0.48→0.70 (+46%, no collapse, KL≤0.004 healthy) but decomposition shows accuracy flat (0.325→0.338) with format reward doubling (0.156→0.362); eval pass@1 0.26→0.26→0.30→0.24→0.22 (flat within 50-item noise, SE≈0.06). Completion audit (base vs ckpt_step100 adapter on 6 eval prompts) ACQUITTED format hacking — completions are legitimate CoT, only 2/6 tagged — and CONVICTED the 256-token cap: base completions run 700-930 chars and truncate mid-reasoning; last-number fallback then extracts noise INCLUDING a false positive ("8 ml/800 calories ×" → 800 = gold, by luck). Trained model is genuinely adapting (more concise: clipped_ratio 0.49→0.375; one audit case finishes at 430 chars and flips wrong→right). Diagnosis: both reward signal and eval metric measured through a truncation-noise channel; instrument noisy, method healthy. DECISION (user-approved): raise max_completion_length 256→512 train+eval (matches cloud config; ES toy will use 512 too — fairness preserved), re-run seed 0. §6/§10 updated. Old run preserved at runs/p1_grpo_qwen0.5b_s0_2026-06-11_maxlen256. Projected ~2.5-3h/seed at 512, within §3 budget. FLOP accounting note: 256-run cost 2.404e15 train FLOPs — within 3% of the 2.47e15 projection; counter solid.
2026-06-11  phase-1  GRPO smoke (20 steps, Qwen2.5-0.5B, MPS) PASSED all 8 checklist points. Steady step time 61.4s → 100-step projection 2.16h/seed incl. evals (≤4h budget ✓). KL 0→7.6e-4 finite; accuracy_reward step-1 0.5, run-mean 0.338; format 0.125; no NaNs on bf16. completions/clipped_ratio=0.53 — half of rollouts hit the 256-token cap (reward-density concern at toy scale, logged not changed). FLOP counter internally exact; 20-step train total 4.93e14 → 2.47e15 per 100-step run (matches design anchor ~2.5e15). metrics.jsonl mirrored 24/24 records; adapter ckpts at eval steps. Eval pass@1 0.26/0.24/0.28 at steps 0/10/20 (flat at this horizon, expected). Step-0 eval 0.26 vs Phase-0 0.32: torch 2.8 kernel change + 50-item noise; not investigated. Earlier 2-step probe found+fixed (commit 3b8812a): trl 0.29.1 needs torch≥2.6 (FSDPModule import; bumped to 2.8.0, vllm cloud extra → ≥0.7, §11 colocate facts need Phase-4 re-verify) and wandb monotonic-step conflict silently dropping flops/eval logs (fixed via define_metric; jsonl mirror caught it, validating §7). TRL 0.29.1 landmines defused in config: loss_type default "dapo"→ set "grpo"; beta default 0.0→ set 0.04; bf16 default-flips-True→ set False. Ops note: harness Bash timeout (10 min) SIGTERM-killed the first smoke attempt; long runs now launched detached (nohup+disown). Next: seed-0 toy run (~2.2h), user reviews curves, then seeds 1+2.
2026-05-12  phase-0  **EXIT GATE 0 PASSED.** Smoke run on M1 Pro MPS: Qwen2.5-0.5B-Instruct, 50 GSM8K test items, max_new_tokens=256, batch_size=4, seed=0 → pass@1 = 0.3200 (16/50). Threshold ≥0.30 satisfied. Wall-clock 5.3 min (well under 20-min budget). Throughput 36.9 tok/s batched bf16 eager-attention. 11,775 completion tokens, 1.163e13 forward FLOPs (counter integration validated against 2·P·T formula exactly). Note: 0.32 is below CLAUDE.md §6 Phase 0 expected 0.40–0.45 — hypotheses: (a) max_new_tokens=256 truncates some chains-of-thought on harder items, (b) 50-item subset noise (SE ≈ 0.07 at p=0.4). Not investigated further — baseline is for the gate, not for method evaluation; clean re-measurement at Phase 1 eval. MPS gotcha discovered + fixed: Qwen2.5 GQA crashes torch sdpa kernel on MPS ("mps_matmul: incompatible dimensions"); load_qwen_model now forces attn_implementation="eager" on MPS (commit cb2fc2d, documented §11). Implementation commits (this session): seed.py fa65609, compute.py e6edd3f, harness.py cb2fc2d, smoke_test.sh 1c4a7ec, baseline + this log entry (pending). Next: open Phase 1 branch (phase-1-grpo-toy), implement src/grpo/train.py with TRL GRPOTrainer + use_vllm=False, run 3-seed toy GRPO per §6 Phase 1.
2026-05-12  plan-update  Hardware reality check: target machine is MacBook Pro M1 Pro 16 GB unified memory, no CUDA. CLAUDE.md was scoped for RTX 4090-class CUDA hardware; full plan unrunnable as written. Rescoped to "toy first, scale on validation": Phases 1–3 now toy-scale on M1 Pro (Qwen2.5-0.5B only, GSM8K-subset, 3 seeds, no vLLM, transformers.generate() for rollouts). Inserted Phase 3.5 decision gate. Phase 4 is now CLOUD headline (Modal, both models, both datasets, 5 seeds, vLLM). Renumbered Phases 4→5 (ablations) and 5→6 (writeup). Updated §3, §4, §6, §8, §10, §11 accordingly. Dropped vllm from default deps, moved to `cloud` optional extra in pyproject.toml. Re-verified dep resolution clean on M1 (no vllm) and on Modal-target (vllm 0.6.6.post1 present). Next: install deps on M1 (`uv pip install -e ".[dev]"`), then implement src/data/gsm8k.py per Phase 0 step 2.
2026-05-12  phase-0  New independent git repo at ~/Desktop/projects/evolution_beats_rl/ (separate from the surrounding Desktop-rooted git repo whose remote is callRounded/F-Project). Scaffolded pyproject.toml, Makefile, README.md, .gitignore, CLAUDE.md, src/ skeleton, tests/, configs/, scripts/, results/spend.md (zero spend). Initial commit. Then: uv pip compile flagged latex2sympy2↔omegaconf antlr4 transitive collision; dropped latex2sympy2 from direct deps (CLAUDE.md §11 says use lm-eval-harness's minerva_math pipeline for MATH-500 latex anyway). Capped transformers <5, trl <1, vllm <0.7, peft <0.15 to stay in CLAUDE.md's tested API surface. Resolved: torch 2.5.1 / transformers 4.57.6 / trl 0.29.1 / vllm 0.6.6.post1 / peft 0.14.0 / lm-eval 0.4.12 / accelerate 1.13.0 (transitive). Commit fc41dff.
2026-05-11  phase-0  Project initialized from CLAUDE.md spec. Next: scaffold repo, install deps.
```
---
## 14. References
- Cognizant AI Lab — *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*, arXiv:2509.24372, github.com/VsonicV/es-fine-tuning-paper
- Salimans et al. — *Evolution Strategies as a Scalable Alternative to RL*, arXiv:1703.03864
- Shao et al. — *DeepSeekMath / GRPO*, arXiv:2402.03300
- Hu et al. — *LoRA*, arXiv:2106.09685
- Aghajanyan et al. — *Intrinsic Dimensionality of Language Model Fine-Tuning*, arXiv:2012.13255
- Wierstra et al. — *Natural Evolution Strategies*, JMLR 2014 (fitness shaping)
- Yue et al. — *Does RLVR Expand the Reasoning Boundary?*, arXiv:2504.13837 (pass@k debate)
