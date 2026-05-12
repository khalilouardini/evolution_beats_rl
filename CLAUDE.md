# CLAUDE.md — ES vs GRPO+RLVR on Verifiable Rewards
You are working on an ML research project autonomously. Read this file completely on every session start. Update the **Progress Log** at the bottom every time you finish a task or hit a decision point. Commit after every successful experiment.
---
## 1. Mission
Compare **zeroth-order optimization (Evolution Strategies in LoRA-space)** against **first-order policy gradient (GRPO with verifiable rewards)** on math reasoning tasks, **matched by compute budget**.
Fill the gap identified in Cognizant AI Lab, arXiv:2509.24372, which benchmarks ES vs vanilla PPO but not GRPO+RLVR — the regime where RL's known weaknesses (high gradient variance under long horizons, reward hacking) are already partially mitigated.
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
- **Hardware**: 1 local GPU, assume 24GB VRAM (RTX 4090 / 3090 / A5000 class). Tune memory params for what you actually find with `nvidia-smi`.
- **Cloud burst**: Modal or RunPod spot allowed for ablations only. **Confirm with user before any run >$50.** Track total spend in `results/spend.md`.
- **Wall-clock**: project should fit in roughly 4 weekends + scattered evenings of GPU time. If a single experiment will take >36h, stop and report.
- **Determinism**: every run must set `seed` from config and write `seed` to wandb. No silent seed defaults.
- **No fabricated numbers**. Ever. If a run failed, the table cell is empty and a note explains why.
---
## 4. Stack
Pin these in `pyproject.toml`:
- **GRPO**: `trl` (TRL ≥ 0.12 has stable GRPO+vLLM colocate). Fallback to `verl` if TRL becomes unstable at scale.
- **Generation**: `vllm` in colocate mode (`vllm_mode="colocate"`, `gpu_memory_utilization=0.3–0.5`)
- **PEFT**: `peft` for LoRA adapters
- **ES**: custom implementation in `src/es/`. Reference: github.com/VsonicV/es-fine-tuning-paper (read it before writing; do not copy verbatim)
- **Eval**: `lm-eval-harness` for GSM8K / MATH-500 standard metrics; custom thin wrapper in `src/eval/`
- **Logging**: `wandb` (project = `es-vs-grpo`). Group runs by `phase` tag.
- **Configs**: hydra or plain yaml + `OmegaConf`. Pick one and stick with it.
Do **not** add: deepspeed, accelerate multi-GPU plumbing, FSDP. Single GPU only — these add complexity that buys nothing here.
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
Each phase has an **exit gate**. Do not start phase N+1 until the gate for N is met. Update Progress Log when a gate passes.
### Phase 0 — Bootstrap (target: half a day)
1. Init repo, write `pyproject.toml`, install deps. Resolve any CUDA/torch version conflicts now, not later.
2. Implement `src/data/gsm8k.py`:
   - Load `openai/gsm8k` (main split).
   - Verifier extracts the final numeric answer (regex on `#### N` or last number) and compares to ground truth.
   - Unit test: 100 reference completions should give ≥95% verifier agreement with the gold labels.
3. Implement `src/eval/harness.py`:
   - Take a model + LoRA adapter, run greedy decode on GSM8K test, return pass@1.
   - Also support `n_samples` for pass@k via temperature 0.7 sampling.
4. Implement `src/eval/compute.py`:
   - FLOP counter: forward ≈ 2 · params · tokens; backward ≈ 4 · params · tokens.
   - For ES, only forward. For GRPO, forward+backward on policy, forward on ref.
   - Returns dict `{forward_flops, backward_flops, total_flops, generated_tokens}` per run.
5. Write `scripts/smoke_test.sh`:
   - Loads Qwen2.5-0.5B-Instruct, runs eval on 50 GSM8K test items, asserts pass@1 ≥ 0.30.
   - Must complete in <5 min on the target GPU.
**Exit gate 0**: `make smoke` passes; baseline Qwen2.5-0.5B-Instruct pass@1 on GSM8K test recorded in `results/tables/00_baseline.md`. Expect ~0.40–0.45.
### Phase 1 — GRPO Baseline (target: 2 days GPU time)
Goal: a GRPO+LoRA+vLLM pipeline that reliably improves a small Qwen on GSM8K, with full compute accounting.
1. Implement `src/grpo/train.py` using TRL `GRPOTrainer`:
   - Model: Qwen2.5-1.5B-Instruct (primary); Qwen2.5-0.5B-Instruct (fast loop).
   - LoRA: r=16, alpha=32, target = all-linear, dropout=0.
   - Reward: accuracy (binary, 1 if verifier matches) + format (small bonus for `<answer>...</answer>` tag). Log both separately.
   - `num_generations=8`, `max_completion_length=512`, `max_prompt_length=512`.
   - `use_vllm=True`, `vllm_mode="colocate"`, `vllm_gpu_memory_utilization=0.35` (tune up if VRAM allows).
   - `learning_rate=5e-6`, `gradient_accumulation_steps=4`, `beta=0.04` (KL coef).
   - Log to wandb: reward mean/std per step, generation length, KL, accuracy on 100-prompt eval slice every 50 steps.
2. Run on Qwen2.5-0.5B-Instruct, 5 seeds, ~500 steps each. Total ~10–15 GPU hours.
3. Run on Qwen2.5-1.5B-Instruct, 5 seeds, ~800 steps each. Total ~30–50 GPU hours. Split across nights.
**Sanity checks before declaring this phase done:**
- Reward curve goes up and plateaus, doesn't collapse.
- Eval pass@1 improves by ≥5 absolute points on 0.5B, ≥3 on 1.5B (Qwen Instruct is already strong).
- Seed-to-seed final-accuracy std is small enough to detect ES differences (<3 absolute points).
- KL to ref grows but doesn't explode (>50 is bad — lower LR or raise beta).
**Exit gate 1**: 5-seed mean ± std GRPO numbers logged for both models in `results/tables/01_grpo_baseline.md`. Total FLOPs and wall-clock per run recorded.
### Phase 2 — ES Implementation (target: 3 days)
Goal: a working LoRA-space ES that demonstrably improves a small model on GSM8K, even if the absolute number is below GRPO.
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
   - Fitness shaping: rank-normalize rewards before computing the update (Wierstra et al.). This is critical for stability — without it ES is brittle.
   - Use Adam on the ES gradient estimate (not raw SGD). LR=0.01 starting.
   - σ schedule: constant 0.02 to start. Add cosine decay if unstable.
3. `src/es/train.py` main loop:
   - For each generation: sample N=40 perturbations (so 80 forward rollouts with antithetic).
   - Each perturbation evaluated on a batch of B=16 prompts; fitness = mean accuracy + small format bonus.
   - Use vLLM in offline mode for the rollouts — swap LoRA weights between perturbations, batch all prompts per perturbation.
   - **Critical optimization**: with vLLM offline + LoRA hot-swap, each generation should take O(minutes), not O(hours). If it doesn't, profile before scaling.
   - Log to wandb: best fitness per gen, mean fitness, fitness std across population, σ, learning rate, accumulated FLOPs.
4. Smoke run: Qwen2.5-0.5B-Instruct, LoRA r=8, N=20, 30 generations. Should show clear upward fitness trend within 1–2 hours. If not, debug before going further.
5. Real run: Qwen2.5-0.5B-Instruct, LoRA r=16, N=40, 100+ generations, 3 seeds. Total ~15–25 GPU hours.
**Exit gate 2**: ES improves Qwen2.5-0.5B-Instruct GSM8K pass@1 by ≥3 absolute points over the SFT baseline; results in `results/tables/02_es_baseline.md`. The improvement does **not** need to match GRPO yet — just demonstrate the method works.
### Phase 3 — Matched Comparison (target: 2 weekends GPU time)
This is the headline experiment. Get it right.
1. Define the **compute budget** $C$ explicitly. Use the FLOP estimate from phase 1's GRPO runs: $C = $ median total FLOPs of a successful GRPO run.
2. Configure each ES run to terminate when its accumulated FLOPs reach $C$ (not after a fixed number of generations).
3. Run, for both Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, both datasets (GSM8K, MATH-500):
   - GRPO: 5 seeds at budget $C$
   - ES: 5 seeds at budget $C$
4. Hyperparameter fairness: do **not** tune ES hyperparameters on the test set. Pick σ, N, LR from phase 2 outcomes, lock them, then run.
5. Evaluate every checkpoint on the held-out test set, plot accuracy-vs-cumulative-FLOPs curves with seed envelopes.
**Exit gate 3**: `results/figures/pareto.pdf` exists, showing both methods with 95% CI bands across 5 seeds, for both models. Numerical summary table in `results/tables/03_matched.md`.
### Phase 4 — Ablations (target: 1 weekend, can use cloud)
Pick **at most 3** of the following based on what's most surprising in phase 3:
- **Rank sweep**: r ∈ {4, 8, 16, 32, 64} — does ES degrade faster with rank?
- **Horizon sweep**: max_completion_length ∈ {256, 512, 1024, 2048} — does ES's horizon-independence show up?
- **Reward density**: GSM8K (sparse) vs a process-reward variant (dense). Hypothesis: ES gap shrinks with denser reward.
- **Population size**: ES N ∈ {20, 40, 80, 160} — sample efficiency Pareto.
- **Base model family**: Llama-3.2-1B and SmolLM2-1.7B as cross-family check.
- **ES variant**: simple Gaussian ES vs sNES vs CMA-ES on small rank.
Do not try to do all of these. Pick the 3 that best support the phase 3 narrative or expose its limits.
**Exit gate 4**: Each chosen ablation has a figure or table in `results/`.
### Phase 5 — Writeup (target: half a week)
1. Draft sections in `paper/` (use ICLR or NeurIPS workshop template):
   - Abstract
   - Introduction (lead with the gap from arXiv:2509.24372)
   - Background (GRPO equations, ES equations, LoRA)
   - Method (LoRA-space ES, fairness protocol, compute accounting)
   - Experiments
   - Limitations (single GPU, two model sizes, English math only)
   - Related work
2. Generate all figures from `results/` via `make figs`. No hand-tweaked plots without script.
3. Push code + README + paper draft. Stop and surface to user for review.
**Exit gate 5**: User reviews paper draft.
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
| Single training run OOMs | Lower `vllm_gpu_memory_utilization` by 0.05, then per-device batch by half. If still OOM, drop model size and note in log. |
| Training run NaNs after >100 steps | Lower LR by 2×, restart from last good checkpoint. After 2 NaN retries on same config, stop and report. |
| GRPO reward collapses (mean drops by >50% over 100 steps) | Raise KL coef β, restart. After 2 retries, stop and report. |
| ES fitness flat for >20 generations | First check σ (too small → no signal; too large → noise drowns signal). Try σ × 2 and σ / 2 each for 10 gens. If still flat, escalate. |
| Cloud cost projected >$50 for the next experiment | **Stop. Ask user before spending.** |
| Surprising result (e.g. ES > GRPO by >5 points) | Re-run with 2 fresh seeds. Verify with a different eval slice. Do **not** publish without replication. |
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
**GRPO:**
- LoRA: r=16, α=32, dropout=0, target=all-linear
- LR=5e-6, gradient_accumulation_steps=4, num_generations=8
- max_completion_length=512, max_prompt_length=512
- β (KL coef)=0.04, ε (clip)=0.2
- Reward = accuracy + 0.1 · format_bonus
- vllm_gpu_memory_utilization=0.35, sleep_level=1
**ES:**
- LoRA: r=16, α=32, dropout=0, target=all-linear
- N=40 (population), antithetic=True
- σ=0.02 constant
- Optimizer: Adam, LR=0.01
- Fitness shaping: centered rank
- Batch B=16 prompts per perturbation evaluation
- Same reward as GRPO for fairness
**Eval:**
- pass@1: greedy (temperature=0)
- pass@k for k>1: temperature=0.7, top_p=0.95
- Always use the test split — never the train split — for reported numbers
---
## 11. Known Gotchas
- **TRL + vLLM colocate** can hang silently if `sleep_level` and `gpu_memory_utilization` aren't tuned for the GPU. If a run is stuck with idle GPU util, kill and lower memory util by 0.05.
- **LoRA flatten/unflatten** is the #1 source of silent ES bugs. Test bit-exact round-trip on a real model, not a toy.
- **vLLM LoRA hot-swap** has a small overhead per swap; batch all evaluations per perturbation together.
- **Qwen2.5 Instruct on GSM8K is already strong** (~50% pass@1 for 0.5B, ~73% for 1.5B). Don't expect huge headline gains from either method — the interesting signal is in stability, sample efficiency curves, and the Pareto frontier.
- **GSM8K format reward** can be hacked: model emits `<answer>X</answer>` for any X to grab the bonus. Keep format bonus small (≤0.1 of accuracy reward) and audit completions early.
- **Seed isolation**: HF `set_seed` does not seed vLLM's sampler. Pass `seed=` to `SamplingParams` explicitly.
- **MATH-500 verifier** is harder than GSM8K's — fractions, surds, equivalent forms. Use the `latex2sympy` route from `lm-eval-harness` rather than rolling your own.
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
2026-05-12  phase-0  New independent git repo at ~/Desktop/projects/evolution_beats_rl/ (separate from the surrounding Desktop-rooted git repo whose remote is callRounded/F-Project). Scaffolded pyproject.toml (torch/trl/vllm/peft/lm-eval/wandb/omegaconf pinned per §4), Makefile, README.md, .gitignore (excludes .claude/ since a worktree of an unrelated repo lives there), src/ skeleton, tests/, configs/, scripts/, results/spend.md (zero spend). Initial commit made. Next: install deps and implement src/data/gsm8k.py, src/eval/harness.py, src/eval/compute.py, scripts/smoke_test.sh per §6 Phase 0.
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
