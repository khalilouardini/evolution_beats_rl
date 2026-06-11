"""TrainerCallbacks for the toy GRPO pipeline.

Three callbacks per CLAUDE.md §7 (logging conventions) and §6 Phase 1:

- FLOPAccountingCallback: translates TRL's per-step token logs into our
  FLOPBudget (src/eval/compute.py) — the source of truth for the Phase 3
  compute-matched comparison. Writes runs/{name}/flops.json every N steps.
- MetricsJSONLCallback: mirrors every wandb-logged scalar to
  runs/{name}/metrics.jsonl line-by-line so a wandb outage can't lose data.
- PeriodicEvalCallback: greedy pass@1 on a held-out test slice every N steps
  (plus a step-0 baseline), reusing src/eval/harness.evaluate. Saves an
  adapter-only checkpoint at each eval point (Phase 3 needs per-checkpoint
  accuracy-vs-FLOPs curves).

TRL 0.29.1 integration facts these callbacks rely on (verified against the
installed package):
- With logging_steps=1, `on_log` receives a fresh logs dict every optimizer
  step containing `num_tokens` (cumulative prompt+completion tokens through
  the policy, train mode) and `completions/mean_length` (mean completion
  length over the generation batch).
- The end-of-training summary log (train_runtime etc.) lacks `num_tokens`
  and must be skipped.
- Callbacks receive the live (unwrapped) PeftModel via kwargs["model"].
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import wandb
from transformers import TrainerCallback

from src.eval.compute import FLOPBudget
from src.eval.harness import evaluate


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


class FLOPAccountingCallback(TrainerCallback):
    """Accumulate training FLOPs from TRL's token logs.

    Accounting model per optimizer step (our toy config: generation cycle and
    optimizer step are 1:1, num_iterations=1, no old-logps forward):
      - generation:  completion tokens only (mean_length × generation_batch_size),
        forward-only — same convention as Phase 0 eval (prompt FLOPs excluded).
      - policy train step: Δnum_tokens (prompt+completion, teacher-forced),
        forward+backward.
      - reference forward (KL, beta≠0): same Δnum_tokens, forward-only.

    `params` is the FULL base-model parameter count taken BEFORE the trainer
    wraps the model with LoRA (≤2% undercount on policy passes vs the wrapped
    model; identical accounting basis as Phase 0 eval and the future ES side —
    keeping the basis constant matters more than the 2%).
    """

    def __init__(
        self,
        run_dir: str | Path,
        params: int,
        beta: float,
        generation_batch_size: int,
        write_every_steps: int = 10,
    ):
        self.run_dir = Path(run_dir)
        self.params = params
        self.beta = beta
        self.generation_batch_size = generation_batch_size
        self.write_every_steps = write_every_steps
        self.budget = FLOPBudget()  # training compute (Phase 3 budget C)
        self.eval_budget = FLOPBudget()  # measurement compute (excluded from C)
        self._last_num_tokens = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "num_tokens" not in logs:
            return  # end-of-train summary log or non-token log

        delta = int(logs["num_tokens"]) - self._last_num_tokens
        if delta < 0:  # defensive: cumulative counter should never decrease
            return
        self._last_num_tokens = int(logs["num_tokens"])

        mean_len = logs.get("completions/mean_length")
        if mean_len is not None:
            completion_tokens = round(float(mean_len) * self.generation_batch_size)
            self.budget.add_generation(params=self.params, n_tokens=completion_tokens)

        self.budget.add_train_step(params=self.params, n_tokens=delta)
        if self.beta != 0.0:
            self.budget.add_reference_forward(params=self.params, n_tokens=delta)

        if wandb.run is not None:
            # No step= kwarg: HF's WandbCallback advances wandb's internal step
            # many times per optimizer step, so explicit step=global_step gets
            # dropped by wandb's monotonicity rule. We embed train/global_step
            # as a field instead; train.py registers it via define_metric as
            # the x-axis for flops/* and eval/*.
            wandb.log(
                {
                    "flops/total": self.budget.total_flops,
                    "flops/forward": self.budget.forward_flops,
                    "flops/backward": self.budget.backward_flops,
                    "flops/generated_tokens": self.budget.generated_tokens,
                    "train/global_step": state.global_step,
                }
            )

        if state.global_step % self.write_every_steps == 0:
            self.write_flops_json(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.write_flops_json(state.global_step)

    def write_flops_json(self, global_step: int) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            self.run_dir / "flops.json",
            {
                "train": self.budget.to_dict(),
                "eval": self.eval_budget.to_dict(),
                "params": self.params,
                "step": global_step,
            },
        )


class MetricsJSONLCallback(TrainerCallback):
    """Mirror every logged scalar to runs/{name}/metrics.jsonl (§7).

    One JSON object per line, flushed immediately, so a wandb outage or a
    crashed run leaves a complete local record up to the last step.
    """

    def __init__(self, run_dir: str | Path, flop_callback: FLOPAccountingCallback | None = None):
        self.run_dir = Path(run_dir)
        self.flop_callback = flop_callback
        self._path = self.run_dir / "metrics.jsonl"

    def _append(self, record: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        scalars = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if not scalars:
            return
        record = {"step": state.global_step, "time": time.time(), **scalars}
        if self.flop_callback is not None:
            record["flops_total"] = self.flop_callback.budget.total_flops
        self._append(record)

    def log_extra(self, step: int, metrics: dict) -> None:
        """Mirror out-of-band scalars (e.g. periodic eval results)."""
        scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        self._append({"step": step, "time": time.time(), **scalars})


class PeriodicEvalCallback(TrainerCallback):
    """Greedy pass@1 on a fixed test slice every `every_steps` optimizer steps.

    Reuses src/eval/harness.evaluate on the LIVE model (LoRA adapter active =
    the current policy). Restores training mode and tokenizer.padding_side in
    a finally block so a mid-eval crash can't silently corrupt training state.

    Deliberately does NOT pass seed= to evaluate(): that would reseed global
    RNGs mid-training and break the rollout-stream determinism guaranteed by
    set_all_seeds at run start. Greedy decoding needs no seed.

    Eval FLOPs go to a separate budget (measurement, not training compute —
    excluded from the Phase 3 budget C).
    """

    def __init__(
        self,
        eval_dataset,
        tokenizer,
        run_dir: str | Path,
        eval_flop_budget: FLOPBudget,
        jsonl_callback: MetricsJSONLCallback | None = None,
        every_steps: int = 25,
        batch_size: int = 4,
        max_new_tokens: int = 256,
        save_checkpoint: bool = True,
        eval_at_start: bool = True,
    ):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.run_dir = Path(run_dir)
        self.eval_flop_budget = eval_flop_budget
        self.jsonl_callback = jsonl_callback
        self.every_steps = every_steps
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.save_checkpoint = save_checkpoint
        self.eval_at_start = eval_at_start
        self.history: list[dict] = []  # [{"step": int, "pass_at_1": float, ...}]

    def on_train_begin(self, args, state, control, **kwargs):
        if self.eval_at_start:
            model = kwargs.get("model")
            if model is not None:
                self._run_eval(model, step=0, checkpoint=False)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_steps == 0:
            model = kwargs.get("model")
            if model is not None:
                self._run_eval(model, step=state.global_step, checkpoint=self.save_checkpoint)

    def _run_eval(self, model, step: int, checkpoint: bool) -> None:
        was_training = model.training
        prev_padding_side = self.tokenizer.padding_side
        try:
            model.eval()
            self.tokenizer.padding_side = "left"
            result = evaluate(
                model,
                self.tokenizer,
                self.eval_dataset,
                n_samples=1,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                batch_size=self.batch_size,
                flop_budget=self.eval_flop_budget,
            )
        finally:
            self.tokenizer.padding_side = prev_padding_side
            if was_training:
                model.train()

        record = {
            "eval/pass_at_1": result.pass_at_1,
            "eval/completion_tokens": result.completion_tokens_total,
            "eval/wall_clock_sec": result.wall_clock_sec,
        }
        self.history.append({"step": step, **record})
        if wandb.run is not None:
            # See FLOPAccountingCallback.on_log for why there's no step= kwarg.
            wandb.log({**record, "train/global_step": step})
        if self.jsonl_callback is not None:
            self.jsonl_callback.log_extra(step, record)

        if checkpoint:
            ckpt_dir = self.run_dir / f"ckpt_step{step}"
            model.save_pretrained(str(ckpt_dir))
