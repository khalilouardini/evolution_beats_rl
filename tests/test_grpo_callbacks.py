"""Tests for src/grpo/callbacks.py — all GPU-free, no TRL trainer needed.

The callbacks are exercised directly with mocked TrainerState / logs dicts,
which pins the integration contract with TRL 0.29.1's logging format:
logs contain cumulative "num_tokens" and per-batch "completions/mean_length".
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import src.grpo.callbacks as callbacks_module
from src.eval.compute import FLOPBudget
from src.eval.harness import EvalResult
from src.grpo.callbacks import (
    FLOPAccountingCallback,
    MetricsJSONLCallback,
    PeriodicEvalCallback,
)

P = 494_032_768  # Qwen2.5-0.5B param count, used as a realistic constant
GEN_BATCH = 16


def state_at(step: int) -> SimpleNamespace:
    return SimpleNamespace(global_step=step)


# -----------------------------------------------------------------------------
# FLOPAccountingCallback
# -----------------------------------------------------------------------------

def make_flop_cb(tmp_path, beta=0.04, write_every=10) -> FLOPAccountingCallback:
    return FLOPAccountingCallback(
        run_dir=tmp_path, params=P, beta=beta,
        generation_batch_size=GEN_BATCH, write_every_steps=write_every,
    )


def test_flop_arithmetic_single_step(tmp_path):
    cb = make_flop_cb(tmp_path)
    # Step 1: cumulative num_tokens=5280 (prompt+completion), mean completion 230.
    cb.on_log(None, state_at(1), None, logs={"num_tokens": 5280, "completions/mean_length": 230.0})

    gen_tokens = round(230.0 * GEN_BATCH)  # 3680
    # forward: generation (2PT) + train fwd (2PT) + ref fwd (2PT)
    assert cb.budget.forward_flops == 2 * P * gen_tokens + 2 * P * 5280 + 2 * P * 5280
    assert cb.budget.backward_flops == 4 * P * 5280
    assert cb.budget.generated_tokens == gen_tokens
    assert cb.budget.training_tokens == 5280


def test_flop_delta_decoding_across_steps(tmp_path):
    cb = make_flop_cb(tmp_path)
    cb.on_log(None, state_at(1), None, logs={"num_tokens": 5280, "completions/mean_length": 230.0})
    # Step 2: cumulative doubles → delta is 5280, not 10560.
    cb.on_log(None, state_at(2), None, logs={"num_tokens": 10560, "completions/mean_length": 230.0})
    assert cb.budget.training_tokens == 10560  # 5280 + 5280
    assert cb.budget.generated_tokens == 2 * round(230.0 * GEN_BATCH)


def test_flop_beta_zero_skips_reference_forward(tmp_path):
    cb = make_flop_cb(tmp_path, beta=0.0)
    cb.on_log(None, state_at(1), None, logs={"num_tokens": 1000, "completions/mean_length": 50.0})
    gen_tokens = round(50.0 * GEN_BATCH)
    # forward: generation + train fwd only — NO ref forward
    assert cb.budget.forward_flops == 2 * P * gen_tokens + 2 * P * 1000


def test_flop_summary_log_without_num_tokens_is_noop(tmp_path):
    cb = make_flop_cb(tmp_path)
    # End-of-training summary log (train_runtime etc.) must not crash or count.
    cb.on_log(None, state_at(20), None, logs={"train_runtime": 123.4, "train_loss": 0.01})
    assert cb.budget.total_flops == 0
    cb.on_log(None, state_at(20), None, logs=None)
    assert cb.budget.total_flops == 0


def test_flops_json_written_at_cadence_only(tmp_path):
    cb = make_flop_cb(tmp_path, write_every=10)
    flops_path = tmp_path / "flops.json"

    cb.on_log(None, state_at(9), None, logs={"num_tokens": 100, "completions/mean_length": 5.0})
    assert not flops_path.exists()

    cb.on_log(None, state_at(10), None, logs={"num_tokens": 200, "completions/mean_length": 5.0})
    assert flops_path.exists()
    payload = json.loads(flops_path.read_text())
    assert set(payload.keys()) == {"train", "eval", "params", "step"}
    assert payload["step"] == 10
    assert payload["params"] == P
    assert payload["train"]["training_tokens"] == 200


def test_flops_json_final_write_on_train_end(tmp_path):
    cb = make_flop_cb(tmp_path, write_every=10)
    cb.on_log(None, state_at(3), None, logs={"num_tokens": 300, "completions/mean_length": 7.0})
    cb.on_train_end(None, state_at(3), None)
    payload = json.loads((tmp_path / "flops.json").read_text())
    assert payload["step"] == 3
    assert payload["train"]["training_tokens"] == 300


# -----------------------------------------------------------------------------
# MetricsJSONLCallback
# -----------------------------------------------------------------------------

def test_jsonl_mirror_appends_parseable_scalar_lines(tmp_path):
    cb = MetricsJSONLCallback(run_dir=tmp_path)
    cb.on_log(None, state_at(1), None, logs={"loss": 0.5, "reward": 1.1, "note": "not-a-scalar"})
    cb.on_log(None, state_at(2), None, logs={"loss": 0.4, "kl": 0.02})

    lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    r1, r2 = (json.loads(line) for line in lines)
    assert r1["step"] == 1 and r1["loss"] == 0.5 and r1["reward"] == 1.1
    assert "note" not in r1  # non-scalars filtered
    assert r2["step"] == 2 and r2["kl"] == 0.02
    assert "time" in r1


def test_jsonl_includes_flops_when_wired(tmp_path):
    flop_cb = make_flop_cb(tmp_path)
    flop_cb.on_log(None, state_at(1), None, logs={"num_tokens": 1000, "completions/mean_length": 50.0})
    cb = MetricsJSONLCallback(run_dir=tmp_path, flop_callback=flop_cb)
    cb.on_log(None, state_at(1), None, logs={"loss": 0.5})
    record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert record["flops_total"] == flop_cb.budget.total_flops > 0


def test_jsonl_log_extra(tmp_path):
    cb = MetricsJSONLCallback(run_dir=tmp_path)
    cb.log_extra(25, {"eval/pass_at_1": 0.34, "ignored": "string"})
    record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert record == pytest.approx({"step": 25, "time": record["time"], "eval/pass_at_1": 0.34})


def test_jsonl_empty_or_nonscalar_logs_skipped(tmp_path):
    cb = MetricsJSONLCallback(run_dir=tmp_path)
    cb.on_log(None, state_at(1), None, logs=None)
    cb.on_log(None, state_at(1), None, logs={"only": "strings"})
    assert not (tmp_path / "metrics.jsonl").exists()


# -----------------------------------------------------------------------------
# PeriodicEvalCallback
# -----------------------------------------------------------------------------

class FakeModel:
    """Minimal stand-in for a PeftModel: training flag + save_pretrained."""

    def __init__(self):
        self.training = True
        self.saved_to: list[str] = []

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def save_pretrained(self, path):
        self.saved_to.append(path)


class FakeTokenizer:
    padding_side = "left"


def canned_result(pass_at_1=0.40) -> EvalResult:
    return EvalResult(
        pass_at_1=pass_at_1, pass_at_k=None, n_items=50, n_samples_per_item=1,
        correct_per_item=[1] * 20 + [0] * 30, completion_tokens_total=11_000,
        wall_clock_sec=300.0, model_params=P,
    )


def make_eval_cb(tmp_path, jsonl=None, **kw) -> PeriodicEvalCallback:
    defaults = dict(
        eval_dataset=[{"question": "q", "answer": "#### 1"}],
        tokenizer=FakeTokenizer(),
        run_dir=tmp_path,
        eval_flop_budget=FLOPBudget(),
        jsonl_callback=jsonl,
        every_steps=25,
        save_checkpoint=True,
        eval_at_start=True,
    )
    defaults.update(kw)
    return PeriodicEvalCallback(**defaults)


def test_eval_fires_at_cadence_only(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(callbacks_module, "evaluate", lambda *a, **k: (calls.append(1), canned_result())[1])
    cb = make_eval_cb(tmp_path, eval_at_start=False)
    model = FakeModel()

    for step in [1, 24, 25, 26, 49, 50]:
        cb.on_step_end(None, state_at(step), None, model=model)
    assert len(calls) == 2  # steps 25 and 50 only
    assert [h["step"] for h in cb.history] == [25, 50]


def test_eval_at_start_baseline_no_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(callbacks_module, "evaluate", lambda *a, **k: canned_result(0.32))
    cb = make_eval_cb(tmp_path)
    model = FakeModel()
    cb.on_train_begin(None, state_at(0), None, model=model)
    assert cb.history == [{
        "step": 0, "eval/pass_at_1": 0.32,
        "eval/completion_tokens": 11_000, "eval/wall_clock_sec": 300.0,
    }]
    assert model.saved_to == []  # step-0 baseline saves no checkpoint


def test_eval_checkpoint_saved_at_eval_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(callbacks_module, "evaluate", lambda *a, **k: canned_result())
    cb = make_eval_cb(tmp_path, eval_at_start=False)
    model = FakeModel()
    cb.on_step_end(None, state_at(25), None, model=model)
    assert model.saved_to == [str(tmp_path / "ckpt_step25")]


def test_eval_restores_state_on_success(tmp_path, monkeypatch):
    def fake_evaluate(model, tokenizer, *a, **k):
        # During eval: model must be in eval mode, tokenizer left-padded.
        assert model.training is False
        assert tokenizer.padding_side == "left"
        return canned_result()

    monkeypatch.setattr(callbacks_module, "evaluate", fake_evaluate)
    cb = make_eval_cb(tmp_path, eval_at_start=False)
    model = FakeModel()
    tok = cb.tokenizer
    tok.padding_side = "right"  # simulate trainer having set something else

    cb.on_step_end(None, state_at(25), None, model=model)
    assert model.training is True  # restored
    assert tok.padding_side == "right"  # restored


def test_eval_restores_state_on_exception(tmp_path, monkeypatch):
    def exploding_evaluate(*a, **k):
        raise RuntimeError("MPS fell over")

    monkeypatch.setattr(callbacks_module, "evaluate", exploding_evaluate)
    cb = make_eval_cb(tmp_path, eval_at_start=False)
    model = FakeModel()

    with pytest.raises(RuntimeError, match="MPS fell over"):
        cb.on_step_end(None, state_at(25), None, model=model)
    assert model.training is True  # finally-block restored despite the crash
    assert cb.tokenizer.padding_side == "left"


def test_eval_mirrors_to_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(callbacks_module, "evaluate", lambda *a, **k: canned_result(0.38))
    jsonl = MetricsJSONLCallback(run_dir=tmp_path)
    cb = make_eval_cb(tmp_path, jsonl=jsonl, eval_at_start=False)
    cb.on_step_end(None, state_at(50), None, model=FakeModel())
    record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert record["step"] == 50
    assert record["eval/pass_at_1"] == 0.38


def test_eval_uses_separate_flop_budget(tmp_path, monkeypatch):
    seen_budgets = []

    def fake_evaluate(*a, flop_budget=None, **k):
        seen_budgets.append(flop_budget)
        return canned_result()

    monkeypatch.setattr(callbacks_module, "evaluate", fake_evaluate)
    eval_budget = FLOPBudget()
    cb = make_eval_cb(tmp_path, eval_flop_budget=eval_budget, eval_at_start=False)
    cb.on_step_end(None, state_at(25), None, model=FakeModel())
    assert seen_budgets == [eval_budget]  # eval budget, never the train budget


def test_eval_never_passes_seed(tmp_path, monkeypatch):
    """Passing seed= would reseed global RNGs mid-training — must not happen."""
    def fake_evaluate(*a, **k):
        assert "seed" not in k or k["seed"] is None
        return canned_result()

    monkeypatch.setattr(callbacks_module, "evaluate", fake_evaluate)
    cb = make_eval_cb(tmp_path, eval_at_start=False)
    cb.on_step_end(None, state_at(25), None, model=FakeModel())
