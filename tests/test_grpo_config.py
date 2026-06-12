"""Tests for src/grpo/train.py config loading + GRPOConfig construction.

Instantiating the real trl.GRPOConfig executes TRL's own batch-divisibility
validation, so test_build_grpo_config doubles as a free integration test of
the batch math (generation_batch_size=16 % num_generations=4 == 0).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.grpo.train import (
    build_grpo_config,
    build_run_name,
    load_config,
)

TOY = "configs/grpo/toy_0.5b.yaml"
SMOKE = "configs/grpo/smoke.yaml"


# -----------------------------------------------------------------------------
# load_config + CLI overrides (pins the Makefile smoke-grpo contract)
# -----------------------------------------------------------------------------

def test_load_toy_config():
    cfg = load_config(["--config", TOY])
    assert cfg.run.seed == 0
    assert cfg.grpo.max_steps == 100
    assert cfg.data.train_n == 500
    assert cfg.run.wandb_mode == "offline"


def test_load_smoke_config():
    cfg = load_config(["--config", SMOKE])
    assert cfg.grpo.max_steps == 20
    assert cfg.data.train_n == 50
    assert cfg.eval.every_steps == 10
    assert "smoke" in list(cfg.run.wandb_tags)


def test_cli_overrides_makefile_contract():
    """The exact flags the Makefile smoke-grpo target passes must work."""
    cfg = load_config(["--config", SMOKE, "--max_steps", "20", "--eval_subset", "50"])
    assert cfg.grpo.max_steps == 20
    assert cfg.data.eval_n == 50


def test_cli_seed_override():
    cfg = load_config(["--config", TOY, "--seed", "7"])
    assert cfg.run.seed == 7


def test_cli_dotlist_override():
    cfg = load_config(["--config", TOY, "--set", "grpo.beta=0.02", "lora.r=8"])
    assert cfg.grpo.beta == 0.02
    assert cfg.lora.r == 8


def test_missing_required_key_raises(tmp_path):
    import yaml

    cfg_dict = yaml.safe_load(Path(TOY).read_text())
    del cfg_dict["run"]["seed"]  # silent seed defaults are forbidden (§3)
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(cfg_dict))
    with pytest.raises(ValueError, match="run.seed"):
        load_config(["--config", str(broken)])


# -----------------------------------------------------------------------------
# build_run_name (§7 naming convention)
# -----------------------------------------------------------------------------

def test_run_name_pattern():
    cfg = load_config(["--config", TOY, "--seed", "42"])
    name = build_run_name(cfg, today="2026-05-12")
    assert name == "p1_grpo_qwen0.5b_s42_2026-05-12"
    assert re.fullmatch(r"p1_grpo_qwen0\.5b_s\d+_\d{4}-\d{2}-\d{2}", name)


# -----------------------------------------------------------------------------
# build_grpo_config — the §10 TOY values, against the REAL trl.GRPOConfig
# -----------------------------------------------------------------------------

def test_build_grpo_config_toy_values(tmp_path):
    cfg = load_config(["--config", TOY])
    gc = build_grpo_config(cfg, tmp_path, "test_run")

    # The two TRL-0.29.1 landmines: defaults are loss_type="dapo", beta=0.0.
    assert gc.loss_type == "grpo"
    assert gc.beta == 0.04

    assert gc.num_generations == 4
    # 512 since the 2026-06-11 plan-update (256 truncated ~50% of rollouts).
    assert gc.max_completion_length == 512
    assert gc.epsilon == 0.2
    assert gc.scale_rewards == "group"
    assert gc.reward_weights == [1.0, 0.1]
    assert gc.temperature == 1.0
    # 2e-5 = LoRA-appropriate LR; the original 5e-6 (full-param-inherited)
    # moved the policy only ~0.01 nats in 100 steps — see Progress Log 2026-06-12.
    assert gc.learning_rate == 2e-5
    assert gc.per_device_train_batch_size == 4
    assert gc.gradient_accumulation_steps == 4
    assert gc.max_steps == 100
    assert gc.use_vllm is False
    assert gc.logging_steps == 1
    assert gc.seed == 0

    # dtype strategy: no HF mixed-precision flags on MPS.
    assert gc.bf16 is False
    assert gc.fp16 is False

    # TRL derives these; their values prove the divisibility validation passed.
    assert gc.generation_batch_size == 16
    assert gc.steps_per_generation == 4


def test_build_grpo_config_smoke_differs_only_where_expected(tmp_path):
    toy = build_grpo_config(load_config(["--config", TOY]), tmp_path, "toy")
    smoke = build_grpo_config(load_config(["--config", SMOKE]), tmp_path, "smoke")
    assert smoke.max_steps == 20 and toy.max_steps == 100
    # Science parameters identical between toy and smoke:
    for field in ["beta", "epsilon", "loss_type", "num_generations", "learning_rate",
                  "temperature", "reward_weights", "max_completion_length"]:
        assert getattr(toy, field) == getattr(smoke, field), field
