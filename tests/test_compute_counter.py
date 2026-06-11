"""Tests for src/eval/compute.py — FLOP accounting math + model param counter."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.eval.compute import (
    BACKWARD_FLOPS_PER_PARAM_PER_TOKEN,
    FORWARD_FLOPS_PER_PARAM_PER_TOKEN,
    FLOPBudget,
    count_model_params,
)


# -----------------------------------------------------------------------------
# Constants — make sure we haven't drifted from Kaplan et al. 2.1 (2·P + 4·P).
# -----------------------------------------------------------------------------

def test_kaplan_constants():
    assert FORWARD_FLOPS_PER_PARAM_PER_TOKEN == 2
    assert BACKWARD_FLOPS_PER_PARAM_PER_TOKEN == 4


# -----------------------------------------------------------------------------
# FLOPBudget — generation (forward only)
# -----------------------------------------------------------------------------

def test_add_generation_basic():
    b = FLOPBudget()
    b.add_generation(params=1_000_000, n_tokens=100)
    assert b.forward_flops == 2 * 1_000_000 * 100
    assert b.backward_flops == 0
    assert b.generated_tokens == 100
    assert b.training_tokens == 0
    assert b.total_flops == 200_000_000


def test_add_generation_accumulates():
    b = FLOPBudget()
    b.add_generation(params=1000, n_tokens=10)
    b.add_generation(params=1000, n_tokens=20)
    assert b.forward_flops == 2 * 1000 * 30
    assert b.generated_tokens == 30


# -----------------------------------------------------------------------------
# FLOPBudget — train step (forward + backward)
# -----------------------------------------------------------------------------

def test_add_train_step_basic():
    b = FLOPBudget()
    b.add_train_step(params=1_000_000, n_tokens=100)
    assert b.forward_flops == 2 * 1_000_000 * 100  # 2·P·T
    assert b.backward_flops == 4 * 1_000_000 * 100  # 4·P·T
    assert b.total_flops == 6 * 1_000_000 * 100  # 6·P·T total
    assert b.training_tokens == 100
    assert b.generated_tokens == 0  # not generation


def test_add_train_step_accumulates():
    b = FLOPBudget()
    b.add_train_step(params=1000, n_tokens=10)
    b.add_train_step(params=1000, n_tokens=20)
    assert b.forward_flops == 2 * 1000 * 30
    assert b.backward_flops == 4 * 1000 * 30
    assert b.training_tokens == 30


# -----------------------------------------------------------------------------
# FLOPBudget — reference forward (GRPO KL term)
# -----------------------------------------------------------------------------

def test_add_reference_forward():
    b = FLOPBudget()
    b.add_reference_forward(params=1_000_000, n_tokens=100)
    assert b.forward_flops == 2 * 1_000_000 * 100
    assert b.backward_flops == 0
    assert b.generated_tokens == 0  # ref forward doesn't generate tokens
    assert b.training_tokens == 0


# -----------------------------------------------------------------------------
# FLOPBudget — mixed scenario (GRPO-ish step: gen + train + ref)
# -----------------------------------------------------------------------------

def test_mixed_grpo_step():
    """One GRPO step: 4 generations × 256 tokens, train on 4×256 tokens, ref forward 4×256."""
    b = FLOPBudget()
    params = 500_000_000  # ~Qwen2.5-0.5B
    completions_per_step = 4
    seq_len = 256

    # Rollout: generate completions × len tokens
    b.add_generation(params=params, n_tokens=completions_per_step * seq_len)
    # Train: forward + backward over rolled-out sequences
    b.add_train_step(params=params, n_tokens=completions_per_step * seq_len)
    # Reference forward (for KL)
    b.add_reference_forward(params=params, n_tokens=completions_per_step * seq_len)

    # Forward: 2·P·T from each of gen, train, ref = 3 * 2 * P * T
    expected_forward = 3 * 2 * params * completions_per_step * seq_len
    # Backward: only from train step = 4 * P * T
    expected_backward = 4 * params * completions_per_step * seq_len

    assert b.forward_flops == expected_forward
    assert b.backward_flops == expected_backward
    assert b.generated_tokens == completions_per_step * seq_len
    assert b.training_tokens == completions_per_step * seq_len


# -----------------------------------------------------------------------------
# FLOPBudget — ES scenario (forward only, antithetic)
# -----------------------------------------------------------------------------

def test_mixed_es_generation():
    """One ES generation: N=10 perturbations, antithetic → 20 forward passes per batch of 8 prompts × 256 tokens."""
    b = FLOPBudget()
    params = 500_000_000
    n_perturbations_antithetic = 20  # N=10 with antithetic
    batch = 8
    seq_len = 256

    total_tokens = n_perturbations_antithetic * batch * seq_len
    b.add_generation(params=params, n_tokens=total_tokens)

    assert b.forward_flops == 2 * params * total_tokens
    assert b.backward_flops == 0
    assert b.training_tokens == 0


# -----------------------------------------------------------------------------
# FLOPBudget — to_dict serialization
# -----------------------------------------------------------------------------

def test_to_dict_includes_total():
    b = FLOPBudget()
    b.add_train_step(params=1000, n_tokens=10)
    d = b.to_dict()
    assert d == {
        "forward_flops": 20_000,
        "backward_flops": 40_000,
        "total_flops": 60_000,
        "generated_tokens": 0,
        "training_tokens": 10,
    }


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def test_negative_params_raises():
    b = FLOPBudget()
    with pytest.raises(ValueError):
        b.add_generation(params=-1, n_tokens=10)


def test_negative_tokens_raises():
    b = FLOPBudget()
    with pytest.raises(ValueError):
        b.add_train_step(params=1000, n_tokens=-1)


def test_zero_tokens_is_a_noop():
    b = FLOPBudget()
    b.add_generation(params=1_000_000, n_tokens=0)
    assert b.forward_flops == 0
    assert b.generated_tokens == 0


# -----------------------------------------------------------------------------
# count_model_params
# -----------------------------------------------------------------------------

def test_count_model_params_total():
    model = nn.Linear(10, 5)
    # Linear(10, 5): weight = 50 params, bias = 5 params → 55 total
    assert count_model_params(model) == 55


def test_count_model_params_only_trainable():
    model = nn.Linear(10, 5)
    # Freeze bias
    model.bias.requires_grad = False
    assert count_model_params(model, only_trainable=True) == 50
    assert count_model_params(model, only_trainable=False) == 55


def test_count_model_params_no_trainable_when_all_frozen():
    model = nn.Linear(10, 5)
    for p in model.parameters():
        p.requires_grad = False
    assert count_model_params(model, only_trainable=True) == 0
    assert count_model_params(model, only_trainable=False) == 55


def test_count_model_params_sequential():
    model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))
    # 10·20 + 20 + 20·5 + 5 = 200 + 20 + 100 + 5 = 325
    assert count_model_params(model) == 325


def test_count_model_params_consistency_with_p_numel():
    """count_model_params must equal sum(p.numel())."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.LayerNorm(128),
        nn.Linear(128, 32),
    )
    expected = sum(p.numel() for p in model.parameters())
    assert count_model_params(model) == expected
    assert count_model_params(model) > 0  # not the trivially-zero case
    # Sanity bound: 64*128 + 128 (Linear) + 2*128 (LayerNorm) + 128*32 + 32 (Linear)
    # = 8192 + 128 + 256 + 4096 + 32 = 12704
    assert expected == 12704


# -----------------------------------------------------------------------------
# Realism check: total FLOPs at toy scale match rough wall-clock expectations.
# -----------------------------------------------------------------------------

def test_toy_phase_grpo_step_flops_in_expected_range():
    """One GRPO step at toy config should be in 10^12-10^13 FLOP range.

    Toy: Qwen2.5-0.5B (~0.5e9 params), 4 generations, 256 max_completion_length,
    batch in trainer ~4. Rough total per step ~ 6 · 0.5e9 · 4 · 256 ≈ 3e12 FLOPs.
    M1 Pro at ~5 TFLOPS bf16 sustained ≈ 0.6s per step optimistically,
    realistically 5-30s with overhead. Sanity check this isn't off by orders.
    """
    b = FLOPBudget()
    params = int(0.5e9)
    completions = 4
    seq_len = 256
    b.add_generation(params=params, n_tokens=completions * seq_len)
    b.add_train_step(params=params, n_tokens=completions * seq_len)
    b.add_reference_forward(params=params, n_tokens=completions * seq_len)

    # Expected: forward = 3·2·P·T, backward = 4·P·T → total = 10·P·T
    total = b.total_flops
    expected_total = 10 * params * completions * seq_len
    assert total == expected_total
    assert 1e12 <= total <= 1e13, f"Toy GRPO step FLOPs {total:.2e} outside 1e12-1e13 sanity range"
