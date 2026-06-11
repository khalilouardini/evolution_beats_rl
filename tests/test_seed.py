"""Tests for src/utils/seed.py — reproducibility under set_all_seeds."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.utils.seed import set_all_seeds


def test_set_all_seeds_python_random():
    set_all_seeds(42)
    a = [random.random() for _ in range(5)]
    set_all_seeds(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_all_seeds_numpy():
    set_all_seeds(42)
    a = np.random.rand(5)
    set_all_seeds(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_all_seeds_torch_cpu():
    set_all_seeds(42)
    a = torch.rand(5)
    set_all_seeds(42)
    b = torch.rand(5)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_set_all_seeds_torch_mps():
    set_all_seeds(42)
    a = torch.rand(5, device="mps")
    set_all_seeds(42)
    b = torch.rand(5, device="mps")
    assert torch.equal(a, b)


def test_different_seeds_produce_different_outputs():
    set_all_seeds(0)
    a = torch.rand(5)
    set_all_seeds(1)
    b = torch.rand(5)
    assert not torch.equal(a, b)


def test_negative_seed_raises():
    with pytest.raises(ValueError):
        set_all_seeds(-1)


def test_set_all_seeds_pythonhashseed_env():
    import os

    set_all_seeds(123)
    assert os.environ["PYTHONHASHSEED"] == "123"
