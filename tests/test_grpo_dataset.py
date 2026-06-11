"""Tests for the GRPO dataset builder — mapping + prompt-length guard.

Uses Dataset.from_dict fixtures and a monkeypatched loader, so no network
or HF Hub access is needed.
"""

from __future__ import annotations

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

import src.grpo.train as train_module
from src.data.gsm8k import SYSTEM_PROMPT
from src.grpo.train import _assert_prompt_lengths, build_train_dataset


class CountingTokenizer:
    """Fake tokenizer: token count = word count. Good enough for the guard."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = " ".join(m["content"] for m in messages)
        if tokenize:
            return text.split()
        return text


def make_cfg(train_n=2, max_prompt_tokens=256):
    return OmegaConf.create(
        {"data": {"train_n": train_n, "eval_n": 50, "data_seed": 0, "max_prompt_tokens": max_prompt_tokens}}
    )


@pytest.fixture
def fake_gsm8k(monkeypatch):
    ds = Dataset.from_dict(
        {
            "question": ["What is 2+3?", "Tom has 4 apples and eats 1. How many left?"],
            "answer": ["2+3=5\n#### 5", "4-1=3\n#### 3"],
        }
    )
    monkeypatch.setattr(train_module, "load_gsm8k_subset", lambda *a, **k: ds)
    return ds


def test_mapped_row_shape(fake_gsm8k):
    out = build_train_dataset(make_cfg(), CountingTokenizer())
    row = out[0]
    # Conversational prompt: system + user, exactly.
    assert row["prompt"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 2+3?"},
    ]
    # answer preserved VERBATIM (the reward needs the raw '#### N' string).
    assert row["answer"] == "2+3=5\n#### 5"
    # question column dropped (keeps reward kwargs minimal).
    assert "question" not in out.column_names
    assert set(out.column_names) == {"prompt", "answer"}


def test_prompt_length_guard_passes_normal(fake_gsm8k):
    # Should not raise: prompts are far below 256 fake-tokens.
    build_train_dataset(make_cfg(max_prompt_tokens=256), CountingTokenizer())


def test_prompt_length_guard_raises_on_oversize(monkeypatch):
    long_question = "word " * 300  # 300 fake-tokens of question alone
    ds = Dataset.from_dict({"question": [long_question], "answer": ["#### 1"]})
    monkeypatch.setattr(train_module, "load_gsm8k_subset", lambda *a, **k: ds)
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        build_train_dataset(make_cfg(max_prompt_tokens=256), CountingTokenizer())


def test_assert_prompt_lengths_direct():
    ds = [{"prompt": [{"role": "user", "content": "three words here"}]}]
    _assert_prompt_lengths(ds, CountingTokenizer(), max_tokens=3)  # exactly at limit: OK
    with pytest.raises(ValueError):
        _assert_prompt_lengths(ds, CountingTokenizer(), max_tokens=2)
