"""Tests for src/data/rewards.py — TRL-compatible reward function signatures."""

from __future__ import annotations

import pytest

from src.data.rewards import (
    accuracy_reward,
    extracted_answer_present_reward,
    format_reward,
)


# -----------------------------------------------------------------------------
# accuracy_reward
# -----------------------------------------------------------------------------

def test_accuracy_reward_plain_strings_correct():
    completions = ["The answer is 18.", "<answer>42</answer>", "#### 7"]
    answers = ["Reasoning.\n#### 18", "Reasoning.\n#### 42", "Reasoning.\n#### 7"]
    assert accuracy_reward(completions=completions, answer=answers) == [1.0, 1.0, 1.0]


def test_accuracy_reward_plain_strings_mixed():
    completions = ["The answer is 18.", "The answer is 99.", "The answer is 7."]
    answers = ["Reasoning.\n#### 18", "Reasoning.\n#### 42", "Reasoning.\n#### 7"]
    assert accuracy_reward(completions=completions, answer=answers) == [1.0, 0.0, 1.0]


def test_accuracy_reward_handles_chat_format():
    """TRL passes completions as [{"role": "assistant", "content": "..."}] in chat mode."""
    completions = [
        [{"role": "assistant", "content": "The answer is 18."}],
        [{"role": "assistant", "content": "The answer is 99."}],
    ]
    answers = ["#### 18", "#### 42"]
    assert accuracy_reward(completions=completions, answer=answers) == [1.0, 0.0]


def test_accuracy_reward_requires_kwargs():
    with pytest.raises(ValueError):
        accuracy_reward(completions=["foo"])  # missing answer
    with pytest.raises(ValueError):
        accuracy_reward(answer=["#### 1"])  # missing completions


def test_accuracy_reward_swallows_extra_dataset_columns():
    # TRL may pass arbitrary dataset columns; the reward should ignore them.
    completions = ["The answer is 5."]
    answers = ["#### 5"]
    out = accuracy_reward(
        completions=completions,
        answer=answers,
        question=["What is 2+3?"],
        random_column=[None],
    )
    assert out == [1.0]


# -----------------------------------------------------------------------------
# format_reward
# -----------------------------------------------------------------------------

def test_format_reward_detects_tag():
    completions = [
        "Reasoning <answer>5</answer>",
        "Reasoning. The answer is 5.",
        "<answer>foo</answer>",  # non-numeric content; format reward only cares about presence
    ]
    assert format_reward(completions=completions) == [1.0, 0.0, 1.0]


def test_format_reward_chat_format():
    completions = [
        [{"role": "assistant", "content": "Reasoning <answer>5</answer>"}],
        [{"role": "assistant", "content": "No tag here."}],
    ]
    assert format_reward(completions=completions) == [1.0, 0.0]


def test_format_reward_case_insensitive():
    completions = ["<ANSWER>5</ANSWER>", "<Answer>5</Answer>"]
    assert format_reward(completions=completions) == [1.0, 1.0]


def test_format_reward_rejects_empty_tag():
    # An empty <answer></answer> shouldn't get the format bonus.
    completions = ["<answer></answer>"]
    assert format_reward(completions=completions) == [0.0]


def test_format_reward_requires_completions():
    with pytest.raises(ValueError):
        format_reward()


# -----------------------------------------------------------------------------
# extracted_answer_present_reward (diagnostic)
# -----------------------------------------------------------------------------

def test_extracted_present_diagnostic():
    completions = [
        "The answer is 5.",  # has number
        "I don't know.",  # no number
        "<answer>42</answer>",  # has number in tag
        "#### 7",  # hash marker
    ]
    assert extracted_answer_present_reward(completions=completions) == [1.0, 0.0, 1.0, 1.0]
