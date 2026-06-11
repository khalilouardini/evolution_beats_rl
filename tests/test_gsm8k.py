"""Tests for GSM8K loader and verifier (src/data/gsm8k.py).

Headline test (per CLAUDE.md §6 Phase 0 step 2): verifier must achieve
>=95% self-agreement on 100 reference completions from the test split.
"""

from __future__ import annotations

import pytest

from src.data.gsm8k import (
    extract_gold,
    extract_model_answer,
    is_correct,
    load_gsm8k_subset,
    parse_number,
)


# -----------------------------------------------------------------------------
# parse_number
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("42", 42.0),
        ("-7", -7.0),
        ("3.14", 3.14),
        ("1,000", 1000.0),
        ("1,234,567", 1234567.0),
        ("-1,234.5", -1234.5),
        ("  42  ", 42.0),  # whitespace
        ("", None),
        (None, None),
        ("not a number", None),
        ("1.2.3", None),
    ],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


# -----------------------------------------------------------------------------
# extract_gold (from "#### N" patterns)
# -----------------------------------------------------------------------------

def test_extract_gold_basic():
    assert extract_gold("Janet's ducks lay 16 eggs.\n#### 18") == 18.0


def test_extract_gold_with_comma():
    assert extract_gold("A complicated calculation.\n#### 1,200") == 1200.0


def test_extract_gold_negative():
    assert extract_gold("It went down.\n#### -5") == -5.0


def test_extract_gold_missing_returns_none():
    assert extract_gold("No marker here, just numbers 42 and 17.") is None


def test_extract_gold_extra_whitespace():
    assert extract_gold("Step by step.\n####    72") == 72.0


# -----------------------------------------------------------------------------
# extract_model_answer (priority: ####, <answer>, last number)
# -----------------------------------------------------------------------------

def test_extract_model_prefers_hash_marker():
    completion = "I think it's 5, then 10, but actually #### 42"
    assert extract_model_answer(completion) == 42.0


def test_extract_model_uses_answer_tag():
    completion = "Reasoning... <answer>17</answer>"
    assert extract_model_answer(completion) == 17.0


def test_extract_model_answer_tag_case_insensitive():
    completion = "Reasoning <ANSWER>9</ANSWER> done."
    assert extract_model_answer(completion) == 9.0


def test_extract_model_answer_tag_with_text():
    completion = "Reasoning <answer>The answer is 7.</answer>"
    assert extract_model_answer(completion) == 7.0


def test_extract_model_falls_back_to_last_number():
    completion = "She has 5 apples and gets 3 more so she has 8."
    assert extract_model_answer(completion) == 8.0


def test_extract_model_with_commas_in_last_number():
    completion = "The total is 1,234,567 dollars."
    assert extract_model_answer(completion) == 1234567.0


def test_extract_model_handles_decimals():
    completion = "Half of 9 is 4.5"
    assert extract_model_answer(completion) == 4.5


def test_extract_model_empty_returns_none():
    assert extract_model_answer("") is None


def test_extract_model_no_numbers_returns_none():
    assert extract_model_answer("I have no idea.") is None


# -----------------------------------------------------------------------------
# is_correct
# -----------------------------------------------------------------------------

def test_is_correct_exact_match():
    assert is_correct("The answer is 18.", 18.0)


def test_is_correct_with_raw_gold_string():
    assert is_correct("#### 18", "Long reasoning.\n#### 18")


def test_is_correct_float_vs_int():
    assert is_correct("The answer is 18.0", 18.0)


def test_is_correct_comma_vs_no_comma():
    assert is_correct("The answer is 1,000.", 1000.0)


def test_is_correct_wrong():
    assert not is_correct("The answer is 17.", 18.0)


def test_is_correct_no_number_in_completion():
    assert not is_correct("I don't know.", 18.0)


def test_is_correct_zero_gold():
    assert is_correct("The answer is 0.", 0.0)
    assert not is_correct("The answer is 1.", 0.0)


def test_is_correct_within_tolerance():
    # rtol=1e-4 default. 18 vs 18.0001 is within tolerance.
    assert is_correct("The answer is 18.0001", 18.0)
    assert not is_correct("The answer is 18.5", 18.0)


# -----------------------------------------------------------------------------
# Headline test: verifier ≥95% self-agreement on 100 GSM8K test references.
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gsm8k_test_100():
    """Load 100 deterministic GSM8K test examples. Skip if network unavailable."""
    try:
        return load_gsm8k_subset("test", n=100, seed=0)
    except Exception as exc:  # network / hf hub down
        pytest.skip(f"Could not load GSM8K (network or HF Hub issue): {exc}")


def test_verifier_self_agreement_on_100_test_examples(gsm8k_test_100):
    """Feed each gold answer string back to the verifier; expect >=95/100 match.

    The gold field IS in '#### N' format, so the extractor's top-priority
    branch should fire. <95 means the extractor is broken on canonical data.
    """
    correct = sum(1 for ex in gsm8k_test_100 if is_correct(ex["answer"], ex["answer"]))
    assert correct >= 95, f"Verifier self-agreement {correct}/100 < 95 (target per CLAUDE.md §6 Phase 0)"


def test_extract_gold_succeeds_on_100_test_examples(gsm8k_test_100):
    """Every test-set example must have a parseable gold answer."""
    parsed = [extract_gold(ex["answer"]) for ex in gsm8k_test_100]
    n_parsed = sum(1 for p in parsed if p is not None)
    assert n_parsed == 100, f"Only {n_parsed}/100 test examples had parseable gold"
