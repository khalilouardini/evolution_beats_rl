"""GSM8K loader + verifier.

Per CLAUDE.md §6 Phase 0 step 2:
- Load openai/gsm8k main split.
- Verifier extracts the final numeric answer.
- Accept three formats from model completions:
    1. "#### N" (matches the gold format)
    2. "<answer>N</answer>" (the format the system prompt requests)
    3. Last standalone number in the completion (fallback)

The verifier is the source of truth for the binary accuracy reward in
src/data/rewards.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = (
    "You are a careful math problem solver. Solve the problem step by step, "
    "showing your reasoning. After your reasoning, write your final numeric "
    "answer inside <answer>...</answer> tags."
)

# "#### -1,234.5" -> captures "-1,234.5"
_GOLD_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

# Matches numbers with optional sign, optional thousands-commas, optional decimals.
# Used to find the *last* standalone number in a string as a fallback.
_NUMBER_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 1,234 or 1,234.5
    r"|-?\d+(?:\.\d+)?"  # 1234 or 1234.5
)

# "<answer>...</answer>" — case-insensitive, multi-line content.
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def parse_number(s: str | None) -> float | None:
    """Parse a number string with optional commas. Returns None if unparseable."""
    if s is None:
        return None
    cleaned = s.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_gold(raw_answer: str) -> float | None:
    """Extract the gold numeric answer from a GSM8K `answer` field (after '####')."""
    m = _GOLD_ANSWER_RE.search(raw_answer)
    return parse_number(m.group(1)) if m else None


def extract_model_answer(completion: str) -> float | None:
    """Extract the model's final numeric answer.

    Priority:
      1. Number after '####' (model follows gold format).
      2. Number inside <answer>...</answer> tag.
      3. Last standalone number in the completion.
    """
    # 1. #### marker
    m = _GOLD_ANSWER_RE.search(completion)
    if m and (n := parse_number(m.group(1))) is not None:
        return n

    # 2. <answer>...</answer> tag
    m = _ANSWER_TAG_RE.search(completion)
    if m:
        inner = m.group(1)
        if (n := parse_number(inner)) is not None:
            return n
        # Number inside the tag if the tag content has text around it
        nums = _NUMBER_RE.findall(inner)
        if nums and (n := parse_number(nums[-1])) is not None:
            return n

    # 3. Last standalone number in the full completion
    nums = _NUMBER_RE.findall(completion)
    if nums:
        return parse_number(nums[-1])

    return None


def is_correct(completion: str, gold: float | str, *, rtol: float = 1e-4) -> bool:
    """Check if a model completion's extracted answer matches the gold answer.

    `gold` can be a numeric value or a raw GSM8K `answer` string (with '####').
    Comparison uses absolute tolerance for 0 and relative tolerance otherwise,
    so 18.0 == 18 and 1000 == 1,000 and 0.500 == 0.5.
    """
    if isinstance(gold, str):
        gold_num = extract_gold(gold)
    else:
        gold_num = float(gold)
    if gold_num is None:
        return False

    pred = extract_model_answer(completion)
    if pred is None:
        return False

    if gold_num == 0.0:
        return abs(pred) <= rtol
    return abs(pred - gold_num) / abs(gold_num) <= rtol


@dataclass(frozen=True)
class GSM8KItem:
    question: str
    raw_answer: str
    gold: float


def load_gsm8k(split: str = "train") -> Dataset:
    """Load `openai/gsm8k` main config. `split` is 'train' (7473) or 'test' (1319)."""
    return load_dataset("openai/gsm8k", "main", split=split)


def load_gsm8k_subset(split: str, n: int, *, seed: int = 0) -> Dataset:
    """Deterministic subset of GSM8K for toy-phase runs."""
    ds = load_gsm8k(split)
    if n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def iter_items(dataset: Dataset) -> Iterator[GSM8KItem]:
    """Yield typed items, skipping any rows whose gold answer fails to parse."""
    for ex in dataset:
        gold = extract_gold(ex["answer"])
        if gold is None:
            continue
        yield GSM8KItem(question=ex["question"], raw_answer=ex["answer"], gold=gold)
