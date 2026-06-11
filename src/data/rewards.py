"""Reward functions for GRPO and ES on GSM8K.

Two atomic rewards (logged separately per CLAUDE.md §6 Phase 1):
- `accuracy_reward`: 1.0 if the verifier finds the model's final answer
  matches the gold, else 0.0.
- `format_reward`: 1.0 if the completion contains a `<answer>...</answer>`
  tag, else 0.0.

For fairness across ES and GRPO (CLAUDE.md §6 Phase 2), the *same* reward
functions are wired into both. TRL sums multiple registered reward functions,
so default per-call weighting is 1.0 each — adjust at the trainer config
level to get `accuracy + 0.1 * format` (the §10 default).

Signature note: TRL 0.x GRPOTrainer passes `completions` as either
`list[str]` (plain mode) or `list[list[dict]]` with conversation turns
(chat mode). Both are handled.
"""

from __future__ import annotations

import re
from typing import Any

from src.data.gsm8k import extract_model_answer, is_correct

_ANSWER_TAG_RE = re.compile(r"<answer>\s*.+?\s*</answer>", re.DOTALL | re.IGNORECASE)


def _to_text(completion: Any) -> str:
    """Extract the assistant's text from a TRL completion (str or chat list)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    return str(completion)


def accuracy_reward(
    prompts: list[Any] | None = None,  # noqa: ARG001 (TRL passes but unused)
    completions: list[Any] | None = None,
    answer: list[str] | None = None,
    **kwargs: Any,  # noqa: ARG001 (catch other dataset cols TRL may pass)
) -> list[float]:
    """Binary accuracy: 1.0 if the verifier matches gold, else 0.0.

    `answer` must be a list of raw GSM8K answer strings (containing '#### N').
    """
    if completions is None or answer is None:
        raise ValueError("accuracy_reward requires completions and answer kwargs")
    return [1.0 if is_correct(_to_text(c), g) else 0.0 for c, g in zip(completions, answer)]


def format_reward(
    prompts: list[Any] | None = None,  # noqa: ARG001
    completions: list[Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> list[float]:
    """1.0 if completion contains a non-empty `<answer>...</answer>` tag.

    Note (§11 gotcha): the tag is independent of correctness — model can emit
    `<answer>0</answer>` for any prompt to grab the bonus. Keep the format
    weight ≤0.1 of accuracy in the trainer config to avoid reward hacking.
    """
    if completions is None:
        raise ValueError("format_reward requires completions kwarg")
    return [1.0 if _ANSWER_TAG_RE.search(_to_text(c)) else 0.0 for c in completions]


def extracted_answer_present_reward(
    prompts: list[Any] | None = None,  # noqa: ARG001
    completions: list[Any] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> list[float]:
    """1.0 if our extractor finds *any* number in the completion.

    Diagnostic — useful for distinguishing "model said nothing parseable" from
    "model said the wrong number." Not part of the trained reward.
    """
    if completions is None:
        raise ValueError("extracted_answer_present_reward requires completions kwarg")
    return [1.0 if extract_model_answer(_to_text(c)) is not None else 0.0 for c in completions]
