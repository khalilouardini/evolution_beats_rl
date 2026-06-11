"""Freeze the reward-function ↔ TRL 0.29.1 integration contract.

TRL 0.29.1 calls reward functions as:
    func(prompts=..., completions=..., completion_ids=..., **dataset_cols,
         trainer_state=<TrainerState>)
with chat-format prompts/completions for conversational datasets. These tests
call our reward functions with EXACTLY that keyword set so any future
signature drift (ours or TRL's) fails here first, not silently mid-run.
"""

from __future__ import annotations

from src.data.rewards import accuracy_reward, format_reward

CHAT_PROMPTS = [
    [{"role": "system", "content": "sys"}, {"role": "user", "content": "What is 2+3?"}],
    [{"role": "system", "content": "sys"}, {"role": "user", "content": "What is 10-1?"}],
]
CHAT_COMPLETIONS = [
    [{"role": "assistant", "content": "2+3=5. <answer>5</answer>"}],
    [{"role": "assistant", "content": "The answer is 8."}],  # wrong (gold 9), no tag
]
COMPLETION_IDS = [[1, 2, 3], [4, 5]]
ANSWERS = ["2+3=5\n#### 5", "10-1=9\n#### 9"]


def trl_kwargs() -> dict:
    """The exact keyword set TRL 0.29.1 passes (trainer_state injected)."""
    return dict(
        prompts=CHAT_PROMPTS,
        completions=CHAT_COMPLETIONS,
        completion_ids=COMPLETION_IDS,
        answer=ANSWERS,  # forwarded dataset column
        trainer_state=object(),  # injected by the trainer
    )


def test_accuracy_reward_with_exact_trl_kwarg_set():
    out = accuracy_reward(**trl_kwargs())
    assert out == [1.0, 0.0]
    assert all(isinstance(v, float) for v in out)


def test_format_reward_with_exact_trl_kwarg_set():
    out = format_reward(**trl_kwargs())
    assert out == [1.0, 0.0]


def test_reward_function_names_preserved():
    """TRL derives wandb metric names (rewards/{__name__}/mean) from __name__.

    A decorator that loses the name would silently rename dashboards.
    """
    assert accuracy_reward.__name__ == "accuracy_reward"
    assert format_reward.__name__ == "format_reward"


def test_weighted_sum_semantics():
    """Document what reward_weights=[1.0, 0.1] produces (pre-normalization).

    TRL computes: combined = w_acc * accuracy + w_fmt * format, then
    group-normalizes per scale_rewards="group". This test pins the
    pre-normalization arithmetic our configs rely on.
    """
    acc = accuracy_reward(**trl_kwargs())  # [1.0, 0.0]
    fmt = format_reward(**trl_kwargs())  # [1.0, 0.0]
    combined = [1.0 * a + 0.1 * f for a, f in zip(acc, fmt)]
    assert combined == [1.1, 0.0]
