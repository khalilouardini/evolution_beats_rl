"""FLOP accounting for compute-matched ES vs GRPO comparisons.

Per CLAUDE.md §6 Phase 0 step 4 and Phase 3/3.5 (matched comparison):
- Forward ≈ 2 · params · tokens  (Kaplan et al. 2020 rule-of-thumb).
- Backward ≈ 4 · params · tokens (twice forward).
- ES: forward only (zeroth-order rollouts).
- GRPO: forward + backward on policy, forward on reference (for KL).

The counter uses **full model parameter count for backward FLOPs**, not the
LoRA trainable count. Backward propagation still pulls gradients through the
frozen base-model activations even when only LoRA adapters are updated; the
Kaplan/Chinchilla 6·P·T rule assumes full-model P. Counting only LoRA params
would undercount GRPO compute and bias the comparison.

Attention's O(L) per-token cost is intentionally omitted. It cancels in the
ES↔GRPO comparison (same generation lengths in both methods), and keeping
the formula at 2·P·T makes the counter cheap and unit-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.nn as nn

# Multipliers per the Kaplan et al. 2020 rule (Section 2.1).
FORWARD_FLOPS_PER_PARAM_PER_TOKEN = 2
BACKWARD_FLOPS_PER_PARAM_PER_TOKEN = 4


@dataclass
class FLOPBudget:
    """Running tally of FLOPs spent during a training/eval run.

    Append-only — call the `add_*` methods as you go; serialize to wandb /
    `runs/{name}/flops.json` per CLAUDE.md §7 logging conventions.
    """

    forward_flops: int = 0
    backward_flops: int = 0
    generated_tokens: int = 0  # tokens emitted by `model.generate()`
    training_tokens: int = 0  # tokens passed through forward+backward training step

    @property
    def total_flops(self) -> int:
        return self.forward_flops + self.backward_flops

    def add_generation(self, params: int, n_tokens: int) -> None:
        """Account for autoregressive generation of `n_tokens` (forward only).

        Used for: ES rollouts, GRPO rollouts (the sampling phase), eval pass@1
        decoding.
        """
        _validate(params, n_tokens)
        self.forward_flops += FORWARD_FLOPS_PER_PARAM_PER_TOKEN * params * n_tokens
        self.generated_tokens += n_tokens

    def add_train_step(self, params: int, n_tokens: int) -> None:
        """Account for one teacher-forced forward+backward pass over `n_tokens`.

        Used for: GRPO policy update.
        """
        _validate(params, n_tokens)
        self.forward_flops += FORWARD_FLOPS_PER_PARAM_PER_TOKEN * params * n_tokens
        self.backward_flops += BACKWARD_FLOPS_PER_PARAM_PER_TOKEN * params * n_tokens
        self.training_tokens += n_tokens

    def add_reference_forward(self, params: int, n_tokens: int) -> None:
        """Account for a forward-only pass for the reference policy (GRPO KL).

        Same FLOP cost as `add_generation` for the forward, but does not count
        the tokens as 'generated' (they were teacher-forced through the ref).
        """
        _validate(params, n_tokens)
        self.forward_flops += FORWARD_FLOPS_PER_PARAM_PER_TOKEN * params * n_tokens

    def to_dict(self) -> dict[str, int]:
        d = asdict(self)
        d["total_flops"] = self.total_flops
        return d


def _validate(params: int, n_tokens: int) -> None:
    if params < 0:
        raise ValueError(f"params must be >= 0, got {params}")
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be >= 0, got {n_tokens}")


def count_model_params(model: nn.Module, *, only_trainable: bool = False) -> int:
    """Count parameters in a PyTorch model.

    Default counts ALL parameters (the relevant number for forward/backward
    FLOPs). Set `only_trainable=True` for the LoRA-adapter-count (used for
    intrinsic-dimensionality discussion in the paper, not for FLOP accounting).
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
