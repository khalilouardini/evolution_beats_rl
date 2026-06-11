"""Single entry point for seeding all randomness in a run.

Per CLAUDE.md §3 "Determinism: every run must set `seed` from config and write
`seed` to wandb. No silent seed defaults." And §11: HF `set_seed` doesn't seed
vLLM's sampler — in cloud phase we'll need to add a vLLM `SamplingParams(seed=...)`
plumbing call here. In the toy phase (no vLLM) we just hit Python `random`, numpy,
torch (CPU + MPS + CUDA), and `transformers.set_seed`.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import set_seed as hf_set_seed


def set_all_seeds(seed: int, *, deterministic_cudnn: bool = False) -> None:
    """Seed every RNG we touch.

    Args:
        seed: integer seed.
        deterministic_cudnn: if True, force cuDNN to deterministic kernels.
            Slows training; off by default. Toy phase doesn't use cuDNN
            (MPS), so this is a no-op on Apple Silicon.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    hf_set_seed(seed)

    if deterministic_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
