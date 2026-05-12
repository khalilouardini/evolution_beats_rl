"""Pass@1 / pass@k eval harness using transformers.generate() — no vLLM.

Per CLAUDE.md §6 Phase 0 step 3 and §10 (Eval defaults):
- pass@1: greedy (temperature=0).
- pass@k for k>1: temperature=0.7, top_p=0.95; unbiased estimator (Chen 2021).
- Test split only — never train (§10).
- Device auto-select: MPS > CUDA > CPU. Cloud phase swaps generation for vLLM.

CLI:
    python -m src.eval.harness \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --split test --n 50 \\
        --max_new_tokens 256 --batch_size 4 \\
        --output results/tables/00_baseline.md
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.gsm8k import SYSTEM_PROMPT, is_correct, load_gsm8k_subset
from src.eval.compute import FLOPBudget, count_model_params
from src.utils.seed import set_all_seeds


def auto_device() -> torch.device:
    """MPS if available (toy phase), else CUDA (cloud phase), else CPU."""
    if torch.backends.mps.is_available():
        # Per CLAUDE.md §11 MPS gotcha — enable CPU fallback for ops without MPS kernels.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_qwen_model(
    name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a Qwen-family causal LM + tokenizer.

    On MPS we force `attn_implementation="eager"` because the default `sdpa`
    kernel crashes on Qwen's grouped-query attention reshape (the broadcasted
    KV-head expansion produces a shape Metal can't infer). Eager attention is
    ~10-20% slower per token but always correct. Cloud phase (CUDA + vLLM)
    sidesteps this entirely. See CLAUDE.md §11 (MPS gotchas).

    Sets pad_token=eos if missing and padding_side='left' (required for
    batched generation so all prompts end at the same column).
    """
    device = device or auto_device()
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict = {"dtype": dtype}
    if device.type == "mps":
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    model = model.to(device).eval()
    return model, tokenizer


def format_prompt(question: str, tokenizer) -> str:
    """Apply Qwen chat template with the GSM8K system prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Chen et al. 2021 unbiased estimator: P(solve with k samples | drew n).

    Args:
        n: total samples drawn for this problem.
        c: number of those samples that were correct.
        k: target k (must satisfy 1 <= k <= n).

    Returns: probability in [0, 1].
    """
    if not (1 <= k <= n):
        raise ValueError(f"pass@k requires 1 <= k <= n; got n={n}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@dataclass
class EvalResult:
    """Output of `evaluate()`."""

    pass_at_1: float
    pass_at_k: float | None  # None when n_samples == 1
    n_items: int
    n_samples_per_item: int
    correct_per_item: list[int]  # how many of n_samples were correct, per problem
    completion_tokens_total: int
    wall_clock_sec: float
    model_params: int

    def to_dict(self) -> dict:
        return {
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "n_items": self.n_items,
            "n_samples_per_item": self.n_samples_per_item,
            "correct_per_item": self.correct_per_item,
            "completion_tokens_total": self.completion_tokens_total,
            "wall_clock_sec": self.wall_clock_sec,
            "model_params": self.model_params,
            "tokens_per_sec": self.completion_tokens_total / max(self.wall_clock_sec, 1e-9),
        }


def evaluate(
    model,
    tokenizer,
    dataset,
    *,
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: torch.device | None = None,
    flop_budget: FLOPBudget | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> EvalResult:
    """Greedy pass@1 (n_samples=1, temperature=0) or pass@k (n_samples=k>1).

    For pass@k, the unbiased Chen-2021 estimator is used so n_samples can be
    larger than k (lower variance) — but we default to n_samples==k for toy phase.

    `flop_budget`: if provided, generation FLOPs are added in-place (per
    CLAUDE.md §6 step 4; counts completion tokens only, not prompt tokens —
    the standard convention since prompt FLOPs are the same for all completions).
    """
    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        set_all_seeds(seed)

    do_sample = temperature > 0
    if n_samples > 1 and not do_sample:
        raise ValueError("pass@k requires temperature>0 for sampling (n_samples>1)")

    model_params = count_model_params(model)
    questions = [ex["question"] for ex in dataset]
    golds = [ex["answer"] for ex in dataset]

    correct_per_item: list[int] = [0] * len(questions)
    completion_tokens_total = 0
    t0 = time.time()

    # Outer loop: each sample index (1 pass for greedy, n_samples passes for sampling)
    for sample_idx in range(n_samples):
        for batch_start in range(0, len(questions), batch_size):
            batch_qs = questions[batch_start : batch_start + batch_size]
            batch_gs = golds[batch_start : batch_start + batch_size]
            prompts = [format_prompt(q, tokenizer) for q in batch_qs]

            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            prompt_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if do_sample:
                    gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
                else:
                    gen_kwargs.update(do_sample=False)
                out = model.generate(**enc, **gen_kwargs)

            new_tokens = out[:, prompt_len:]
            # Pad-token count varies per sequence; subtract.
            n_completion_tokens = (new_tokens != tokenizer.pad_token_id).sum().item()
            completion_tokens_total += n_completion_tokens

            if flop_budget is not None:
                flop_budget.add_generation(params=model_params, n_tokens=n_completion_tokens)

            completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for j, (comp, gold) in enumerate(zip(completions, batch_gs)):
                if is_correct(comp, gold):
                    correct_per_item[batch_start + j] += 1

            if verbose:
                done_in_sample = batch_start + len(batch_qs)
                print(
                    f"  sample {sample_idx + 1}/{n_samples}: {done_in_sample}/{len(questions)} items "
                    f"({100 * done_in_sample / len(questions):.0f}%)",
                    flush=True,
                )

    wall_clock = time.time() - t0

    # pass@1: fraction with at least 1 correct out of n_samples drawn at temperature 0
    # (for greedy, that just means correct).
    pass_1 = sum(1 for c in correct_per_item if c >= 1) / len(correct_per_item)
    pass_k = None
    if n_samples > 1:
        pass_k = sum(pass_at_k_unbiased(n=n_samples, c=c, k=n_samples) for c in correct_per_item) / len(correct_per_item)

    return EvalResult(
        pass_at_1=pass_1,
        pass_at_k=pass_k,
        n_items=len(questions),
        n_samples_per_item=n_samples,
        correct_per_item=correct_per_item,
        completion_tokens_total=completion_tokens_total,
        wall_clock_sec=wall_clock,
        model_params=model_params,
    )


def _format_markdown_report(result: EvalResult, model_name: str, split: str) -> str:
    """Format an EvalResult as a Phase 0 baseline markdown table."""
    d = result.to_dict()
    return f"""# Phase 0 baseline — `{model_name}` on GSM8K `{split}`

| Metric | Value |
|---|---|
| pass@1 | {d["pass_at_1"]:.4f} |
| pass@{result.n_samples_per_item} (unbiased) | {f'{d["pass_at_k"]:.4f}' if d["pass_at_k"] is not None else "n/a"} |
| Items evaluated | {d["n_items"]} |
| Samples per item | {d["n_samples_per_item"]} |
| Total completion tokens | {d["completion_tokens_total"]:,} |
| Wall-clock (s) | {d["wall_clock_sec"]:.2f} |
| Generation tok/s | {d["tokens_per_sec"]:.1f} |
| Model parameters | {d["model_params"]:,} |

Generated by `src/eval/harness.py`.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="GSM8K pass@1 / pass@k evaluator (no vLLM)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n", type=int, default=50, help="Number of GSM8K items to evaluate")
    p.add_argument("--n_samples", type=int, default=1, help="Samples per item (1=greedy)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="Override device (default: auto)")
    p.add_argument("--output", default=None, help="Path to write markdown report")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    set_all_seeds(args.seed)

    device = torch.device(args.device) if args.device else auto_device()
    print(f"Device: {device}", flush=True)

    print(f"Loading {args.model}...", flush=True)
    model, tokenizer = load_qwen_model(args.model, device=device)
    print(f"  loaded; params = {count_model_params(model):,}", flush=True)

    print(f"Loading GSM8K {args.split} (n={args.n}, seed={args.seed})...", flush=True)
    dataset = load_gsm8k_subset(args.split, n=args.n, seed=args.seed)

    print(
        f"Evaluating: n_samples={args.n_samples} temperature={args.temperature} "
        f"max_new_tokens={args.max_new_tokens} batch_size={args.batch_size}...",
        flush=True,
    )
    flop_budget = FLOPBudget()
    result = evaluate(
        model,
        tokenizer,
        dataset,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=device,
        flop_budget=flop_budget,
        seed=args.seed,
        verbose=args.verbose,
    )

    print("")
    print(f"pass@1                  = {result.pass_at_1:.4f}")
    if result.pass_at_k is not None:
        print(f"pass@{result.n_samples_per_item} (unbiased)    = {result.pass_at_k:.4f}")
    print(f"items                   = {result.n_items}")
    print(f"completion tokens total = {result.completion_tokens_total:,}")
    print(f"wall-clock              = {result.wall_clock_sec:.1f}s")
    print(f"throughput              = {result.completion_tokens_total / max(result.wall_clock_sec, 1e-9):.1f} tok/s")
    print(f"forward FLOPs (counter) = {flop_budget.forward_flops:.3e}")

    if args.output:
        md = _format_markdown_report(result, args.model, args.split)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(md)
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
