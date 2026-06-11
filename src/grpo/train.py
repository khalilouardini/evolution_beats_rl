"""Config-driven GRPO training entrypoint (toy phase: no vLLM, MPS-safe).

Per CLAUDE.md §6 Phase 1. CLI matches the Makefile `smoke-grpo` contract:

    python -m src.grpo.train --config configs/grpo/toy_0.5b.yaml \\
        [--seed N] [--max_steps N] [--eval_subset N] [--set key=value ...]

Design notes (verified against installed TRL 0.29.1 — see Phase 1 plan):
- loss_type MUST be "grpo": the TRL default is "dapo", which would silently
  benchmark a different algorithm than the project compares.
- beta MUST be set explicitly: the TRL default is 0.0 (no KL), CLAUDE.md
  §10 specifies 0.04.
- With a PEFT model and beta≠0, TRL skips the separate reference model and
  computes ref logps via adapter-disable — no extra model in memory.
- TRL 0.29.1 has NO max_prompt_length: we assert-guard prompt lengths at
  dataset build time instead (GSM8K prompts are short; fail fast if not).
- bf16/fp16 Trainer flags stay False: base model is loaded bf16 by our
  loader (with the MPS eager-attention fix), peft gives fp32 LoRA adapters,
  and we avoid the untested accelerate-autocast-on-MPS path.
- lr_scheduler_type="constant" deviates from the HF default (linear decay):
  CLAUDE.md §10 specifies a flat LR and the ES side runs constant-LR Adam;
  letting LR decay would make the nominal 5e-6 a lie over the run.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.data.gsm8k import SYSTEM_PROMPT, load_gsm8k_subset
from src.data.rewards import accuracy_reward, format_reward

REQUIRED_KEYS = [
    "run.seed",
    "run.wandb_mode",
    "model.name",
    "data.train_n",
    "data.eval_n",
    "data.data_seed",
    "data.max_prompt_tokens",
    "lora.r",
    "grpo.max_steps",
    "grpo.learning_rate",
    "grpo.num_generations",
    "grpo.beta",
    "reward.weights",
    "eval.every_steps",
]


def load_config(argv: list[str] | None = None) -> DictConfig:
    p = argparse.ArgumentParser(description="Toy GRPO training (no vLLM)")
    p.add_argument("--config", required=True, help="Path to yaml config")
    p.add_argument("--seed", type=int, default=None, help="Override run.seed")
    p.add_argument("--max_steps", type=int, default=None, help="Override grpo.max_steps")
    p.add_argument("--eval_subset", type=int, default=None, help="Override data.eval_n")
    p.add_argument("--set", nargs="*", default=[], help="Dotlist overrides, e.g. grpo.beta=0.02")
    args = p.parse_args(argv)

    cfg = OmegaConf.load(args.config)
    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.max_steps is not None:
        cfg.grpo.max_steps = args.max_steps
    if args.eval_subset is not None:
        cfg.data.eval_n = args.eval_subset
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))

    missing = [k for k in REQUIRED_KEYS if OmegaConf.select(cfg, k) is None]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg


def build_run_name(cfg: DictConfig, today: str | None = None) -> str:
    """§7 pattern: {phase}_{method}_{model}_{seed}_{date}."""
    date = today or _dt.date.today().isoformat()
    return f"{cfg.run.phase}_{cfg.run.method}_{cfg.run.model_tag}_s{cfg.run.seed}_{date}"


def build_train_dataset(cfg: DictConfig, tokenizer):
    """GSM8K subset → conversational prompts; keep `answer` for the reward.

    data_seed is FIXED (decoupled from run.seed) so every run seed trains on
    the identical prompt subset — seed variance must come from init/sampling,
    not from data draw.
    """
    ds = load_gsm8k_subset("train", n=int(cfg.data.train_n), seed=int(cfg.data.data_seed))

    def to_prompt(ex):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": ex["answer"],
        }

    ds = ds.map(to_prompt, remove_columns=["question"])
    _assert_prompt_lengths(ds, tokenizer, int(cfg.data.max_prompt_tokens))
    return ds


def _assert_prompt_lengths(ds, tokenizer, max_tokens: int) -> None:
    """TRL 0.29.1 has no max_prompt_length — fail fast on oversized prompts."""
    lengths = [
        len(tokenizer.apply_chat_template(ex["prompt"], tokenize=True, add_generation_prompt=True))
        for ex in ds
    ]
    longest = max(lengths)
    if longest > max_tokens:
        raise ValueError(
            f"Longest prompt is {longest} tokens > max_prompt_tokens={max_tokens}; "
            "TRL 0.29.1 does not truncate prompts — fix the data or raise the limit."
        )
    print(f"Prompt lengths: max={longest}, mean={sum(lengths) / len(lengths):.1f} (limit {max_tokens})")


def build_eval_dataset(cfg: DictConfig):
    """Raw GSM8K test rows — harness.evaluate formats its own prompts."""
    return load_gsm8k_subset("test", n=int(cfg.data.eval_n), seed=int(cfg.data.data_seed))


def build_lora_config(cfg: DictConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=str(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
        bias="none",
    )


def build_grpo_config(cfg: DictConfig, run_dir: Path, run_name: str):
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(run_dir / "trainer"),
        run_name=run_name,
        seed=int(cfg.run.seed),
        # core GRPO — beta and loss_type MUST be explicit (TRL defaults: 0.0, "dapo")
        num_generations=int(cfg.grpo.num_generations),
        max_completion_length=int(cfg.grpo.max_completion_length),
        beta=float(cfg.grpo.beta),
        epsilon=float(cfg.grpo.epsilon),
        loss_type=str(cfg.grpo.loss_type),
        scale_rewards=str(cfg.grpo.scale_rewards),
        num_iterations=1,
        reward_weights=[float(w) for w in cfg.reward.weights],
        # rollout sampling (do_sample=True is forced by TRL)
        temperature=float(cfg.grpo.temperature),
        top_p=float(cfg.grpo.top_p),
        # batch math → generation_batch_size derived = bs × grad_accum (= 16 toy)
        per_device_train_batch_size=int(cfg.grpo.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.grpo.gradient_accumulation_steps),
        # optimizer
        learning_rate=float(cfg.grpo.learning_rate),
        lr_scheduler_type=str(cfg.grpo.lr_scheduler_type),
        max_grad_norm=1.0,
        max_steps=int(cfg.grpo.max_steps),
        # plumbing
        use_vllm=False,
        disable_dropout=bool(cfg.grpo.disable_dropout),
        logging_steps=int(cfg.grpo.logging_steps),
        save_strategy="no",  # checkpointing handled by PeriodicEvalCallback
        report_to=["wandb"],
        # TRL's BaseConfig.__post_init__ flips bf16 to True when left None
        # (base_config.py:105) — NOT the HF default. Explicit False keeps the
        # dtype strategy: pure-bf16 weights from our loader, fp32 LoRA, no
        # accelerate autocast on MPS.
        bf16=False,
        fp16=False,
        # deliberately NOT set: generation_batch_size / steps_per_generation
        # (derived by TRL), max_prompt_length (does not exist in TRL 0.29.1)
    )


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(argv)

    # Device + seeds before any model/data work. auto_device sets
    # PYTORCH_ENABLE_MPS_FALLBACK=1 — must happen before model load.
    from src.eval.harness import auto_device, load_qwen_model
    from src.utils.seed import set_all_seeds

    device = auto_device()
    set_all_seeds(int(cfg.run.seed))

    run_name = build_run_name(cfg)
    run_dir = Path(str(cfg.run.output_root)) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_name}\nDevice: {device}\nDir: {run_dir}")

    # Provenance per §7: seed, git_sha, model_revision in config + wandb.
    git_sha = _git_sha()

    import wandb

    resolved = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=str(cfg.run.wandb_project),
        name=run_name,
        group=str(cfg.run.wandb_group),
        tags=[str(t) for t in cfg.run.wandb_tags],
        mode=str(cfg.run.wandb_mode),
        config={**resolved, "git_sha": git_sha, "seed": int(cfg.run.seed)},
    )

    print(f"Loading {cfg.model.name}...")
    model, tokenizer = load_qwen_model(str(cfg.model.name), device=device)
    model_revision = getattr(model.config, "_commit_hash", None) or "unknown"

    from src.eval.compute import count_model_params

    params = count_model_params(model)  # pre-LoRA-wrap, documented accounting basis
    print(f"  params={params:,} revision={model_revision}")

    OmegaConf.save(
        OmegaConf.create({**resolved, "git_sha": git_sha, "model_revision": model_revision}),
        run_dir / "config_resolved.yaml",
    )

    print("Building datasets...")
    train_ds = build_train_dataset(cfg, tokenizer)
    eval_ds = build_eval_dataset(cfg)
    print(f"  train={len(train_ds)} eval={len(eval_ds)}")

    from src.grpo.callbacks import (
        FLOPAccountingCallback,
        MetricsJSONLCallback,
        PeriodicEvalCallback,
    )

    generation_batch_size = int(cfg.grpo.per_device_train_batch_size) * int(
        cfg.grpo.gradient_accumulation_steps
    )
    flop_cb = FLOPAccountingCallback(
        run_dir=run_dir,
        params=params,
        beta=float(cfg.grpo.beta),
        generation_batch_size=generation_batch_size,
        write_every_steps=int(cfg.flops.write_every_steps),
    )
    jsonl_cb = MetricsJSONLCallback(run_dir=run_dir, flop_callback=flop_cb)
    eval_cb = PeriodicEvalCallback(
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        run_dir=run_dir,
        eval_flop_budget=flop_cb.eval_budget,
        jsonl_callback=jsonl_cb,
        every_steps=int(cfg.eval.every_steps),
        batch_size=int(cfg.eval.batch_size),
        max_new_tokens=int(cfg.eval.max_new_tokens),
        save_checkpoint=bool(cfg.eval.save_checkpoint),
        eval_at_start=bool(cfg.eval.eval_at_start),
    )

    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[accuracy_reward, format_reward],
        args=build_grpo_config(cfg, run_dir, run_name),
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
        callbacks=[flop_cb, jsonl_cb, eval_cb],
    )

    print("Training...")
    trainer.train()

    trainer.model.save_pretrained(str(run_dir / "adapter_final"))
    flop_cb.write_flops_json(trainer.state.global_step)

    final_eval = eval_cb.history[-1] if eval_cb.history else None
    baseline_eval = eval_cb.history[0] if eval_cb.history else None
    print("\n=== Run summary ===")
    print(f"  run            : {run_name}")
    if baseline_eval and final_eval:
        print(f"  pass@1 step 0  : {baseline_eval['eval/pass_at_1']:.4f}")
        print(f"  pass@1 final   : {final_eval['eval/pass_at_1']:.4f} (step {final_eval['step']})")
    print(f"  train FLOPs    : {flop_cb.budget.total_flops:.3e}")
    print(f"  eval FLOPs     : {flop_cb.eval_budget.total_flops:.3e}")
    print(f"  generated tok  : {flop_cb.budget.generated_tokens:,}")

    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
