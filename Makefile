.PHONY: help install install-dev install-cloud smoke smoke-grpo smoke-es test repro figs clean

help:
	@echo "Targets:"
	@echo "  install        - install runtime deps (toy phase, no vllm)"
	@echo "  install-dev    - install runtime + dev deps (toy phase, no vllm)"
	@echo "  install-cloud  - install runtime + dev + cloud deps (adds vllm; CUDA Linux only)"
	@echo "  smoke          - Phase 0 smoke (Qwen2.5-0.5B on 50 GSM8K prompts via MPS, pass@1 >= 0.30)"
	@echo "  smoke-grpo     - Phase 1 fast loop (20 GRPO steps, toy config)"
	@echo "  smoke-es       - Phase 2 fast loop (5 ES generations, N=10)"
	@echo "  test           - run pytest"
	@echo "  repro          - re-run all headline configs (TBD after Phase 4)"
	@echo "  figs           - regenerate all figures from results/ (TBD after Phase 4)"
	@echo "  clean          - remove caches and build artifacts"

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

install-cloud:
	uv pip install -e ".[dev,cloud]"

smoke:
	bash scripts/smoke_test.sh

smoke-grpo:
	uv run python -m src.grpo.train --config configs/grpo/smoke.yaml --max_steps 20 --eval_subset 50

smoke-es:
	uv run python -m src.es.train --config configs/es/smoke.yaml

test:
	uv run pytest -v

repro:
	@echo "Re-running all headline configs (TBD — implement after Phase 3)"
	@exit 1

figs:
	@echo "Regenerating figures from results/ (TBD — implement after Phase 3)"
	@exit 1

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
