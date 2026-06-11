# =============================================================================
# Path-VQA Med-GaMMa — Common tasks
# =============================================================================

.PHONY: help smoke-test train train-runpod evaluate serve demo lint format test clean

help:
	@echo ""
	@echo "  Path-VQA Med-GaMMa — available targets"
	@echo ""
	@echo "  Training:"
	@echo "    make smoke-test       5-step pipeline sanity check"
	@echo "    make train            Full training run (config/config.yaml)"
	@echo "    make train-runpod     Full training via RunPod script (with logging)"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make evaluate         Evaluate outputs/final on test split"
	@echo ""
	@echo "  Serving:"
	@echo "    make serve            FastAPI server on :8000"
	@echo "    make demo             Gradio demo"
	@echo ""
	@echo "  Dev:"
	@echo "    make lint             ruff + black check"
	@echo "    make format           ruff --fix + black"
	@echo "    make test             pytest"
	@echo "    make clean            Remove outputs, wandb, metrics, caches"
	@echo ""

# ── Training ──────────────────────────────────────────────────────────────────
smoke-test:
	python -m src.train --smoke-test

train:
	python -m src.train --config config/config.yaml

train-runpod:
	bash scripts/train_runpod.sh

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate:
	python -m src.evaluate --model outputs/final --output-json metrics/eval.json

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

demo:
	python hf_spaces_app.py

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	black --check src/ tests/ hf_spaces_app.py

format:
	ruff check --fix src/ tests/ hf_spaces_app.py
	black src/ tests/ hf_spaces_app.py

test:
	pytest tests/ -v

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/ wandb/ metrics/ mlruns/ logs/ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
