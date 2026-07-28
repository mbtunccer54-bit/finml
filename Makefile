# =============================================================================
# FinML Platform — developer entrypoints
# =============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash

VENV        ?= .venv
ifeq ($(OS),Windows_NT)
PY          := $(VENV)/Scripts/python.exe
else
PY          := $(VENV)/bin/python
endif
PYTEST      := $(PY) -m pytest
MLFLOW_URI  ?= sqlite:///mlflow.db
API_PORT    ?= 8000
UI_PORT     ?= 8501
MLFLOW_PORT ?= 5000

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Environment -------------------------------------------------------------
.PHONY: venv
venv:  ## Create the virtual environment
	python -m venv $(VENV)

.PHONY: install
install: venv  ## Install the package with dev extras (editable)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

.PHONY: install-nlp
install-nlp:  ## Install the heavy transformer/topic-model extras
	$(PY) -m pip install -e ".[nlp]"

.PHONY: hooks
hooks:  ## Register pre-commit hooks
	$(PY) -m pre_commit install

# --- Quality -----------------------------------------------------------------
.PHONY: lint
lint:  ## Ruff lint
	$(PY) -m ruff check src tests

.PHONY: format
format:  ## Ruff format (writes)
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

.PHONY: typecheck
typecheck:  ## Mypy strict
	$(PY) -m mypy src

.PHONY: security
security:  ## Bandit security scan
	$(PY) -m bandit -c pyproject.toml -r src

.PHONY: test
test:  ## Full test suite with coverage gate
	$(PYTEST) --cov=src --cov-report=term-missing --cov-report=xml

.PHONY: test-fast
test-fast:  ## Unit tests only (pre-commit hook)
	$(PYTEST) -m "unit" -q --no-cov

.PHONY: check
check: lint typecheck security test  ## Everything CI runs

# --- Pipelines ---------------------------------------------------------------
.PHONY: train
train:  ## Run the training pipeline (Hydra; override with ARGS="model=lightgbm")
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PY) -m application.train_pipeline $(ARGS)

.PHONY: train-fast
train-fast:  ## Training smoke run: no tuning, small synthetic dataset
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PY) -m application.train_pipeline \
		model.tuning.enabled=false data.n_entities=300 data.n_periods=8 $(ARGS)

.PHONY: infer
infer:  ## Batch inference over the demo dataset
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PY) -m application.inference_pipeline $(ARGS)

# --- Services ----------------------------------------------------------------
.PHONY: serve
serve:  ## Run the FastAPI service
	$(PY) -m uvicorn infrastructure.api.fastapi_app:app --host 0.0.0.0 --port $(API_PORT) --reload

.PHONY: dashboard
dashboard:  ## Run the Streamlit dashboard
	$(PY) -m streamlit run src/presentation/streamlit_app.py --server.port $(UI_PORT)

.PHONY: mlflow-ui
mlflow-ui:  ## Run the MLflow tracking UI
	$(PY) -m mlflow ui --backend-store-uri $(MLFLOW_URI) --port $(MLFLOW_PORT)

# --- Docker ------------------------------------------------------------------
.PHONY: docker-build
docker-build:  ## Build the container image
	docker compose build

.PHONY: up
up:  ## Start api + dashboard + mlflow
	docker compose up -d

.PHONY: down
down:  ## Stop the stack
	docker compose down

.PHONY: logs
logs:  ## Tail stack logs
	docker compose logs -f

# --- Housekeeping ------------------------------------------------------------
.PHONY: clean
clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-artifacts
clean-artifacts:  ## Remove trained models, feature store and MLflow runs
	rm -rf artifacts mlruns data/feature_store outputs
