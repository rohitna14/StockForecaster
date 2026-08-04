.DEFAULT_GOAL := help
PY := backend/.venv/Scripts/python.exe
ifeq ($(OS),)
PY := backend/.venv/bin/python
endif

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

install: ## Create the venv and install backend + frontend deps
	python -m venv backend/.venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e "backend[data,ml,api,dev]"
	cd frontend && npm install

seed: ## Create the schema and load instruments + universes
	$(PY) -m forecaster.cli db init
	$(PY) -m forecaster.cli seed

ingest: ## Fetch 5y of history for the demo universe into both tiers
	$(PY) -m forecaster.cli ingest --universe demo --hot --years 5

evaluate: ## Run the full experiment and regenerate docs/RESULTS.md
	$(PY) -m forecaster.cli evaluate --write-results

status: ## Show coverage across hot and cold tiers
	$(PY) -m forecaster.cli status

api: ## Run the API with reload
	cd backend && .venv/Scripts/python.exe -m uvicorn forecaster.api.main:app --reload

web: ## Run the Next.js frontend
	cd frontend && npm run dev

streamlit: ## Run the Streamlit client
	cd streamlit_app && streamlit run app.py

test: ## Run the full test suite (excluding deep + network)
	cd backend && .venv/Scripts/python.exe -m pytest tests -m "not network and not deep"

test-leakage: ## Run only the lookahead-bias guards
	cd backend && .venv/Scripts/python.exe -m pytest tests/unit/test_leakage.py tests/unit/test_windowing.py -v

test-deep: ## Run the sequence-model tests (needs TensorFlow)
	cd backend && .venv/Scripts/python.exe -m pytest tests/unit/test_deep_models.py -v

lint: ## Lint and format-check
	cd backend && .venv/Scripts/python.exe -m ruff check src tests
	cd backend && .venv/Scripts/python.exe -m ruff format --check src tests

fmt: ## Auto-format
	cd backend && .venv/Scripts/python.exe -m ruff format src tests
	cd backend && .venv/Scripts/python.exe -m ruff check --fix src tests

up: ## Full stack via docker compose
	docker compose up --build

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.pytest_cache backend/.ruff_cache backend/.mypy_cache frontend/.next

.PHONY: help install seed ingest evaluate status api web streamlit test test-leakage test-deep lint fmt up clean
