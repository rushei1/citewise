PIP_EXTRA = --extra-index-url https://download.pytorch.org/whl/cpu

.PHONY: install lint format test run-api run-ui docker-up eval eval-quick

install:
	pip install -r requirements.txt -r requirements-frontend.txt -r requirements-dev.txt $(PIP_EXTRA)

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest --cov=app --cov-report=term-missing

run-api:
	uvicorn app.main:app --reload

run-ui:
	streamlit run frontend/app.py

docker-up:
	docker compose up --build

# Full evaluation: needs the models (requirements.txt) and network on the first run; arXiv and LLM
# calls are cached under .eval_cache. The LLM arm runs only if LLM_MODEL (+ key or base URL) is set.
eval:
	python -m eval.build_benchmark
	python -m eval.run_clustering
	python -m eval.run_summarization --arm hf
	python -m eval.run_summarization --arm llm
	python -m eval.run_summarization --arm nli
	python -m eval.failure_cases
	python -m eval.report

# Tiny model-free smoke test for CI; writes to eval/results_quick/, not real results.
eval-quick:
	python -m eval.run_clustering --quick
	python -m eval.run_summarization --quick
	python -m eval.report --quick
