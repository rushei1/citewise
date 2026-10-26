PIP_EXTRA = --extra-index-url https://download.pytorch.org/whl/cpu

.PHONY: install lint format test run-api run-ui docker-up eval

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

eval:
	@echo "The evaluation harness is added in Phase 4 of the upgrade; nothing to run yet." && exit 1
