.PHONY: install fmt lint type test run eval up down

BACKEND := backend

install:
	cd $(BACKEND) && uv sync --all-extras --all-groups
	uv run --directory $(BACKEND) pre-commit install

fmt:
	cd $(BACKEND) && uv run ruff format . && uv run ruff check --fix .

lint:
	cd $(BACKEND) && uv run ruff format --check . && uv run ruff check . && uv run lint-imports

type:
	cd $(BACKEND) && uv run mypy

test:
	cd $(BACKEND) && uv run pytest

run:
	cd $(BACKEND) && uv run research $(Q)

eval:
	cd $(BACKEND) && uv run python -m evals.run_eval

up:
	docker compose up -d --build

down:
	docker compose down
