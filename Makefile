.PHONY: install lint format check build train run

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run python ../agent-dev-harness/python-styleguide/docstring_length.py .
	uv run mypy .

build:
	docker build -t weather-model .

train:
	docker run --rm \
		-v $(PWD)/src/data:/app/src/data \
		weather-model python src/train_model.py

run:
	docker run --rm \
		-v $(PWD)/src/data:/data \
		-v $(PWD)/src/data/scheduled_runs:/output \
		weather-model
