.PHONY: setup lint format typecheck test integration reproduce report serve clean

UV ?= uv
RUN = $(UV) run --frozen

setup:
	$(UV) venv --python 3.13 .venv
	$(UV) sync --frozen

lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format:
	$(RUN) ruff format src tests
	$(RUN) ruff check --fix src tests

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest --cov --cov-report=term-missing:skip-covered

# Starts the real I2 and S6 from ../point-in-time-store and ../research-ledger (their own .venv).
integration:
	$(RUN) pytest -m integration -p no:cacheprovider -o addopts="-ra --strict-markers" tests/live

reproduce:
	$(RUN) bt reproduce

report:
	$(RUN) bt report

serve:
	$(RUN) bt serve
