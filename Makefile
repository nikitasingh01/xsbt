.PHONY: help setup lint format format-check typecheck test test-cov check demo clean docker docker-run

PY ?= python
UNIVERSE ?= configs/universe_us_liquid.csv
START ?= 2010-01-01
END ?= 2026-08-01

help:
	@echo "setup        install the package and dev dependencies (editable)"
	@echo "lint         ruff check"
	@echo "format       ruff format + fix imports"
	@echo "format-check ruff format --check, no writes"
	@echo "typecheck    mypy on src/"
	@echo "test         pytest (offline, no network)"
	@echo "test-cov     pytest with coverage report"
	@echo "check        everything CI runs, in the same order"
	@echo "demo         fetch data, run both strategies, build both reports"
	@echo "docker       build the container image"
	@echo "clean        remove caches and build artifacts"

setup:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

format-check:
	$(PY) -m ruff format --check src tests

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest

test-cov:
	$(PY) -m pytest --cov --cov-report=term-missing

check: lint format-check typecheck test

# The only target that uses the network is the fetch. Both runs are --offline, which is
# also a check that the snapshot the fetch just wrote is complete.
demo:
	$(PY) -m xsbt fetch --universe $(UNIVERSE) --start $(START) --end $(END)
	$(PY) -m xsbt run --config configs/momentum.yaml --out runs/momentum --offline
	$(PY) -m xsbt run --config configs/reversal.yaml --out runs/reversal --offline
	@echo ""
	@echo "reports: runs/momentum/report.html and runs/reversal/report.html"

docker:
	docker build -t xsbt:latest .

docker-run:
	docker run --rm -v "$(PWD)/data:/app/data" -v "$(PWD)/runs:/app/runs" xsbt:latest --help

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
