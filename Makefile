.PHONY: help install test lint ui fetch backtest clean run monitor

# Suppress pyenv global py2app/pkg_resources noise when Poetry bootstraps
export PYTHONWARNINGS ?= ignore::UserWarning:py2app,ignore:pkg_resources is deprecated:UserWarning

TF ?= 15m
DATA ?= .data/X_ADAUSD_2024-01-01_2026-06-15_15minute.parquet

help:
	@echo "ADA Seller-Exhaustion — make targets"
	@echo ""
	@echo "  make install     Poetry install"
	@echo "  make test        pytest (warnings filtered)"
	@echo "  make run ARGS=   Run cli.py, e.g. ARGS='paper-top-stats --tf 15m'"
	@echo "  make monitor     Full paper-scheduler cycle (TF=15m DATA=...)"
	@echo "  make ui          PySide6 UI"
	@echo ""

install:
	poetry install

test:
	poetry run pytest tests/ -v

run:
	poetry run python cli.py $(ARGS)

monitor:
	poetry run python cli.py paper-scheduler --tf $(TF) --data $(DATA) --refresh

validate:
	poetry run python cli.py validate-candidate --tf $(TF) --data $(DATA)

lint:
	poetry run ruff check .

ui:
	poetry run python cli.py ui

fetch:
	poetry run python cli.py fetch --from 2024-01-01 --to 2025-01-13

backtest:
	poetry run python cli.py backtest --from 2024-01-01 --to 2025-01-13

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
