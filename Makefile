.PHONY: help install test lint ui fetch backtest clean

help:
	@echo "ADA Seller-Exhaustion Agent - Available targets:"
	@echo ""
	@echo "  make install    - Install dependencies with Poetry"
	@echo "  make test       - Run all tests"
	@echo "  make ui         - Launch PySide6 UI"
	@echo "  make fetch      - Fetch sample data"
	@echo "  make backtest   - Run backtest on sample data"
	@echo "  make lint       - Run ruff linter"
	@echo "  make clean      - Remove generated files"
	@echo ""

install:
	poetry install

test:
	poetry run pytest tests/ -v

lint:
	poetry run ruff check .

ui:
	poetry run python cli.py ui

fetch:
	poetry run python cli.py fetch --from 2024-01-01 --to 2025-01-13

backtest:
	poetry run python cli.py backtest --from 2024-01-01 --to 2025-01-13

clean:
	rm -rf .data __pycache__ **/__pycache__ .pytest_cache .ruff_cache
	rm -f trades.csv features.csv
	find . -name "*.pyc" -delete
