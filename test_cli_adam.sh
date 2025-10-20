#!/usr/bin/env bash
# ADAM Optimizer Test - Forces epsilon=0.02 via environment variable

# IMPORTANT: Environment variables OVERRIDE .env file in pydantic!
# This is why we explicitly set it here to ensure it's used
export ADAM_EPSILON=0.02

# Default values
POPULATION="${1:-populations/118576.json}"
DATA="${2:-.data/X_ADAUSD_2024-01-14_2025-10-17_15minute.parquet}"
TIMEFRAME="${3:-15m}"
GENERATIONS="${4:-1}"

echo "============================================"
echo "ADAM Optimizer TEST (Forced epsilon=0.02)"
echo "============================================"
echo "Population:  $POPULATION"
echo "Data:        $DATA"
echo "Timeframe:   $TIMEFRAME"
echo "Generations: $GENERATIONS"
echo "ADAM_EPSILON: $ADAM_EPSILON (explicitly set)"
echo "============================================"
echo ""

poetry run python cli.py optimize --optimizer adam --init-from "$POPULATION" --data "$DATA" --tf "$TIMEFRAME" --generations "$GENERATIONS"
