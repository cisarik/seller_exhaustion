#!/usr/bin/env bash
# Genetic Algorithm Optimizer Test

# Default values
POPULATION="${1:-populations/118576.json}"
DATA="${2:-.data/X_ADAUSD_2024-01-14_2025-10-17_15minute.parquet}"
TIMEFRAME="${3:-15m}"
GENERATIONS="${4:-5}"

echo "============================================"
echo "GA Optimizer TEST"
echo "============================================"
echo "Population:  $POPULATION"
echo "Data:        $DATA"
echo "Timeframe:   $TIMEFRAME"
echo "Generations: $GENERATIONS"
echo "============================================"
echo ""

poetry run python cli.py optimize --optimizer evolutionary --init-from "$POPULATION" --data "$DATA" --tf "$TIMEFRAME" --generations "$GENERATIONS"
