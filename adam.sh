#!/usr/bin/env bash
# ADAM Optimizer - Quick Launch Script

# Default values
POPULATION="${1:-populations/118576.json}"
DATA="${2:-.data/X_ADAUSD_2025-09-14_2025-10-14_15minute.parquet}"
TIMEFRAME="${3:-15m}"
GENERATIONS="${4:-10}"

echo "============================================"
echo "ADAM Optimizer"
echo "============================================"
echo "Population:  $POPULATION"
echo "Data:        $DATA"
echo "Timeframe:   $TIMEFRAME"
echo "Generations: $GENERATIONS"
echo "============================================"
echo ""

poetry run python cli.py optimize --optimizer adam --init-from "$POPULATION" --data "$DATA" --tf "$TIMEFRAME" --generations "$GENERATIONS"
