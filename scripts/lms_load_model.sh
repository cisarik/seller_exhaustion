#!/bin/bash
# Load model into LM Studio
# Usage: ./lms_load_model.sh [model_name] [gpu_ratio] [context_length]

MODEL="${1:-google/gemma-3-12b}"
GPU="${2:-0.6}"
CONTEXT="${3:-5000}"

echo "🔄 Loading model: $MODEL"
echo "   GPU offload: ${GPU} (0.0-1.0)"
echo "   Context length: $CONTEXT tokens"

lms load "$MODEL" \
    --gpu "$GPU" \
    --context-length "$CONTEXT" \
    --yes

if [ $? -eq 0 ]; then
    echo "✅ Model loaded successfully"
    lms ps  # Show loaded models
else
    echo "❌ Failed to load model"
    exit 1
fi
