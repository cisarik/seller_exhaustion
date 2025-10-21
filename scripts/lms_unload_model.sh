#!/bin/bash
# Unload model from LM Studio
# Usage: ./lms_unload_model.sh [model_identifier]

MODEL_ID="${1}"

if [ -z "$MODEL_ID" ]; then
    echo "Usage: $0 <model_identifier>"
    echo "Run 'lms ps' to see loaded models"
    exit 1
fi

echo "🔄 Unloading model: $MODEL_ID"

lms unload "$MODEL_ID"

if [ $? -eq 0 ]; then
    echo "✅ Model unloaded successfully"
else
    echo "❌ Failed to unload model"
    exit 1
fi
