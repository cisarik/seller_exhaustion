#!/usr/bin/env bash
# Fix ADAM_EPSILON environment variable issue

echo "Unsetting ADAM_EPSILON environment variable..."
unset ADAM_EPSILON

echo "✓ Environment variable cleared"
echo ""
echo "Verifying settings load correctly:"
poetry run python -c "from config.settings import settings; print(f'adam_epsilon: {settings.adam_epsilon}')"
echo ""
echo "Now run: ./adam.sh"
