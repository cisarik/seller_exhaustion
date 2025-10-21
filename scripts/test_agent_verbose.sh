#!/bin/bash
# Test agent with maximum verbosity and logging
# Usage: ./test_agent_verbose.sh

echo "🔬 Running Agent Test in Verbose Mode"
echo "======================================"
echo ""

# Set environment variables for max logging
export COACH_DEBUG_PAYLOADS=true
export PYTHONPATH=/home/agile/seller_exhaustion:$PYTHONPATH

# Enable Python logging at DEBUG level
export PYTHONUNBUFFERED=1

cd /home/agile/seller_exhaustion

echo "📋 Environment:"
echo "   COACH_DEBUG_PAYLOADS: $COACH_DEBUG_PAYLOADS"
echo "   Python logging: DEBUG"
echo ""

# Run with Python logging
poetry run python -u test_agent.py 2>&1 | tee agent_test_verbose.log

echo ""
echo "✅ Test complete"
echo "   Full log saved to: agent_test_verbose.log"
echo ""
echo "🔍 To see LM Studio's view, run in another terminal:"
echo "   lms log stream"
