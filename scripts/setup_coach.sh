#!/bin/bash
# Setup Evolution Coach with LM Studio

set -e

echo "=================================================="
echo "Evolution Coach Setup"
echo "=================================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python3 --version || { echo "❌ Python 3 not found"; exit 1; }
echo "✅ Python OK"
echo ""

# Install LM Studio Python SDK
echo "2. Installing LM Studio Python SDK..."
pip install lmstudio || { echo "⚠️  Warning: lmstudio package not available yet"; }
echo "✅ SDK installation attempted"
echo ""

# Check if LM Studio is running
echo "3. Checking LM Studio server..."
if curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo "✅ LM Studio server is running"
    echo ""
    echo "Available models:"
    curl -s http://localhost:1234/v1/models | python3 -m json.tool
else
    echo "⚠️  LM Studio server not detected"
    echo ""
    echo "To setup LM Studio:"
    echo "  1. Download from: https://lmstudio.ai/"
    echo "  2. Install and open LM Studio"
    echo "  3. Go to 'Discover' tab"
    echo "  4. Search for 'google/gemma-3-12b' or 'gemma-2-9b-it'"
    echo "  5. Download the model"
    echo "  6. Go to 'My Models' and load it"
    echo "  7. Enable server mode (port 1234)"
    echo ""
    echo "Then run this script again"
    exit 1
fi
echo ""

# Create coach_prompts directory if it doesn't exist
echo "4. Setting up prompt directory..."
mkdir -p coach_prompts
if [ -f "coach_prompts/async_coach_v1.txt" ]; then
    echo "✅ Prompt directory ready"
else
    echo "⚠️  Default prompt not found"
    echo "   Make sure coach_prompts/async_coach_v1.txt exists"
fi
echo ""

# Test coach connection
echo "5. Testing coach connection..."
python3 << EOF
import asyncio
from backtest.coach_protocol import EvolutionState
from backtest.llm_coach import GemmaCoachClient

async def test():
    try:
        client = GemmaCoachClient(verbose=False)
        print("✅ Coach client initialized")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

result = asyncio.run(test())
if not result:
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "✅ Coach connection test passed"
else
    echo "❌ Coach connection test failed"
    exit 1
fi
echo ""

# Final summary
echo "=================================================="
echo "Setup Complete! 🎉"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Run example:"
echo "     python3 examples/evolution_with_coach.py --generations 20"
echo ""
echo "  2. Or use in your code:"
echo "     from backtest.coach_manager import CoachManager"
echo "     coach = CoachManager()"
echo ""
echo "  3. Read documentation:"
echo "     docs/EVOLUTION_COACH_ASYNC_GUIDE.md"
echo ""
echo "=================================================="
