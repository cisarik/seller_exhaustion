#!/bin/bash
# Debug agent with live LM Studio logs
# Usage: ./debug_agent_with_logs.sh

echo "🔍 Starting Agent Debug Mode"
echo "This will run test_agent.py with live LM Studio logs"
echo ""

# Check if lms is available
if ! command -v lms &> /dev/null; then
    echo "❌ lms CLI not found"
    echo "   Install: https://lmstudio.ai/docs/cli"
    exit 1
fi

# Check if server is running
if ! lms server status 2>&1 | grep -q "running\|port"; then
    echo "❌ LM Studio server not running"
    echo "   Start server: lms server start"
    exit 1
fi

echo "✅ LM Studio server is running"
echo ""

# Start log streaming in background
echo "📺 Starting log stream (Ctrl+C to stop)..."
echo "   This will show the EXACT prompts sent to the model"
echo ""
echo "=" $(tput cols)
lms log stream --filter llm.prediction &
LOG_PID=$!

# Give logs a moment to start
sleep 2

# Run the agent test in foreground
echo ""
echo "🤖 Running agent test..."
echo "=" $(tput cols)
poetry run python test_agent.py

# Kill log stream
kill $LOG_PID 2>/dev/null

echo ""
echo "✅ Debug session complete"
echo ""
echo "💡 Tips:"
echo "   • Check the logs above for the exact prompt sent to model"
echo "   • Look for 'llm.prediction.input' entries"
echo "   • Verify JSON format in system prompt"
echo "   • Adjust temperature if responses are too creative"
