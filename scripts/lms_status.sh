#!/bin/bash
# Check LM Studio status and loaded models
# Usage: ./lms_status.sh

echo "📊 LM Studio Status"
echo "==================="
echo ""

echo "🖥️  Server Status:"
lms server status
echo ""

echo "📦 Loaded Models:"
lms ps
echo ""

echo "💾 Downloaded Models:"
lms ls | head -20
