#!/bin/bash
# Start LM Studio server
# Usage: ./lms_server_start.sh

echo "🚀 Starting LM Studio server..."

lms server start

if [ $? -eq 0 ]; then
    echo "✅ Server started successfully"
    echo "   Endpoint: http://localhost:1234"
    sleep 2
    lms server status
else
    echo "❌ Failed to start server"
    exit 1
fi
