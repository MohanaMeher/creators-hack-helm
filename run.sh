#!/bin/bash
# Helm Startup Script
# Note: Set ANTHROPIC_API_KEY environment variable or use .env file

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: ANTHROPIC_API_KEY not set. Please set it as an environment variable or in .env file"
    echo "Example: export ANTHROPIC_API_KEY='your_key_here'"
    echo ""
fi

echo "Starting Helm with REAL agents..."
echo "Access the control plane at: http://localhost:8001"
echo ""

python3 main.py
