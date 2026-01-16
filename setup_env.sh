#!/bin/bash
echo "Helm - Real Agent Control Plane Setup"
echo ""
echo "You need to provide your Anthropic API key."
echo "Get it from: https://console.anthropic.com/"
echo ""
read -p "Enter your ANTHROPIC_API_KEY: " api_key
echo "ANTHROPIC_API_KEY=$api_key" > .env
echo ""
echo "✓ .env file created!"
echo "Now run: python main.py"
