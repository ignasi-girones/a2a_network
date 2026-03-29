#!/bin/bash
# Start all agents and frontend for local development.
# Usage: bash start_all.sh
# Requires: .env file configured with API keys.

set -e

export PYTHONPATH="$(pwd)"

echo "Starting MCP Tools Server..."
python -m agents.mcp_tools.server &

echo "Starting Normalizer Agent (port 9001)..."
python -m agents.normalizer &

echo "Starting Feedback Agent (port 9004)..."
python -m agents.feedback &

echo "Starting Specialized Agent AE1 (port 9002)..."
python -m agents.specialized --port 9002 --agent-id ae1 &

echo "Starting Specialized Agent AE2 (port 9003)..."
python -m agents.specialized --port 9003 --agent-id ae2 &

sleep 2

echo "Starting Orchestrator Agent (port 9000)..."
python -m agents.orchestrator &

echo ""
echo "All agents started. Starting frontend..."
cd frontend && npm run dev &

echo ""
echo "=========================================="
echo "  A2A Debate Network is running!"
echo "  Frontend: http://localhost:5173"
echo "  Orchestrator: http://localhost:9000"
echo "=========================================="

wait
