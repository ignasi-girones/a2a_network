#!/bin/bash
# Start all agents and frontend for local development.
# Usage: bash start_all.sh
# Requires: .env file configured with API keys.
#
# Ports match the UPC FIB VM tunnel (nattech.fib.upc.edu:40530-40539 → 8080-8089).

set -e

export PYTHONPATH="$(pwd)"

echo "Starting MCP Tools Server (port 8085)..."
python -m agents.mcp_tools.server &

echo "Starting Normalizer Agent (port 8081)..."
python -m agents.normalizer &

echo "Starting Feedback Agent (port 8084)..."
python -m agents.feedback &

echo "Starting Specialized Agent AE1 (port 8082)..."
python -m agents.specialized --port 8082 --agent-id ae1 &

echo "Starting Specialized Agent AE2 (port 8083)..."
python -m agents.specialized --port 8083 --agent-id ae2 &

sleep 2

echo "Starting Orchestrator Agent (port 8080)..."
python -m agents.orchestrator &

echo ""
echo "All agents started. Starting frontend..."
cd frontend && npm run dev &

echo ""
echo "=========================================="
echo "  A2A Debate Network is running!"
echo "  Frontend:     http://localhost:8086"
echo "  Orchestrator: http://localhost:8080"
echo "=========================================="

wait
