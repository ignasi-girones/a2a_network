#!/bin/bash
# Start all agents and frontend for local development.
# Usage: bash start_all.sh
# Requires: .env file configured with API keys.
#
# Ports match the UPC FIB VM tunnel (nattech.fib.upc.edu:40530-40539 → 8080-8089).
#
# Each process writes its own log to ./logs/<service>.log so you can grep /
# tail / share a single agent's output without the others' noise interleaved.
# Live tail example:   tail -f logs/orchestrator.log

set -e

export PYTHONPATH="$(pwd)"

# Make sure the logs directory exists and is empty for this run so old
# output doesn't get confused with the new run.
LOG_DIR="$(pwd)/logs"
mkdir -p "$LOG_DIR"
: > "$LOG_DIR/mcp_tools.log"
: > "$LOG_DIR/normalizer.log"
: > "$LOG_DIR/feedback.log"
: > "$LOG_DIR/ae1.log"
: > "$LOG_DIR/ae2.log"
: > "$LOG_DIR/ae3.log"
: > "$LOG_DIR/orchestrator.log"
: > "$LOG_DIR/frontend.log"

echo "Starting MCP Tools Server (port 8085)..."
python -m agents.mcp_tools.server > "$LOG_DIR/mcp_tools.log" 2>&1 &

echo "Starting Normalizer Agent (port 8081)..."
python -m agents.normalizer > "$LOG_DIR/normalizer.log" 2>&1 &

echo "Starting Feedback Agent (port 8084)..."
python -m agents.feedback > "$LOG_DIR/feedback.log" 2>&1 &

echo "Starting Specialized Agent AE1 (port 8082)..."
python -m agents.specialized --port 8082 --agent-id ae1 > "$LOG_DIR/ae1.log" 2>&1 &

echo "Starting Specialized Agent AE2 (port 8083)..."
python -m agents.specialized --port 8083 --agent-id ae2 > "$LOG_DIR/ae2.log" 2>&1 &

echo "Starting Specialized Agent AE3 (port 8087, independent evaluator)..."
python -m agents.specialized --port 8087 --agent-id ae3 > "$LOG_DIR/ae3.log" 2>&1 &

sleep 2

echo "Starting Orchestrator Agent (port 8080)..."
python -m agents.orchestrator > "$LOG_DIR/orchestrator.log" 2>&1 &

echo ""
echo "All agents started. Starting frontend..."
(cd frontend && npm run dev > "$LOG_DIR/frontend.log" 2>&1) &

echo ""
echo "=========================================="
echo "  A2A Debate Network is running!"
echo "  Frontend:     http://localhost:8086"
echo "  Orchestrator: http://localhost:8080"
echo ""
echo "  Logs:         $LOG_DIR/<service>.log"
echo "  Live tail:    tail -f $LOG_DIR/orchestrator.log"
echo "=========================================="

wait
