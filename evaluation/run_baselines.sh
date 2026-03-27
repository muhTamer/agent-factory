#!/usr/bin/env bash
# Run AutoGen and LangGraph baselines sequentially with logging.
# Tail the logs:
#   tail -f evaluation/logs/autogen_baseline.log
#   tail -f evaluation/logs/langgraph_baseline.log

set -e
cd "$(dirname "$0")/.."

mkdir -p evaluation/logs

echo "============================================================"
echo "  BASELINE EVALUATION SUITE — $(date)"
echo "  Log files: evaluation/logs/autogen_baseline.log"
echo "             evaluation/logs/langgraph_baseline.log"
echo "============================================================"

echo ""
echo "[$(date +%H:%M:%S)] Starting AutoGen baseline (60 scenarios)..."
python -m evaluation.autogen_baseline 2>&1 | tee evaluation/logs/autogen_baseline_stdout.log
AUTOGEN_EXIT=$?
echo "[$(date +%H:%M:%S)] AutoGen baseline finished (exit=$AUTOGEN_EXIT)"

echo ""
echo "[$(date +%H:%M:%S)] Starting LangGraph baseline (60 scenarios)..."
python -m evaluation.langgraph_baseline 2>&1 | tee evaluation/logs/langgraph_baseline_stdout.log
LANGGRAPH_EXIT=$?
echo "[$(date +%H:%M:%S)] LangGraph baseline finished (exit=$LANGGRAPH_EXIT)"

echo ""
echo "============================================================"
echo "  ALL BASELINES COMPLETE — $(date)"
echo "  AutoGen exit: $AUTOGEN_EXIT"
echo "  LangGraph exit: $LANGGRAPH_EXIT"
echo "============================================================"
