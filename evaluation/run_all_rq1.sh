#!/usr/bin/env bash
# Run all 3 RQ1 evaluations: Agent Factory → AutoGen → LangGraph
# 60 scenarios each, same ground_truth.json, full tool parity.
#
# Monitor progress:
#   tail -f evaluation/logs/agent_factory.log
#   tail -f evaluation/logs/autogen_baseline.log
#   tail -f evaluation/logs/langgraph_baseline.log

set -e
cd "$(dirname "$0")/.."

mkdir -p evaluation/logs

echo "============================================================"
echo "  RQ1 FULL EVALUATION SUITE — $(date)"
echo "  60 scenarios x 3 systems (Agent Factory, AutoGen, LangGraph)"
echo "  Logs: evaluation/logs/{agent_factory,autogen_baseline,langgraph_baseline}.log"
echo "============================================================"

echo ""
echo "[$(date +%H:%M:%S)] === 1/3 Agent Factory (hybrid) ==="
python -m evaluation.run_evaluation --output evaluation/results/rq1 2>&1
AF_EXIT=$?
echo "[$(date +%H:%M:%S)] Agent Factory finished (exit=$AF_EXIT)"

echo ""
echo "[$(date +%H:%M:%S)] === 2/3 AutoGen baseline ==="
python -m evaluation.autogen_baseline 2>&1
AG_EXIT=$?
echo "[$(date +%H:%M:%S)] AutoGen finished (exit=$AG_EXIT)"

echo ""
echo "[$(date +%H:%M:%S)] === 3/3 LangGraph baseline ==="
python -m evaluation.langgraph_baseline 2>&1
LG_EXIT=$?
echo "[$(date +%H:%M:%S)] LangGraph finished (exit=$LG_EXIT)"

echo ""
echo "============================================================"
echo "  ALL RQ1 EVALUATIONS COMPLETE — $(date)"
echo "  Agent Factory: exit=$AF_EXIT"
echo "  AutoGen:       exit=$AG_EXIT"
echo "  LangGraph:     exit=$LG_EXIT"
echo "============================================================"
