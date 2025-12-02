#!/bin/bash
# Run REAL adaptive parameter experiments on GCP

echo "=========================================="
echo "Real Adaptive Parameter Experiments"
echo "=========================================="

cd ~/myproject/6694-DynaPrompt
git pull origin zk2295

echo ""
echo "Running real experiments with HybridDynaPrompt..."
echo "This will take approximately 1 hour (10 prompts × 2 methods × 50 steps × ~3min per generation)"
echo ""

nohup python -u scripts/run_real_adaptive_experiments.py > nohup_real_adaptive.out 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo ""
echo "To monitor: tail -f nohup_real_adaptive.out"
echo "After completion: scp zk2295@136.107.82.176:myproject/6694-DynaPrompt/outputs/adaptive_results_real.json ./outputs/"
echo "=========================================="
