#!/bin/bash
# Run adaptive parameter experiments on GCP
# This script will take approximately 3-4 hours to complete

echo "=========================================="
echo "Adaptive Parameter Selection Experiments"
echo "=========================================="

# Navigate to project directory
cd ~/myproject/6694-DynaPrompt

# Activate conda environment (handle different conda locations)
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Try to activate environment, if it fails just use base python
conda activate dynaprompt 2>/dev/null || echo "Using base python environment"

# Run experiments with nohup to prevent SSH disconnection issues
echo "Running with 12 test prompts..."
echo "Method 1: Will complete in ~30 minutes"
echo "Method 4: Skipped by default (add --no-skip-method4 to enable, adds ~2 hours)"
echo ""
echo "Running in background with nohup..."
echo "Output will be saved to: nohup_adaptive.out"

nohup python -u scripts/run_adaptive_experiments.py \
    --test-size 12 \
    --output outputs/adaptive_results.json \
    --device cuda \
    --skip-method4 > nohup_adaptive.out 2>&1 &

PID=$!
echo "Process ID: $PID"
echo ""
echo "=========================================="
echo "Experiment started in background!"
echo "=========================================="
echo ""
echo "To monitor progress:"
echo "  tail -f nohup_adaptive.out"
echo ""
echo "To check if still running:"
echo "  ps aux | grep $PID"
echo ""
echo "After completion, download results:"
echo "  scp zk2295@136.107.82.176:myproject/6694-DynaPrompt/outputs/adaptive_results.json ./outputs/"
echo "=========================================="
