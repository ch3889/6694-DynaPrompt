#!/bin/bash
# Run adaptive parameter experiments on GCP
# This script will take approximately 3-4 hours to complete

echo "=========================================="
echo "Adaptive Parameter Selection Experiments"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dynaprompt

# Navigate to project directory
cd ~/myproject/6694-DynaPrompt

# Run experiments (start with smaller test to verify it works)
echo "Running with 12 test prompts..."
echo "Method 1: Will complete in ~30 minutes"
echo "Method 4: Skipped by default (add --no-skip-method4 to enable, adds ~2 hours)"

python scripts/run_adaptive_experiments.py \
    --test-size 12 \
    --output outputs/adaptive_results.json \
    --device cuda \
    --skip-method4

echo "=========================================="
echo "Experiments complete!"
echo "Results saved to: outputs/adaptive_results.json"
echo "=========================================="

# Download results locally:
# scp zk2295@136.107.82.176:myproject/6694-DynaPrompt/outputs/adaptive_results.json ./outputs/
