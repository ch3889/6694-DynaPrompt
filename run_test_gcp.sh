#!/bin/bash
# Run this script on your GCP VM to test DynaPrompt

echo "=========================================="
echo "DynaPrompt Testing on GCP"
echo "=========================================="

# Navigate to project directory
cd ~/6694-DynaPrompt || { echo "Error: Project directory not found"; exit 1; }

# Pull latest changes
echo "Pulling latest code from zk2295 branch..."
git pull origin zk2295

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dynaprompt

# Run the test
echo ""
echo "=========================================="
echo "Running Baseline vs Hybrid Test"
echo "This will take ~5-10 minutes..."
echo "=========================================="
echo ""

python scripts/baseline_vs_hybrid.py

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results saved to: outputs/baseline_vs_hybrid/"
echo ""
echo "To download results to your local machine:"
echo "gcloud compute scp --recurse dynaprompt-vm:~/6694-DynaPrompt/outputs/baseline_vs_hybrid ./outputs/ --zone=us-central1-a"
