#!/bin/bash

# Restart Method 1 experiment with robust error handling
echo "Restarting Method 1 experiment with checkpointing..."

cd ~/myproject/6694-DynaPrompt

# Pull latest code
echo "Pulling latest changes..."
git pull origin zk2295

# Activate environment
source ~/.bashrc
conda activate base

# Kill any existing processes
pkill -f "run_method1_robust.py" 2>/dev/null || true

# Run with nohup
nohup python scripts/run_method1_robust.py > nohup_method1_robust.out 2>&1 &

# Get PID
PID=$!
echo "Process started with PID: $PID"

# Save PID for monitoring
echo $PID > method1_experiment.pid

echo ""
echo "Monitoring commands:"
echo "  tail -f ~/myproject/6694-DynaPrompt/nohup_method1_robust.out"
echo "  ps aux | grep $PID"
echo ""
echo "Check progress:"
echo "  cat ~/myproject/6694-DynaPrompt/outputs/method1_checkpoint.json"
