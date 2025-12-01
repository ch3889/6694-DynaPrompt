#!/bin/bash
# Run DrawBench Phase 1 Evaluation on GCP
# Subset: 50 prompts across 5 categories
# Expected time: ~2 hours on T4
# Expected cost: ~$8

set -e  # Exit on error

echo "========================================"
echo "DrawBench Phase 1 Evaluation - GCP"
echo "========================================"
echo ""

# Check GPU
echo "Checking GPU availability..."
python check_gpu.py

# Install CLIP if not already installed
echo ""
echo "Installing CLIP model..."
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git || echo "CLIP already installed"

# Create data directory
mkdir -p data
mkdir -p outputs/drawbench_phase1

# Step 1: Download DrawBench prompts
echo ""
echo "Step 1: Downloading DrawBench prompts..."
python scripts/download_drawbench.py

# Step 2: Run evaluation on subset (5 categories, 50 prompts)
echo ""
echo "Step 2: Running evaluation (LOW MEMORY MODE)..."
echo "  Categories: Colors, Positional, Counting, Descriptions, Conflicting"
echo "  Methods: Baseline, Hybrid"
echo "  Prompts: 50 total"
echo "  Batch size: 10 (reloads model every 10 prompts)"
echo "  Estimated time: 2-3 hours"
echo ""

python scripts/evaluate_drawbench_lowmem.py \
  --categories Colors Positional Counting Descriptions Conflicting \
  --methods baseline hybrid \
  --steps 50 \
  --guidance 7.5 \
  --seed 42 \
  --output outputs/drawbench_phase1

# Step 3: Analyze spatial relationships
echo ""
echo "Step 3: Analyzing spatial relationships..."
python scripts/analyze_spatial_failures.py \
  --results outputs/drawbench_phase1/results_detailed.json \
  --summary outputs/drawbench_phase1/results_summary.json \
  --output outputs/drawbench_phase1

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved to: outputs/drawbench_phase1/"
echo ""
echo "Next steps:"
echo "1. Review results_summary.json"
echo "2. Check spatial_analysis_report.txt"
echo "3. Visually inspect Positional category images"
echo ""
