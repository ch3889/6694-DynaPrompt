#!/bin/bash
# Quick GCP Setup and Execution Script
# Run this on your GCP VM to set up and execute DrawBench Phase 1

echo "================================================"
echo "  DrawBench Phase 1 - GCP Quick Setup"
echo "================================================"
echo ""

# 1. Check if in correct directory
if [ ! -f "run_drawbench_phase1.sh" ]; then
    echo "Error: Not in project root directory"
    echo "Please cd to ~/6694-DynaPrompt first"
    exit 1
fi

# 2. Pull latest changes
echo "Step 1: Pulling latest code from GitHub..."
git pull origin zk2295

# 3. Check GPU
echo ""
echo "Step 2: Checking GPU..."
python check_gpu.py || {
    echo "Error: GPU not available"
    echo "Please ensure VM has GPU attached and drivers installed"
    exit 1
}

# 4. Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
pip install -q ftfy regex tqdm || echo "Some packages already installed"
pip install -q git+https://github.com/openai/CLIP.git || echo "CLIP already installed"

# 5. Verify models
echo ""
echo "Step 4: Verifying Stable Diffusion model..."
if [ ! -d "models/stable_diffusion_compvis" ]; then
    echo "Warning: Stable Diffusion model not found"
    echo "Downloading... (this may take 10-15 minutes)"
    python -c "from dynaprompt.sd_loader import load_stable_diffusion; load_stable_diffusion('cuda')"
fi

# 6. Run evaluation
echo ""
echo "Step 5: Starting DrawBench Phase 1 evaluation..."
echo "  - 50 prompts (5 categories × 10 prompts each)"
echo "  - Estimated time: 2 hours"
echo "  - Estimated cost: ~$1.40"
echo ""
read -p "Press Enter to start evaluation (or Ctrl+C to cancel)..."

bash run_drawbench_phase1.sh

echo ""
echo "================================================"
echo "  Setup and Evaluation Complete!"
echo "================================================"
echo ""
echo "Results: outputs/drawbench_phase1/"
echo ""
