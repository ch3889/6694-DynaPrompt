# Run DrawBench Phase 1 Evaluation on GCP (PowerShell)
# Subset: 50 prompts across 5 categories
# Expected time: ~2 hours on T4
# Expected cost: ~$8

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "DrawBench Phase 1 Evaluation - GCP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check GPU
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
python check_gpu.py

# Install CLIP if not already installed
Write-Host ""
Write-Host "Installing CLIP model..." -ForegroundColor Yellow
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# Create data directory
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "outputs/drawbench_phase1" | Out-Null

# Step 1: Download DrawBench prompts
Write-Host ""
Write-Host "Step 1: Downloading DrawBench prompts..." -ForegroundColor Green
python scripts/download_drawbench.py

# Step 2: Run evaluation on subset (5 categories, 50 prompts)
Write-Host ""
Write-Host "Step 2: Running evaluation..." -ForegroundColor Green
Write-Host "  Categories: Colors, Positional, Counting, Descriptions, Conflicting"
Write-Host "  Methods: Baseline, Hybrid"
Write-Host "  Prompts: 50 total"
Write-Host "  Estimated time: 2 hours"
Write-Host ""

python scripts/evaluate_drawbench.py `
  --categories Colors Positional Counting Descriptions Conflicting `
  --methods baseline hybrid `
  --steps 50 `
  --guidance 7.5 `
  --seed 42 `
  --output outputs/drawbench_phase1

# Step 3: Analyze spatial relationships
Write-Host ""
Write-Host "Step 3: Analyzing spatial relationships..." -ForegroundColor Green
python scripts/analyze_spatial_failures.py `
  --results outputs/drawbench_phase1/results_detailed.json `
  --summary outputs/drawbench_phase1/results_summary.json `
  --output outputs/drawbench_phase1

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Evaluation Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to: outputs/drawbench_phase1/"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Review results_summary.json"
Write-Host "2. Check spatial_analysis_report.txt"
Write-Host "3. Visually inspect Positional category images"
Write-Host ""
