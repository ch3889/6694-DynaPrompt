# Running DrawBench Phase 1 on GCP - Quick Guide

## Prerequisites

1. **GCP VM with GPU** (recommended: n1-standard-4 with T4 GPU)
2. **Repository cloned** on the VM
3. **Python environment** activated with requirements installed

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
# On GCP VM
cd ~/6694-DynaPrompt
bash run_drawbench_phase1.sh
```

This will:
1. Check GPU availability
2. Install CLIP model
3. Download DrawBench prompts (50 prompts)
4. Run evaluation (baseline + hybrid)
5. Analyze spatial relationships
6. Generate reports

**Time**: ~2 hours  
**Cost**: ~$8

### Option 2: Manual Step-by-Step

```bash
# Step 1: Download prompts
python scripts/download_drawbench.py

# Step 2: Run evaluation
python scripts/evaluate_drawbench.py \
  --categories Colors Positional Counting Descriptions Conflicting \
  --methods baseline hybrid \
  --steps 50 \
  --output outputs/drawbench_phase1

# Step 3: Analyze results
python scripts/analyze_spatial_failures.py \
  --results outputs/drawbench_phase1/results_detailed.json \
  --summary outputs/drawbench_phase1/results_summary.json
```

## GCP VM Setup (If Starting Fresh)

### 1. Create VM with GPU

```bash
# Create n1-standard-4 with T4 GPU
gcloud compute instances create dynaprompt-eval \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

### 2. SSH into VM

```bash
gcloud compute ssh dynaprompt-eval --zone=us-central1-a
```

### 3. Clone Repository

```bash
cd ~
git clone https://github.com/YourUsername/6694-DynaPrompt.git
cd 6694-DynaPrompt
git checkout zk2295
```

### 4. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements_gpu.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

### 5. Verify GPU

```bash
python check_gpu.py
```

Should show:
```
CUDA Available: True
Device: cuda
GPU: Tesla T4
```

### 6. Run Evaluation

```bash
bash run_drawbench_phase1.sh
```

## Monitoring Progress

### Check Running Status

```bash
# In another terminal
tail -f outputs/drawbench_phase1/evaluation.log
```

### Monitor GPU Usage

```bash
watch -n 1 nvidia-smi
```

Should show ~80-90% GPU utilization during evaluation.

### Estimated Timeline

- **Setup**: 5 minutes
- **Prompt download**: 1 minute
- **Evaluation**: ~2 hours
  - Baseline: ~1 hour (50 prompts × 50 steps × ~1.4s)
  - Hybrid: ~1 hour (50 prompts × 50 steps × ~1.5s)
- **Analysis**: 2 minutes

**Total**: ~2 hours 10 minutes

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce image resolution
```bash
# Edit configs/dynaprompt_config.yaml
image_size: 256  # Instead of 512
```

Then re-run:
```bash
python scripts/evaluate_drawbench.py --categories Colors Positional --steps 30
```

### Issue: Slow Progress

**Solution**: Use fewer steps for faster evaluation
```bash
python scripts/evaluate_drawbench.py \
  --categories Colors Positional \
  --steps 30 \
  --output outputs/drawbench_quick
```

### Issue: CLIP Not Found

**Solution**: Install CLIP
```bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### Issue: Evaluation Interrupted

**Solution**: Resume from checkpoint
```bash
# The script saves results incrementally
# Check outputs/drawbench_phase1/results_detailed.json
# Re-run will skip completed prompts (if you modify script to check existing images)
```

## Output Structure

After completion:

```
outputs/drawbench_phase1/
├── baseline/
│   ├── Colors/
│   │   ├── A_blue_colored_dog.png
│   │   ├── A_red_colored_car.png
│   │   └── ... (10 images)
│   ├── Positional/
│   │   ├── A_car_to_the_left_of_a_house.png
│   │   └── ... (10 images)
│   ├── Counting/          (10 images)
│   ├── Descriptions/      (10 images)
│   └── Conflicting/       (10 images)
├── hybrid/
│   ├── Colors/            (10 images)
│   ├── Positional/        (10 images)
│   ├── Counting/          (10 images)
│   ├── Descriptions/      (10 images)
│   └── Conflicting/       (10 images)
├── results_detailed.json          # Per-prompt metrics
├── results_summary.json           # Aggregated statistics
└── spatial_analysis_report.txt   # Spatial relationship analysis
```

## Retrieving Results

### Download to Local Machine

```bash
# From local machine
gcloud compute scp --recurse \
  dynaprompt-eval:~/6694-DynaPrompt/outputs/drawbench_phase1 \
  ./outputs/ \
  --zone=us-central1-a
```

### View Results

```bash
# Summary statistics
cat outputs/drawbench_phase1/results_summary.json | python -m json.tool

# Spatial analysis
cat outputs/drawbench_phase1/spatial_analysis_report.txt
```

## Cost Estimation

**VM Configuration**: n1-standard-4 + T4 GPU
- **Compute**: $0.35/hour
- **GPU**: $0.35/hour
- **Total**: $0.70/hour

**Phase 1 Evaluation**:
- **Time**: 2 hours
- **Cost**: ~$1.40

**With setup time** (3 hours total): ~$2.10

**Note**: Previous estimate of $8 was conservative. Actual cost is much lower!

## Cleanup

After evaluation completes:

```bash
# Stop VM (to avoid charges)
gcloud compute instances stop dynaprompt-eval --zone=us-central1-a

# Or delete VM (if done)
gcloud compute instances delete dynaprompt-eval --zone=us-central1-a
```

## Next Steps After Phase 1

1. **Review results_summary.json** - Check overall performance
2. **Analyze spatial_analysis_report.txt** - Validate hypothesis
3. **Visual inspection** - Look at Positional category images
4. **Compare with 2-prompt results** - Consistency check
5. **Decide on Phase 2** - If results promising, run full 150 prompts

## Expected Results

Based on 2-prompt testing:

| Metric | Expected Value |
|--------|---------------|
| Overall Comp Δ | +5-8% |
| Overall CLIP Δ | +0-2% |
| Colors Comp Δ | +12-18% |
| Positional Comp Δ | +8-12% (but visual quality issues) |
| Counting Comp Δ | +5-10% |

**Key Validation**: Positional category should show improved metrics BUT visual inspection reveals incorrect spatial relationships.
