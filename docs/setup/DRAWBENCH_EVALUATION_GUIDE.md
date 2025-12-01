# DrawBench Evaluation - Quick Start Guide

## Overview

This directory contains scripts to evaluate DynaPrompt on the DrawBench benchmark (150 prompts across 11 categories).

## Files

- `download_drawbench.py` - Download/create DrawBench prompt set
- `evaluate_drawbench.py` - Run baseline and hybrid evaluation
- `analyze_spatial_failures.py` - Analyze spatial relationship performance

## Quick Start

### 1. Download DrawBench Prompts

```bash
python scripts/download_drawbench.py
```

This creates `data/drawbench_prompts.json` with 100 prompts across 10 categories.

### 2. Run Evaluation (Subset - Recommended Start)

Evaluate on 50 prompts (5 categories):

```bash
python scripts/evaluate_drawbench.py \
  --categories Colors Positional Counting Descriptions Conflicting \
  --methods baseline hybrid \
  --steps 50 \
  --output outputs/drawbench_subset
```

**Time**: ~2 hours on T4 GPU  
**Cost**: ~$8 on GCP

### 3. Run Full Evaluation

Evaluate all 100 prompts:

```bash
python scripts/evaluate_drawbench.py \
  --methods baseline hybrid \
  --steps 50 \
  --output outputs/drawbench_full
```

**Time**: ~4 hours on T4 GPU  
**Cost**: ~$15 on GCP

### 4. Analyze Spatial Performance

Focus on Positional category (validates our hypothesis):

```bash
python scripts/analyze_spatial_failures.py \
  --results outputs/drawbench_subset/results_detailed.json \
  --summary outputs/drawbench_subset/results_summary.json
```

## Command Line Options

### evaluate_drawbench.py

```
--prompts PATH          Path to DrawBench prompts JSON (default: data/drawbench_prompts.json)
--methods [METHOD ...]  Methods to evaluate: baseline, hybrid (default: both)
--output PATH           Output directory (default: outputs/drawbench)
--categories [CAT ...]  Subset of categories (default: all)
--steps INT            Number of inference steps (default: 50)
--guidance FLOAT       Classifier-free guidance scale (default: 7.5)
--seed INT             Random seed (default: 42)
--device STR           Device: cuda or cpu (default: cuda)
```

### Examples

**Evaluate only Colors and Positional:**
```bash
python scripts/evaluate_drawbench.py --categories Colors Positional
```

**Evaluate only Hybrid method:**
```bash
python scripts/evaluate_drawbench.py --methods hybrid
```

**Use fewer steps (faster, lower quality):**
```bash
python scripts/evaluate_drawbench.py --steps 30
```

## Output Structure

```
outputs/drawbench/
├── baseline/
│   ├── Colors/
│   │   ├── A_blue_colored_dog.png
│   │   └── ...
│   ├── Positional/
│   └── ...
├── hybrid/
│   ├── Colors/
│   ├── Positional/
│   └── ...
├── results_detailed.json   # Per-prompt results
├── results_summary.json    # Aggregated statistics
└── spatial_analysis_report.txt  # Spatial relationship analysis
```

## Expected Results (Based on 2-Prompt Testing)

### Overall Performance
- **Compositional Accuracy**: +5-8% (hybrid vs baseline)
- **CLIP Score**: +0-2% (hybrid vs baseline)

### Per-Category Predictions

| Category | Expected Comp Δ | Expected CLIP Δ | Notes |
|----------|----------------|----------------|-------|
| **Colors** | +12-18% | +1-3% | Strong - attribute binding |
| **Positional** | +8-12% | +0-2% | ⚠️ Metrics improve, visual degrades |
| **Counting** | +5-10% | -1-2% | Moderate improvement |
| **Descriptions** | +6-10% | +0-2% | General improvement |
| **Conflicting** | +0-3% | -2-4% | Difficult - unusual compositions |

### Critical Finding to Validate

**Hypothesis**: On Positional category, quantitative metrics improve BUT spatial relationships not preserved.

Example: "car to the left of house"
- Baseline: Car missing (low comp) BUT house in correct position
- Hybrid: Both present (high comp) BUT wrong positions (car on right)

→ Metrics improve, visual quality degrades

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or image resolution in config:
```yaml
# configs/dynaprompt_config.yaml
image_size: 512  # Try 256
```

### Slow Evaluation

Options to speed up:
1. Use fewer steps: `--steps 30`
2. Evaluate subset: `--categories Colors Positional`
3. Evaluate single method: `--methods hybrid`

### Missing CLIP Model

Install CLIP:
```bash
pip install git+https://github.com/openai/CLIP.git
```

## Next Steps

1. **Run subset evaluation** (50 prompts, $8, 2 hours)
2. **Analyze results** - Check if patterns match our 2-prompt findings
3. **Visual inspection** - Manually check Positional category images
4. **Human evaluation** (optional) - Rate spatial correctness 1-10
5. **Update reports** - Add DrawBench results to presentation and technical report

## Integration with Existing Tests

Compare DrawBench with current 2-prompt testing:

```bash
# Current testing
python scripts/baseline_vs_hybrid.py  # 2 prompts

# DrawBench testing
python scripts/evaluate_drawbench.py --categories Colors Positional  # 20 prompts
```

Results should be consistent:
- Current: +6.37% comp, +0.85% CLIP average
- DrawBench: Expected +5-8% comp, +0-2% CLIP average
