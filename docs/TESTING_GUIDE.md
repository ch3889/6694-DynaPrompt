# Testing Guide for DynaPrompt

## Quick Test (2 prompts, ~5 minutes)

### Option 1: Using existing test script (Windows PowerShell)

```powershell
# Activate your Python environment
conda activate base  # or your environment name

# Run the comparison test
python scripts/baseline_vs_hybrid.py
```

This will:
- Generate 2 test images (baseline + hybrid)
- Save results to `outputs/baseline_vs_hybrid/`
- Print quantitative metrics (CLIP score, compositional accuracy)

---

## Full Test Suite (5 prompts, ~15 minutes)

Edit `scripts/baseline_vs_hybrid.py` line 270-280 to use all 5 prompts:

```python
test_prompts = [
    "a fluffy white cat wearing a tiny red hat sitting next to a blue flower vase",
    "a wooden table with a green apple, yellow banana, and orange carrot arranged in a row",
    "a golden bicycle next to a silver car in a sunny parking lot",
    "a blue ceramic vase sitting next to a vintage wooden clock",
    "a brown dog holding a red frisbee in a grassy park"
]
```

---

## Individual Component Tests

### Test ZK2295 Only (embedding feedback)
```python
python scripts/test_zk2295_only.py
```

### Test CH3889 Only (attention boosting)
```python
python scripts/test_attention_only.py
```

### Test Hybrid (both combined)
```python
python scripts/test_hybrid_dynaprompt.py
```

---

## Understanding Results

### Good Results:
- ✅ Compositional Accuracy: Hybrid > ZK2295 > Baseline
- ✅ Hybrid shows +10-15% improvement over baseline
- ⚠️ CLIP Score might be slightly lower (trade-off for better composition)

### Problem Indicators:
- ❌ Compositional Accuracy: Hybrid < Baseline → parameters too aggressive
- ❌ Very low CLIP scores (< 20) → embeddings corrupted
- ❌ All black/noise images → model not loading correctly

---

## Output Files

Results saved to `outputs/baseline_vs_hybrid/`:
- `*_comparison.png` - Side-by-side visual comparison
- `*_baseline.png` - Baseline image
- `*_hybrid.png` - Hybrid image  
- `*_metrics.json` - Quantitative metrics

---

## Troubleshooting

### DLL Error (torch)
```powershell
# Use conda environment instead
conda activate base
python scripts/baseline_vs_hybrid.py
```

### Out of Memory (GPU)
Edit config: `configs/dynaprompt_config.yaml`
```yaml
sampling:
  H: 256  # Reduce from 512
  W: 256
```

### Model Not Found
Download model:
```powershell
python scripts/download_models.sh
```

---

## Expected Runtime

| Test Type | Prompts | Time (GPU) | Time (CPU) |
|-----------|---------|------------|------------|
| Quick | 2 | ~5 min | ~30 min |
| Full | 5 | ~15 min | ~75 min |
| Single Hybrid | 1 | ~2 min | ~15 min |

---

## Verifying Parameters

Check if code matches presentation:
```powershell
# Should see: alpha=0.08, boost=1.3
python -c "import yaml; print(yaml.safe_load(open('configs/dynaprompt_config.yaml'))['prompt_update']['update_alpha'])"
python -c "import yaml; print(yaml.safe_load(open('configs/dynaprompt_config.yaml'))['attention']['boost_factor'])"
```

Expected output:
```
0.08
1.3
```
