# GCP Experiment Status - Method 1 Real Results

## Current Status
- **Experiment PID**: 1445761
- **Started**: ~10-15 minutes ago
- **Expected Duration**: ~40-60 minutes (20 images × 2-3 min each)
- **Output File**: `outputs/adaptive_results_real.json`

## Commands to Monitor Progress

### Check if experiment is still running:
```bash
ssh zk2295@136.107.82.176
ps aux | grep 1445761
```

### Watch real-time output:
```bash
ssh zk2295@136.107.82.176
cd myproject/6694-DynaPrompt
tail -f nohup_real_adaptive.out
```

### Check for completion (output file exists):
```bash
ssh zk2295@136.107.82.176
ls -lh myproject/6694-DynaPrompt/outputs/adaptive_results_real.json
```

### Download results when complete:
```powershell
scp zk2295@136.107.82.176:myproject/6694-DynaPrompt/outputs/adaptive_results_real.json ./outputs/
```

## Expected Output Format

The `adaptive_results_real.json` will contain:

```json
{
  "fixed_baseline": [
    {
      "prompt": "a blue cube on top of a red sphere",
      "final_clipscore": 59.1,
      "compositional_accuracy": 0.85,
      "generation_time": 5.2
    },
    ...10 prompts total
  ],
  "method1_adaptive": [
    {
      "prompt": "a blue cube on top of a red sphere",
      "baseline_clip": 58.2,
      "tier": "strong",
      "alpha": 0.03,
      "boost_factor": 1.1,
      "frequency": 6,
      "final_clipscore": 59.8,
      "compositional_accuracy": 0.87,
      "generation_time": 5.7
    },
    ...10 prompts total
  ]
}
```

## Test Prompts (10 DrawBench)

1. "a blue cube on top of a red sphere" (Est baseline: 58.2 - Strong)
2. "a golden bicycle next to a silver car" (Est baseline: 67.3 - Very Strong)
3. "a cat wearing a red hat" (Est baseline: 41.7 - Weak)
4. "three red apples on a wooden table" (Est baseline: 52.8 - Medium)
5. "a small dog sitting under a large tree" (Est baseline: 63.1 - Strong)
6. "colorful balloons floating in the sky" (Est baseline: 36.4 - Weak)
7. "a white vase with pink flowers" (Est baseline: 69.2 - Very Strong)
8. "a person riding a horse" (Est baseline: 48.9 - Medium)
9. "a green frog on a lily pad" (Est baseline: 44.3 - Weak)
10. "a castle on a mountain peak" (Est baseline: 59.7 - Strong)

## Once Results Arrive

Run the helper script to format results:
```powershell
python update_method1_results.py
```

This will:
1. Load `outputs/adaptive_results_real.json`
2. Generate formatted tables for presentation and report
3. Calculate summary statistics
4. Save to `outputs/formatted_method1_results.md`
5. Display copy-paste ready results

Then update documentation:
1. Copy tables from `outputs/formatted_method1_results.md`
2. Replace "PENDING" sections in `docs/presentations/PRESENTATION_FINAL.md` Slide 6
3. Replace "PENDING" sections in `docs/reports/REPORT_HYBRID_FINAL.md` Section 3.5.2
4. Commit with message: "Add real Method 1 experimental results from GCP"

## Hypothesis Validation

Expected findings:
- ✅ Method 1 prevents over-optimization on strong baselines (CLIP >60)
- ✅ Method 1 maintains strong gains on weak baselines (CLIP <45)
- ✅ Average improvement: +0.8% to +1.5%
- ✅ Win rate: 80-100% (vs 30% for fixed params)
- ✅ Validates CLIP ceiling effect hypothesis
