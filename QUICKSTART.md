# Quick Reference: Testing DynaPrompt

## Fastest Way to Test

```powershell
# 1. Activate environment
conda activate base

# 2. Run test (2 prompts, ~5 min on GPU)
python scripts/baseline_vs_hybrid.py

# 3. Check results
ls outputs/baseline_vs_hybrid/
```

## What You'll See

Terminal output:
```
================================================================================
BASELINE VS HYBRID EVALUATION
================================================================================

TEST 1/2: "a fluffy white cat wearing a tiny red hat..."

Baseline Complete: CLIP 34.60, Comp 0.631
Hybrid Complete:   CLIP 34.48, Comp 0.649

Improvement: +2.9% compositional accuracy
================================================================================
```

Output files in `outputs/baseline_vs_hybrid/`:
- `*_comparison.png` - Side-by-side visual
- `*_metrics.json` - Quantitative data

## Expected Results

### Good (Working Correctly):
✅ Hybrid compositional accuracy > Baseline  
✅ ~10-15% improvement  
✅ CLIP score slightly lower (expected trade-off)

### Problem:
❌ Hybrid < Baseline → Check parameters in `configs/dynaprompt_config.yaml`  
❌ Very low CLIP (<20) → Model issue  
❌ Black images → GPU/model loading issue

## Key Parameters (Should Match Presentation)

Check: `configs/dynaprompt_config.yaml`
```yaml
update_alpha: 0.08      # Should be 0.08
boost_factor: 1.3       # Should be 1.3
feedback_frequency: 4   # Every 4 steps
feedback_start_step: 5  # Start at 5
feedback_end_step: 30   # End at 30
```

## Troubleshooting

**DLL Error:**
```powershell
conda activate base
python scripts/baseline_vs_hybrid.py
```

**Out of Memory:**
Edit `configs/dynaprompt_config.yaml`:
```yaml
sampling:
  H: 256  # Reduce from 512
  W: 256
```

**More Help:** See [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md)

---

## Repository Organization

After cleanup, structure is:
```
DynaPrompt/
├── README.md                    # Project overview
├── docs/
│   ├── TESTING_GUIDE.md        # ⭐ Detailed testing
│   ├── presentations/           # Slides
│   ├── reports/                # Technical reports
│   ├── analysis/               # Analysis docs
│   └── setup/                  # Setup guides
├── dynaprompt/                 # Core code
├── configs/                    # Configuration
├── scripts/                    # Test scripts
└── outputs/                    # Results
```

All loose .md files moved to organized folders!
