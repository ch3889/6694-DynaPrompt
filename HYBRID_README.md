# DynaPrompt Hybrid: Combining zk2295 + ch3889

## Overview

**DynaPrompt Hybrid** combines two complementary techniques for improving compositional text-to-image generation:

1. **zk2295**: External CLIP-based embedding feedback (global + selective boosting)
2. **ch3889**: Internal U-Net attention amplification

## Why Hybrid?

Both approaches address different aspects of the generation process:

| Aspect | zk2295 (Embedding) | ch3889 (Attention) | Hybrid |
|--------|-------------------|-------------------|---------|
| **What it changes** | Input embeddings | Attention weights | Both |
| **Where** | External feedback loop | Inside U-Net | Both phases |
| **Strength** | Stable, model-agnostic | Fast, integrated | Double reinforcement |
| **Can add objects?** | Partially | No | Better chance |

**Result**: Weak concepts get boosted in BOTH the input specification AND the processing mechanism.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Hybrid DynaPrompt Pipeline                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Every 4 steps during denoising:                        │
│                                                          │
│  ┌─────────────────────────────────────┐               │
│  │ Phase 1: Embedding Feedback (zk2295)│               │
│  │                                      │               │
│  │ 1. Decode intermediate image        │               │
│  │ 2. CLIP per-token analysis           │               │
│  │    • Unigrams, bigrams, trigrams     │               │
│  │    • Detect weak concepts            │               │
│  │ 3. Update embeddings:                │               │
│  │    • Global gradient (α=0.08)        │               │
│  │    • Selective boost (β=1.5)         │               │
│  │                                      │               │
│  │ Output: Updated embedding e'         │               │
│  └─────────────────────────────────────┘               │
│                    ↓                                     │
│  ┌─────────────────────────────────────┐               │
│  │ Phase 2: Attention Boost (ch3889)   │               │
│  │                                      │               │
│  │ 1. Map weak concepts → token indices│               │
│  │ 2. Enable attention hooks            │               │
│  │ 3. During U-Net forward:             │               │
│  │    • Intercept attention computation │               │
│  │    • Amplify weak tokens (1.3-3x)    │               │
│  │    • Re-normalize attention          │               │
│  │                                      │               │
│  │ Output: Denoised latent x_{t-1}      │               │
│  └─────────────────────────────────────┘               │
│                                                          │
│  Result: Both input AND processing reinforced           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

The hybrid system uses files from both branches:

```bash
# Already on zk2295 branch with your code
git status  # Should show: On branch zk2295

# Files added for hybrid (already in dynaprompt/):
# - attention_modifier.py (from ch3889)
# - hybrid.py (new - combines both)

# Test script:
# - scripts/test_hybrid_dynaprompt.py
```

---

## Configuration

Edit `configs/dynaprompt_config.yaml`:

```yaml
# Phase 1: Embedding feedback (zk2295)
feedback:
  enabled: true
  feedback_frequency: 4
  feedback_start_step: 5
  feedback_end_step: 42

prompt_update:
  update_alpha: 0.08      # Global gradient strength
  boost_factor: 1.5       # Selective token boosting (in core.py)

# Phase 2: Attention boosting (ch3889)
attention:
  boost_factor: 1.3       # Attention amplification
  threshold: 0.05         # Weak token detection threshold
  start_step: 0           # Early intervention
  end_step: 20            # Structure formation phase
```

---

## Usage

### Basic Generation

```python
from dynaprompt.hybrid import HybridDynaPrompt

# Initialize hybrid system
hybrid = HybridDynaPrompt(device='cuda')

# Generate with both feedbacks
result = hybrid.generate(
    prompt="a silver car parked next to a golden bicycle",
    steps=50,
    cfg_scale=7.5,
    seed=42,
    embedding_feedback=True,   # Phase 1: zk2295
    attention_feedback=True    # Phase 2: ch3889
)

# Access results
image = result['image']
clip_score = result['final_clipscore']
comp_accuracy = result['compositional_accuracy']
weak_tokens = result['weak_tokens_history']

# Cleanup
hybrid.cleanup()
```

### Compare All Methods

```bash
# Run comprehensive comparison
python scripts/test_hybrid_dynaprompt.py
```

This will generate:
- Baseline (no feedback)
- Hybrid (embedding + attention)

Results saved to `outputs/hybrid_comparison/`

---

## Expected Results

Based on the individual approaches:

### Baseline
- CLIP Score: 20.15
- Compositional Accuracy: ~0.65
- Issues: Missing objects, weak attributes

### Hybrid (Expected)
- CLIP Score: 20.25+ (+0.5%+)
- Compositional Accuracy: 0.80+
- Improvements: Double reinforcement of weak concepts
- Overhead: ~18-20% slower (VAE decode + attention hooks)

---

## How It Works: Step-by-Step

### Step 1: Detection (Shared)

Both phases use the same weak token detection from zk2295:

```python
# CLIP analysis on intermediate image
concepts = ["silver", "car", "golden", "bicycle", "golden bicycle"]
scores = {
    "silver": 22.5,   # Strong ✓
    "car": 21.8,      # Strong ✓
    "golden": 14.2,   # Weak ✗
    "bicycle": 12.8   # Weak ✗
}

# Threshold: mean - 0.5*std = 17.8
weak_tokens = {"golden": 14.2, "bicycle": 12.8}
```

### Step 2: Phase 1 - Embedding Update (zk2295)

```python
# Global gradient: Pull entire embedding toward CLIP alignment
clip_img_features = CLIP_vision(intermediate_image)
clip_txt_features = CLIP_text(prompt)
gradient = clip_img_features - clip_txt_features
embedding += 0.08 * gradient  # Conservative alpha

# Selective boost: Amplify weak token positions
for concept in ["golden", "bicycle"]:
    token_positions = find_positions(concept)  # [4, 5]
    weakness = (20 - score) / 20
    for pos in token_positions:
        embedding[0, pos, :] *= (1 + 1.5 * weakness)

# Renormalize to preserve magnitude
embedding *= (original_norm / current_norm)
```

### Step 3: Phase 2 - Attention Amplification (ch3889)

```python
# Map weak concepts to token indices
weak_indices = [4, 5]  # Positions of "golden", "bicycle"

# During U-Net forward pass (hooked):
def modified_attention(Q, K, V):
    attn = softmax(Q @ K.T / scale)
    
    # Boost weak tokens
    for idx in [4, 5]:
        current_attn = attn[:, :, idx].mean()
        
        # Adaptive boost
        if current_attn < 0.001:
            boost = 1.3 * 3.0  # Very weak: 3.9x
        elif current_attn < 0.005:
            boost = 1.3 * 2.0  # Weak: 2.6x
        else:
            boost = 1.3        # Moderate: 1.3x
        
        attn[:, :, idx] *= boost
    
    # Re-normalize
    attn = attn / attn.sum(dim=-1, keepdim=True)
    
    return attn @ V
```

---

## Advantages Over Individual Approaches

### vs. zk2295 Alone
- ✅ **Stronger reinforcement**: Attention adds second layer of emphasis
- ✅ **Internal + External**: Changes both input and processing
- ✅ **Complementary**: Embedding update enables attention boost to work better

### vs. ch3889 Alone
- ✅ **Can create features**: Embedding updates provide signal for attention to amplify
- ✅ **More stable**: Embedding feedback guides attention to right tokens
- ✅ **Quantitative metrics**: zk2295's compositional accuracy

### Combined Benefits
- **Double signal**: Weak concepts boosted in input AND processing
- **Synergy**: Updated embeddings may contain features attention can amplify
- **Better than sum**: Embedding boost creates signal, attention amplifies it

---

## Limitations

⚠️ **Computational Cost**: ~18-20% slower than baseline
- zk2295: +13% (VAE decoding)
- ch3889: +5% (attention hooks)

⚠️ **Complexity**: Two systems to tune and maintain

⚠️ **Potential Interference**: Both modifying different parts - could conflict

⚠️ **Diminishing Returns**: May not be fully additive

---

## Tuning Guide

### Conservative (Stable)
```yaml
prompt_update:
  update_alpha: 0.05    # Gentle embedding updates
attention:
  boost_factor: 1.2     # Mild attention amplification
  start_step: 5         # After initial structure
```

### Balanced (Recommended)
```yaml
prompt_update:
  update_alpha: 0.08    # Moderate updates
attention:
  boost_factor: 1.3     # Standard boost
  start_step: 0         # Early intervention
```

### Aggressive (Maximum Effect)
```yaml
prompt_update:
  update_alpha: 0.12    # Strong updates (risk: corruption)
attention:
  boost_factor: 2.0     # High boost (risk: artifacts)
  start_step: 0         # Immediate intervention
```

**Warning**: Aggressive settings may cause:
- Embedding space corruption (if alpha > 0.15)
- Visual artifacts (if boost > 2.5)
- Random pixels (if both too high)

---

## Evaluation

Run systematic evaluation:

```bash
python scripts/test_hybrid_dynaprompt.py
```

**Test Prompts** (known challenging cases):
1. "a silver car parked next to a golden bicycle"
2. "a red cube and a blue sphere on a wooden table"
3. "a golden retriever playing with a red ball in a snowy park"
4. "a tiny red bicycle next to a giant blue umbrella"
5. "a purple elephant wearing a pink hat"

**Metrics Tracked**:
- CLIP Score (semantic alignment)
- Compositional Accuracy (concept completeness)
- Weak tokens detected (interpretability)
- Generation time (efficiency)

**Output**:
- Side-by-side comparison grid
- Individual images per method
- JSON metrics with improvements

---

## Troubleshooting

### "Gradient flow broken" Error
This is expected - ch3889 doesn't use gradients in the hybrid version. We use direct attention modification, not gradient-based refinement.

### Images Corrupted / Random Pixels
- Reduce `update_alpha` to 0.05-0.08
- Reduce `boost_factor` to 1.2-1.5
- Check both parameters aren't too aggressive simultaneously

### No Improvement Over zk2295
- Increase `boost_factor` to 1.5-2.0
- Ensure `attention.start_step = 0` (early intervention)
- Check weak tokens are being detected (print `weak_tokens_history`)

### Attention Hooks Not Working
- Verify `attention_modifier.py` exists in `dynaprompt/`
- Check U-Net has `CrossAttention` layers
- Enable debug: `attention_modifier.enabled = True`

---

## Files Structure

```
dynaprompt/
├── core.py                  # zk2295 feedback computation
├── wrapper.py               # zk2295 SD integration
├── sd_loader.py             # Model loading
├── attention_modifier.py    # ch3889 attention hooks (NEW)
└── hybrid.py                # Hybrid pipeline (NEW)

scripts/
└── test_hybrid_dynaprompt.py  # Comparison testing (NEW)

configs/
└── dynaprompt_config.yaml   # Updated with attention config

outputs/
└── hybrid_comparison/       # Generated comparisons
    ├── *_grid.png          # Side-by-side comparison
    ├── *_baseline.png      # Baseline result
    ├── *_zk2295.png        # zk2295 result
    ├── *_hybrid.png        # Hybrid result
    └── *_metrics.json      # Quantitative metrics
```

---

## Next Steps

### Immediate Testing
1. Run `test_hybrid_dynaprompt.py` to generate comparisons
2. Evaluate if hybrid shows improvements over zk2295 alone
3. Tune parameters if needed

### Advanced Experiments
1. **Ablation study**: Test each component separately
2. **Parameter sweep**: Find optimal alpha + boost combinations
3. **Difficult prompts**: Test on systematically hard cases
4. **Timing analysis**: Measure overhead breakdown

### Potential Improvements
1. **Gradient-based refinement**: Implement ch3889's V4 (requires fixing gradient flow)
2. **Multi-CLIP ensemble**: Use multiple CLIP models for detection
3. **Learned rewards**: Train small model to predict weak tokens
4. **Spatial attention**: Add location-aware boosting

---

## Citation

If you use this hybrid approach, please credit both techniques:

```bibtex
@misc{dynaprompt_zk2295,
  author = {zk2295},
  title = {DynaPrompt: External CLIP-based Embedding Feedback},
  year = {2025},
  note = {Embedding update with dual strategy (global + selective)}
}

@misc{dynaprompt_ch3889,
  author = {ch3889},
  title = {DynaPrompt: Internal Attention Amplification},
  year = {2025},
  note = {U-Net attention boosting inspired by Attend-and-Excite}
}

@misc{dynaprompt_hybrid,
  title = {DynaPrompt Hybrid: Combining Embedding and Attention Feedback},
  year = {2025},
  note = {Integrates zk2295 and ch3889 approaches}
}
```

---

## References

- **zk2295 Documentation**: `ARCHITECTURE.md`, `COMPARISON.md`
- **ch3889 Findings**: See `origin/ch3889:FINDINGS.md`
- **Technique Comparison**: `TECHNIQUE_COMPARISON.md`
- **Attend-and-Excite Paper**: https://yuval-alaluf.github.io/Attend-and-Excite/

---

**Version**: 1.0  
**Date**: November 28, 2025  
**Authors**: Integration by zk2295, combining zk2295 + ch3889 techniques
