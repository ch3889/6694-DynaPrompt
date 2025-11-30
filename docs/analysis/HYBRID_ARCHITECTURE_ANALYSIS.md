# Hybrid Architecture Analysis
**Date**: November 30, 2025  
**Branch**: zk2295  
**Status**: Ready for testing with latest fixes

## Architecture Overview

### Current Implementation: zk2295 (Embedding Feedback) + ch3889 (Attention Boosting)

**zk2295 Approach** (from `dynaprompt/core.py`):
- CLIP gradient-based embedding updates
- Computes per-token CLIP alignment scores
- Updates embeddings using: `embedding += alpha * clip_gradient`
- **NOT naive multiplication** - uses proper gradient descent

**ch3889 Approach** (from `dynaprompt/attention_modifier.py`):
- U-Net cross-attention map amplification
- Boosts attention to underrepresented tokens
- Adaptive boosting based on current attention levels

### Key Difference from ch3889's Failed V1

| Ch3889 V1 (Failed) | Your zk2295 (Valid) |
|-------------------|---------------------|
| `embedding *= 1.5` | `embedding += alpha * gradient` |
| Arbitrary scaling | Gradient-based optimization |
| Corrupts embedding space | Preserves semantic structure |
| -31.9% degradation | Theoretically sound |

## Commit History Analysis

### Phase 1: Over-Correction (Commits d01aa50 - a7b3da4)
**Problem**: Too aggressive (alpha 0.50, boost 3.0)
**Result**: Progressively worse results
**Lesson**: Embedding updates must be gentle

### Phase 2: Architecture Fix (Commit 09842e9)
**Change**: Restored proper CLIP gradient feedback
**Config**: alpha 0.10, boost 1.8x
**Result**: Still negative, but architecture correct

### Phase 3: Enhancement Implementation (Commits 57a64d4 - 11bbe53)
**Added**:
1. Adaptive per-token boosting (1.0x - 4.0x based on CLIP)
2. Stage-based decomposition (subjects → attributes → objects)
3. Dynamic negative prompts

**Result**: Still negative due to bugs

### Phase 4: Critical Bug Fixes (Commits 44eb686 - 9980907)

**Fix 1** (44eb686): Move embedding feedback outside no_grad
- **Problem**: Updates had no effect inside `with torch.no_grad()`
- **Fix**: Moved feedback outside the block
- **Result**: Updates now work, but still negative

**Fix 2** (9980907): Stage emphasis calculation
- **Problem**: Averaging all tokens diluted boost (2.0x → 0.78x)
- **Fix**: Use max of boosted tokens only
- **Result**: Now reaches intended 2.0x emphasis

**Fix 3** (5bb589e): Negative prompt generation
- **Problem**: Weak tokens are phrases ("red hat") not matching mapping
- **Fix**: Extract keywords from phrases
- **Result**: Negative prompts should now generate

### Phase 5: Alpha Tuning (Commit 5bb589e - Current)
**Change**: Reduced alpha from 0.15 → 0.12
**Reasoning**: Prevent embedding drift while maintaining effectiveness
**Max Alpha**: 0.12 * 2.0 = 0.24 (instead of 0.30)

## Current Configuration

```yaml
Embedding Feedback (zk2295):
  alpha: 0.12 (base)
  alpha_max: 0.24 (with 2.0x stage emphasis)
  frequency: every 4 steps
  range: steps 5-35
  method: CLIP gradient descent

Attention Boosting (ch3889):  
  base_boost: 1.8x
  adaptive_boost: 1.0x - 4.0x (based on CLIP score)
  range: steps 0-35
  method: Cross-attention amplification

Stage Decomposition:
  Stage 1 (0-33%): 2.0x emphasis on subjects
  Stage 2 (34-66%): 2.0x emphasis on attributes  
  Stage 3 (67-100%): 2.0x emphasis on objects

Negative Prompts:
  threshold: CLIP < 15
  blending: 50/50 with unconditional
  extraction: Keywords from phrase-based weak tokens
```

## Why This Should Work (Theory)

### 1. Non-Conflicting Mechanisms

**Embedding Updates** (external, pre-U-Net):
- Modifies text embeddings fed to U-Net
- Changes WHAT the model receives as input
- Gradient-based, preserves semantic structure

**Attention Boosting** (internal, during U-Net):
- Modifies cross-attention maps inside U-Net
- Changes HOW the model processes those embeddings
- Amplifies weak signals in existing embeddings

These operate at different stages → should complement, not conflict

### 2. Adaptive Feedback Loop

```
Step N:
  1. Decode current latent → intermediate image
  2. Compute CLIP scores per token
  3. Identify weak tokens (CLIP < threshold)
  4. Update embeddings: c += alpha * gradient (zk2295)
  5. Compute adaptive boosts: 1.0x-4.0x (ch3889)
  6. Apply stage emphasis: 0.5x-2.0x multiplier
  7. Generate negative prompts for missing concepts
  8. Continue denoising with updated c and boosted attention
```

This creates **double reinforcement** of weak concepts.

### 3. What Was Fixed

| Bug | Impact | Fix | Commit |
|-----|--------|-----|--------|
| Feedback inside no_grad | Updates didn't apply | Moved outside | 44eb686 |
| Stage emphasis averaging | 0.78x instead of 2.0x | Use max of boosted | 9980907 |
| Negative prompts not generating | No concept suppression | Extract keywords | 5bb589e |
| Alpha too aggressive | Embedding drift | 0.15 → 0.12 | 5bb589e |

## Expected Results After Latest Fixes

### Test Output Should Show:

```
[Step 8/31] Stage 1 (Subjects), Emphasis: 2.00x
  Alpha: 0.120 * 2.00 = 0.240
  CLIP Score: 18.02
  Weak tokens: ['hat', 'red', 'blue', 'vase']  ← Individual words, not phrases
  Negative prompt: 'no hat, bare head, wrong color, not red, wrong color, not blue, no vase'
```

### Quantitative Results Should Show:

**Test 1: Cat with hat and vase**
- Baseline CLIP: ~18.65
- **Hybrid CLIP: 18.70-19.00 (+0.5% to +2%)**  ← Positive improvement
- Compositional: Equal or better

**Test 2: Table with fruits**
- Baseline CLIP: ~22.35
- **Hybrid CLIP: 22.40-22.70 (+0.2% to +1.5%)**  ← Positive improvement  
- Compositional: Equal or better

## If Results Are Still Negative

### Diagnostic Test: Attention-Only vs Hybrid

Run `scripts/test_attention_only.py`:

```bash
python scripts/test_attention_only.py
```

This tests three configurations:
1. **Baseline**: No feedback
2. **Attention-only**: ch3889 approach alone
3. **Hybrid**: Embedding + attention

**If attention-only is positive but hybrid is negative**:
→ Embedding feedback is interfering (reduce alpha to 0.05-0.08)

**If both are negative**:
→ Ch3889's approach may not work with this SD implementation

**If hybrid is best**:
→ ✅ Architecture validated, ready for production

## Comparison to Ch3889's Approaches

| Approach | Ch3889 Branch | Your Hybrid |
|----------|---------------|-------------|
| Embedding updates | ❌ V1 failed (naive multiplication) | ✓ Gradient-based |
| Attention boosting | ✓ V3/V6 working | ✓ Same as ch3889 |
| Stage decomposition | ❌ Not implemented | ✓ Added |
| Negative prompts | ❌ Not implemented | ✓ Added (now fixed) |
| Adaptive boosting | Partial (fixed 2.5x) | ✓ Full (1.0x-4.0x) |

Your approach is **more sophisticated** than ch3889's, combining:
- Their working attention approach (V6)
- Your novel gradient-based embedding updates
- Additional enhancements (stages, negatives, adaptive)

## Action Items

### Immediate Test (GCP)
```bash
git pull origin zk2295
python scripts/baseline_vs_hybrid.py
```

### Expected Outcome
- ✅ Negative prompts now appear in logs
- ✅ Stage emphasis reaches 2.0x
- ✅ Weak tokens are individual words
- ✅ CLIP scores show positive improvement

### If Still Negative
1. Run `python scripts/test_attention_only.py`
2. Compare attention-only vs hybrid
3. If attention-only better → reduce alpha to 0.05
4. If both negative → disable embedding, use attention only

## Conclusion

**Current Status**: All known bugs fixed, configuration tuned

**Theory**: Sound - gradient-based updates + attention boosting should complement

**Practical**: Ready for final test with:
- Stage emphasis working (2.0x)
- Negative prompts generating
- Alpha moderate (0.24 max)
- Adaptive boosting (1.0x-4.0x)

**Next Step**: Test on GCP and analyze results. If positive → success. If negative → run diagnostic test to isolate which component helps/hurts.
