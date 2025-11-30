# DynaPrompt Improvements to Fix Noise Issue

## Critical Bug Fixed
**Problem**: The feedback loop was adding a **scalar value** to the embedding tensor, corrupting the embeddings and causing noise.

**Original buggy code**:
```python
feedback_signal = 1.0 - clipscore  # scalar
updated_embedding = prompt_embedding + alpha * feedback_signal  # Broadcasting scalar to tensor!
```

**Fixed code**: Now uses proper gradient-based feedback with feature alignment:
- Extracts CLIP text and image features
- Computes alignment direction as a tensor
- Normalizes gradients to prevent explosion
- Uses conservative alpha (0.05 instead of 0.3)

## Key Improvements

### 1. **Proper Feature Alignment** ✓
- Use CLIP text/image features to compute alignment direction
- Create gradient tensor with correct dimensions
- Scale feedback by misalignment score

### 2. **Conservative Update Strategy** ✓
- Reduced `update_alpha` from 0.3 → 0.05
- Normalized gradients before applying
- Prevents embedding corruption

### 3. **Better Feedback Schedule** ✓
- Start earlier: step 5 instead of 10
- More frequent: every 5 steps instead of 10
- Stop at step 40 to allow final refinement

### 4. **Fixed DDIM Sampling**
- Corrected parameter names in `p_sample_ddim` call
- Proper handling of predicted x0

## Additional Recommendations

### Short-term Improvements:

1. **Add Embedding Regularization**
   - Clip embedding norms to prevent drift
   - Add loss term to stay close to original embedding

2. **Adaptive Feedback Strength**
   - Strong feedback early (high noise)
   - Weak feedback late (refinement)
   ```python
   alpha = 0.1 * (1 - step / total_steps)  # Decay over time
   ```

3. **Selective Token Updates**
   - Only update tokens relevant to misalignment
   - Use attention scores to identify important tokens

4. **Better Intermediate Decoding**
   - Only decode every 10 steps (expensive operation)
   - Use latent-space CLIP if available

### Medium-term Improvements:

5. **Learnable Feedback Module**
   - Train small MLP to predict optimal embedding updates
   - Learn from successful generations

6. **Multi-scale Feedback**
   - Apply feedback at different latent resolutions
   - Coarse feedback early, fine feedback late

7. **Contrastive Feedback**
   - Generate negative examples
   - Push away from what prompt is NOT describing

8. **Attention Map Guidance**
   - Use cross-attention maps from U-Net
   - Identify which tokens need more emphasis

### Long-term Improvements:

9. **RL-based Optimization**
   - Treat embedding update as policy
   - Reward: CLIP score improvement
   - Train update strategy end-to-end

10. **Semantic Parsing**
    - Parse prompt into objects, attributes, relations
    - Apply targeted feedback per component
    - Use scene graphs for compositional generation

## Testing Strategy

1. **Baseline Comparison**
   ```bash
   python quick_compare.py
   ```

2. **Visual Inspection**
   - Check for noise vs. coherent structures
   - Verify objects match prompt

3. **Metric Tracking**
   - CLIP score should increase over steps
   - FID score should be reasonable (< 50)
   - Compositional accuracy > 0.5

4. **Ablation Studies**
   - Disable feedback (baseline)
   - Try different alpha values (0.01, 0.05, 0.1)
   - Vary feedback frequency (1, 5, 10 steps)

## Configuration Changes

Updated `configs/dynaprompt_config.yaml`:
```yaml
feedback:
  feedback_frequency: 5   # More frequent
  feedback_start_step: 5  # Earlier start
  feedback_end_step: 40   # Earlier stop

prompt_update:
  update_alpha: 0.05      # More conservative
```

## Expected Results

After fixes:
- ✓ No noise/corruption
- ✓ Recognizable objects
- ✓ Better text-image alignment
- ✓ CLIP score improvement trajectory
- ✓ Stable generation process

## Run Commands

Test the fixes:
```bash
# Quick test (30 steps)
python run_dynaprompt.py

# Full comparison
python compare_baseline.py

# Fast comparison (reuse existing)
python quick_compare.py
```
