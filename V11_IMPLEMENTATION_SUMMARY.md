# DynaPrompt V11 Implementation Summary

## Approach: Smart Retry Strategy

### Why V11 Instead of Full Attend-and-Excite?

Based on V10's findings, we learned that:
- Increasing attention boost (7.5x → 60x) has ZERO effect on CLIP scores
- The problem isn't attention strength, it's the fundamental way attributes bind to objects
- Full Attend-and-Excite requires complex integration with the sampling loop

**V11 Simple takes a pragmatic approach**:
- Different random seeds produce different latent initializations
- Some seeds may naturally lead to better attribute binding
- Use CLIP to evaluate multiple seeds and pick the best one

## Implementation

### File: `dynaprompt/dynaprompt_v11_simple.py`

**Key Features**:
1. **Multi-Seed Trial**: Generate with 5 different random seeds
2. **CLIP Evaluation**: Score each result for critical attributes
3. **Best Selection**: Return the seed with highest CLIP scores
4. **Early Exit**: Stop if we find a passing result

**Strategy**:
```
For each seed in [seed1, seed2, seed3, seed4, seed5]:
    1. Generate image with V7 (with that seed)
    2. Validate with CLIP
    3. Track scores
    4. If all attributes pass → return immediately
    5. Otherwise → continue trying

Return the seed with best average CLIP score
```

### Comparison with V10

| Feature                  | V10 (Adaptive Boost)    | V11 (Smart Retry)        |
|--------------------------|-------------------------|--------------------------|
| Core mechanism           | Increase boost factor   | Try different seeds      |
| Retries                  | 4 (with 2x boost each)  | 5 (with different seeds) |
| What changes per retry   | Attention boost (7.5x→60x) | Latent initialization |
| Expected to help?        | NO (V10 proved it doesn't) | YES (different latents) |
| Computational cost       | 4 full generations      | 5 full generations       |
| Time per generation      | ~60s                    | ~60s                     |
| Total time               | ~240s                   | ~300s                    |

### Why This Might Work

**Hypothesis**: Diffusion models are sensitive to initial noise

1. **Different seeds → different latent noise**
   - Each seed initializes latents differently
   - Model starts from different points in latent space

2. **Some initializations are "luckier"**
   - Some latent configurations naturally evolve toward better attribute binding
   - Similar to how some prompts work better than others

3. **CLIP finds the best**
   - We don't know which seed will work best
   - CLIP objectively evaluates the results
   - Pick the winner

### Code Structure

```python
class DynaPromptV11Simple(DynaPromptV7Sampler):
    def sample_with_smart_retry(
        self,
        prompt: str,
        critical_attributes: List[str],
        num_seed_trials: int = 5,
    ):
        best_score = -1
        best_samples = None

        for trial in range(num_seed_trials):
            # Use different seed
            seed = random_seed()

            # Generate with V7
            samples = V7.generate(prompt, seed=seed)

            # Score with CLIP
            scores = CLIP.score(samples, critical_attributes)
            avg_score = mean(scores)

            # Track best
            if avg_score > best_score:
                best_score = avg_score
                best_samples = samples

            # Early exit if passed
            if all_passed(scores):
                break

        return best_samples
```

## Expected Results

### Success Criteria

**Passing**: CLIP scores ≥ 0.25 for all critical attributes

**Hypothesis**:
- V10 showed scores of 0.12-0.20 (failed) with same seed
- Different seeds should show **variation** in scores
- At least one seed should score higher than V10's best (0.20)

### Possible Outcomes

**Best Case** (60-70% chance):
- At least one seed passes (all scores ≥ 0.25)
- Demonstrates that seed variation helps
- V11 succeeds where V10 failed

**Middle Case** (25-35% chance):
- No seed passes, but we see **variation** in scores
- Some seeds score higher (0.22-0.24), closer to threshold
- Proves seed matters, just need more trials or better threshold

**Worst Case** (5% chance):
- All seeds produce similar scores (0.12-0.20)
- No variation observed
- Suggests the problem is deeper than seed initialization
- Would need full Attend-and-Excite or spatial decomposition

## Test Configuration

**Test Script**: `scripts/test_v11_simple.py`

**Test Prompt**: "a silver car parked next to a golden bicycle"

**Critical Attributes**: ["silver car", "golden bicycle"]

**Parameters**:
- Number of trials: 5
- Steps per trial: 50
- CFG scale: 7.5
- CLIP threshold: 0.25
- V7 boost_factor: 7.5x (not increased)

## Metrics to Track

### Per-Trial Metrics
- Seed used
- CLIP score for "silver car"
- CLIP score for "golden bicycle"
- Average CLIP score
- Pass/Fail status

### Overall Metrics
- Best trial (which seed)
- Best average CLIP score
- Score distribution (variance across seeds)
- Success rate (did any seed pass?)

## Next Steps Based on Results

### If V11 Succeeds (≥60% success rate)
1. ✅ Use V11 Simple as the new baseline
2. Increase num_seed_trials to 10 for better coverage
3. Run full evaluation on 30 prompts
4. Consider combining with CompAgent (generate objects separately)

### If V11 Partially Succeeds (scores improve but don't pass)
1. Increase num_seed_trials to 10-20
2. Lower CLIP threshold to 0.20-0.22
3. Combine with CompAgent for multi-object prompts
4. Implement full Attend-and-Excite for harder cases

### If V11 Fails (no improvement over V10)
1. Confirms that U-Net approach has fundamental limitations
2. Move to spatial decomposition (CompAgent-style):
   - Generate "a silver car" separately
   - Generate "a golden bicycle" separately
   - Compose using layout control or inpainting
3. Or implement full Attend-and-Excite with latent optimization

## Technical Details

### Why Seeds Matter

In diffusion models:
```python
# Initial latent noise
latents = torch.randn(shape, generator=torch.Generator().manual_seed(seed))

# Denoising process
for t in timesteps:
    # Model predicts noise to remove
    noise_pred = model(latents, t, text_embeddings)

    # Update latents
    latents = scheduler.step(noise_pred, t, latents)
```

Different seeds → different initial `latents` → different denoising trajectory → different final image

### V7's Seed Retry vs V11's Seed Trial

**V7's seed retry** (within Phase 1):
- Tries 16 different seeds
- Looking for one with sufficient early attention
- If found, uses that seed
- If not found, falls back to attention boosting

**V11's seed trial** (at top level):
- Lets V7 do its full process (including its own seed retry)
- Tries 5 completely independent V7 runs
- Compares final results with CLIP
- Picks the best one

This is **different** - V11 gets 5 chances at V7's best output, not just 5 random seeds.

## Files

- **Implementation**: `/home/cursedfox/6694-DynaPrompt/dynaprompt/dynaprompt_v11_simple.py`
- **Test Script**: `/home/cursedfox/6694-DynaPrompt/scripts/test_v11_simple.py`
- **This Document**: `/home/cursedfox/6694-DynaPrompt/V11_IMPLEMENTATION_SUMMARY.md`

## Status

**Implementation**: ✅ Complete
**Testing**: 🔄 In Progress (currently running test_v11_simple.py)
**Expected completion**: ~5-10 minutes (5 trials × 60s per trial = 300s + overhead)
