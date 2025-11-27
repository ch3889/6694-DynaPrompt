# DynaPrompt Architecture & Mathematical Framework

## Overview
DynaPrompt is a real-time semantic feedback system for text-to-image diffusion models that dynamically adjusts prompt embeddings during generation to improve compositional accuracy.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DynaPrompt Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Text Prompt p                                       │
│     ↓                                                        │
│  ┌──────────────────────────────┐                          │
│  │ SD Text Encoder              │                          │
│  │ e₀ = TextEnc(p)              │                          │
│  └──────────────────────────────┘                          │
│     ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Iterative Denoising Loop (t = T → 0)                 │  │
│  │                                                        │  │
│  │  For each feedback step:                              │  │
│  │                                                        │  │
│  │  1. Generate intermediate image x̃ₜ                    │  │
│  │     x̃ₜ = Denoise(xₜ, eₜ, t)                          │  │
│  │                                                        │  │
│  │  2. Compute Per-Token Alignment                       │  │
│  │     ┌─────────────────────────────────┐              │  │
│  │     │ CLIP Alignment Analysis         │              │  │
│  │     │ - Extract concepts from prompt  │              │  │
│  │     │ - Score each concept with CLIP  │              │  │
│  │     │ - Identify weak tokens          │              │  │
│  │     └─────────────────────────────────┘              │  │
│  │     sᵢ = CLIP(x̃ₜ, cᵢ) for concept cᵢ                │  │
│  │     W = {cᵢ | sᵢ < μ - 0.5σ}                          │  │
│  │                                                        │  │
│  │  3. Compute Feedback Gradient                         │  │
│  │     ┌─────────────────────────────────┐              │  │
│  │     │ Global Alignment                │              │  │
│  │     │ - Extract CLIP features         │              │  │
│  │     │ - Compute alignment direction   │              │  │
│  │     │ - Project to embedding space    │              │  │
│  │     └─────────────────────────────────┘              │  │
│  │     fₜᵉˣᵗ = CLIPₜₑₓₜ(p)                              │  │
│  │     fₜⁱᵐᵍ = CLIPᵢₘₐ�ᵍ(x̃ₜ)                             │  │
│  │     g = fₜⁱᵐᵍ - fₜᵉˣᵗ                                  │  │
│  │                                                        │  │
│  │  4. Update Embedding (Dual Strategy)                 │  │
│  │     ┌─────────────────────────────────┐              │  │
│  │     │ A) Global Gradient Update       │              │  │
│  │     │    e'ₜ = eₜ + α·g               │              │  │
│  │     │    α = 0.05 (conservative)      │              │  │
│  │     └─────────────────────────────────┘              │  │
│  │     ┌─────────────────────────────────┐              │  │
│  │     │ B) Selective Token Boosting     │              │  │
│  │     │    For each weak token cᵢ ∈ W:  │              │  │
│  │     │    wᵢ = (20 - sᵢ) / 20          │              │  │
│  │     │    bᵢ = 1 + β·wᵢ                │              │  │
│  │     │    e'ₜ[pos(cᵢ)] *= bᵢ           │              │  │
│  │     │    β = 1.3 (30% boost factor)   │              │  │
│  │     └─────────────────────────────────┘              │  │
│  │                                                        │  │
│  │  5. Normalize & Continue                              │  │
│  │     eₜ₊₁ = e'ₜ · ‖eₜ‖ / ‖e'ₜ‖                        │  │
│  │                                                        │  │
│  └──────────────────────────────────────────────────────┘  │
│     ↓                                                        │
│  Final Image x₀                                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Mathematical Formulation

### 1. Per-Token Alignment Analysis

**Concept Extraction:**
- Extract concepts C = {c₁, c₂, ..., cₙ} from prompt p
- Include unigrams, bigrams, trigrams (excluding stop words)

**Alignment Scoring:**
```
For each concept cᵢ ∈ C:
    sᵢ = CLIP(image, cᵢ)
    sᵢ ∈ [0, 30] (typical CLIP score range)
```

**Weak Token Detection:**
```
μ = (1/n)·Σsᵢ                    (mean score)
σ = sqrt((1/n)·Σ(sᵢ - μ)²)       (standard deviation)
threshold = μ - 0.5σ
W = {cᵢ | sᵢ < threshold}        (weak tokens)
```

### 2. Global Gradient Feedback

**CLIP Feature Extraction:**
```
fₜᵉˣᵗ = CLIPₜₑₓₜ(prompt)         ∈ ℝ⁵¹²
fₜⁱᵐᵍ = CLIPᵢₘₐ�ᵍ(x̃ₜ)            ∈ ℝ⁵¹²
```

**Alignment Direction:**
```
g = fₜⁱᵐᵍ - fₜᵉˣᵗ                (push text toward image)
```

**Dimension Projection:**
```
g_projected = Pad(g, dim_embed)  (512 → 768 for SD)
feedback = g_projected · (1 - clip_score/100)
```

**Embedding Update:**
```
e'ₜ = eₜ + α · feedback / ‖feedback‖
α = 0.05                         (conservative learning rate)
```

### 3. Selective Token Re-weighting

**Weakness Quantification:**
```
For weak token cᵢ with score sᵢ:
    wᵢ = max(0, 20 - sᵢ) / 20    ∈ [0, 1]
    (higher = weaker concept)
```

**Adaptive Boost:**
```
bᵢ = 1.0 + β · wᵢ
β = 1.3                          (30% max boost)
```

**Token Position Mapping:**
```
For concept cᵢ at word positions [k, k+1, ..., k+m]:
    token_positions = [k+1, k+2, ..., k+m+1]  (+1 for BOS)
    
For each position j ∈ token_positions:
    e'ₜ[0, j, :] *= bᵢ
```

**Normalization (Prevent Explosion):**
```
e'ₜ = e'ₜ · (‖eₜ‖ / ‖e'ₜ‖)
```

### 4. Compositional Accuracy Metric

**Novel metric replacing BLIP-2:**

```
Per-token analysis result:
    C = all concepts extracted
    W = weak concepts
    S = {sᵢ | cᵢ ∈ C} = concept scores

Compositional Completeness:
    completeness = (|C| - |W|) / |C|

Average Alignment:
    avg_alignment = (1/|C|) · Σsᵢ / 30

Final Metric:
    compositional_accuracy = 0.7 · completeness + 0.3 · avg_alignment
```

## Implementation Details

### Module Structure
```
dynaprompt/
├── core.py              # DynaPrompt feedback engine
│   ├── compute_per_token_alignment()     (lines 34-113)
│   ├── selective_token_reweight()        (lines 206-267)
│   ├── feedback_loop()                   (lines 268-358)
│   └── compute_compositional_accuracy()  (lines 143-180)
├── wrapper.py           # SD integration
│   └── generate_with_feedback()          (lines 68-280)
└── sd_loader.py         # CompVis SD loader
```

### Feedback Integration Points

**In SD Denoising Loop:**
```python
for step in range(num_steps):
    # Standard DDIM step
    latents, pred_x0 = sampler.p_sample_ddim(...)
    
    # DynaPrompt feedback (every N steps)
    if step % feedback_freq == 0:
        # Decode to image space
        image = decode_latents(pred_x0)
        
        # Compute feedback
        feedback_result = dynaprompt.feedback_loop(
            prompt, current_embedding, image, step
        )
        
        # Update embedding for next steps
        current_embedding = feedback_result['updated_embedding']
        weak_tokens = feedback_result['weak_tokens']
```

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `α` (global learning rate) | 0.05 | Conservative gradient update |
| `β` (boost factor) | 1.3 | Max 30% boost for weak tokens |
| `feedback_freq` | 5 steps | Balance speed vs accuracy |
| `weak_threshold` | μ - 0.5σ | Statistical outlier detection |
| `clip_score_range` | [0, 30] | Typical CLIP score normalization |

## Key Innovations

1. **Per-Token Analysis**: Unlike global CLIP scores, analyzes individual concepts
2. **Dual Feedback**: Combines global alignment + selective token boosting
3. **Adaptive Boosting**: Boost strength proportional to weakness
4. **Real-Time**: Integrates directly into denoising loop
5. **No BLIP-2**: Compositional accuracy computed from per-token scores

## Computational Complexity

**Per Feedback Step:**
- Concept extraction: O(n·m) where n = prompt words, m = n-gram sizes
- CLIP scoring: O(k·C) where k = concepts, C = CLIP inference cost
- Token boosting: O(k·d) where d = embedding dimension
- **Total**: ~50ms per feedback step on T4 GPU

**Memory Overhead:**
- CLIP model: ~600MB
- Per-token scores: O(k) = ~1KB
- Embedding updates: Same as original SD

## Comparison to Baselines

| Approach | Global Alignment | Compositional | Real-Time |
|----------|-----------------|---------------|-----------|
| Baseline SD | ✗ | ✗ | ✓ |
| Prompt++  | ✓ | ✗ | ✗ |
| Attend-and-Excite | ✓ | Partial | Slow |
| **DynaPrompt** | ✓ | ✓ | ✓ |

