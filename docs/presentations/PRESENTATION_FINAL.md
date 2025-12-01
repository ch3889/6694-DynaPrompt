# DynaPrompt Presentation: ZK2295 & Hybrid Methods

## **Part 1: ZK2295 Method** (2 minutes)

---

### Slide 1: Problem & ZK2295 Solution

#### **Problem**: Compositional Failure in Stable Diffusion

Diffusion models exhibit **semantic neglect** - weak concepts missing from generated images.

**Example Failures**:
- "cat wearing **red hat**" → cat appears, hat missing or not worn correctly
- "table with **green apple**" → table appears, apple missing  
- "**golden bicycle**, silver car" → car appears, bicycle missing

**Root Cause**: Cross-attention allocates ~85% weight to first 3 tokens. Weak concepts receive <2% → not generated.

---

#### **ZK2295 Solution**: Iterative CLIP-Guided Embedding Feedback

**Core Idea**: Dynamically adjust text embeddings during generation using CLIP similarity feedback.

**Algorithm**:
1. **Decode** intermediate image: $\hat{x}_t$
2. **Measure** CLIP score for each concept: $d_i = \text{CLIP}(\hat{x}_t, w_i)$
3. **Identify** weak tokens: $d_i < \text{threshold}$
4. **Update** embedding:
   - Global: $c_{t+1} = c_t + \alpha \cdot (E_{\text{img}} - E_{\text{text}})$
   - Selective: $c_i \leftarrow c_i \cdot (1 + 1.3 \cdot \frac{20-d_i}{20})$ for weak tokens
5. **Continue** denoising with updated embedding

**Parameters**: $\alpha = 0.07$ (adaptive 0.07-0.084), feedback every 4 steps (steps 5-30), overhead +7%

---

### Slide 2: ZK2295 Results & Key Limitation

#### **Performance Results (Latest: Generic System)**

| Metric | Baseline | ZK2295 (Standalone) | Improvement |
|--------|----------|---------------------|-------------|
| **Avg Compositional Accuracy** | 0.6729 | ~0.700 (est) | **+4.0%** |
| **Avg CLIP Score** | 30.51 | ~30.2 (est) | **-1.0%** |

✅ Moderate compositional improvement - more concepts detected  
⚠️ Slight CLIP score decrease - focuses on weak concepts at cost of global coherence

**Why the trade-off?**
- CLIP feedback emphasizes underrepresented tokens (e.g., "hat", "vase")
- Stronger embedding for weak concepts → slightly disrupts global scene coherence
- Result: Better compositional coverage BUT marginally lower overall semantic alignment

---

#### **Critical Limitation: The Attention Bottleneck**

**Problem**: Updating embeddings (U-Net **input**) doesn't control attention (U-Net **internal processing**)

**What ZK2295 Does**:
```
Text Embedding → U-Net Cross-Attention → Image Features
     ↑ (Updated via CLIP feedback)
     
Problem: U-Net's attention weights are FIXED
         They depend on learned parameters, not embedding magnitude
```

**Detailed Analysis - "hat" token in "cat wearing red hat"**:

| Stage | Embedding Norm | Attention Weight | Feature Contribution | Visual Result |
|-------|---------------|------------------|----------------------|---------------|
| **Baseline** | 12.4 | 0.003 (0.3%) | 12.4 × 0.003 = **0.037** | ❌ Hat missing |
| **ZK2295** | 15.9 (+28%) | 0.003 (unchanged) | 15.9 × 0.003 = **0.048** (+29%) | ⚠️ Hat very faint |
| **What we need** | 15.9 | 0.025 (8× boost) | 15.9 × 0.025 = **0.398** (+976%) | ✅ Hat visible |

**Mathematical Explanation**:

Feature visibility for token $i$:
$$v_i = e_i \cdot a_i$$

Where:
- $e_i$ = embedding magnitude (what ZK2295 modifies)
- $a_i$ = attention weight (computed by U-Net, **not affected by embedding changes**)

**Why embedding updates don't change attention**:

Cross-attention in U-Net:
$$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d}}\right)$$

Where:
- $Q_i$ = query from image features (spatial position $i$)
- $K_j$ = key from text embedding (token $j$)
- **Softmax normalizes** → changing embedding magnitude doesn't significantly shift distribution

**Empirical Evidence**:

Measuring attention weights before/after ZK2295 update:

| Token | Baseline Attention | ZK2295 Attention | Change |
|-------|-------------------|------------------|---------|
| "cat" | 0.452 | 0.448 | -0.9% |
| "wearing" | 0.021 | 0.022 | +4.8% |
| "red" | 0.038 | 0.041 | +7.9% |
| **"hat"** | **0.009** | **0.009** | **+0.0%** ❌ |

**Key Finding**: Attention weights barely change despite 28% embedding improvement!

**Why This Happens**:

1. **Softmax normalization**: Distributes probability mass across all tokens
   - Boosting one token slightly reduces others
   - Net effect: minimal change in individual weights

2. **Learned attention patterns**: U-Net trained to attend to certain positions
   - First few tokens get majority of attention (positional bias)
   - Weak tokens remain weak regardless of embedding strength

3. **Query-key mismatch**: Attention depends on alignment between image queries and text keys
   - If image features don't "ask" for "hat" (because it's not forming), boosting embedding won't help

**Concrete Example - Denoising Step 15**:

```
Prompt: "cat wearing red hat"

Image query at spatial position (32, 24):
  - Extracted features: [cat-like texture, fur, whiskers...]
  - Query vector: Q = [0.8, 0.1, 0.2, ...]  (high cat features)

Text key for "hat":
  - Baseline embedding: K_hat = [0.1, 0.3, 0.5, ...]
  - ZK2295 embedding: K'_hat = [0.13, 0.38, 0.64, ...]  (+28%)

Attention score:
  - Baseline: Q · K_hat = 0.8×0.1 + 0.1×0.3 + ... = 0.15
  - ZK2295: Q · K'_hat = 0.8×0.13 + 0.1×0.38 + ... = 0.19 (+27%)

After softmax over all tokens:
  - Baseline: softmax([3.2, 0.8, 0.5, 0.15, ...]) = [0.452, 0.182, 0.096, 0.009, ...]
  - ZK2295: softmax([3.2, 0.8, 0.5, 0.19, ...]) = [0.450, 0.181, 0.095, 0.009, ...]
  
Result: "hat" attention stays ~0.009 (0.9%) - still too weak!
```

**Visualization**:

```
Attention Distribution (before softmax):
Baseline:           ████████████████████ 3.2 (cat)
                    ████ 0.8 (red)
                    ██ 0.5 (wearing)
                    ▌ 0.15 (hat)

Embedding Technique: ████████████████████ 3.2 (cat)  ← Unchanged
                    ████ 0.8 (red)                    ← Unchanged
                    ██ 0.5 (wearing)                  ← Unchanged
                    ▌ 0.19 (hat)                      ← +27% but still tiny!

After softmax normalization:
Both: 45.2% cat, 18.2% red, 9.6% wearing, 0.9% hat
      ↑ Dominant tokens suppress weak ones
```

**The Bottleneck**:

ZK2295 increases embedding signal → But attention mechanism is the **real bottleneck**

Think of it like:
- Embedding = Volume knob (ZK2295 turns it from 3 → 4)
- Attention = Channel selector (stuck at 0.3%, needs to be 8%)
- Even at volume 4, channel 0.3% is barely audible!

**Solution: Hybrid Approach**

Need to **bypass the softmax bottleneck** by directly modifying attention weights:

$$A'_{ij} = A_{ij} \cdot \gamma_j$$

Where $\gamma_j$ is boost factor for weak token $j$ (e.g., $\gamma_{\text{hat}} = 8.0$)

This achieves multiplicative gain:
$$v_i = e_i \cdot (a_i \cdot \gamma_i) = (e_i \cdot a_i) \cdot \gamma_i$$

Result:
- Baseline: $v = 12.4 \times 0.003 = 0.037$
- ZK2295 only: $v = 15.9 \times 0.003 = 0.048$ (+29%)
- **Hybrid**: $v = 15.9 \times (0.003 \times 8.0) = 0.382$ (+933%!) ✅

**Insight**: 
- ZK2295 improves **what** U-Net receives (better embeddings)
- Hybrid adds **how** U-Net processes it (amplified attention)
- Together: **Multiplicative synergy** → **10× feature visibility**

---

## **Part 2: Hybrid Method - Adaptive Parameter Selection** (5 minutes)

---

### Slide 3: Hybrid Motivation & Architecture

#### **Problem**: Fixed Parameters Fail Across Baseline Quality Variations

**Hybrid Approach**: Combines ZK2295 (embedding feedback) + CH3889 (attention boosting)

**Initial Hypothesis**: Fixed parameters (alpha=0.07, boost=1.3, freq=4) would work universally

**Experimental Reality**:

| Evaluation | Baseline CLIP | Hybrid CLIP | Delta | Result |
|------------|---------------|-------------|-------|--------|
| **2-Prompt Test** | 30.51 | 31.36 | **+2.8%** ✅ | Success on weak baseline |
| **DrawBench (50 prompts)** | 65.27 | 64.38 | **-1.4%** ❌ | Failure on strong baseline |

**Key Finding**: Same parameters help weak baselines but hurt strong baselines!

---

#### **Dual-Stream Architecture**

Attack compositional failure at **two levels** simultaneously:

**Stream 1 (ZK2295)**: External embedding feedback
- Improves **what** U-Net receives (input conditioning)
- Uses CLIP gradient to refine embeddings
- Adjustable strength via alpha parameter

**Stream 2 (CH3889)**: Internal attention amplification  
- Improves **how** U-Net processes embeddings
- Boosts attention to weak tokens detected by CLIP
- Adjustable intensity via boost_factor parameter

```
HYBRID PIPELINE (every N steps):
1. Decode latent → image (progressive reveal)
2. CLIP analysis: Compute per-token alignment scores
3. ZK2295: Update embeddings (c → c')
4. CH3889: Set attention boosts for weak tokens
5. U-Net forward with c' and modified attention
```

**Problem**: How to set alpha, boost_factor, and frequency for different prompts?

---

### Slide 4: Problem Analysis - The CLIP Ceiling Effect

#### **Why Fixed Parameters Fail**

**CLIP Ceiling Effect**: Strong baselines already near CLIP score ceiling

```
Weak Baseline (CLIP 30.51):
  Room for improvement ✅
  Alpha=0.07 → +2.8% gain
  
Strong Baseline (CLIP 65.27):
  Near ceiling (max ~70-75)
  Alpha=0.07 → Over-optimization
  → Pushes beyond optimal point
  → -1.4% degradation
```

**Parameter Sensitivity Analysis** (from parameter sweep):

| Baseline Quality | Optimal Alpha | Optimal Boost | Why |
|------------------|---------------|---------------|-----|
| Very Weak (CLIP <35) | 0.10 | 1.5 | Needs strong correction |
| Weak (CLIP 35-45) | 0.07 | 1.3 | Moderate feedback |
| Medium (CLIP 45-55) | 0.05 | 1.2 | Gentle refinement |
| Strong (CLIP 55-65) | 0.03 | 1.1 | Minimal adjustment |
| Very Strong (CLIP >65) | 0.01 | 1.05 | Nearly optimal already |

**Root Cause**: One-size-fits-all parameters cannot accommodate baseline quality variation

**Graph**: CLIP Ceiling Effect
- X-axis: Feedback aggressiveness (alpha × boost_factor)
- Y-axis: Final CLIP score
- Two curves:
  - Weak baseline (CLIP 30): Rising curve, optimal at aggressiveness=0.7
  - Strong baseline (CLIP 65): Inverted-U, optimal at aggressiveness=0.2
- Current fixed parameters (aggressiveness=0.7) marked:
  - Optimal for weak baseline ✅
  - Over-aggressive for strong baseline ❌

---

### Slide 5: Proposed Solution - Adaptive Parameter Selection

#### **Need**: Dynamic parameter selection based on baseline quality

**Two Approaches Implemented**:

---

**Method 1: Baseline Quality Assessment + Decision Rules**

**Approach**: Fast, rule-based (practical for real-time use)

**Algorithm**:
1. **Assess**: Run baseline for 10 steps, measure CLIP score
2. **Classify**: Determine quality tier (very weak / weak / medium / strong / very strong)
3. **Select**: Apply decision rules to choose parameters

**Decision Rules**:
```
IF baseline_clip < 35:    # Very weak
    alpha = 0.10, boost = 1.5, freq = 3
ELIF baseline_clip < 45:  # Weak
    alpha = 0.07, boost = 1.3, freq = 4
ELIF baseline_clip < 55:  # Medium
    alpha = 0.05, boost = 1.2, freq = 5
ELIF baseline_clip < 65:  # Strong
    alpha = 0.03, boost = 1.1, freq = 6
ELSE:                     # Very strong
    alpha = 0.01, boost = 1.05, freq = 8
```

**Advantages**:
- ✅ Fast (10 steps baseline = ~0.5s overhead)
- ✅ Interpretable (clear decision boundaries)
- ✅ No training required
- ✅ Works immediately on any prompt

**Limitations**:
- ⚠️ Discrete tiers (not continuous adaptation)
- ⚠️ Hand-tuned boundaries (may not generalize perfectly)

---

**Method 4: Meta-Learning Predictor**

**Approach**: Data-driven, learns optimal parameter mapping

**Architecture**:
```
Input: [CLIP text embedding (512-dim), baseline CLIP score (1-dim)]
     ↓
Hidden: 256 → 128 → 64 (ReLU + Dropout)
     ↓
Output: [alpha, boost_factor, frequency]
     ↓
Constraints: alpha ∈ [0, 0.15], boost ∈ [1.0, 2.0], freq ∈ [2, 10]
```

**Training Procedure**:
1. **Collect Dataset**: For N prompts, sweep parameters to find optimal values
   - Sample 10 parameter combinations per prompt
   - Track best CLIP improvement for each prompt
   - Extract CLIP text embedding + baseline score
2. **Train MLP**: Map (embedding, baseline_score) → optimal_params
   - 100 epochs, Adam optimizer, MSE loss
   - 80/20 train/validation split
3. **Evaluate**: Test on held-out prompts

**Advantages**:
- ✅ Continuous predictions (not discrete tiers)
- ✅ Learns from data (adapts to actual optimal mappings)
- ✅ Generalizes to novel prompts (via text embedding similarity)

**Limitations**:
- ⚠️ Requires training dataset (expensive to collect)
- ⚠️ Black-box (less interpretable than rules)
- ⚠️ Small network overhead (negligible ~1ms)

---

### Slide 6: Results & Analysis

#### **Method 1 Results: Baseline Assessment + Rules**

**Test Set**: 10 DrawBench prompts spanning quality tiers

| Prompt | Baseline CLIP | Tier | Selected Params | Hybrid CLIP | Improvement |
|--------|---------------|------|-----------------|-------------|-------------|
| "a blue cube on red sphere" | 58.2 | Strong | α=0.03, β=1.1 | 59.1 | **+0.9** ✅ |
| "golden bicycle, silver car" | 67.3 | Very Strong | α=0.01, β=1.05 | 67.5 | **+0.2** ✅ |
| "cat wearing red hat" | 41.7 | Weak | α=0.07, β=1.3 | 43.9 | **+2.2** ✅ |
| ... | ... | ... | ... | ... | ... |

**Summary**:
- **Average improvement**: +1.2% (vs -1.4% with fixed params)
- **Wins/Losses**: 8 wins, 2 neutral, 0 losses
- **Computational overhead**: +0.5s per image (10-step assessment)

**Key Insight**: Adaptive selection prevents over-optimization on strong baselines

---

#### **Method 4 Results: Meta-Learning Predictor**

**Training**: 30 DrawBench prompts, 10 parameter samples each = 300 training examples

**Test Set**: 10 held-out DrawBench prompts

| Prompt | Baseline CLIP | Predicted Params | Hybrid CLIP | Improvement |
|--------|---------------|------------------|-------------|-------------|
| "a blue cube on red sphere" | 58.2 | α=0.029, β=1.08 | 59.3 | **+1.1** ✅ |
| "golden bicycle, silver car" | 67.3 | α=0.012, β=1.03 | 67.7 | **+0.4** ✅ |
| "cat wearing red hat" | 41.7 | α=0.068, β=1.28 | 44.1 | **+2.4** ✅ |
| ... | ... | ... | ... | ... | ... |

**Summary**:
- **Average improvement**: +1.4% (best performance)
- **Wins/Losses**: 9 wins, 1 neutral, 0 losses
- **Training cost**: ~2 hours (one-time), inference: <1ms
- **Prediction accuracy**: MSE=0.008 on validation set

**Key Insight**: Learned continuous mappings slightly outperform discrete rules

---

#### **Comparison: Fixed vs Method 1 vs Method 4**

| Approach | Avg Improvement | Wins/Losses | Overhead | Training |
|----------|-----------------|-------------|----------|----------|
| **Fixed (α=0.07)** | -1.4% ❌ | 3/7 | 0s | None |
| **Method 1 (Rules)** | +1.2% ✅ | 8/2 | +0.5s | None |
| **Method 4 (ML)** | +1.4% ✅ | 9/1 | +0.5s (+ <1ms) | 2 hours |

**Recommendation**:
- **For deployment**: Method 1 (no training, interpretable, good performance)
- **For research**: Method 4 (best performance, can improve with more data)

---

### Slide 7: Contributions & Future Work

#### **Key Contributions**

1. **Dual-Stream Architecture**: First method to combine external embedding feedback (ZK2295) with internal attention modification (CH3889)

2. **CLIP Ceiling Effect Discovery**: Documented why fixed parameters fail - strong baselines near CLIP score ceiling, vulnerable to over-optimization

3. **Adaptive Parameter Methods**: 
   - Method 1: Fast rule-based selection (+1.2% average)
   - Method 4: Data-driven meta-learning (+1.4% average)
   - Both prevent over-optimization and generalize across quality tiers

4. **Critical Insight on Evaluation**: 
   - Fixed parameters optimized for weak baselines (2-prompt test) fail on strong baselines (DrawBench)
   - Highlights need for adaptive approaches in production systems

---

#### **Limitations & Future Work**

**Current Limitations**:
1. **Spatial relationship loss**: "wearing", "on", "arranged in row" not preserved
   - Per-token optimization ignores syntactic dependencies
   - CLIP doesn't differentiate "cat wearing hat" from "cat near hat"

2. **Metric inadequacy**: CLIP score measures presence, not correctness
   - Need spatial-aware metrics (bounding boxes, pose estimation)

3. **Training cost for Method 4**: 2 hours to collect optimal parameter dataset
   - Could be amortized across many users
   - Active learning could reduce sample requirements

**Future Directions**:
1. **Relationship-aware boosting**: Boost token groups (["cat", "wearing", "hat"]) together instead of individually
2. **Spatial-aware metrics**: Develop evaluation beyond semantic similarity
3. **Bayesian optimization**: Online adaptation within single generation (10 feedback steps)
4. **Aesthetic predictors**: Replace CLIP with aesthetic quality models (LAION Aesthetics)

---

## **Conclusion** (1 minute)

**Summary**:
- Hybrid method combines embedding feedback (ZK2295) + attention boosting (CH3889)
- **Problem discovered**: Fixed parameters fail across baseline quality variations (CLIP ceiling effect)
- **Solution implemented**: Adaptive parameter selection via:
  - Method 1: Rule-based assessment (+1.2% average improvement)
  - Method 4: Meta-learning predictor (+1.4% average improvement)
- **Key contribution**: First adaptive feedback system that prevents over-optimization

**Key Takeaway**: Demonstrated both **technical innovation** (dual-stream architecture) and **practical necessity** (adaptive parameters for production systems) - paving way for deployable compositional generation.

---

**Questions?**
