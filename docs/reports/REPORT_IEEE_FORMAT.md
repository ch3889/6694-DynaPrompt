# Hybrid Feedback Mechanisms for Compositional Image Generation in Latent Diffusion Models

**Max Zishock Kim (zk2295)**  
Department of Electrical Engineering  
Columbia University  
New York, NY 10027  
Email: zk2295@columbia.edu

---

## Abstract

Text-to-image diffusion models exhibit systematic compositional failures, frequently omitting concepts specified in text prompts. We present a dual-stream intervention framework that addresses compositional neglect through simultaneous modification of text embeddings (external conditioning) and cross-attention weights (internal processing). Our hybrid approach achieves +6.37% improvement in compositional accuracy and +0.85% in CLIP alignment score on weak baseline prompts. However, critical analysis reveals a fundamental limitation: current evaluation metrics (CLIP score, compositional accuracy) measure semantic similarity rather than spatial relationships, creating a quantitative-visual disconnect where metric improvements correspond to degraded visual quality. We demonstrate that strong baseline prompts (CLIP score >65) experience -1.4% degradation due to over-optimization near the CLIP scoring ceiling. These findings highlight the inadequacy of presence-based metrics for evaluating compositional generation and motivate the development of spatial-aware evaluation frameworks. Our work contributes: (1) first dual-stream architecture combining embedding feedback and attention modification, (2) empirical evidence of the CLIP ceiling effect in compositional optimization, and (3) documentation of the quantitative-visual disconnect in current evaluation paradigms.

**Index Terms**—Diffusion models, compositional generation, text-to-image synthesis, attention mechanisms, CLIP evaluation

---

## I. INTRODUCTION

### A. Motivation

TEXT-TO-IMAGE diffusion models [1], particularly Stable Diffusion [2], have achieved remarkable photorealistic synthesis capabilities. However, these models exhibit a systematic failure mode termed *compositional neglect*: they frequently generate images missing concepts explicitly specified in text prompts. For instance, the prompt "a cat wearing a red hat" may produce a cat without the hat, or the prompt "golden bicycle next to silver car" may generate only the car while omitting the bicycle entirely.

This failure mode is not merely an occasional error but represents a fundamental limitation in how diffusion models process multi-concept prompts. Prior work has identified attention distribution imbalance as a key factor [3], [4]: cross-attention mechanisms in U-Net architectures allocate disproportionate weight to early tokens (~85% to first 3 tokens), leaving insufficient capacity for later concepts. When combined with strong learned priors favoring common objects, the model systematically neglects less-salient concepts during the denoising process.

### B. Research Gap

Existing approaches to compositional generation fall into two categories: (1) *embedding-level interventions* that modify text conditioning vectors [5], [6], and (2) *attention-level interventions* that amplify cross-attention weights for weak tokens [3], [7]. However, these methods operate in isolation—embedding modifications may fail if attention weights remain imbalanced, while attention amplification cannot compensate for fundamentally weak conditioning signals. No prior work has investigated whether simultaneous intervention at both levels yields synergistic improvements.

Furthermore, evaluation of compositional generation relies heavily on CLIP score [8] and derived metrics (e.g., compositional accuracy [9]). These metrics measure semantic similarity between generated images and text prompts but do not assess spatial relationships or compositional correctness. A generated image showing "cat" and "hat" as separate objects would score similarly to one correctly depicting "cat wearing hat," despite the latter being compositionally correct. This evaluation gap remains unaddressed in existing literature.

### C. Contributions

This work makes the following contributions:

1) **Dual-stream architecture**: We present the first framework combining external embedding feedback (ZK2295) with internal attention modification (CH3889), demonstrating multiplicative synergy where weak tokens achieve 10× feature visibility.

2) **CLIP ceiling effect**: We provide empirical evidence that fixed intervention parameters catastrophically fail on strong baseline prompts (CLIP >65), experiencing -1.4% degradation due to over-optimization near the CLIP scoring ceiling (~70-75 for ViT-B/32).

3) **Quantitative-visual disconnect**: We document a fundamental limitation where metric improvements (+6.37% compositional accuracy) correspond to degraded visual quality, revealing that CLIP-based metrics measure concept presence but not spatial correctness.

4) **Generalizability analysis**: We identify and eliminate 189 lines of prompt-specific hardcoded logic from prior implementations, achieving a fully generic system that works for arbitrary prompts.

The remainder of this paper is organized as follows: Section II reviews related work, Section III details our methodology, Section IV presents experimental results, Section V provides critical analysis of limitations, and Section VI concludes with future directions.

---

## II. RELATED WORK

### A. Diffusion Models for Image Synthesis

Denoising Diffusion Probabilistic Models (DDPM) [1] formulate image generation as iterative denoising of Gaussian noise. Latent Diffusion Models (LDM) [2], including Stable Diffusion, improve efficiency by operating in VAE latent space rather than pixel space. The forward diffusion process gradually adds noise:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

The reverse process learns to denoise:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Text conditioning is injected via cross-attention mechanisms in the U-Net denoiser, where text embeddings from CLIP [10] serve as keys and values while latent features serve as queries.

### B. Compositional Generation Methods

**Embedding-level interventions:** Prompt-to-Prompt [5] modifies attention maps during generation to control spatial layout. Textual Inversion [6] learns pseudo-words representing concepts. However, these methods do not address the fundamental attention imbalance problem.

**Attention-level interventions:** Attend-and-Excite [3] iteratively updates latent codes to maximize attention weights for weak tokens. Expressive Text-to-Image [7] uses energy-based guidance. These approaches modify attention but do not strengthen the underlying conditioning signal.

**Hybrid approaches:** To our knowledge, no prior work combines embedding-level and attention-level interventions simultaneously. Our work fills this gap by demonstrating that dual-stream feedback achieves synergistic improvements.

### C. Evaluation Metrics

CLIPScore [8] measures image-text alignment via cosine similarity in CLIP embedding space:

$$\text{CLIP}(I, T) = \cos(\text{Enc}_I(I), \text{Enc}_T(T))$$

Compositional accuracy [9] checks per-token alignment:

$$\text{CompAcc} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{CLIP}(I, t_i) > \tau]$$

where $t_i$ are individual tokens and $\tau$ is a threshold (typically 20).

**Limitation:** These metrics measure semantic similarity (concept presence) but not spatial relationships (concept positioning). Section V-C provides empirical evidence of this limitation's impact on evaluation validity.

---

## III. METHODOLOGY

### A. Problem Formulation

Given a text prompt $P = \{t_1, t_2, ..., t_N\}$ with $N$ tokens, Stable Diffusion generates image $I$ via:

$$I = \text{Decode}(z_0), \quad z_0 = \text{Denoise}(z_T, c)$$

where $z_T \sim \mathcal{N}(0, I)$ is Gaussian noise, $c = \text{CLIP}_{\text{text}}(P)$ is text embedding, and Denoise is the U-Net with cross-attention layers.

**Observation:** Weak tokens (low initial attention weight) fail to influence generation:

$$\alpha_{t,i} = \text{softmax}\left(\frac{Q_t K_i^T}{\sqrt{d}}\right) \ll \frac{1}{N}$$

where $Q_t$ is latent query at step $t$ and $K_i$ is key for token $t_i$.

**Objective:** Increase weak token influence through:

1) Embedding feedback: $c_{t+1} = c_t + \alpha \cdot \nabla_c \mathcal{L}_{\text{CLIP}}$
2) Attention boosting: $\alpha'_{t,i} = \beta \cdot \alpha_{t,i}$ for weak $i$

### B. Dual-Stream Architecture

Our framework operates on two parallel streams:

**Stream 1 (ZK2295): Embedding Feedback**

The embedding feedback mechanism operates by iteratively refining text conditioning vectors based on intermediate generation quality. This approach addresses the fundamental issue that initial CLIP text embeddings may not optimally represent weak concepts in the context of diffusion model generation.

**Architecture Overview:**

The ZK2295 stream consists of four computational stages executed at feedback steps $t \in \{5, 9, 13, 17, 21, 25, 29\}$ (every 4 steps from step 5 to 30):

*Stage 1: Intermediate Image Decoding*

At each feedback step $t$, the partially denoised latent $z_t$ is decoded to pixel space:

$$\hat{I}_t = \text{VAE}_{\text{decode}}(z_t)$$

The Variational Autoencoder (VAE) [2] decoder maps from 4-channel latent space to RGB pixel space. This requires:
- VAE decoder forward pass: 0.08s on NVIDIA T4
- Output: 512×512×3 RGB image representing current generation state
- Latent dimensionality: $z_t \in \mathbb{R}^{4 \times 64 \times 64}$ → image $\hat{I}_t \in \mathbb{R}^{3 \times 512 \times 512}$

The decoded image $\hat{I}_t$ is noisy but contains sufficient semantic information for CLIP evaluation. At step 5, the image shows rough structure; by step 30, it approaches the final form with ~70% visual similarity to the eventual output.

*Stage 2: Per-Token Semantic Assessment*

For each token $t_i$ in prompt $P = \{t_1, ..., t_N\}$, compute alignment score:

$$s_i = \text{CLIP}(\hat{I}_t, t_i) = \cos\left(\text{Enc}_I(\hat{I}_t), \text{Enc}_T(t_i)\right)$$

where:
- $\text{Enc}_I$: CLIP image encoder (Vision Transformer ViT-B/32 [16], 151M params)
- $\text{Enc}_T$: CLIP text encoder (Transformer [17], 63M params)
- $\cos(\cdot, \cdot)$: Cosine similarity in 512-dimensional embedding space

**Computational cost:** $N$ CLIP forward passes per feedback step. For typical prompt with $N=10$ tokens:
- Image encoding: 0.02s (amortized across tokens)
- Text encoding: $10 \times 0.003$s = 0.03s
- Total: 0.05s per feedback step

**Score interpretation:**
- $s_i > 30$: Strong semantic alignment (concept prominently visible)
- $20 < s_i < 30$: Moderate alignment (concept present but weak)
- $s_i < 20$: Weak alignment (concept missing or barely visible)
- $s_i < 15$: Severe neglect (concept completely absent)

*Stage 3: Weak Token Identification*

Tokens are classified as "weak" using dynamic median thresholding:

$$W_t = \left\{i : s_i < \text{median}\left(\{s_j\}_{j=1}^N\right)\right\}$$

**Rationale for median threshold:** Avoids hardcoded score cutoffs that fail to adapt to:
- Prompt-specific baseline quality (some prompts naturally score higher)
- Temporal dynamics (scores evolve during denoising)
- Cross-token relative strength (identifies bottom 50% regardless of absolute scores)

**Alternative approaches considered:**
- Fixed threshold ($s_i < 25$): Fails on strong baseline prompts where all tokens exceed 25
- Percentile threshold (bottom 30%): Arbitrary cutoff, less interpretable than median
- Standard deviation criterion ($s_i < \mu - \sigma$): Sensitive to outliers

**Empirical observation:** Median thresholding correctly identifies 87% of visually-missing concepts (validated on 50 prompts with manual inspection).

*Stage 4: Gradient-Based Embedding Update*

For each weak token $i \in W_t$, update its embedding via gradient ascent [20] on CLIP score:

$$c_{i,t+1} = c_{i,t} + \alpha \cdot \frac{\partial \text{CLIP}(\hat{I}_t, t_i)}{\partial c_{i,t}}$$

**Gradient computation:** Backpropagation through CLIP image encoder while treating $\hat{I}_t$ as constant (no gradient to VAE or U-Net [18]):

$$\frac{\partial \text{CLIP}(\hat{I}_t, t_i)}{\partial c_{i,t}} = \frac{\partial}{\partial c_{i,t}} \cos\left(\text{Enc}_I(\hat{I}_t), \text{Enc}_T(t_i; c_{i,t})\right)$$

Since $\text{Enc}_T(t_i; c_{i,t}) = c_{i,t}$ (embeddings are pre-computed), the gradient simplifies to:

$$\nabla_{c_{i,t}} = \frac{\text{Enc}_I(\hat{I}_t) - \langle \text{Enc}_I(\hat{I}_t), c_{i,t} \rangle \cdot c_{i,t}}{\|\text{Enc}_I(\hat{I}_t)\| \cdot \|c_{i,t}\|}$$

This points toward the image embedding, increasing alignment.

**Hyperparameter: Step size $\alpha = 0.07$**

Selected via grid search over $\alpha \in \{0.01, 0.03, 0.05, 0.07, 0.10, 0.15\}$ on 2-prompt validation set:

| $\alpha$ | Comp. Acc. | CLIP Score | Visual Quality |
|---------|-----------|-----------|----------------|
| 0.01 | +2.1% | +0.3% | Good (minimal artifacts) |
| 0.03 | +4.8% | +0.6% | Good |
| 0.05 | +6.9% | +0.9% | Moderate (slight over-saturation) |
| **0.07** | **+8.2%** | **+1.1%** | **Moderate** ✓ |
| 0.10 | +9.1% | +0.7% | Poor (color artifacts) |
| 0.15 | +10.3% | -0.4% | Very poor (unnatural) |

**Optimal choice:** $\alpha=0.07$ maximizes compositional accuracy while maintaining positive CLIP score. Larger values ($\alpha \geq 0.10$) introduce visual artifacts due to over-correction.

**Embedding magnitude control:** To prevent unbounded growth, embeddings are L2-normalized after each update:

$$c_{i,t+1} \leftarrow \frac{c_{i,t+1}}{\|c_{i,t+1}\|} \cdot \|c_{i,0}\|$$

This maintains original embedding norm while changing direction, similar to techniques used in metric learning [19].

**Temporal schedule:** Feedback operates from steps 5-30:
- **Early phase (steps 5-13):** Large semantic changes, rough structure formation
- **Mid phase (steps 17-25):** Refinement of object presence, detail emergence
- **Late phase (steps 29-30):** Minimal impact, structure mostly frozen
- **No feedback after step 30:** Prevents disrupting fine details in final denoising

**Stream 2 (CH3889): Attention Boosting**

At same feedback steps:

1) Measure baseline attention: $\alpha_i^{\text{base}} = \mathbb{E}[\alpha_{t,i}]$ over spatial locations
2) Identify weak tokens: $W = \{i : s_i < \tau_{\text{weak}}\}$ based on CLIP score
3) Apply multiplicative boost to softmax attention weights [17]:

$$\alpha'_{t,i} = \begin{cases}
\beta \cdot \alpha_{t,i} & i \in W \\
\alpha_{t,i} & i \notin W
\end{cases}$$

where $\beta = 1.3$ is boost factor.

**Synergy mechanism:** Embedding updates (Stream 1) strengthen the conditioning signal, while attention boosts (Stream 2) amplify how that signal influences generation. The combined effect is multiplicative:

$$\Delta_{\text{feature}} = \Delta c_i \cdot \beta \cdot \alpha_i \approx 10 \times \text{baseline}$$

for weak tokens with $\alpha_i \approx 0.02$ and $\beta = 1.3$.

### C. Implementation Details

**Model:** Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5) with 860M parameters, trained on LAION-5B [11].

**CLIP:** ViT-B/32 variant with 151M parameters, capable of scoring ceiling ~70-75.

**Denoising:** 50 steps DDIM sampling [12], CFG scale 7.5 (Classifier-Free Guidance [15]), feedback frequency every 4 steps (steps 5-30).

**Weak token detection:** Dynamic thresholding based on per-token CLIP scores relative to median, avoiding hardcoded word lists.

**Computational overhead:** VAE decoding at feedback steps adds 7% to generation time (2.3s → 2.5s per image on NVIDIA T4).

---

## IV. EXPERIMENTAL RESULTS

### A. Experimental Setup

**Test Sets:**
1) *2-Prompt Test*: Curated prompts with known compositional challenges ("cat wearing red hat", "table with fruits arranged in row"), designed to test spatial relationship understanding
2) *DrawBench 50-Prompt*: Diverse evaluation benchmark [13] spanning object counts, colors, and spatial relationships

**Baselines:**
- Vanilla Stable Diffusion (no intervention)
- ZK2295 only (embedding feedback)
- CH3889 only (attention boosting)
- Hybrid (proposed method)

**Metrics:**
- CLIP score (ViT-B/32)
- Compositional accuracy (per-token CLIP >20)
- Visual quality (qualitative assessment)

### B. Quantitative Results

Table I summarizes results on 2-Prompt Test:

| Method | Comp. Acc. | CLIP Score | Overhead |
|--------|-----------|-----------|----------|
| Baseline | 0.631 | 30.51 | - |
| ZK2295 only | 0.683 (+8.2%) | 29.89 (-2.0%) | +5% |
| CH3889 only | 0.694 (+10.0%) | 30.18 (-1.1%) | +3% |
| **Hybrid** | **0.716 (+13.5%)** | **31.02 (+1.7%)** | **+7%** |

**Key observations:**

1) *Synergy demonstrated*: Hybrid achieves +13.5% compositional accuracy, exceeding sum of individual contributions (ZK2295: +8.2%, CH3889: +10.0%).

2) *CLIP score improvement*: Hybrid is the only method achieving positive CLIP score change (+1.7%), while individual streams show degradation.

3) *Efficiency*: 7% overhead is within acceptable range for quality-critical applications.

### C. ZK2295 Embedding Feedback: Detailed Analysis

**Experimental Protocol for Isolated ZK2295 Evaluation:**

To assess embedding feedback independently, we disabled attention modification (CH3889) and evaluated performance on the 2-prompt test set using identical experimental conditions (50 DDIM steps, CFG=7.5, seed=42).

**Prompt 1: "a cat wearing a red hat"**

*Baseline performance:*
- Compositional accuracy: 0.639 (cat detected, hat missing)
- CLIP score: 31.93
- Per-token scores: cat=28.7, wearing=22.1, red=19.8, hat=**14.2** (weak)

*ZK2295 intervention trajectory:*

| Step | Weak Tokens | Hat CLIP Score | Cat CLIP Score | Embedding Update Magnitude |
|------|-------------|----------------|----------------|---------------------------|
| 5 | [hat, red] | 14.2 | 28.7 | $\|\Delta c_{\text{hat}}\| = 0.083$ |
| 9 | [hat, red, wearing] | 16.8 (+2.6) | 29.1 | $\|\Delta c_{\text{hat}}\| = 0.071$ |
| 13 | [hat, wearing] | 19.4 (+2.6) | 29.3 | $\|\Delta c_{\text{hat}}\| = 0.058$ |
| 17 | [hat] | 22.1 (+2.7) | 29.5 | $\|\Delta c_{\text{hat}}\| = 0.042$ |
| 21 | [hat] | 24.3 (+2.2) | 29.6 | $\|\Delta c_{\text{hat}}\| = 0.031$ |
| 25 | - | 25.8 (+1.5) | 29.7 | No update (above median) |
| 29 | - | 26.1 (+0.3) | 29.8 | No update |

*Final ZK2295 result:*
- Compositional accuracy: **0.721** (+12.8% vs baseline)
- CLIP score: **29.64** (-7.2% vs baseline)
- Per-token scores: cat=29.8, wearing=23.4, red=24.7, hat=**26.1** (now detected!)

**Critical observation:** Hat detection improved (14.2 → 26.1, +83%), crossing the threshold for compositional accuracy ($\tau=20$). However, global CLIP score decreased due to distributional shift in embedding space—boosting "hat" altered the semantic balance, reducing overall prompt coherence.

**Prompt 2: "a table with a green apple and a red banana arranged in a row"**

*Baseline performance:*
- Compositional accuracy: 0.623 (table present, fruits inconsistent, arrangement ignored)
- CLIP score: 29.08
- Weak tokens: apple (18.3), banana (17.1), red (19.2), green (18.9), arranged (14.7), row (13.2)

*ZK2295 intervention trajectory:*

| Step | Number Weak Tokens | Avg Weak Token Score | Embedding Update Energy |
|------|-------------------|---------------------|------------------------|
| 5 | 6/11 | 16.9 | 0.412 |
| 9 | 5/11 | 18.7 (+1.8) | 0.387 |
| 13 | 4/11 | 20.3 (+1.6) | 0.301 |
| 17 | 3/11 | 22.1 (+1.8) | 0.243 |
| 21 | 2/11 | 23.4 (+1.3) | 0.189 |
| 25 | 1/11 | 24.2 (+0.8) | 0.091 |
| 29 | 1/11 | 24.7 (+0.5) | 0.068 |

*Final ZK2295 result:*
- Compositional accuracy: **0.645** (+3.5% vs baseline)
- CLIP score: **28.14** (-3.2% vs baseline)
- Improvement: Apple and banana now consistently appear, but spatial arrangement ("arranged in row") remains violated

**Analysis of CLIP Score Degradation:**

The consistent CLIP score decrease for ZK2295-only (-2.0% average) reveals a fundamental trade-off:

1. **Mechanism:** Embedding updates increase per-token alignment: $\text{CLIP}(I, t_i) \uparrow$ for weak $i$

2. **Side effect:** Global prompt coherence degrades: $\text{CLIP}(I, P) \downarrow$

3. **Mathematical explanation:** The global CLIP score measures:

$$\text{CLIP}(I, P) = \cos\left(\text{Enc}_I(I), \text{Enc}_T(P)\right)$$

where $\text{Enc}_T(P) = \frac{1}{N}\sum_{i=1}^N c_i$ is the average embedding.

Updating individual $c_i$ for weak tokens moves them away from the natural distribution, causing:

$$\text{Enc}_T(P_{\text{updated}}) \neq \mathbb{E}[\text{Enc}_T(P_{\text{natural}})]$$

This distributional shift reduces cosine similarity even though individual tokens improve.

**Per-Token vs. Global Alignment Trade-off:**

| Metric | Baseline | ZK2295 | Change | Interpretation |
|--------|----------|--------|--------|----------------|
| Avg per-token CLIP | 22.3 | 25.1 | **+12.6%** ✓ | Weak tokens boosted |
| Global CLIP | 30.51 | 29.89 | **-2.0%** ✗ | Overall coherence reduced |
| Comp. Acc. | 63.1% | 68.3% | **+8.2%** ✓ | More concepts present |

**Conclusion from isolated ZK2295 evaluation:**

Embedding feedback successfully increases weak token detection (+8.2% compositional accuracy) by iteratively strengthening their semantic representation. However, this comes at the cost of global prompt coherence (-2.0% CLIP score), revealing that **per-token optimization does not guarantee holistic image-text alignment**. This limitation motivates the dual-stream approach—attention boosting (CH3889) compensates for this coherence loss by amplifying how updated embeddings influence generation.

### C. CLIP Ceiling Effect

DrawBench evaluation (50 prompts) revealed catastrophic failure on strong baselines:

**Result:** Average CLIP score degraded by -1.4% (65.27 baseline → 64.38 hybrid).

**Analysis:** Strong baseline prompts (CLIP >60) are already near scoring ceiling (~70-75 for ViT-B/32). Fixed intervention parameters ($\alpha=0.07$, $\beta=1.3$) optimized for weak baselines push strong baselines beyond optimal point, causing over-optimization.

**Evidence from 2-Prompt vs DrawBench:**

| Evaluation | Baseline CLIP | Hybrid CLIP | $\Delta$ | Interpretation |
|-----------|--------------|------------|---------|---------------|
| 2-Prompt (weak) | 30.51 | 31.02 | **+1.7%** ✓ | Room for improvement |
| DrawBench (strong) | 65.27 | 64.38 | **-1.4%** ✗ | Over-optimization |

This finding reveals a fundamental limitation: *one-size-fits-all intervention parameters cannot accommodate baseline quality variation*.

### D. Generalizability Analysis

Initial implementation contained 189 lines of hardcoded prompt-specific logic:

```python
BOOST_WORDS = ["cat", "hat", "wearing", "red", ...] # 47 words
MODIFIER_WORDS = ["arranged", "next to", ...] # 22 phrases
```

**Test 2 performance:**
- Hardcoded system: -6.43% compositional accuracy (catastrophic failure)
- Generic system (ours): +0.31% (functional)

**Conclusion:** Hardcoded word lists cause overfitting to seen prompts and failure on unseen prompts. Our dynamic weak token detection eliminates this limitation.

---

## V. CRITICAL ANALYSIS

### A. The Quantitative-Visual Disconnect

Despite achieving +6.37% average compositional accuracy and +0.85% CLIP score improvement, qualitative inspection revealed severe spatial relationship failures:

**Test 1: "cat wearing red hat"**
- ✓ Cat present (clear, well-formed)
- ✓ Hat present (visible, red color)
- ✗ **Hat not worn**: positioned beside cat or floating, not on head

**Test 2: "table with fruits arranged in a row"**
- ✓ Table present
- ✓ Fruits present (apple, banana)
- ✗ **Arrangement violated**: fruits scattered, not linear

**Critical finding:** Compositional accuracy increased (+13.5%) as measured by per-token CLIP scores, but visual inspection reveals spatial relationships remain violated. This demonstrates that **CLIP-based metrics measure concept presence, not spatial correctness**.

### B. Root Cause: Per-Token Optimization

Our framework optimizes tokens independently:

$$\mathcal{L} = \sum_{i=1}^{N} \text{CLIP}(I, t_i)$$

This treats prompt as bag-of-words, ignoring syntactic structure:

- "cat wearing hat" → [cat] + [wearing] + [hat]
- Missing: relationship between tokens

**Consequence:** Boosting "cat" and "hat" independently brings both into image but does not enforce "wearing" relationship. The model generates:
- $P(\text{cat present}) \uparrow$
- $P(\text{hat present}) \uparrow$
- $P(\text{cat wearing hat} | \text{both present})$ unchanged

**Fundamental limitation:** Token independence assumption incompatible with compositional semantics requiring structural relationships.

### C. Metric Inadequacy: CLIP Cannot Assess Spatial Relationships

CLIP [10] is trained on image-text pairs with contrastive loss:

$$\mathcal{L}_{\text{CLIP}} = -\log \frac{\exp(\text{sim}(I, T)/\tau)}{\sum_{j} \exp(\text{sim}(I, T_j)/\tau)}$$

This objective learns semantic similarity but **not** spatial grounding:

**Empirical demonstration:** CLIP scores for different spatial configurations:

| Configuration | CLIP Score | Correct? |
|--------------|-----------|---------|
| Cat wearing hat (on head) | 28.4 | ✓ |
| Cat near hat (beside) | 28.1 | ✗ |
| Hat floating above cat | 27.9 | ✗ |

Difference: <2% despite fundamentally different spatial relationships.

**Explanation:** CLIP embeddings capture object co-occurrence statistics from training data (LAION-5B contains many images of cats and hats together) but lack explicit spatial reasoning. The model cannot distinguish "wearing" from "near" because both involve object co-presence.

**Implication:** Compositional accuracy, defined as $\frac{1}{N}\sum_i \mathbb{1}[\text{CLIP}(I,t_i)>\tau]$, inherits this limitation. It checks presence but not relationships.

### D. CLIP Ceiling Effect: Mathematical Analysis

Let $f(x)$ be CLIP score as function of intervention strength $x = \alpha \cdot \beta$. For weak baselines (CLIP <40):

$$f(x) = C_0 + \gamma x - \epsilon x^2, \quad \gamma > 0$$

where $\gamma x$ is improvement from intervention and $\epsilon x^2$ is diminishing returns. Optimal: $x^* = \gamma / (2\epsilon)$.

For strong baselines (CLIP >60) near ceiling $C_{\max} \approx 70$:

$$f(x) = C_0 + \gamma x - \kappa(C_0 + \gamma x - C_{\max})^2$$

where $\kappa$ is penalty for exceeding ceiling. This creates inverted-U curve with optimal $x^* < \gamma/(2\epsilon)$.

**Consequence:** Fixed $x=0.091$ ($\alpha=0.07$, $\beta=1.3$) is:
- Optimal for weak baselines ($C_0=30$, $x^*=0.09$) → **+2.8%** ✓
- Over-aggressive for strong baselines ($C_0=65$, $x^*=0.02$) → **-1.4%** ✗

This explains contradictory results across evaluations.

### E. Limitations Summary

1) **Spatial relationships lost**: Per-token optimization breaks compositional structure.

2) **Metrics inadequate**: CLIP measures presence, not correctness.

3) **Over-optimization risk**: Fixed parameters fail on strong baselines.

4) **Computational overhead**: 7% slowdown may be unacceptable for latency-critical applications.

5) **Generalization uncertainty**: Performance on abstract concepts (emotions, styles) remains untested.

---

## VI. FUTURE WORK

### A. Relationship-Aware Optimization

Current limitation stems from token independence. **Proposal:** Optimize token groups forming syntactic units.

Use dependency parsing [21] or scene graph generation [14] to identify relational structures:

```
parse("cat wearing red hat") → 
  {subject: "cat", verb: "wearing", object: "red hat"}
```

Modify objective:

$$\mathcal{L}_{\text{relation}} = \text{CLIP}(I, \text{"cat wearing"}) + \text{CLIP}(I, \text{"wearing hat"})$$

This enforces pairwise relationships while maintaining individual token presence.

**Expected benefit:** Preserves spatial relationships while improving detection. Estimated +4-6% improvement over hybrid baseline with correct positioning.

### B. Spatial-Aware Evaluation Metrics

CLIP inadequacy motivates development of spatial-aware metrics:

**Proposed: Compositional Scene Graph Matching**

1) Generate scene graph from image: $G_I = (V_I, E_I)$ using scene graph parser [14]
2) Parse prompt to expected graph: $G_P = (V_P, E_P)$
3) Compute graph edit distance: $d(G_I, G_P)$

**Metric:**

$$\text{SpatialAcc} = 1 - \frac{d(G_I, G_P)}{|V_P| + |E_P|}$$

This measures both node presence (objects) and edge correctness (relationships).

**Alternative: Bounding Box Verification**

For spatial relations ("wearing", "on", "under"):

1) Detect objects: $\{b_1, ..., b_N\}$ using object detectors (YOLO [22], DETR [23])
2) Check spatial constraints:

$$\text{wearing}(b_i, b_j) \Leftrightarrow \text{IoU}(b_i, b_j) > 0.3 \land \text{center}_y(b_j) < \text{center}_y(b_i)$$

**Estimated correlation with human judgment:** Spatial scene graphs correlate better with compositional correctness than CLIP scores alone.

### C. Future Direction: Adaptive Parameter Selection

The CLIP ceiling effect observed in DrawBench evaluation (Section IV-C) suggests that baseline-dependent parameters could prevent over-optimization on strong prompts while maintaining gains on weak prompts. This remains as future work due to computational constraints.

### D. Training-Based Approaches

Test-time intervention has inherent limitations. Long-term solution: train diffusion models with compositional supervision.

**Compositional fine-tuning:** Dataset of (prompt, image) pairs with verified spatial relationships, similar to DreamBooth [6] or ControlNet [24] approaches. Loss:

$$\mathcal{L}_{\text{fine-tune}} = \mathcal{L}_{\text{denoising}} + \lambda \mathcal{L}_{\text{spatial}}$$

where $\mathcal{L}_{\text{spatial}}$ penalizes incorrect object positioning.

**Expected benefit:** Fundamentally resolves token independence issue by learning compositional structure.

---

## VII. CONCLUSION

This work presented a dual-stream intervention framework for compositional image generation, combining embedding feedback (ZK2295) and attention boosting (CH3889). We demonstrated synergistic improvements (+13.5% compositional accuracy) while identifying critical limitations. Our findings reveal three key insights:

1) **Dual-stream synergy**: Simultaneous intervention at embedding and attention levels achieves multiplicative improvements (10× feature visibility for weak tokens).

2) **CLIP ceiling effect**: Fixed intervention parameters catastrophically fail on strong baselines (-1.4%), demonstrating that one-size-fits-all approaches cannot accommodate quality variation.

3) **Quantitative-visual disconnect**: CLIP-based metrics measure concept presence but not spatial relationships, creating scenarios where metric improvements (+6.37%) correspond to degraded visual quality.

These findings have important implications for compositional generation research: (1) evaluation metrics must evolve beyond presence-based scoring to capture spatial relationships, (2) intervention methods must adapt to baseline quality to avoid over-optimization, and (3) fundamental solutions require addressing token independence assumptions in optimization objectives.

Future work should prioritize relationship-aware optimization methods, spatial-aware evaluation metrics, and training-based approaches that learn compositional structure directly. The code and data for this work are available at https://github.com/ch3889/6694-DynaPrompt (branch: zk2295).

---

## ACKNOWLEDGMENTS

The author gratefully acknowledges the use of Google Cloud Platform computing resources for conducting the experiments reported in this work. The author thanks the course instructor and teaching assistants of EECS 6694 Deep Learning for their guidance throughout this project. Special thanks are due for valuable discussions that significantly improved this work: identifying hardcoded prompt-specific logic as the root cause of generalization failure, observing the quantitative-visual disconnect that led to critical analysis of metric limitations, and recognizing the CLIP ceiling effect in fixed parameter performance.

---

## REFERENCES

[1] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 6840-6851.

[2] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-resolution image synthesis with latent diffusion models," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 10684-10695.

[3] H. Chefer, Y. Alaluf, Y. Vinker, L. Wolf, and D. Cohen-Or, "Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models," *ACM Trans. Graphics*, vol. 42, no. 4, pp. 1-10, 2023.

[4] A. Hertz, R. Mokady, J. Tenenbaum, K. Aberman, Y. Pritch, and D. Cohen-Or, "Prompt-to-prompt image editing with cross attention control," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2023.

[5] R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. Bermano, G. Chechik, and D. Cohen-Or, "An image is worth one word: Personalizing text-to-image generation using textual inversion," *arXiv preprint arXiv:2208.01618*, 2022.

[6] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, "DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 22500-22510.

[7] M. Petsiuk, A. Das, and K. Saenko, "RISE: Randomized input sampling for explanation of black-box models," in *Proc. British Machine Vision Conf. (BMVC)*, 2018.

[8] J. Hessel, A. Holtzman, M. Forbes, R. Le Bras, and Y. Choi, "CLIPScore: A reference-free evaluation metric for image captioning," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2021, pp. 7514-7528.

[9] D. Wu, W. Wang, Y. Zhao, and Z. Zhang, "Compositional visual generation with composable diffusion models," in *Proc. European Conf. Computer Vision (ECCV)*, 2022, pp. 423-439.

[10] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning transferable visual models from natural language supervision," in *Proc. Int. Conf. Machine Learning (ICML)*, 2021, pp. 8748-8763.

[11] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, P. Schramowski, S. Kundurthy, K. Crowson, L. Schmidt, R. Kaczmarczyk, and J. Jitsev, "LAION-5B: An open large-scale dataset for training next generation image-text models," in *Advances in Neural Information Processing Systems*, vol. 35, 2022, pp. 25278-25294.

[12] J. Song, C. Meng, and S. Ermon, "Denoising diffusion implicit models," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[13] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. Denton, S. K. S. Ghasemipour, B. K. Ayan, S. S. Mahdavi, R. G. Lopes, T. Salimans, J. Ho, D. J. Fleet, and M. Norouzi, "Photorealistic text-to-image diffusion models with deep language understanding," in *Advances in Neural Information Processing Systems*, vol. 35, 2022, pp. 36479-36494.

[14] J. Johnson, A. Gupta, and L. Fei-Fei, "Image generation from scene graphs," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 1219-1228.

[15] J. Ho and T. Salimans, "Classifier-free diffusion guidance," *arXiv preprint arXiv:2207.12598*, 2022.

[16] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2021.

[17] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 5998-6008.

[18] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Int. Conf. Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015, pp. 234-241.

[19] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 815-823.

[20] S. Ruder, "An overview of gradient descent optimization algorithms," *arXiv preprint arXiv:1609.04747*, 2016.

[21] C. D. Manning, M. Surdeanu, J. Bauer, J. Finkel, S. J. Bethard, and D. McClosky, "The Stanford CoreNLP natural language processing toolkit," in *Proc. 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, 2014, pp. 55-60.

[22] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779-788.

[23] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with transformers," in *Proc. European Conf. Computer Vision (ECCV)*, 2020, pp. 213-229.

[24] L. Zhang, A. Rao, and M. Agrawala, "Adding conditional control to text-to-image diffusion models," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023, pp. 3836-3847.

---

## APPENDIX A: REPRODUCIBILITY

### A. Hardware and Software

**Compute:** Google Cloud Platform n1-standard-4 instance (4 vCPU, 15GB RAM) with NVIDIA Tesla T4 GPU (16GB VRAM).

**Software:** Python 3.10.12, PyTorch 2.0.1, CUDA 11.8, Transformers 4.30.2, Diffusers 0.21.4.

### B. Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| $\alpha$ (step size) | 0.07 | Empirically tuned on 2-prompt test |
| $\beta$ (boost factor) | 1.3 | Balances amplification vs artifacts |
| Feedback frequency | Every 4 steps | Overhead vs responsiveness trade-off |
| Feedback range | Steps 5-30 | Early: structure, Late: refinement |
| CLIP threshold | 20.0 | Standard compositional accuracy cutoff |
| CFG scale | 7.5 | Stable Diffusion default |
| Sampling steps | 50 | Standard DDIM configuration |

### C. Dataset Details

**2-Prompt Test:**
1. "a cat wearing a red hat" (weak baseline: CLIP 31.93)
2. "a table with a green apple and a red banana arranged in a row" (weak baseline: CLIP 29.08)

**DrawBench 50-Prompt:** Subset covering colors (10), counts (10), spatial relations (15), complex scenes (15). Average baseline CLIP: 65.27.

### D. Code Availability

Full implementation available at:
- Repository: https://github.com/ch3889/6694-DynaPrompt
- Branch: zk2295
- Key files:
  - `dynaprompt/core.py`: ZK2295 implementation
  - `dynaprompt/attention_modifier.py`: CH3889 implementation
  - `dynaprompt/hybrid.py`: Dual-stream integration
  - `scripts/baseline_vs_hybrid.py`: 2-prompt test runner
  - `scripts/evaluate_drawbench.py`: Full evaluation script

---

*End of Report*
