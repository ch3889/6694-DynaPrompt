# DynaPrompt: Code vs Proposal Gap Analysis

## ✅ IMPLEMENTED (Matching Proposal)

1. **Real-time feedback loop during denoising** ✓
2. **CLIP semantic similarity computation** ✓
3. **Dynamic prompt embedding updates** ✓
4. **Model-agnostic external feedback** ✓
5. **Closed-loop sampling system** ✓
6. **CLIPScore metric** ✓
7. **FID Score** ✓ (implemented but needs real images)
8. **Generation time tracking** ✓
9. **BLIP-2 compositional accuracy** ✓ (basic implementation)

## ❌ MISSING (From Proposal)

### Critical Missing Features:

1. **Per-Token Underrepresentation Detection**
   - Proposal: "Detect underrepresented concepts (e.g., 'red ball')"
   - Current: Only global CLIP score, no per-token analysis
   - **Need to add**: Cross-attention map analysis or per-token CLIP scoring

2. **Selective Token Re-weighting**
   - Proposal: "Re-weight corresponding token embeddings"
   - Current: Uniform update to entire embedding
   - **Need to add**: Identify weak tokens and boost only those

3. **Attention Map Analysis**
   - Proposal: Related to Attend-and-Excite approach
   - Current: Not using U-Net cross-attention maps
   - **Need to add**: Extract attention maps to see which tokens are ignored

### Missing Metrics:

4. **ImageReward**
   - Proposal explicitly mentions this
   - Current: Not implemented
   - **Need to add**: Install and integrate ImageReward model

5. **Human Evaluation Framework**
   - Proposal: "Ask participants to choose which image matches text"
   - Current: None
   - **Need to add**: A/B testing interface or survey

### Missing Experiments:

6. **Ablation Studies**
   - Proposal: "Varying feedback frequency (5, 10, 20 steps)"
   - Current: Hardcoded to 5 steps
   - **Need to add**: Scripts to run with different configs

7. **Alignment Curves**
   - Proposal: "Plot CLIP similarity evolution through denoising"
   - Current: Metrics tracked but not visualized
   - **Need to add**: Plotting scripts

8. **Baseline Comparisons**
   - Proposal: Static prompt, dynamic CFG, prompt rewrite
   - Current: Only static baseline comparison
   - **Need to add**: Dynamic CFG and prompt rewrite baselines

### Missing Dataset:

9. **COCO 2017 Prompts**
   - Proposal: "200 curated COCO prompts for compositional understanding"
   - Current: Single test prompts
   - **Need to add**: Dataset loader with COCO prompts

## 🔧 IMPLEMENTATION PRIORITIES

### Phase 1: Core Algorithm (Critical)
These directly affect whether your method works as proposed:

1. **Per-token CLIP scoring**
   ```python
   # Instead of one global score, compute score per token
   def compute_per_token_alignment(self, image, prompt_tokens):
       scores = []
       for token in prompt_tokens:
           score = self.compute_clipscore(image, token)
           scores.append(score)
       return scores  # Find which tokens have low scores
   ```

2. **Selective token re-weighting**
   ```python
   # Only update embeddings for underrepresented tokens
   def selective_update(self, embedding, token_scores, threshold=0.7):
       weak_tokens = [i for i, score in enumerate(token_scores) if score < threshold]
       # Boost only weak_tokens in embedding
       for idx in weak_tokens:
           embedding[0, idx, :] *= 1.2  # Increase weight
   ```

3. **Cross-attention extraction**
   ```python
   # Extract which tokens U-Net is attending to
   def extract_attention_maps(self, unet, timestep):
       # Hook into cross-attention layers
       # Identify which tokens have low attention
   ```

### Phase 2: Metrics & Evaluation

4. **ImageReward**
   ```bash
   pip install image-reward
   ```
   ```python
   from ImageReward import ImageReward
   model = ImageReward.load("ImageReward-v1.0")
   reward = model.score(prompt, image)
   ```

5. **Alignment curve plotting**
   ```python
   import matplotlib.pyplot as plt
   plt.plot(steps, clip_scores)
   plt.xlabel("Denoising Step")
   plt.ylabel("CLIP Score")
   plt.title("Semantic Alignment Evolution")
   ```

6. **COCO dataset loader**
   ```python
   from pycocotools.coco import COCO
   coco = COCO('coco/annotations/captions_val2017.json')
   prompts = [coco.loadAnns(ann_id)[0]['caption'] for ann_id in selected_ids]
   ```

### Phase 3: Experiments

7. **Ablation script**
   ```python
   for freq in [5, 10, 20]:
       results = run_dynaprompt_generation(
           prompt=prompt,
           feedback_frequency=freq
       )
       save_results(freq, results)
   ```

8. **Baseline implementations**
   - Dynamic CFG: Vary cfg_scale over timesteps
   - Prompt rewrite: Re-run with modified prompt if CLIP < threshold

9. **Human evaluation interface**
   - Generate image pairs (baseline vs DynaPrompt)
   - Create survey asking which is better
   - Collect responses and compute preference percentage

## 📊 WHAT YOU CAN REPORT NOW

Even without the missing features, you can still report:

✓ **Working proof-of-concept**: Real-time feedback loop functional  
✓ **CLIP score improvement**: Show trajectory from your test runs  
✓ **Generation time**: CPU vs GPU comparison  
✓ **Qualitative results**: Visual comparison of baseline vs DynaPrompt  
✓ **Implementation details**: Architecture and integration approach  

## 🎯 MINIMUM VIABLE FOR PROPOSAL MATCH

To claim your code "reflects the proposal", you MUST add:

1. **Per-token underrepresentation detection** (core claim)
2. **Selective re-weighting** (core claim)
3. **ImageReward metric** (explicitly promised)
4. **COCO prompts evaluation** (explicitly promised)
5. **Ablation studies** (explicitly promised)

## 🚀 QUICK WINS (Add in 1-2 hours)

1. **Add ImageReward**: `pip install image-reward` + 10 lines of code
2. **Plot alignment curves**: Use matplotlib on existing metrics_history
3. **Test multiple feedback frequencies**: Just change config and re-run
4. **Download 50 COCO prompts**: Quick API call to COCO dataset

## 📝 RECOMMENDATION

Your current implementation is a **strong foundation** but only implements ~60% of the proposal. The **core feedback loop works**, but you're missing the **token-level granularity** that differentiates DynaPrompt from naive CLIP-based guidance.

**Priority order**:
1. Add per-token scoring (critical for proposal claims)
2. Add selective re-weighting (critical for proposal claims)
3. Add ImageReward (explicitly promised)
4. Run ablations (quick to do)
5. Add COCO evaluation (dataset work)
6. Human eval (time-consuming, can be last)

**Time estimate**:
- Phase 1 (core algorithm): 4-6 hours
- Phase 2 (metrics): 2-3 hours  
- Phase 3 (experiments): 3-4 hours
- **Total**: ~10-15 hours to full proposal match
