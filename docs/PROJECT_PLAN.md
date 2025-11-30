# DynaPrompt Project Plan

**Course:** EECS 6694 Deep Learning
**Team:** Charles Hou (ch3889), Max Kim (zk2295), Swapnil Banerjee (sb5041)
**Presentation:** December 2, 2025
**Timeline:** 5 weeks (Oct 25 - Dec 2)

---

## Project Overview

DynaPrompt is a dynamic prompt guidance system for text-to-image diffusion models that uses real-time CLIP feedback to maintain semantic alignment during image generation.

**Key Innovation:** External feedback loop that adaptively re-weights prompt embeddings at intermediate denoising steps.

---

## Timeline & Milestones

| Week | Dates | Phase | Deliverables |
|------|-------|-------|-------------|
| 1 | Oct 25 - Nov 1 | Setup & Infrastructure | Working SD + CLIP + 200 prompts |
| 2 | Nov 2 - Nov 9 | Core Implementation | DynaPrompt feedback loop working |
| 3 | Nov 10 - Nov 17 | Baselines & Experiments | All methods running on 200 prompts |
| 4 | Nov 18 - Nov 25 | Evaluation & Ablations | All metrics + human eval complete |
| 5 | Nov 26 - Dec 2 | Analysis & Report | Final report + presentation |

---

## Week 1: Setup & Infrastructure (Oct 25 - Nov 1)

### Status: ✅ IN PROGRESS

### Team Tasks (Everyone)
- [x] Environment setup (conda, dependencies)
- [x] Repository structure created
- [x] CompVis Stable Diffusion working
- [x] SD v1.5 weights downloaded
- [x] GCP instance with T4 GPU configured

### Charles - Infrastructure Lead
**Tasks:**
- [x] Get CompVis SD working on both Mac and GCP
- [x] Fix cross-platform compatibility (MPS + CUDA)
- [ ] Generate 10 test images with varied prompts
- [ ] Create inference wrapper: `models/stable_diffusion_compvis/inference.py`
- [ ] Document generation performance (time per image)
- [ ] Create test notebook: `notebooks/01_environment_test.ipynb`

**Deliverable:** Clean SD inference pipeline ready for integration

### Max - Core Algorithm Lead
**Tasks:**
- [ ] Install and test CLIP on sample images
- [ ] Implement `dynaprompt/feedback.py`:
  - Load CLIP model (ViT-B/32)
  - Tokenize prompt into individual tokens
  - Compute per-token image-text similarity
  - Identify underrepresented tokens (similarity < threshold)
- [ ] Create demo: `notebooks/02_clip_feedback_demo.ipynb`
- [ ] Test on 10 generated images from Charles

**Deliverable:** CLIP per-token similarity scores working

### Swapnil - Data Lead
**Tasks:**
- [ ] Download COCO 2017 validation set (~1GB)
- [ ] Implement `scripts/prepare_dataset.py`:
  - Extract captions using pycocotools
  - Filter for compositional prompts:
    - 3+ distinct objects
    - Color/size attributes
    - Spatial relations (on, next to, behind)
    - Action verbs
- [ ] Curate 200 prompts → `data/prompts/coco_curated_200.json`
- [ ] Create 20 test prompts → `data/prompts/test_20.json`
- [ ] Document dataset statistics in notebook

**Deliverable:** 200 diverse, compositional prompts ready

### Week 1 Integration Test (Nov 1)
1. Swapnil provides 1 test prompt
2. Charles generates image with SD
3. Max computes CLIP per-token similarity
4. Identify which tokens are underrepresented

---

## Week 2: Core DynaPrompt Implementation (Nov 2 - Nov 9)

### Goal: DynaPrompt generating images with real-time feedback

### Charles - Infrastructure Lead
**Tasks:**
- [ ] Study CompVis sampling loop in `ldm/models/diffusion/ddim.py`
- [ ] Identify injection points for feedback
- [ ] Implement `dynaprompt/prompt_updater.py`:
  - Token re-weighting based on CLIP scores
  - Embedding normalization (L2, softmax)
  - Handle edge cases
- [ ] Create wrapper that integrates updater with SD

**Deliverable:** Prompt updater module working

### Max - Core Algorithm Lead
**Tasks:**
- [ ] Implement `dynaprompt/controller.py`:
  - Main feedback loop
  - Call SD sampler with custom hooks
  - At every N steps:
    - Decode latent to image
    - Get CLIP feedback
    - Update prompt embedding
    - Continue sampling
- [ ] Add logging and debugging
- [ ] Test on 20 prompts from Swapnil
- [ ] Tune hyperparameters:
  - Feedback frequency (5, 10, 20 steps)
  - Update step size (0.1, 0.3, 0.5)

**Deliverable:** Full DynaPrompt pipeline working

### Swapnil - Data & Integration Lead
**Tasks:**
- [ ] Test DynaPrompt on diverse prompts
- [ ] Debug integration issues
- [ ] Document failures and edge cases
- [ ] Create comparison notebook showing:
  - Static SD output
  - DynaPrompt output
  - CLIP similarity curves

**Deliverable:** DynaPrompt tested on 20+ prompts

### Week 2 Deliverable
Working DynaPrompt system generating images with adaptive feedback

---

## Week 3: Baselines & Full Experiments (Nov 10 - Nov 17)

### Goal: Run all methods on 200 prompts

### Charles - Baselines Lead
**Tasks:**
- [ ] Implement `baselines/static_prompt.py`:
  - Vanilla SD with fixed prompt
- [ ] Implement `baselines/dynamic_cfg.py`:
  - Adaptive CFG schedule (linear, cosine)
- [ ] Implement `baselines/prompt_rewrite.py`:
  - Generate → check CLIP → retry if low
- [ ] Test each baseline on 10 prompts

**Deliverable:** 3 baseline methods working

### Max - Experiments Lead
**Tasks:**
- [ ] Create `experiments/run_all.py`:
  - Loop over 200 prompts
  - Run each method (DynaPrompt + 3 baselines)
  - Save images to organized folders
  - Log generation times
- [ ] Run full experiment suite (~800 images)
- [ ] Optimize batch processing
- [ ] Monitor and restart if failures

**Critical:** This generates 200 prompts × 4 methods = 800 images
**Estimated time:** 800 images × 30 sec/image = 6-7 GPU hours

**Deliverable:** All 800 images generated

### Swapnil - Evaluation Prep
**Tasks:**
- [ ] Implement `evaluation/metrics.py`:
  - CLIPScore (ViT-B/32, ViT-L/14)
  - FID score (need reference images)
  - Generation time tracking
- [ ] Implement `evaluation/compositional_accuracy.py`:
  - Load BLIP-2
  - Caption all generated images
  - Extract objects/attributes
  - Compute recall vs prompt tokens
- [ ] Download COCO reference images for FID

**Deliverable:** Evaluation pipeline ready

### Week 3 Deliverable
800 images generated, ready for evaluation

---

## Week 4: Evaluation & Ablations (Nov 18 - Nov 25)

### Goal: Complete all metrics and ablations

### Charles - Metrics Computation
**Tasks:**
- [ ] Run all metrics on 800 images:
  - CLIPScore for each image
  - FID per method
  - Compositional accuracy per method
  - Generation time statistics
- [ ] Save results to `results/metrics/`:
  - `clip_scores.csv`
  - `fid_scores.csv`
  - `compositional_accuracy.csv`
  - `generation_times.csv`
- [ ] Create comparison tables

**Deliverable:** All metrics computed

### Max - Ablation Studies
**Tasks:**
- [ ] Implement `experiments/ablation_studies.py`
- [ ] Run ablations (each on 50 prompts):
  - **Ablation 1:** Feedback frequency (5, 10, 20 steps)
  - **Ablation 2:** Update step size (0.1, 0.3, 0.5)
  - **Ablation 3:** CLIP model (ViT-B/32 vs L/14)
- [ ] Compute metrics for each ablation
- [ ] Analyze which settings work best

**Deliverable:** Ablation results showing optimal settings

### Swapnil - Visualization & Human Eval
**Tasks:**
- [ ] Implement `evaluation/alignment_curves.py`:
  - Plot CLIP similarity over denoising steps
  - Compare static vs DynaPrompt trajectories
- [ ] Create visualizations:
  - Bar charts comparing methods
  - Scatter plots (CLIP vs FID)
  - Qualitative comparisons (grid of images)
- [ ] Setup human evaluation:
  - Select 50 diverse prompts
  - Create pairwise comparison interface
  - Recruit 5-10 evaluators
  - Run evaluation (DynaPrompt vs baselines)
  - Collect and analyze votes

**Deliverable:** All plots + human eval results

### Week 4 Deliverable
Complete evaluation results + visualizations

---

## Week 5: Analysis & Report (Nov 26 - Dec 2)

### Goal: Final report and presentation ready

### All Team Members - Report Writing (Nov 26-30)

**Charles:**
- [ ] Write: Introduction
- [ ] Write: Related Work
- [ ] Write: Method Description
- [ ] Create method diagrams/figures

**Max:**
- [ ] Write: Experimental Setup
- [ ] Write: Results section
- [ ] Write: Ablation Studies
- [ ] Create results tables

**Swapnil:**
- [ ] Write: Analysis & Discussion
- [ ] Write: Limitations
- [ ] Write: Conclusion
- [ ] Format references

**Shared:**
- [ ] Create figures (architecture, results plots)
- [ ] Iterate on draft (2-3 rounds)
- [ ] Proofread and polish

### Presentation Prep (Nov 30 - Dec 2)
- [ ] Create slides (15-20 minutes)
- [ ] Prepare demo video or live demo
- [ ] Practice presentation (each person's part)
- [ ] Prepare Q&A responses
- [ ] Final rehearsal

### Dec 2: Presentation Day

---

## Technical Architecture

### DynaPrompt System Design

```
┌─────────────────────────────────────────────────┐
│           DynaPrompt Controller                 │
│  (dynaprompt/controller.py)                     │
└─────────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│  Stable Diffusion│    │  CLIP Feedback   │
│  Sampling Loop   │◄───┤  Module          │
│                  │    │  (feedback.py)   │
└──────────────────┘    └──────────────────┘
        │                         ▲
        │ Intermediate Image      │
        └─────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Prompt Updater        │
        │  (prompt_updater.py)   │
        │  - Re-weight tokens    │
        │  - Normalize embedding │
        └────────────────────────┘
```

### Key Injection Points in CompVis SD

**File:** `ldm/models/diffusion/ddim.py`
**Function:** `DDIMSampler.sample()` or `p_sample_ddim()`

**Pseudocode:**
```python
def sample_with_dynaprompt(self, steps, prompt, ...):
    for t in timesteps:
        # Standard denoising
        noise_pred = unet(x_t, t, text_embedding)
        x_t_minus_1 = denoise_step(x_t, noise_pred)

        # DynaPrompt feedback (every N steps)
        if t % feedback_freq == 0 and feedback_enabled:
            # Decode to pixel space
            img = vae_decode(x_t_minus_1)

            # Get CLIP feedback
            token_scores = clip_feedback(img, prompt_tokens)

            # Update embedding
            text_embedding = update_embedding(
                text_embedding,
                token_scores,
                alpha=0.3
            )

    return x_t_minus_1
```

---

## Evaluation Metrics

### Quantitative Metrics

1. **CLIPScore** (Higher = Better)
   - Measures text-image alignment
   - Computed per image using CLIP ViT-B/32 or L/14
   - Range: 0-1

2. **FID Score** (Lower = Better)
   - Measures image quality vs real distribution
   - Compare against COCO validation images
   - Typical range: 10-30 (lower is better)

3. **Compositional Accuracy** (Higher = Better)
   - Use BLIP-2 to caption generated images
   - Extract mentioned objects/attributes
   - Compute recall: (detected objects) / (prompt objects)
   - Range: 0-1

4. **Generation Time** (Lower = Better)
   - Seconds per image
   - Measure overhead of DynaPrompt vs baselines

### Qualitative Metrics

1. **Alignment Curves**
   - Plot CLIP similarity over denoising timesteps
   - Compare trajectories: static vs DynaPrompt
   - Show where DynaPrompt corrects drift

2. **Human Evaluation**
   - 50 prompts × 2 methods = 100 pairwise comparisons
   - Ask: "Which image better matches the text?"
   - Compute win rate for DynaPrompt

---

## Dataset Specification

### Source
COCO 2017 Validation Set

### Filtering Criteria
Select captions with:
- **Multi-object:** ≥3 distinct nouns
- **Attributes:** Colors, sizes, materials
- **Spatial relations:** "on", "next to", "behind", "in front of"
- **Actions:** Verbs describing activities
- **Counting:** "two cats", "three balls"

### Categories (distribute 200 prompts across):
- 50 prompts: Multi-object scenes
- 50 prompts: Attribute binding (color + object)
- 50 prompts: Spatial relationships
- 50 prompts: Complex (all above)

### Example Prompts
```json
{
  "multi_object": "A golden retriever playing with a red ball in a snowy park",
  "attribute_binding": "A large blue truck parked next to a small red car",
  "spatial": "A cat sitting on top of a wooden table behind a vase",
  "complex": "Two birds flying over a green lake with mountains in the background"
}
```

---

## Baseline Methods

### 1. Static Prompt (Vanilla SD)
- No feedback
- Fixed prompt embedding throughout sampling
- Serves as main comparison baseline

### 2. Dynamic CFG
- Adaptive classifier-free guidance schedule
- Varies guidance scale over timesteps
- Schedule types: linear, cosine
- Shows if just adjusting guidance helps

### 3. Prompt Rewrite
- Generate image
- Compute CLIP score
- If score < threshold: regenerate with same prompt
- Max 3 retries
- Shows value of real-time vs post-hoc feedback

---

## Computing Resources

### GCP Setup
- **Instance:** n1-standard-4 with 1× NVIDIA T4
- **Cost:** ~$0.35/hour
- **Budget:** $100 = ~285 GPU hours
- **Estimated usage:**
  - Development/debugging: 20 hours
  - Main experiments (800 images): 7 hours
  - Ablations (400 images): 4 hours
  - Re-runs/fixes: 10 hours
  - **Total:** ~41 hours = ~$15

### Backup: Google Colab Pro
- If GCP credits run low
- ~$10/month
- T4 GPU access

---

## Success Criteria

### Minimum Viable Product (MVP)
- DynaPrompt working on 100 prompts
- 1 baseline (static prompt)
- 2 metrics (CLIPScore + compositional accuracy)
- Basic report (5-6 pages)

### Target Deliverables
- DynaPrompt on 200 prompts
- 3 baselines
- 4 metrics + alignment curves
- Human evaluation
- Full report (8-10 pages)

### Stretch Goals
- Both CompVis + Diffusers implementations
- Additional CLIP models tested
- More extensive ablations
- Interactive demo

---

## Risk Mitigation

### Risk 1: CLIP doesn't provide useful feedback
**Mitigation:** Test early (Week 1), have fallback to attention-based methods

### Risk 2: Compute budget insufficient
**Mitigation:** Reduce to 100 prompts, 2 baselines, use Colab Pro backup

### Risk 3: Integration complexity
**Mitigation:** Start simple, iterate, allocate extra debugging time

### Risk 4: Time crunch
**Mitigation:** Prioritize core method over fancy ablations, cut scope if needed

---

## Current Status

**As of Oct 26, 2025:**

### Completed ✅
- Repository structure created
- CompVis Stable Diffusion installed and tested
- Cross-platform compatibility (Mac MPS + GCP CUDA)
- SD v1.5 weights downloaded
- GCP instance with T4 GPU configured
- SETUP.md documentation created
- Initial commits pushed to GitHub

### In Progress 🔄
- Week 1 tasks for Charles, Max, Swapnil
- Testing SD generation on GCP

### Next Immediate Steps
1. Charles: Generate 10 test images on GCP
2. Max: Install CLIP and test per-token similarity
3. Swapnil: Download COCO and start prompt curation
4. All: Weekly sync meeting to review progress

---

## Contact & Resources

**Team:**
- Charles Hou: ch3889@columbia.edu
- Max Kim: zk2295@columbia.edu
- Swapnil Banerjee: sb5041@columbia.edu

**Repository:** https://github.com/ch3889/6694-DynaPrompt

**Documentation:**
- `README.md` - Project overview
- `SETUP.md` - Setup instructions
- `docs/PROPOSAL.md` - Original proposal
- `patches/compvis_mac_compatibility.patch` - Platform fixes

**References:**
- CompVis SD: https://github.com/CompVis/stable-diffusion
- CLIP: https://github.com/openai/CLIP
- COCO Dataset: https://cocodataset.org/
- HuggingFace Diffusers: https://huggingface.co/docs/diffusers
