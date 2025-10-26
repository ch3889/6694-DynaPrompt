# DynaPrompt: Dynamic Prompt Guidance for Text-to-Image Diffusion Models

**EECS 6694 Deep Learning Project**
**Team Members:** Charles Chaoyu Hou (ch3889), Max Zishock Kim (zk2295), Swapnil Banerjee (sb5041)
**Presentation Date:** December 2, 2025

---

## Overview

Text-to-image diffusion models like Stable Diffusion can generate highly realistic images but often fail to maintain semantic alignment with text prompts throughout the generation process. DynaPrompt addresses this limitation by introducing a **real-time feedback loop** that dynamically adjusts prompt embeddings during denoising, ensuring continuous semantic fidelity.

### The Problem

Given the prompt *"A golden retriever playing with a red ball in a snowy park"*, current diffusion models might:
- Omit the red ball entirely
- Generate the wrong color ball
- Misrepresent the snowy environment

This happens because prompt conditioning remains **fixed** after sampling begins, causing gradual semantic drift.

### Our Solution

DynaPrompt implements a **closed-loop feedback system**:

1. At intermediate denoising steps, evaluate the partially generated image
2. Use CLIP to compute semantic similarity between image and text tokens
3. Detect underrepresented concepts (e.g., "red ball", "snowy park")
4. Adaptively re-weight prompt embeddings to emphasize missing elements
5. Continue sampling with updated conditioning

Think of it as a painter continuously checking the original description while creating the artwork.

---

## Key Contributions

- **Feedback-Driven**: Transforms diffusion sampling into a closed-loop system with real-time semantic correction
- **Model-Agnostic**: Works externally without retraining or architectural modifications
- **Semantic Generalization**: Handles both global and fine-grained prompt semantics beyond attention-based methods

---

## Repository Structure

```
DynaPrompt/
├── configs/                          # Configuration files
├── models/                           # Stable Diffusion implementations
│   ├── stable_diffusion_compvis/    # CompVis (primary)
│   └── stable_diffusion_diffusers/  # HuggingFace (backup)
├── dynaprompt/                       # Core DynaPrompt implementation
│   ├── controller.py                # Main feedback loop controller
│   ├── feedback.py                  # CLIP-based semantic feedback
│   └── prompt_updater.py            # Token re-weighting logic
├── baselines/                        # Baseline implementations
│   ├── static_prompt.py             # Vanilla SD
│   ├── dynamic_cfg.py               # Dynamic CFG schedule
│   └── prompt_rewrite.py            # Post-hoc retry
├── data/                             # Datasets and generated images
│   ├── prompts/                     # Curated 200 COCO prompts
│   ├── coco/                        # COCO validation set
│   └── images/generated/            # Output images
├── evaluation/                       # Evaluation metrics
│   ├── metrics.py                   # CLIP, FID, ImageReward, CLIPScore
│   ├── compositional_accuracy.py    # BLIP-2 object extraction
│   └── alignment_curves.py          # CLIP similarity over time
├── experiments/                      # Experiment scripts
├── notebooks/                        # Jupyter notebooks for demos
├── scripts/                          # Utility scripts
├── results/                          # Experimental results
└── docs/                             # Documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:ch3889/6694-DynaPrompt.git
cd 6694-DynaPrompt
```

### 2. Create Python Environment

```bash
# Using conda
conda create -n dynaprompt python=3.10
conda activate dynaprompt

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 5. Download Stable Diffusion Weights

```bash
# Run the download script
bash scripts/download_models.sh

# Or manually download SD v1.5
# Place in: models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt
```

---

## Quick Start

### Test Vanilla Stable Diffusion

```bash
python models/stable_diffusion_compvis/scripts/txt2img.py \
  --prompt "A golden retriever playing with a red ball in a snowy park" \
  --ckpt models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt \
  --n_samples 1 \
  --outdir outputs/test
```

### Run DynaPrompt

```bash
python experiments/run_dynaprompt.py \
  --prompt "A golden retriever playing with a red ball in a snowy park" \
  --feedback_freq 10 \
  --update_alpha 0.3
```

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

---

## Project Timeline

| Week | Dates | Phase | Deliverables |
|------|-------|-------|-------------|
| 1 | Oct 25 - Nov 1 | Setup & Infrastructure | Working SD + CLIP + 200 prompts |
| 2 | Nov 2 - Nov 9 | Core Implementation | DynaPrompt feedback loop |
| 3 | Nov 10 - Nov 17 | Baselines & Dataset | All baselines + full experiments |
| 4 | Nov 18 - Nov 25 | Experiments & Evaluation | Metrics + ablations + human eval |
| 5 | Nov 26 - Dec 2 | Analysis & Report | Final report + presentation |

---

## Work Division

- **Charles (ch3889)**: Infrastructure & Baselines Lead
- **Max (zk2295)**: Core Algorithm & Experiments Lead
- **Swapnil (sb5041)**: Data & Evaluation Lead

---

## Evaluation Metrics

### Quantitative
- **CLIPScore**: Text-image semantic alignment
- **FID Score**: Image quality vs. real distribution
- **Compositional Accuracy**: Object/attribute recall via BLIP-2
- **Generation Time**: Runtime overhead analysis

### Qualitative
- **Alignment Curves**: CLIP similarity evolution over denoising steps
- **Human Evaluation**: Pairwise preference comparisons

---

## Baselines

1. **Static Prompt**: Vanilla Stable Diffusion (no feedback)
2. **Dynamic CFG**: Adaptive classifier-free guidance schedule
3. **Prompt Rewrite**: Post-hoc regeneration with CLIP filtering

---

## Related Work

- **Prompt-to-Prompt** (P2P): Cross-attention manipulation for local edits
- **Attend-and-Excite**: Dynamic attention re-weighting for neglected tokens
- **Dynamic CFG**: Varying guidance strength across timesteps
- **Composable Diffusion** & **GLIGEN**: Spatial/compositional conditioning

DynaPrompt differs by providing **external, model-agnostic feedback** rather than internal parameter adjustments.

---

## Dataset

**Primary**: COCO 2017 Validation Set
- 200 curated prompts with multi-object, attribute-rich descriptions
- Focus on compositional understanding and attribute binding

**Not Using**: LAION-5B (too large, noisy captions, designed for training)

---

## Computing Resources

- **Platform**: Google Colab Pro (T4 GPU)
- **Budget**: ~$100
- **Strategy**: Optimize inference, batch processing, overnight runs

---

## Acknowledgments

- [CompVis Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- COCO Dataset
