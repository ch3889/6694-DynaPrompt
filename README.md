# DynaPrompt: Iterative CLIP-Guided Embedding Feedback for Stable Diffusion

**Improving compositional accuracy in text-to-image generation through dynamic prompt refinement.**

**EECS 6694 Deep Learning Project**  
**Team:** Charles Chaoyu Hou (ch3889), Max Zishock Kim (zk2295), Swapnil Banerjee (sb5041)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick test (2 prompts, ~5 minutes)
python scripts/baseline_vs_hybrid.py
```

📖 **See [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for detailed testing instructions**

---

## Overview

### The Problem
Stable Diffusion fails to generate ~40% of prompt concepts:
- "cat wearing **red hat**" → cat appears, hat missing
- "table with **green apple**" → table appears, apple missing

### Our Solution

**Two complementary methods:**

1. **ZK2295**: CLIP-guided embedding feedback
   - Iteratively refines text embeddings using CLIP similarity
   - +8% compositional accuracy

2. **Hybrid** (Final Solution): ZK2295 + Attention Boosting
   - Combines embedding refinement with attention amplification
   - **+14.7% compositional accuracy**
   - Best performance vs overhead trade-off

---

## Results

| Method | Compositional Accuracy | CLIP Score | Overhead |
|--------|----------------------|------------|----------|
| Baseline | 0.611 | 28.60 | - |
| Prompt-to-Prompt | 0.632 (+3.4%) | 28.52 | +12% |
| Attend-and-Excite | 0.679 (+11.1%) | 26.16 | **+45%** |
| ZK2295 (ours) | 0.660 (+8.0%) | 28.27 | +7% |
| **Hybrid (ours)** | **0.701 (+14.7%)** | **27.94** | **+9%** |

✅ **Best compositional gains with lowest overhead**

---

## Repository Structure

```
DynaPrompt/
├── dynaprompt/              # Core implementation
│   ├── core.py             # ZK2295 CLIP feedback
│   ├── hybrid.py           # Hybrid method (final)
│   ├── attention_modifier.py
│   └── adaptive_reweighting.py
├── configs/
│   └── dynaprompt_config.yaml  # α=0.08, boost=1.3
├── scripts/
│   ├── baseline_vs_hybrid.py   # Main test
│   └── test_hybrid_dynaprompt.py
├── docs/                    # 📚 All documentation
│   ├── TESTING_GUIDE.md    # ⭐ Start here
│   ├── presentations/       # Slides
│   │   ├── PRESENTATION_FINAL.md  # Main (6 slides)
│   │   └── PRESENTATION_SLIDES.md # Detailed
│   ├── reports/            # Technical reports
│   │   ├── REPORT_ZK2295_METHOD.md
│   │   └── REPORT_HYBRID_METHOD.md
│   ├── analysis/           # Analysis docs
│   │   ├── ARCHITECTURE.md
│   │   └── TECHNIQUE_COMPARISON.md
│   └── setup/              # Setup guides
└── outputs/                # Generated images & metrics
```

---

## Documentation

### 🚀 Getting Started
- **[TESTING_GUIDE.md](docs/TESTING_GUIDE.md)** - How to test (start here!)
- **[SETUP.md](docs/setup/SETUP.md)** - Installation guide
- **[INTEGRATION_GUIDE.md](docs/setup/INTEGRATION_GUIDE.md)** - Integration

### 📊 Presentations
- **[PRESENTATION_FINAL.md](docs/presentations/PRESENTATION_FINAL.md)** - Main slides (6 slides)
- **[PRESENTATION_SLIDES.md](docs/presentations/PRESENTATION_SLIDES.md)** - Detailed version

### 📝 Technical Reports
- **[REPORT_ZK2295_METHOD.md](docs/reports/REPORT_ZK2295_METHOD.md)** - ZK2295 method
- **[REPORT_HYBRID_METHOD.md](docs/reports/REPORT_HYBRID_METHOD.md)** - Hybrid method
- **[REPORT_BENCHMARKS.md](docs/reports/REPORT_BENCHMARKS.md)** - Benchmarks

### 🔬 Analysis
- **[ARCHITECTURE.md](docs/analysis/ARCHITECTURE.md)** - System architecture
- **[TECHNIQUE_COMPARISON.md](docs/analysis/TECHNIQUE_COMPARISON.md)** - vs prior work
- **[COMPARISON.md](docs/analysis/COMPARISON.md)** - Method comparisons

---

## How It Works

### ZK2295 Algorithm

```
For each feedback step (every 4 steps, steps 5-30):
  1. Decode latent → intermediate image
  2. Compute CLIP score for each concept
  3. Identify weak tokens (CLIP < threshold)
  4. Update embedding:
     - Global: c ← c + α·(E_img - E_text)
     - Selective: c_i ← c_i × (1 + β·weakness)
  5. Continue denoising with updated c
```

### Hybrid = ZK2295 + Attention Boosting

**Key Innovation**: Multiplicative synergy  
- Visibility = embedding_strength × attention_weight
- Hybrid improves BOTH → superlinear gains
- +416% for weak tokens vs +30% (ZK2295) or +303% (attention only)

---

## Key Features

✅ **Moderate Parameters**: α=0.08, boost=1.3 (stable, proven)  
✅ **Efficient**: 9% overhead (vs 45% for Attend-and-Excite)  
✅ **Multiplicative Synergy**: Dual-stream feedback  
✅ **Best Trade-off**: Highest comp gains / lowest overhead  
✅ **Well-Documented**: Comprehensive reports & analysis

---

## Installation

```bash
# Clone repository
git clone https://github.com/ch3889/6694-DynaPrompt.git
cd 6694-DynaPrompt

# Install dependencies
pip install -r requirements.txt

# Download models (if needed)
# Model will auto-download on first run
```

---

## Quick Test

```bash
# Run baseline vs hybrid comparison (2 prompts, ~5 min)
python scripts/baseline_vs_hybrid.py

# Results saved to: outputs/baseline_vs_hybrid/
```

Expected output:
```
Average Compositional Accuracy:
  Baseline: 0.611
  Hybrid:   0.701
  Improvement: +14.7%
```

---

## Configuration

Edit `configs/dynaprompt_config.yaml`:

```yaml
prompt_update:
  update_alpha: 0.08  # Embedding update rate

attention:
  boost_factor: 1.3   # Attention amplification

feedback:
  feedback_frequency: 4     # Every N steps
  feedback_start_step: 5    # Start step
  feedback_end_step: 30     # End step
```

---

## Citation

```bibtex
@misc{dynaprompt2024,
  title={DynaPrompt: Iterative CLIP-Guided Embedding Feedback for Compositional Text-to-Image Generation},
  author={Hou, Charles Chaoyu and Kim, Max Zishock and Banerjee, Swapnil},
  year={2024},
  school={Columbia University}
}
```

---

## Acknowledgments

This project builds upon:
- [Stable Diffusion by CompVis](https://github.com/CompVis/stable-diffusion)
- [CLIP by OpenAI](https://github.com/openai/CLIP)

---

## License

See individual component licenses. This project is for academic research purposes.

---

## Contact

For questions or issues, please open a GitHub issue or contact the team.
