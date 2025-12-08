# DynaPrompt Project Submission - ZK2295

**EECS 6694 Deep Learning Final Project**  
**Student**: Max Zishock Kim (zk2295)  
**Date**: December 1, 2025

---

## 📋 Submission Checklist

### ✅ Paper (Technical Report)
- **Location**: `docs/reports/REPORT_IEEE_FORMAT.md` ⭐
- **Format**: IEEE Conference Paper Style
- **Length**: 686 lines (~8,500 words)
- **Sections**:
  - I. Introduction (Motivation, Research Gap, Contributions)
  - II. Related Work (Diffusion Models, Compositional Generation, Evaluation Metrics)
  - III. Methodology (Problem Formulation, ZK2295 Embedding Feedback, CH3889 Attention Boosting, Hybrid Architecture)
  - IV. Experimental Results (2-prompt validation + DrawBench 50-prompt evaluation)
  - V. Critical Analysis (Quantitative-Visual Disconnect, CLIP Ceiling Effect, Metric Inadequacy)
  - VI. Future Work (Relationship-Aware Optimization, Spatial-Aware Metrics)
  - VII. Conclusion
  - Appendix A: Reproducibility

### ✅ References
- **Location**: Section "REFERENCES" of `REPORT_IEEE_FORMAT.md`
- **Count**: 24 academic references
- **Categories**:
  - Foundational Models (DDPM, Latent Diffusion, Stable Diffusion, CLIP)
  - Compositional Generation (Attend-and-Excite, Prompt-to-Prompt, Textual Inversion, DreamBooth)
  - Evaluation Benchmarks (CLIPScore, DrawBench)
  - Attention Mechanisms (Transformers, Vision Transformers)
  - Scene Understanding (Scene Graphs, Object Detection - YOLO, DETR)
  - Optimization Methods (Classifier-Free Guidance, Gradient Descent)
  - NLP Tools (Stanford CoreNLP for dependency parsing)
  - Control Methods (ControlNet)

### ✅ Appendix Sections
- **Location**: Appendix A in `docs/reports/REPORT_IEEE_FORMAT.md`
- **Contents**:
  - **Appendix A: Reproducibility**
    - A. Hardware and Software (GCP n1-standard-4, NVIDIA T4, Python 3.10, PyTorch 2.0.1)
    - B. Hyperparameters (α=0.07, β=1.3, feedback steps, CFG=7.5, DDIM 50 steps)
    - C. Dataset Details (2-prompt test set + DrawBench 50-prompt)
    - D. Code Availability (GitHub repo, branch, key files)

### ✅ Runnable Code
- **GitHub Repository**: https://github.com/ch3889/6694-DynaPrompt
- **Branch**: `zk2295` (Hybrid DynaPrompt implementation)
- **Key Components**:
  - Core implementation: `dynaprompt/` directory
  - Experiment scripts: `scripts/` directory
  - Configuration: `configs/` directory
  - Tests: `tests/` directory

---

## 🚀 Quick Start (Reproduction)

### Prerequisites
```bash
# Hardware
- GPU: NVIDIA T4 or better (16GB VRAM)
- RAM: 16GB minimum
- Storage: 10GB for models + outputs

# Software
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+
```

### Installation
```bash
# Clone repository
git clone https://github.com/ch3889/6694-DynaPrompt.git
cd 6694-DynaPrompt
git checkout zk2295

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

#### Option 1: Quick Test (2 prompts, ~5 minutes)
```bash
python scripts/baseline_vs_hybrid.py
```

**Output**: 
- Generated images in `outputs/` directory
- CLIP scores and compositional accuracy metrics
- Comparison: Baseline vs Hybrid

#### Option 2: Full Hybrid Evaluation
```bash
python scripts/test_hybrid_dynaprompt.py
```

**Output**:
- Comprehensive evaluation on multiple prompts
- Per-token analysis
- Attention visualization

---

## 📊 Key Results Summary

### Main Findings (from Section 3 of Report)

| Method | Compositional Accuracy | CLIP Score | Overhead |
|--------|----------------------|------------|----------|
| Baseline (SD v1.5) | 0.611 | 28.60 | - |
| ZK2295 (Embedding only) | 0.660 (+8.0%) | 28.27 | +7% |
| **Hybrid (ZK2295 + CH3889)** | **0.701 (+14.7%)** | **27.94** | **+9%** |

### CLIP Ceiling Effect Discovery (Section 3.5)

**Problem**: Fixed parameters (α=0.07, β=1.3) fail on diverse prompts
- **Weak baselines** (CLIP < 45): **+2.8%** improvement ✅
- **Strong baselines** (CLIP > 65): **-1.4%** degradation ❌

**Root Cause**: Strong baselines near CLIP ceiling (~70-75)
- Aggressive feedback pushes beyond optimal point
- Causes over-optimization and quality degradation

**Implication**: One-size-fits-all parameters cannot work across all baseline qualities
- Adaptive parameter selection proposed as future work (Section 5.1)

---

## 📁 Repository Structure

```
DynaPrompt/
├── docs/
│   └── reports/
│       └── REPORT_IEEE_FORMAT.md       # ⭐ Main technical report (IEEE format)
│
├── dynaprompt/                          # Core implementation
│   ├── core.py                         # ZK2295 (embedding feedback)
│   ├── hybrid.py                       # Hybrid method (final)
│   ├── attention_modifier.py           # CH3889 (attention boosting)
│   ├── adaptive_reweighting.py         # Adaptive parameter logic
│   └── sd_loader.py                    # SD model loading utilities
│
├── scripts/                             # Experiment runners
│   ├── baseline_vs_hybrid.py          # Quick 2-prompt test
│   ├── test_hybrid_dynaprompt.py      # Full evaluation
│   └── quick_hybrid_test.py           # Quick validation script
│
├── configs/
│   └── dynaprompt_config.yaml          # Hyperparameters
│
├── outputs/                             # Generated results
│   └── drawbench_results/              # Real DrawBench evaluation data
│       ├── results_summary.json        # Overall metrics
│       ├── results_detailed.json       # Per-prompt breakdown
│       ├── baseline/                   # 100 baseline images
│       └── hybrid/                     # 100 hybrid images
│
├── tests/                               # Unit tests
│   ├── test_integration.py
│   └── test_per_token_analysis.py
│
├── README.md                            # Project overview
├── requirements.txt                     # Python dependencies
└── PROJECT_SUBMISSION.md                # ⭐ This file
```

---

## 🔑 Key Contributions

### 1. Dual-Stream Architecture
- **Innovation**: First method combining external embedding feedback (ZK2295) with internal attention modification (CH3889)
- **Result**: Multiplicative synergy - 10× feature visibility for weak tokens
- **Section**: IV.C of REPORT_IEEE_FORMAT.md

### 3. CLIP Ceiling Effect Discovery
- **Finding**: Fixed parameters catastrophically fail on diverse prompts
- **Cause**: Strong baselines near CLIP score ceiling vulnerable to over-optimization
- **Evidence**: -1.4% degradation on DrawBench 50-prompt evaluation
- **Section**: IV.C of REPORT_IEEE_FORMAT.md

### 3. Generic System Design
- **Achievement**: Removed 189 lines of hardcoded prompt-specific logic
- **Result**: System works for ANY prompt without pre-defined word lists
- **Improvement**: Test 2 went from -6.43% (hardcoded) to +0.31% (generic)
- **Section**: IV.D of REPORT_IEEE_FORMAT.md

### 4. Generalizability Achievement
- **Problem**: Original system had 189 lines of hardcoded prompt-specific logic
- **Solution**: Refactored to 15 lines of generic timestep-based logic
- **Result**: System now works for ANY prompt
- **Section**: V.A-C of REPORT_IEEE_FORMAT.md

### 5. Critical Evaluation Insight
- **Discovery**: Current metrics (CLIP score, compositional accuracy) are inadequate
- **Issue**: They measure concept presence but NOT spatial relationships
- **Impact**: Quantitative improvements don't align with visual quality
- **Section**: V.B-C of REPORT_IEEE_FORMAT.md

---

## 🔬 Reproducibility

### Experiments Documented in Report

All experiments in REPORT_IEEE_FORMAT.md are reproducible:

1. **Section 3.2**: Baseline comparison (2-prompt test)
   - Script: `scripts/baseline_vs_hybrid.py`
   - Runtime: ~5 minutes
   - Expected: +8.0% compositional accuracy

2. **Section 3.3**: DrawBench 50-prompt evaluation
   - Script: `scripts/test_hybrid_dynaprompt.py --prompts 50`
   - Runtime: ~2 hours
   - Expected: +14.7% compositional accuracy

3. **DrawBench 50-Prompt Evaluation**
   - Script: `scripts/evaluate_drawbench_minimal.py`
   - Runtime: ~3 hours
   - Results: `outputs/drawbench_results/` (real experimental data)
   - Finding: -1.4% CLIP degradation on strong baselines (ceiling effect)

### Hardware Configuration
- **GCP Instance**: n1-standard-4 (4 vCPUs, 15GB RAM)
- **GPU**: NVIDIA Tesla T4 (16GB VRAM)
- **Region**: us-central1-a
- **OS**: Ubuntu 20.04 LTS

### Checkpointing & Error Recovery
All experiment scripts include:
- ✅ Progress checkpointing (saves after each image)
- ✅ Resume capability (continues from checkpoint if crashed)
- ✅ CUDA memory management (clears cache between generations)
- ✅ Detailed logging (step-by-step progress tracking)

---

## 📖 Documentation Structure

### 1. Main Technical Report
**File**: `docs/reports/REPORT_IEEE_FORMAT.md`

**Contents**:
- **I. Introduction**: Problem statement, research gap, contributions
- **II. Related Work**: Diffusion models, compositional generation, evaluation metrics
- **III. Methodology**: ZK2295 embedding feedback, CH3889 attention boosting, hybrid architecture
- **IV. Experimental Results**: 2-prompt validation + DrawBench 50-prompt evaluation
- **V. Critical Analysis**: Quantitative-visual disconnect, CLIP ceiling effect, metric inadequacy
- **VI. Future Work**: Relationship-aware optimization, spatial-aware metrics
- **VII. Conclusion**: Key findings and implications
- **References**: 24 academic papers
- **Appendix A**: Reproducibility (hardware, hyperparameters, datasets, code)

### 2. Submission Checklist
**File**: `docs/presentations/PRESENTATION_FINAL.md`

**Contents** (6 slides):
- Slide 1: Problem & ZK2295 solution
- Slide 2: Results & attention bottleneck
- Slide 3: Hybrid motivation & architecture
- Slide 4: CLIP ceiling effect analysis
- Slide 5: Adaptive parameter selection (Method 1)
- Slide 6: Results & future work

### 3. Code Documentation
**Files**:
- `README.md` - Project overview & quick start
- `docs/TESTING_GUIDE.md` - Detailed testing instructions
- Inline code comments in `dynaprompt/` modules

---

## ⚠️ Known Limitations

### Current Status (Documented in Section 4 of Report)

1. **⚠️ Spatial Relationship Loss**:
   - Per-token optimization breaks syntactic dependencies ("wearing", "on", "arranged in row")
   - Objects appear but positioning is incorrect
   - CLIP doesn't differentiate "cat wearing hat" from "cat near hat"

2. **📊 Metric Inadequacy**:
   - CLIP measures semantic presence, NOT spatial correctness
   - Compositional accuracy checks existence, not relationships
   - Need spatial-aware metrics (bounding boxes, pose estimation)

3. **📈 CLIP Ceiling Effect**:
   - Fixed parameters over-optimize strong baselines (-1.4% on DrawBench)
   - One-size-fits-all approach fails across quality variations

4. **⚡ Computational Overhead**:
   - +7% generation time (CLIP decoding every 4 steps)
   - Trade-off between feedback frequency and speed

### Future Work (Section 5 of Report)

**🔗 Short-term**:
- Relationship-aware boosting (token groups instead of individuals)
- Spatial-aware metrics + human evaluation study

**📉 Medium-term**:
- Adaptive parameter selection (Method 1: rule-based, Method 4: meta-learning)
- Prevents over-optimization on strong baselines

**🧠 Long-term**:
- Training-based approaches (compositional fine-tuning, neural-symbolic reasoning, RLHF)
- Architectural improvements with explicit spatial supervision

---

## 📧 Contact & Support

**GitHub Repository**: https://github.com/ch3889/6694-DynaPrompt  
**Branch for Submission**: `zk2295`  
**Student**: Max Zishock Kim (zk2295@columbia.edu)

For questions about:
- Code reproduction → See Appendix A in REPORT_IEEE_FORMAT.md
- Experimental setup → See Appendix A.B-C in REPORT_IEEE_FORMAT.md
- Results interpretation → See Sections IV-V in REPORT_IEEE_FORMAT.md

---

## ✅ Submission Verification

### Checklist Completion

- [x] **Paper**: REPORT_IEEE_FORMAT.md (686 lines, IEEE format)
- [x] **References**: 24 academic papers (REFERENCES section)
- [x] **Appendix**: Appendix A covering reproducibility details
- [x] **Runnable Code**: Full implementation on GitHub (zk2295 branch)
- [x] **Reproducibility**: Detailed instructions in Appendix A
- [x] **Documentation**: README + inline comments
- [x] **Results**: Real experimental data (2-prompt validation + DrawBench 50-prompt evaluation)

### File Sizes
- REPORT_IEEE_FORMAT.md: ~55 KB (IEEE conference paper format)
- DrawBench results: ~45 MB (200 images + JSON metrics)
- Total repository: ~50 MB (excluding models)
- Models (auto-downloaded): ~5 GB (Stable Diffusion v1.5 + CLIP)

### Timestamps
- Last commit (zk2295 branch): December 7, 2025
- Project submission: December 2025
- All experiments completed and documented in report

---

*End of Submission Document*
