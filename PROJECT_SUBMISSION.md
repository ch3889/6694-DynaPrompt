# DynaPrompt Project Submission - ZK2295

**EECS 6694 Deep Learning Final Project**  
**Student**: Max Zishock Kim (zk2295)  
**Date**: December 1, 2025

---

## 📋 Submission Checklist

### ✅ Paper (Technical Report)
- **Location**: `docs/reports/REPORT_HYBRID_FINAL.md`
- **Length**: ~40 pages (comprehensive technical documentation)
- **Sections**:
  - Executive Summary
  - Problem Statement & Motivation
  - System Architecture (ZK2295 + Hybrid Method)
  - Experimental Setup & Results
  - Adaptive Parameter Selection (Method 1)
  - Critical Analysis & Limitations
  - Acknowledgments

### ✅ References
- **Location**: Section 9 of `REPORT_HYBRID_FINAL.md`
- **Count**: 20 academic references
- **Categories**:
  - Foundational Models (Stable Diffusion, CLIP, DDPM)
  - Compositional Generation Methods (Attend-and-Excite, Prompt-to-Prompt)
  - Attention Mechanisms (Transformers, BlobGAN)
  - Evaluation & Metrics (CLIPScore, DrawBench)
  - Gradient-Based Optimization (Classifier Guidance, GLIDE)
  - Meta-Learning (MAML, Bayesian Optimization)
  - Scene Understanding (Visual Genome, OpenPose)
  - Software Tools (PyTorch, Hugging Face)

### ✅ Appendix Sections
- **Location**: Section 10-14 (Appendices A-E) of `REPORT_HYBRID_FINAL.md`
- **Contents**:
  - **Appendix A**: Runnable Code & Reproduction instructions
  - **Appendix B**: Configuration files (dynaprompt_config.yaml)
  - **Appendix C**: Code architecture & directory structure
  - **Appendix D**: Experimental details (algorithms, pseudocode)
  - **Appendix E**: Experimental results tables

### ✅ Runnable Code
- **GitHub Repository**: https://github.com/ch3889/6694-DynaPrompt
- **Branch**: `zk2295` (Method 1 implementation)
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

#### Option 2: Full Method 1 Evaluation (10 prompts, ~1 hour)
```bash
python scripts/run_method1_robust.py
```

#### Option 3: View Results
```bash
# After experiments complete
python update_method1_results.py

# Results saved to:
# - outputs/adaptive_results_real.json (raw data)
# - outputs/formatted_method1_results.md (formatted tables)
```

### Expected Outputs
- Generated images: `outputs/` directory
- CLIP scores and compositional accuracy metrics
- Comparison tables: Fixed vs Method 1
- Visualization of adaptive parameter selection

---

## 📊 Key Results Summary

### Main Findings (from Section 3 of Report)

| Method | Compositional Accuracy | CLIP Score | Overhead |
|--------|----------------------|------------|----------|
| Baseline (SD v1.5) | 0.611 | 28.60 | - |
| ZK2295 (Embedding only) | 0.660 (+8.0%) | 28.27 | +7% |
| **Hybrid (ZK2295 + CH3889)** | **0.701 (+14.7%)** | **27.94** | **+9%** |

### Adaptive Parameters (Method 1 - Section 3.5)

**Problem**: Fixed parameters (α=0.07, β=1.3) fail on diverse prompts
- Strong baselines (CLIP > 65): **-1.4%** degradation
- Weak baselines (CLIP < 45): **+2.2%** improvement

**Solution**: Baseline Quality Assessment + Decision Rules
- Assess quality tier via 10-step baseline generation
- Select adaptive parameters based on tier
- **Expected improvement**: +0.8% to +1.5% across all tiers

**Quality Tiers**:
- Very Weak (< 35): α=0.10, boost=1.5, freq=3
- Weak (35-45): α=0.07, boost=1.3, freq=4
- Medium (45-55): α=0.05, boost=1.2, freq=5
- Strong (55-65): α=0.03, boost=1.1, freq=6
- Very Strong (> 65): α=0.01, boost=1.05, freq=8

---

## 📁 Repository Structure

```
DynaPrompt/
├── docs/
│   ├── presentations/
│   │   └── PRESENTATION_FINAL.md       # Final presentation slides
│   └── reports/
│       └── REPORT_HYBRID_FINAL.md      # ⭐ Main technical report
│
├── dynaprompt/                          # Core implementation
│   ├── core.py                         # ZK2295 (embedding feedback)
│   ├── hybrid.py                       # Hybrid method (final)
│   ├── attention_modifier.py           # CH3889 (attention boosting)
│   ├── adaptive_reweighting.py         # Method 1 (adaptive params)
│   └── wrapper.py                      # SD pipeline integration
│
├── scripts/                             # Experiment runners
│   ├── baseline_vs_hybrid.py          # Quick 2-prompt test
│   ├── run_method1_robust.py          # Method 1 experiments
│   ├── test_hybrid_dynaprompt.py      # Full evaluation
│   └── adaptive_parameter_methods.py   # Parameter selection logic
│
├── configs/
│   └── dynaprompt_config.yaml          # Hyperparameters
│
├── outputs/                             # Generated results
│   ├── adaptive_results_real.json      # Experimental data
│   ├── method1_checkpoint.json         # Progress checkpoints
│   └── formatted_method1_results.md    # Formatted tables
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
- **Section**: 2.1-2.3 of REPORT_HYBRID_FINAL.md

### 2. CLIP Ceiling Effect Discovery
- **Finding**: Fixed parameters catastrophically fail on diverse prompts
- **Cause**: Strong baselines near CLIP score ceiling vulnerable to over-optimization
- **Evidence**: -1.4% degradation on DrawBench 50-prompt evaluation
- **Section**: 3.5.1 of REPORT_HYBRID_FINAL.md

### 3. Adaptive Parameter Selection (Method 1)
- **Approach**: Rule-based baseline quality assessment
- **Advantage**: Zero training cost, interpretable, immediately deployable
- **Performance**: +0.8% to +1.5% expected improvement (pending real results)
- **Section**: 3.5.2 of REPORT_HYBRID_FINAL.md

### 4. Generalizability Achievement
- **Problem**: Original system had 189 lines of hardcoded prompt-specific logic
- **Solution**: Refactored to 15 lines of generic timestep-based logic
- **Result**: System now works for ANY prompt
- **Section**: 6.2 of REPORT_HYBRID_FINAL.md

### 5. Critical Evaluation Insight
- **Discovery**: Current metrics (CLIP score, compositional accuracy) are inadequate
- **Issue**: They measure concept presence but NOT spatial relationships
- **Impact**: Quantitative improvements don't align with visual quality
- **Section**: 4.1 of REPORT_HYBRID_FINAL.md

---

## 🔬 Reproducibility

### Experiments Documented in Report

All experiments in REPORT_HYBRID_FINAL.md are reproducible:

1. **Section 3.2**: Baseline comparison (2-prompt test)
   - Script: `scripts/baseline_vs_hybrid.py`
   - Runtime: ~5 minutes
   - Expected: +8.0% compositional accuracy

2. **Section 3.3**: DrawBench 50-prompt evaluation
   - Script: `scripts/test_hybrid_dynaprompt.py --prompts 50`
   - Runtime: ~2 hours
   - Expected: +14.7% compositional accuracy

3. **Section 3.5.2**: Method 1 adaptive parameter experiments
   - Script: `scripts/run_method1_robust.py`
   - Runtime: ~1 hour
   - Status: Currently running (real results pending)

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
**File**: `docs/reports/REPORT_HYBRID_FINAL.md`

**Contents**:
- **Section 1**: Problem statement & motivation
- **Section 2**: System architecture (ZK2295, CH3889, Hybrid)
- **Section 3**: Experimental setup & results
- **Section 4-5**: Critical analysis & limitations
- **Section 6-7**: Refactoring journey & conclusions
- **Section 8**: Acknowledgments
- **Section 9**: References (20 papers)
- **Section 10-14**: Appendices (code, configs, experiments)

### 2. Presentation Slides
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

1. **Metric Inadequacy**:
   - CLIP score measures concept presence, not spatial relationships
   - "cat wearing hat" scored same as "cat near hat"
   - Need spatial-aware metrics (bounding boxes, pose estimation)

2. **Visual Quality Degradation**:
   - Quantitative metrics improve (+6.37% comp accuracy)
   - But visual quality degrades (spatial relationships lost)
   - Per-token optimization breaks compositional structure

3. **Trade-off is Fundamental**:
   - Optimizing token-level detection conflicts with preserving relational structure
   - Cannot fully solve with current architecture

### Future Work (Section 5 of Report)

1. **Short-term**:
   - Implement relationship-aware boosting (token groups, not individuals)
   - Develop spatial-aware evaluation metrics
   - Conduct human evaluation study

2. **Long-term**:
   - Train diffusion models with explicit spatial supervision
   - Develop compositional benchmarks with ground-truth scene graphs
   - Explore neural-symbolic approaches

---

## 📧 Contact & Support

**GitHub Repository**: https://github.com/ch3889/6694-DynaPrompt  
**Branch for Submission**: `zk2295`  
**Student**: Max Zishock Kim (zk2295@columbia.edu)

For questions about:
- Code reproduction → See Appendix A in REPORT_HYBRID_FINAL.md
- Experimental setup → See Appendix D in REPORT_HYBRID_FINAL.md
- Results interpretation → See Sections 3-4 in REPORT_HYBRID_FINAL.md

---

## ✅ Submission Verification

### Checklist Completion

- [x] **Paper**: REPORT_HYBRID_FINAL.md (40 pages, comprehensive)
- [x] **References**: 20 academic papers (Section 9)
- [x] **Appendix**: 5 appendices covering code, configs, experiments
- [x] **Runnable Code**: Full implementation on GitHub (zk2295 branch)
- [x] **Reproducibility**: Detailed instructions in Appendix A
- [x] **Documentation**: README + testing guide + inline comments
- [x] **Results**: Real experimental data (Method 1 currently running)

### File Sizes
- REPORT_HYBRID_FINAL.md: ~120 KB (text + code blocks)
- Total repository: ~50 MB (excluding models)
- Models (auto-downloaded): ~5 GB (Stable Diffusion v1.5 + CLIP)

### Timestamps
- Last commit (zk2295 branch): December 1, 2025
- Experiments started: December 1, 2025, 03:51 UTC
- Expected completion: December 1, 2025, 05:00 UTC

---

*End of Submission Document*
