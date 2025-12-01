# Benchmark Evaluation Plan: DrawBench for DynaPrompt

## Motivation

Current evaluation uses only **2 test prompts**:
1. "a cat wearing a red hat"
2. "a table with a green apple and a red banana arranged in a row"

**Limitations**:
- ❌ Too small sample size (not statistically significant)
- ❌ Biased toward specific composition types (worn objects, spatial arrangements)
- ❌ Doesn't test diverse compositional challenges (counting, colors, conflicting descriptions, etc.)
- ❌ Can't generalize findings to broader prompt distributions

**Solution**: Evaluate on **DrawBench** - the gold standard benchmark for text-to-image models.

---

## DrawBench Overview

**Source**: Google Research, used in Imagen paper (NIPS 2022)
- **Size**: 150 prompts across 11 categories
- **Focus**: Compositional challenges in text-to-image generation
- **Used by**: Imagen, Stable Diffusion, DALL-E 3, Parti papers

### Categories (with examples)

1. **Colors** (e.g., "A blue colored dog", "A red colored car")
   - Tests attribute binding

2. **Counting** (e.g., "Three cats", "Five apples on a tree")
   - Tests numeracy understanding

3. **Conflicting** (e.g., "A horse riding an astronaut")
   - Tests handling of reversed/unusual compositions

4. **DALL-E** (prompts from DALL-E paper)
   - Canonical challenging prompts

5. **Descriptions** (detailed scene descriptions)
   - Tests complex multi-object compositions

6. **Gary Marcus** (adversarial prompts by AI critic Gary Marcus)
   - Specifically designed to break AI systems

7. **Misspellings** (e.g., "A rde car", "A grene apple")
   - Tests robustness to typos

8. **Positional** (e.g., "A car to the left of a house")
   - Tests spatial relationships (CRITICAL for us!)

9. **Rare Words** (e.g., "A malachite colored bird")
   - Tests uncommon concepts

10. **Reddit** (prompts from Reddit users)
    - Real-world diverse requests

11. **Text** (e.g., "A sign that says 'Hello World'")
    - Tests text rendering (expected failure for SD)

---

## Evaluation Protocol

### Phase 1: Subset Testing (Recommended Start)

**Prompts**: 50 selected from DrawBench
- 10 Colors (full category)
- 10 Positional (spatial relationships - our focus)
- 10 Counting (compositional challenge)
- 10 Descriptions (complex scenes)
- 10 Conflicting (stress test)

**Methods**:
1. Baseline (Stable Diffusion v1.5)
2. Hybrid DynaPrompt (generic system)

**Metrics**:
1. **Compositional Accuracy** (automated - CLIP per-concept)
2. **Global CLIP Score** (automated)
3. **Human Evaluation** (optional - spatial correctness rating 1-10)

**Computational Cost**:
- 50 prompts × 2 methods × 50 steps × 2.4s = **2 hours on T4**
- GCP cost: ~$6-8

### Phase 2: Full DrawBench (If Phase 1 shows promise)

**Prompts**: All 150 from DrawBench

**Computational Cost**:
- 150 prompts × 2 methods × 50 steps × 2.4s = **6 hours on T4**
- GCP cost: ~$20-25

### Phase 3: Human Evaluation (Validate findings)

**Sample**: 30 prompts (20 from Phase 1 + 10 new)
- Focus on **Positional** and **Descriptions** categories
- Rate spatial correctness: 1-10 scale
- Preference test: Baseline vs Hybrid

---

## Implementation Plan

### Step 1: Download DrawBench Prompts

```python
# scripts/download_drawbench.py
import json
import requests

DRAWBENCH_URL = "https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts/drawbench.json"

def download_drawbench():
    """Download DrawBench prompts from official repo"""
    response = requests.get(DRAWBENCH_URL)
    prompts = response.json()
    
    with open("data/drawbench_prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"Downloaded {len(prompts)} DrawBench prompts")
    return prompts

if __name__ == "__main__":
    download_drawbench()
```

### Step 2: Create Evaluation Script

```python
# scripts/evaluate_drawbench.py
import torch
import json
from tqdm import tqdm
from dynaprompt.wrapper import StableDiffusionWrapper
from evaluation.metrics import compute_compositional_accuracy, compute_clip_score

def evaluate_drawbench(
    prompts_file: str = "data/drawbench_prompts.json",
    methods: list = ["baseline", "hybrid"],
    output_dir: str = "outputs/drawbench",
    subset_categories: list = None  # None = all, or ["Colors", "Positional", ...]
):
    """
    Evaluate methods on DrawBench benchmark
    
    Args:
        prompts_file: Path to DrawBench prompts JSON
        methods: List of methods to evaluate
        output_dir: Where to save generated images and results
        subset_categories: If provided, only evaluate these categories
    """
    # Load prompts
    with open(prompts_file) as f:
        drawbench = json.load(f)
    
    # Filter by category if requested
    if subset_categories:
        drawbench = {
            k: v for k, v in drawbench.items() 
            if k in subset_categories
        }
    
    # Flatten prompts
    all_prompts = []
    for category, prompts in drawbench.items():
        for prompt in prompts:
            all_prompts.append({"category": category, "prompt": prompt})
    
    print(f"Evaluating {len(all_prompts)} prompts across {len(drawbench)} categories")
    
    # Initialize models
    models = {}
    if "baseline" in methods:
        models["baseline"] = StableDiffusionWrapper(use_dynaprompt=False)
    if "hybrid" in methods:
        models["hybrid"] = StableDiffusionWrapper(
            use_dynaprompt=True,
            use_hybrid=True,
            alpha=0.07,
            boost_factor=1.3
        )
    
    # Evaluation loop
    results = {method: [] for method in methods}
    
    for item in tqdm(all_prompts, desc="Evaluating"):
        prompt = item["prompt"]
        category = item["category"]
        
        for method_name, model in models.items():
            # Generate image
            image = model.generate(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                seed=42  # Fixed seed for reproducibility
            )
            
            # Save image
            img_path = f"{output_dir}/{method_name}/{category}/{prompt[:50]}.png"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            image.save(img_path)
            
            # Compute metrics
            comp_acc = compute_compositional_accuracy(image, prompt)
            clip_score = compute_clip_score(image, prompt)
            
            results[method_name].append({
                "prompt": prompt,
                "category": category,
                "compositional_accuracy": comp_acc,
                "clip_score": clip_score,
                "image_path": img_path
            })
    
    # Aggregate results by category
    summary = {}
    for method_name, method_results in results.items():
        summary[method_name] = {}
        
        # Overall
        summary[method_name]["overall"] = {
            "comp_acc": np.mean([r["compositional_accuracy"] for r in method_results]),
            "clip_score": np.mean([r["clip_score"] for r in method_results])
        }
        
        # Per category
        for category in drawbench.keys():
            cat_results = [r for r in method_results if r["category"] == category]
            summary[method_name][category] = {
                "comp_acc": np.mean([r["compositional_accuracy"] for r in cat_results]),
                "clip_score": np.mean([r["clip_score"] for r in cat_results]),
                "count": len(cat_results)
            }
    
    # Save results
    with open(f"{output_dir}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(f"{output_dir}/results_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("DRAWBENCH EVALUATION RESULTS")
    print("="*80)
    
    for method_name, method_summary in summary.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Overall Comp Acc: {method_summary['overall']['comp_acc']:.4f}")
        print(f"  Overall CLIP: {method_summary['overall']['clip_score']:.2f}")
        print(f"\n  Per Category:")
        for category, cat_stats in method_summary.items():
            if category != "overall":
                print(f"    {category:20s}: Comp={cat_stats['comp_acc']:.4f}, CLIP={cat_stats['clip_score']:.2f}")
    
    return results, summary

if __name__ == "__main__":
    # Phase 1: Subset evaluation
    subset_categories = ["Colors", "Positional", "Counting", "Descriptions", "Conflicting"]
    
    results, summary = evaluate_drawbench(
        subset_categories=subset_categories,
        methods=["baseline", "hybrid"]
    )
```

### Step 3: Analyze Spatial Relationship Performance

```python
# scripts/analyze_spatial_failures.py
import json
from collections import defaultdict

def analyze_spatial_failures(results_file: str):
    """
    Deep dive into spatial relationship performance
    Focus on 'Positional' category to validate our hypothesis
    """
    with open(results_file) as f:
        results = json.load(f)
    
    # Extract Positional prompts
    spatial_prompts = {
        method: [r for r in method_results if r["category"] == "Positional"]
        for method, method_results in results.items()
    }
    
    print("SPATIAL RELATIONSHIP ANALYSIS")
    print("="*80)
    
    for method, prompts in spatial_prompts.items():
        print(f"\n{method.upper()}:")
        
        # Check if our hypothesis holds: metrics improve but quality degrades
        avg_comp = np.mean([p["compositional_accuracy"] for p in prompts])
        avg_clip = np.mean([p["clip_score"] for p in prompts])
        
        print(f"  Avg Compositional Accuracy: {avg_comp:.4f}")
        print(f"  Avg CLIP Score: {avg_clip:.2f}")
        
        # List individual results
        print(f"\n  Individual Prompts:")
        for p in prompts[:10]:  # Show first 10
            print(f"    {p['prompt'][:60]:60s} | Comp: {p['compositional_accuracy']:.3f} | CLIP: {p['clip_score']:.1f}")
    
    # Compare baseline vs hybrid on spatial prompts
    if "baseline" in spatial_prompts and "hybrid" in spatial_prompts:
        baseline_comp = np.mean([p["compositional_accuracy"] for p in spatial_prompts["baseline"]])
        hybrid_comp = np.mean([p["compositional_accuracy"] for p in spatial_prompts["hybrid"]])
        
        baseline_clip = np.mean([p["clip_score"] for p in spatial_prompts["baseline"]])
        hybrid_clip = np.mean([p["clip_score"] for p in spatial_prompts["hybrid"]])
        
        print("\n" + "="*80)
        print("SPATIAL PROMPTS: BASELINE VS HYBRID")
        print("="*80)
        print(f"Compositional Accuracy: {baseline_comp:.4f} → {hybrid_comp:.4f} ({(hybrid_comp/baseline_comp-1)*100:+.2f}%)")
        print(f"CLIP Score: {baseline_clip:.2f} → {hybrid_clip:.2f} ({(hybrid_clip/baseline_clip-1)*100:+.2f}%)")
        
        # Hypothesis check
        if hybrid_comp > baseline_comp:
            print("\n⚠️  HYPOTHESIS TO VALIDATE:")
            print("   Quantitative metrics improved on spatial prompts")
            print("   → Need human evaluation to check if spatial relationships actually correct")
            print("   → Expected: Metrics improve BUT visual quality degrades (objects present but wrong position)")

if __name__ == "__main__":
    analyze_spatial_failures("outputs/drawbench/results_detailed.json")
```

---

## Expected Findings

### Hypothesis 1: Generic System Performance

**Current results (2 prompts)**:
- Test 1: +13.23% comp, -7.30% CLIP
- Test 2: +0.31% comp, +11.51% CLIP
- Average: +6.37% comp, +0.85% CLIP

**DrawBench prediction**:
- Colors: **Strong improvement** (+10-15% comp) - attribute binding is our strength
- Positional: **Metrics improve, quality degrades** - our documented limitation
- Counting: **Moderate improvement** (+5-8% comp) - if objects are detected
- Descriptions: **Variable** - depends on spatial vs presence requirements
- Conflicting: **Baseline performance** - unusual compositions outside training distribution

### Hypothesis 2: Category-Specific Performance

| Category | Expected Comp Δ | Expected CLIP Δ | Rationale |
|----------|----------------|----------------|-----------|
| **Colors** | +12-18% | +1-3% | Our strength: per-token boosting helps attribute binding |
| **Positional** | +8-12% | +0-2% | ⚠️ Metrics improve but visual quality likely poor |
| **Counting** | +5-10% | -1-2% | Helps if objects missing; breaks if spatial arrangement matters |
| **Descriptions** | +6-10% | +0-2% | General improvement for complex scenes |
| **Conflicting** | +0-3% | -2-4% | Unusual compositions - method may struggle |
| **DALL-E** | +4-8% | +0-2% | Mixed bag of challenges |

### Hypothesis 3: Validation of Metric Inadequacy

**What we'll see**:
1. **Positional category**: High comp/CLIP scores but **incorrect spatial arrangements**
   - "car to the left of house" → both present ✅ but wrong positions ❌
   
2. **Human evaluation**: Low correlation (r < 0.4) between metrics and human ratings

3. **Per-category variance**: Huge spread in performance (some categories benefit, others don't)

---

## Timeline & Resources

### Phase 1: Subset (50 prompts)
- **Setup**: 1 hour (download prompts, test scripts)
- **Execution**: 2 hours on GCP T4
- **Analysis**: 2 hours (generate plots, tables)
- **Total**: ~5 hours, $8 cost

### Phase 2: Full DrawBench (150 prompts)
- **Execution**: 6 hours on GCP T4
- **Analysis**: 4 hours
- **Total**: ~10 hours, $25 cost

### Phase 3: Human Evaluation
- **Raters**: 3 people × 30 prompts × 2 min = 3 hours
- **Analysis**: 2 hours (correlation, preference statistics)

---

## Deliverables

1. **Results Table**: Performance by category
2. **Visualization**: Bar charts comparing baseline vs hybrid per category
3. **Error Analysis**: Which prompts fail and why
4. **Updated Report**: Add DrawBench results to REPORT_ZK2295_METHOD.md
5. **Updated Presentation**: Replace 2-prompt results with DrawBench statistics
6. **Human Evaluation Data** (if Phase 3): Validate metric inadequacy hypothesis

---

## Decision Point

**Recommendation**: Start with **Phase 1 (50 prompts, $8, 5 hours)**

This will:
- ✅ Validate our findings on broader distribution
- ✅ Test category-specific performance hypotheses
- ✅ Identify which prompt types benefit from our method
- ✅ Provide statistically significant results (50 prompts >> 2 prompts)
- ✅ Low cost/time investment to prove value before full evaluation

**If Phase 1 shows**:
- Strong performance on Colors/Counting → confirms generic system works
- Poor spatial quality despite good metrics → validates our limitation analysis
- Category-specific patterns → actionable insights for future work

→ Then proceed to **Phase 2 (full 150 prompts)** for paper-quality evaluation

---

## Alternative: PartiPrompts (Stretch Goal)

If DrawBench validation is successful, consider **PartiPrompts subset**:
- Sample 200 prompts across difficulty levels (Simple, Basic, Challenge, Complex)
- Focus on compositional categories (Spatial, Counting, Colors, Attributes)
- Estimated cost: ~$30-40, 8 hours computation

This would provide even more comprehensive validation for publication-quality evaluation.
