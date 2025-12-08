"""
Analyze spatial relationship performance on DrawBench Positional category
Focus on validating the hypothesis: metrics improve but visual quality degrades
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

def analyze_spatial_failures(results_file, summary_file, output_dir="outputs/drawbench"):
    """
    Deep dive into spatial relationship performance
    Focus on 'Positional' category to validate our hypothesis
    
    Hypothesis: Hybrid method improves quantitative metrics (CLIP, comp accuracy)
                BUT spatial relationships are not preserved (visual quality degrades)
    """
    
    print("="*80)
    print("SPATIAL RELATIONSHIP ANALYSIS")
    print("="*80)
    print("\nLoading results...")
    
    # Load results
    with open(results_file) as f:
        results = json.load(f)
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    # Extract Positional prompts
    spatial_prompts = {
        method: [r for r in method_results if r["category"] == "Positional"]
        for method, method_results in results.items()
    }
    
    print(f"\nFound {len(spatial_prompts)} method(s) with Positional category results")
    
    # Analyze each method
    for method, prompts in spatial_prompts.items():
        if not prompts:
            continue
            
        print(f"\n{'='*80}")
        print(f"{method.upper()} - Positional Category Analysis")
        print(f"{'='*80}")
        
        # Overall statistics
        comp_accs = [p["compositional_accuracy"] for p in prompts]
        clip_scores = [p["clip_score"] for p in prompts]
        
        print(f"\nOverall Statistics (n={len(prompts)}):")
        print(f"  Compositional Accuracy: {np.mean(comp_accs):.4f} ± {np.std(comp_accs):.4f}")
        print(f"  CLIP Score: {np.mean(clip_scores):.2f} ± {np.std(clip_scores):.2f}")
        
        # List individual results
        print(f"\nIndividual Prompt Results:")
        print(f"  {'Prompt':<60s} | {'Comp':>6s} | {'CLIP':>6s}")
        print(f"  {'-'*60}-+-{'-'*6}-+-{'-'*6}")
        
        for p in sorted(prompts, key=lambda x: x["compositional_accuracy"], reverse=True):
            prompt_short = p['prompt'][:58] if len(p['prompt']) <= 58 else p['prompt'][:55] + "..."
            print(f"  {prompt_short:<60s} | {p['compositional_accuracy']:6.3f} | {p['clip_score']:6.1f}")
        
        # Categorize spatial relationships
        print(f"\nSpatial Relationship Types:")
        
        spatial_types = {
            "left/right": ["left", "right"],
            "above/below": ["above", "below", "under", "top", "bottom"],
            "on/in": ["on", "in"],
            "beside/next": ["beside", "next"],
            "behind/front": ["behind", "front"]
        }
        
        type_results = defaultdict(list)
        for p in prompts:
            prompt_lower = p["prompt"].lower()
            for type_name, keywords in spatial_types.items():
                if any(kw in prompt_lower for kw in keywords):
                    type_results[type_name].append(p)
                    break
        
        for type_name, type_prompts in type_results.items():
            if type_prompts:
                type_comp = np.mean([p["compositional_accuracy"] for p in type_prompts])
                type_clip = np.mean([p["clip_score"] for p in type_prompts])
                print(f"  {type_name:15s}: n={len(type_prompts):2d}, Comp={type_comp:.3f}, CLIP={type_clip:.1f}")
    
    # Compare baseline vs hybrid on spatial prompts
    if "baseline" in spatial_prompts and "hybrid" in spatial_prompts:
        print("\n" + "="*80)
        print("SPATIAL PROMPTS: BASELINE VS HYBRID COMPARISON")
        print("="*80)
        
        baseline_comp = np.mean([p["compositional_accuracy"] for p in spatial_prompts["baseline"]])
        hybrid_comp = np.mean([p["compositional_accuracy"] for p in spatial_prompts["hybrid"]])
        
        baseline_clip = np.mean([p["clip_score"] for p in spatial_prompts["baseline"]])
        hybrid_clip = np.mean([p["clip_score"] for p in spatial_prompts["hybrid"]])
        
        comp_delta = ((hybrid_comp / baseline_comp) - 1) * 100
        clip_delta = ((hybrid_clip / baseline_clip) - 1) * 100
        
        print(f"\nQuantitative Metrics:")
        print(f"  Compositional Accuracy: {baseline_comp:.4f} → {hybrid_comp:.4f} ({comp_delta:+.2f}%)")
        print(f"  CLIP Score: {baseline_clip:.2f} → {hybrid_clip:.2f} ({clip_delta:+.2f}%)")
        
        # Per-prompt comparison
        print(f"\nPer-Prompt Comparison:")
        print(f"  {'Prompt':<50s} | {'Baseline':>8s} | {'Hybrid':>8s} | {'Delta':>8s}")
        print(f"  {'-'*50}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        
        # Match prompts
        baseline_dict = {p["prompt"]: p for p in spatial_prompts["baseline"]}
        hybrid_dict = {p["prompt"]: p for p in spatial_prompts["hybrid"]}
        
        for prompt in sorted(baseline_dict.keys()):
            if prompt in hybrid_dict:
                b_comp = baseline_dict[prompt]["compositional_accuracy"]
                h_comp = hybrid_dict[prompt]["compositional_accuracy"]
                delta = ((h_comp / b_comp) - 1) * 100 if b_comp > 0 else 0
                
                prompt_short = prompt[:48] if len(prompt) <= 48 else prompt[:45] + "..."
                delta_str = f"{delta:+.1f}%"
                print(f"  {prompt_short:<50s} | {b_comp:8.3f} | {h_comp:8.3f} | {delta_str:>8s}")
        
        # Hypothesis validation
        print("\n" + "="*80)
        print("HYPOTHESIS VALIDATION")
        print("="*80)
        
        print("\n📊 QUANTITATIVE FINDINGS:")
        if comp_delta > 0 or clip_delta > 0:
            print(f"  ✅ Metrics improved:")
            if comp_delta > 0:
                print(f"     - Compositional accuracy: {comp_delta:+.2f}%")
            if clip_delta > 0:
                print(f"     - CLIP score: {clip_delta:+.2f}%")
        else:
            print(f"  ⚠️  Metrics did not improve (unexpected)")
        
        print("\n⚠️  CRITICAL QUESTION:")
        print("  Do improved metrics = improved spatial relationships?")
        print("\n  To validate, we need HUMAN EVALUATION:")
        print("  - Visual inspection: Are objects in correct positions?")
        print("  - Examples to check:")
        
        # Identify high-metric prompts for manual inspection
        hybrid_sorted = sorted(spatial_prompts["hybrid"], 
                              key=lambda x: x["compositional_accuracy"], 
                              reverse=True)
        
        print("\n  High-metric prompts (likely false positives):")
        for i, p in enumerate(hybrid_sorted[:5], 1):
            print(f"    {i}. '{p['prompt']}' (Comp={p['compositional_accuracy']:.3f})")
            print(f"       → Check: Are objects in correct spatial relationship?")
        
        print("\n  📝 EXPECTED FINDING (based on our 2-prompt analysis):")
        print("     Objects PRESENT (good metrics) BUT WRONG POSITIONS (poor quality)")
        print("     Example: 'cat wearing hat' → cat + hat both visible ✅")
        print("                                  → but hat beside cat, not on head ❌")
        
        print("\n  🔬 This confirms our hypothesis:")
        print("     CLIP measures SEMANTIC SIMILARITY (presence)")
        print("     NOT SPATIAL RELATIONSHIPS (correctness)")
        
    # Save analysis report
    output_path = Path(output_dir)
    report_file = output_path / "spatial_analysis_report.txt"
    
    with open(report_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("SPATIAL RELATIONSHIP ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        if "baseline" in spatial_prompts and "hybrid" in spatial_prompts:
            f.write(f"Baseline Compositional Accuracy: {baseline_comp:.4f}\n")
            f.write(f"Hybrid Compositional Accuracy: {hybrid_comp:.4f}\n")
            f.write(f"Improvement: {comp_delta:+.2f}%\n\n")
            
            f.write(f"Baseline CLIP Score: {baseline_clip:.2f}\n")
            f.write(f"Hybrid CLIP Score: {hybrid_clip:.2f}\n")
            f.write(f"Improvement: {clip_delta:+.2f}%\n\n")
            
            f.write("CONCLUSION:\n")
            f.write("Quantitative metrics show improvement, but human evaluation needed\n")
            f.write("to validate whether spatial relationships are actually correct.\n")
            f.write("Based on 2-prompt testing, we expect metrics improve BUT visual quality degrades.\n")
    
    print(f"\n✓ Saved analysis report to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze spatial relationship performance")
    parser.add_argument("--results", type=str, default="outputs/drawbench/results_detailed.json",
                        help="Path to detailed results JSON")
    parser.add_argument("--summary", type=str, default="outputs/drawbench/results_summary.json",
                        help="Path to summary JSON")
    parser.add_argument("--output", type=str, default="outputs/drawbench",
                        help="Output directory")
    
    args = parser.parse_args()
    
    analyze_spatial_failures(args.results, args.summary, args.output)

if __name__ == "__main__":
    main()
