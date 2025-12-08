"""
Extract sample results from DrawBench evaluation for presentation
"""

import json
import sys
from pathlib import Path

def main():
    results_file = Path('outputs/drawbench_phase1/results_detailed.json')
    
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        print("Please download results from GCP first:")
        print("  scp -r zk2295@136.107.82.176:myproject/6694-DynaPrompt/outputs/drawbench_phase1 ./outputs/")
        sys.exit(1)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    baseline = data['baseline']
    hybrid = data['hybrid']
    
    # Create lookup for hybrid results
    hybrid_dict = {r['prompt']: r for r in hybrid}
    
    # Select 10 representative prompts (2 from each category)
    categories = {
        'Colors': 2,
        'Positional': 2,
        'Counting': 2,
        'Descriptions': 2,
        'Conflicting': 2
    }
    
    selected = []
    for cat, count in categories.items():
        cat_results = [r for r in baseline if r['category'] == cat][:count]
        selected.extend(cat_results)
    
    # Print results table
    print("\n" + "="*120)
    print("SAMPLE RESULTS: 10 Prompts from DrawBench Evaluation")
    print("="*120)
    print(f"{'#':<3} | {'Prompt':<45} | {'Category':<15} | {'Base CLIP':<10} | {'Hybrid CLIP':<10} | {'Delta':<8} | Visual")
    print("-"*120)
    
    total_base = 0
    total_hybrid = 0
    
    for idx, base_result in enumerate(selected, 1):
        prompt = base_result['prompt']
        category = base_result['category']
        base_clip = base_result['clip_score']
        
        # Find matching hybrid result
        if prompt in hybrid_dict:
            hybrid_result = hybrid_dict[prompt]
            hybrid_clip = hybrid_result['clip_score']
        else:
            hybrid_clip = 0
            print(f"Warning: No hybrid result for '{prompt}'")
        
        delta = hybrid_clip - base_clip
        delta_str = f"{delta:+.1f}"
        
        # Visual quality assessment based on delta
        if delta < -2.0:
            visual = "Much worse ❌❌"
        elif delta < -0.5:
            visual = "Worse ❌"
        elif delta < 0.2:
            visual = "Similar ≈"
        else:
            visual = "Better ✓"
        
        # Truncate prompt if too long
        prompt_short = prompt[:43] + "..." if len(prompt) > 45 else prompt
        
        print(f"{idx:<3} | {prompt_short:<45} | {category:<15} | {base_clip:<10.2f} | {hybrid_clip:<10.2f} | {delta_str:<8} | {visual}")
        
        total_base += base_clip
        total_hybrid += hybrid_clip
    
    print("-"*120)
    avg_base = total_base / len(selected)
    avg_hybrid = total_hybrid / len(selected)
    avg_delta = avg_hybrid - avg_base
    print(f"{'AVG':<3} | {'(10 prompts)':<45} | {'ALL':<15} | {avg_base:<10.2f} | {avg_hybrid:<10.2f} | {avg_delta:+.2f}   | {'Hybrid worse overall' if avg_delta < 0 else 'Hybrid better overall'}")
    print("="*120)
    
    # Generate markdown table
    print("\n\n" + "="*120)
    print("MARKDOWN TABLE FOR PRESENTATION")
    print("="*120)
    print("\n```markdown")
    print("| # | Prompt | Category | Baseline CLIP | Hybrid CLIP | Delta | Visual Quality |")
    print("|---|--------|----------|---------------|-------------|-------|----------------|")
    
    for idx, base_result in enumerate(selected, 1):
        prompt = base_result['prompt']
        category = base_result['category']
        base_clip = base_result['clip_score']
        
        if prompt in hybrid_dict:
            hybrid_result = hybrid_dict[prompt]
            hybrid_clip = hybrid_result['clip_score']
        else:
            hybrid_clip = 0
        
        delta = hybrid_clip - base_clip
        
        if delta < -2.0:
            visual = "Much worse ❌❌"
        elif delta < -0.5:
            visual = "Worse ❌"
        elif delta < 0.2:
            visual = "Similar ≈"
        else:
            visual = "Better ✓"
        
        # Truncate prompt for table
        prompt_short = prompt[:40] + "..." if len(prompt) > 42 else prompt
        
        print(f"| {idx} | {prompt_short} | {category} | {base_clip:.1f} | {hybrid_clip:.1f} | {delta:+.1f} | {visual} |")
    
    print(f"| | **AVERAGE** | | **{avg_base:.1f}** | **{avg_hybrid:.1f}** | **{avg_delta:+.1f}** | **{'Hybrid worse' if avg_delta < 0 else 'Hybrid better'}** |")
    print("```\n")
    
    print("\n" + "="*120)
    print("Copy the markdown table above into your presentation!")
    print("="*120)


if __name__ == "__main__":
    main()
