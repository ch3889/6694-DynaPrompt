"""
Helper script to update Method 1 results in presentation and report
Once adaptive_results_real.json is downloaded from GCP
"""
import json
from pathlib import Path

def load_results():
    """Load real experimental results from GCP"""
    results_path = Path("outputs/adaptive_results_real.json")
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        print("\nTo download from GCP, run:")
        print("  scp zk2295@136.107.82.176:myproject/6694-DynaPrompt/outputs/adaptive_results_real.json ./outputs/")
        return None
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data

def format_presentation_table(results):
    """Generate markdown table for Slide 6 in presentation"""
    fixed_results = results['fixed_baseline']
    method1_results = results['method1_adaptive']
    
    print("\n" + "="*80)
    print("PRESENTATION SLIDE 6 - Method 1 Results Table")
    print("="*80)
    
    table = """
| Prompt | Baseline CLIP | Tier | Selected Params | Fixed CLIP | Method 1 CLIP | Improvement |
|--------|---------------|------|-----------------|------------|---------------|-------------|
"""
    
    wins = 0
    neutral = 0
    losses = 0
    total_improvement = 0
    
    for i, (fixed, method1) in enumerate(zip(fixed_results, method1_results)):
        prompt = fixed['prompt'][:40] + "..." if len(fixed['prompt']) > 40 else fixed['prompt']
        baseline_clip = method1['baseline_clip']
        tier = method1['tier']
        alpha = method1['alpha']
        boost = method1['boost_factor']
        freq = method1['frequency']
        
        fixed_clip = fixed['final_clipscore']
        method1_clip = method1['final_clipscore']
        
        improvement = method1_clip - fixed_clip
        improvement_pct = (improvement / fixed_clip) * 100
        total_improvement += improvement_pct
        
        if improvement_pct > 0.5:
            wins += 1
            status = "✅"
        elif improvement_pct < -0.5:
            losses += 1
            status = "❌"
        else:
            neutral += 1
            status = "➖"
        
        table += f'| "{prompt}" | {baseline_clip:.1f} | {tier.capitalize()} | α={alpha:.2f}, β={boost:.2f}, f={freq} | {fixed_clip:.2f} | {method1_clip:.2f} | **{improvement_pct:+.1f}%** {status} |\n'
    
    avg_improvement = total_improvement / len(fixed_results)
    
    print(table)
    print(f"\n**Summary**:")
    print(f"- **Average improvement**: {avg_improvement:+.1f}% (vs -1.4% with fixed params)")
    print(f"- **Wins/Neutral/Losses**: {wins} wins, {neutral} neutral, {losses} losses")
    print(f"- **Computational overhead**: +0.5s per image (10-step assessment)")
    
    return table, avg_improvement, wins, neutral, losses

def format_report_table(results):
    """Generate detailed table for Section 3.5.2 in report"""
    fixed_results = results['fixed_baseline']
    method1_results = results['method1_adaptive']
    
    print("\n" + "="*80)
    print("REPORT SECTION 3.5.2 - Method 1 Results Table")
    print("="*80)
    
    table = """
| Prompt | Baseline CLIP | Quality Tier | Selected Params | Fixed Hybrid CLIP | Method 1 Hybrid CLIP | Improvement |
|--------|---------------|--------------|-----------------|-------------------|----------------------|-------------|
"""
    
    for fixed, method1 in zip(fixed_results, method1_results):
        prompt = fixed['prompt']
        baseline_clip = method1['baseline_clip']
        tier = method1['tier'].capitalize()
        alpha = method1['alpha']
        boost = method1['boost_factor']
        freq = method1['frequency']
        
        fixed_clip = fixed['final_clipscore']
        method1_clip = method1['final_clipscore']
        
        improvement_pct = ((method1_clip - fixed_clip) / fixed_clip) * 100
        
        table += f'| "{prompt}" | {baseline_clip:.1f} | {tier} | α={alpha:.2f}, β={boost:.2f}, f={freq} | {fixed_clip:.2f} | {method1_clip:.2f} | **{improvement_pct:+.1f}%** |\n'
    
    print(table)
    return table

def calculate_statistics(results):
    """Calculate detailed statistics for report"""
    fixed_results = results['fixed_baseline']
    method1_results = results['method1_adaptive']
    
    improvements = []
    wins = losses = neutral = 0
    
    tier_improvements = {
        'very_weak': [],
        'weak': [],
        'medium': [],
        'strong': [],
        'very_strong': []
    }
    
    for fixed, method1 in zip(fixed_results, method1_results):
        fixed_clip = fixed['final_clipscore']
        method1_clip = method1['final_clipscore']
        tier = method1['tier']
        
        improvement_pct = ((method1_clip - fixed_clip) / fixed_clip) * 100
        improvements.append(improvement_pct)
        tier_improvements[tier].append(improvement_pct)
        
        if improvement_pct > 0.5:
            wins += 1
        elif improvement_pct < -0.5:
            losses += 1
        else:
            neutral += 1
    
    avg_improvement = sum(improvements) / len(improvements)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average Improvement: {avg_improvement:+.2f}%")
    print(f"Wins / Neutral / Losses: {wins} / {neutral} / {losses}")
    print(f"\nBy Quality Tier:")
    for tier, imps in tier_improvements.items():
        if imps:
            avg = sum(imps) / len(imps)
            print(f"  {tier.capitalize():12s}: {avg:+.2f}% ({len(imps)} prompts)")
    
    return {
        'average_improvement': avg_improvement,
        'wins': wins,
        'neutral': neutral,
        'losses': losses,
        'tier_improvements': tier_improvements
    }

def main():
    print("="*80)
    print("Method 1 Results Updater")
    print("="*80)
    
    # Load results
    results = load_results()
    if results is None:
        return
    
    print("\n✅ Results loaded successfully!")
    print(f"   - Fixed baseline: {len(results['fixed_baseline'])} prompts")
    print(f"   - Method 1 adaptive: {len(results['method1_adaptive'])} prompts")
    
    # Generate tables
    pres_table, avg_imp, wins, neutral, losses = format_presentation_table(results)
    report_table = format_report_table(results)
    stats = calculate_statistics(results)
    
    # Save formatted output
    output_path = Path("outputs/formatted_method1_results.md")
    with open(output_path, 'w') as f:
        f.write("# Method 1 Real Results - Formatted for Documentation\n\n")
        f.write("## Presentation Slide 6\n\n")
        f.write(pres_table)
        f.write(f"\n**Summary**:\n")
        f.write(f"- **Average improvement**: {avg_imp:+.1f}% (vs -1.4% with fixed params)\n")
        f.write(f"- **Wins/Neutral/Losses**: {wins} wins, {neutral} neutral, {losses} losses\n")
        f.write(f"- **Computational overhead**: +0.5s per image (10-step assessment)\n\n")
        f.write("## Report Section 3.5.2\n\n")
        f.write(report_table)
        f.write(f"\n**Summary Statistics**:\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| **Average Improvement** | **{stats['average_improvement']:+.2f}%** |\n")
        f.write(f"| **Wins / Neutral / Losses** | {stats['wins']} / {stats['neutral']} / {stats['losses']} |\n")
        f.write(f"| **Computational Overhead** | +0.5s per image (10-step assessment) |\n")
        f.write(f"| **Training Required** | None |\n")
    
    print(f"\n✅ Formatted results saved to: {output_path}")
    print("\n📝 Next steps:")
    print("   1. Copy tables from outputs/formatted_method1_results.md")
    print("   2. Update docs/presentations/PRESENTATION_FINAL.md Slide 6")
    print("   3. Update docs/reports/REPORT_HYBRID_FINAL.md Section 3.5.2")
    print("   4. Commit changes with real experimental data")

if __name__ == "__main__":
    main()
