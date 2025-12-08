"""
Analyze V8 evaluation results and compare with V7 baseline.
"""

import json
from pathlib import Path

# Load V7 results
v7_path = Path("data/images/v7_cleaned_eval/evaluation_results.json")
with open(v7_path) as f:
    v7_results = json.load(f)

# Load V8 results
v8_path = Path("data/images/v8_eval/evaluation_results.json")
with open(v8_path) as f:
    v8_results = json.load(f)

print("="*80)
print("DynaPrompt V7 vs V8 Analysis")
print("="*80)

# Overall comparison
print("\n1. PERFORMANCE COMPARISON:")
print("-" * 80)
print(f"V7 Average time: {v7_results['summary']['average_time_per_image']:.2f}s")
print(f"V8 Average time: {v8_results['summary']['average_time_per_image']:.2f}s")
print(f"V8 is {v7_results['summary']['average_time_per_image'] / v8_results['summary']['average_time_per_image']:.2f}x faster")

# CLIP scores by difficulty
print("\n2. CLIP SCORES BY DIFFICULTY:")
print("-" * 80)

for difficulty in ["easy", "medium", "hard"]:
    v8_scores = [r["metrics"]["avg_final_clip_score"] for r in v8_results["results"] if r["difficulty"] == difficulty]
    avg_score = sum(v8_scores) / len(v8_scores) if v8_scores else 0

    print(f"\n{difficulty.upper()}:")
    print(f"  Average CLIP score: {avg_score:.3f}")
    print(f"  Min: {min(v8_scores):.3f}, Max: {max(v8_scores):.3f}")

    # List prompts with low scores (< 0.25)
    low_score_prompts = [r for r in v8_results["results"]
                         if r["difficulty"] == difficulty and r["metrics"]["avg_final_clip_score"] < 0.25]

    if low_score_prompts:
        print(f"\n  Low CLIP scores (< 0.25):")
        for r in low_score_prompts:
            print(f"    - {r['prompt']}: {r['metrics']['avg_final_clip_score']:.3f}")
            # Show which attributes failed
            for attr, score in r["metrics"]["final_clip_scores"].items():
                if score < 0.25:
                    print(f"      ⚠ '{attr}': {score:.3f}")

# CLIP guidance effectiveness
print("\n3. CLIP GUIDANCE EFFECTIVENESS:")
print("-" * 80)

total_guidance_steps = sum(r["metrics"]["guidance_applied_steps"] for r in v8_results["results"])
total_steps = sum(r["metrics"]["total_steps"] for r in v8_results["results"])
print(f"CLIP guidance applied: {total_guidance_steps}/{total_steps} steps ({100*total_guidance_steps/total_steps:.1f}%)")

# By difficulty
for difficulty in ["easy", "medium", "hard"]:
    diff_results = [r for r in v8_results["results"] if r["difficulty"] == difficulty]
    diff_guidance = sum(r["metrics"]["guidance_applied_steps"] for r in diff_results)
    diff_total = sum(r["metrics"]["total_steps"] for r in diff_results)
    print(f"{difficulty.upper()}: {diff_guidance}/{diff_total} steps ({100*diff_guidance/diff_total:.1f}%)")

# Problem identification
print("\n4. PROBLEM IDENTIFICATION:")
print("-" * 80)

# Find the worst performing prompts
all_results = sorted(v8_results["results"], key=lambda x: x["metrics"]["avg_final_clip_score"])
print("\nTop 10 WORST performing prompts (lowest CLIP scores):")
for i, r in enumerate(all_results[:10], 1):
    print(f"{i}. [{r['difficulty'].upper()}] {r['prompt']}")
    print(f"   CLIP score: {r['metrics']['avg_final_clip_score']:.3f}")
    print(f"   Attribute scores:")
    for attr, score in r["metrics"]["final_clip_scores"].items():
        status = "✓" if score >= 0.25 else "✗"
        print(f"     {status} '{attr}': {score:.3f}")
    print()

print("\nTop 10 BEST performing prompts (highest CLIP scores):")
for i, r in enumerate(reversed(all_results[-10:]), 1):
    print(f"{i}. [{r['difficulty'].upper()}] {r['prompt']}")
    print(f"   CLIP score: {r['metrics']['avg_final_clip_score']:.3f}")

# Key findings
print("\n5. KEY FINDINGS:")
print("-" * 80)

# Count how many prompts have avg score < 0.25
low_score_count = sum(1 for r in v8_results["results"] if r["metrics"]["avg_final_clip_score"] < 0.25)
print(f"• Prompts with average CLIP score < 0.25: {low_score_count}/30 ({100*low_score_count/30:.1f}%)")

# Count individual attribute failures
attr_failures = 0
total_attrs = 0
for r in v8_results["results"]:
    for attr, score in r["metrics"]["final_clip_scores"].items():
        total_attrs += 1
        if score < 0.25:
            attr_failures += 1

print(f"• Individual attribute failures (< 0.25): {attr_failures}/{total_attrs} ({100*attr_failures/total_attrs:.1f}%)")

# Average CLIP score overall
overall_avg = sum(r["metrics"]["avg_final_clip_score"] for r in v8_results["results"]) / len(v8_results["results"])
print(f"• Overall average CLIP score: {overall_avg:.3f}")

print("\n" + "="*80)
