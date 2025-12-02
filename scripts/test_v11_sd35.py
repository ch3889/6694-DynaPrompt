"""
Test DynaPrompt V11 with Stable Diffusion 3.

This tests if upgrading the base model solves our compositional generation problem.

Expected outcome:
- SD 3 Medium has 2B parameters (vs SD 1.5's 983M)
- Better prompt adherence and compositional understanding
- Should achieve CLIP scores > 0.30 (vs SD 1.5's best of 0.226)
- Note: Using SD 3 Medium instead of SD 3.5 Large (ungated access)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from dynaprompt.dynaprompt_v11_sd35 import DynaPromptV11SD35


def main():
    print("="*80)
    print("DynaPrompt V11 + SD 3 Medium Test")
    print("="*80)

    # Test prompt (same as V11 Simple)
    test_prompt = "a silver car parked next to a golden bicycle"
    critical_attributes = ["silver car", "golden bicycle"]

    print(f"\nTest prompt: {test_prompt}")
    print(f"Critical attributes: {critical_attributes}")
    print(f"\nPrevious results:")
    print(f"  V11 + SD 1.5: Best avg 0.195 (silver: 0.226, golden: 0.171)")
    print(f"  Hypothesis: SD 3 Medium should score 0.30+ due to better composition")
    print(f"\n{'='*80}\n")

    # Initialize V11 with SD 3 Medium
    print("Initializing DynaPrompt V11 with SD 3 Medium...")
    print("Note: This will download ~5GB if not cached")

    sampler = DynaPromptV11SD35(
        sd35_model_id="stabilityai/stable-diffusion-3-medium-diffusers",
        clip_model_id="openai/clip-vit-large-patch14",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )

    # Generate with smart retry
    print("\n" + "="*80)
    print("Generating with smart retry...")
    print("="*80)

    best_image, metrics = sampler.sample_with_smart_retry(
        prompt=test_prompt,
        critical_attributes=critical_attributes,
        num_inference_steps=28,  # SD 3.5 default
        guidance_scale=7.0,
        height=512,  # Reduced for memory (15GB GPU)
        width=512,
        clip_threshold=0.25,
        num_seed_trials=5,
        verbose=True,
    )

    # Save result
    output_dir = Path("data/images/v11_sd35_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "silver_car_golden_bicycle.png"

    sampler.save_image(best_image, str(output_path))

    # Print final results
    print("\n" + "="*80)
    print("✓ Test Complete!")
    print("="*80)
    print(f"\nMetrics:")
    print(f"  Total trials: {metrics['num_trials']}")
    print(f"  Validation passed: {metrics['validation_passed']}")
    print(f"  Best average CLIP score: {metrics['best_avg_score']:.3f}")
    print(f"\n  Best individual scores:")
    for attr, score in metrics['best_scores'].items():
        status = "✓" if score >= 0.25 else "✗"
        print(f"    {status} '{attr}': {score:.3f}")

    # Compare with SD 1.5
    print(f"\n{'='*80}")
    print(f"Comparison: SD 3.5 vs SD 1.5")
    print(f"{'='*80}")
    print(f"\nSD 1.5 (V11 Simple) best scores:")
    print(f"  Silver car: 0.226")
    print(f"  Golden bicycle: 0.171")
    print(f"  Average: 0.195")

    print(f"\nSD 3 Medium (V11 SD3) best scores:")
    print(f"  Silver car: {metrics['best_scores'].get('silver car', 0):.3f}")
    print(f"  Golden bicycle: {metrics['best_scores'].get('golden bicycle', 0):.3f}")
    print(f"  Average: {metrics['best_avg_score']:.3f}")

    improvement_silver = metrics['best_scores'].get('silver car', 0) - 0.226
    improvement_golden = metrics['best_scores'].get('golden bicycle', 0) - 0.171
    improvement_avg = metrics['best_avg_score'] - 0.195

    print(f"\nImprovement:")
    print(f"  Silver car: {improvement_silver:+.3f} ({improvement_silver/0.226*100:+.1f}%)")
    print(f"  Golden bicycle: {improvement_golden:+.3f} ({improvement_golden/0.171*100:+.1f}%)")
    print(f"  Average: {improvement_avg:+.3f} ({improvement_avg/0.195*100:+.1f}%)")

    # Score distribution
    print(f"\n{'='*80}")
    print(f"Score Distribution Across {metrics['num_trials']} Trials:")
    print(f"{'='*80}")
    for i, attempt in enumerate(metrics['all_attempts']):
        print(f"Trial {i+1}: avg={attempt['avg_score']:.3f} ", end="")
        for attr, score in attempt['scores'].items():
            print(f"[{attr[:15]}:{score:.2f}] ", end="")
        print()

    print(f"\nImage saved to: {output_path}")
    print(f"\nConclusion:")
    if metrics['validation_passed']:
        print(f"  ✓ SD 3 Medium SOLVED the problem!")
        print(f"  The model was indeed the bottleneck.")
    elif metrics['best_avg_score'] > 0.25:
        print(f"  ⚠ SD 3 Medium improved scores but didn't fully pass.")
        print(f"  Consider: try SD 3.5 Large (requires auth) or spatial decomposition.")
    else:
        print(f"  ✗ SD 3 Medium didn't improve enough.")
        print(f"  Need to try SD 3.5 Large or spatial decomposition.")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
