"""
Simple test of DynaPrompt V8 (SDXL + CLIP Guidance).

Tests with one prompt to verify the implementation works before full evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynaprompt.dynaprompt_v8_dit_clip import DynaPromptV8CLIP


def main():
    print("="*80)
    print("DynaPrompt V8 Simple Test")
    print("Testing: SDXL + CLIP Guidance")
    print("="*80)

    # Test prompt (the problematic one from V7)
    test_prompt = "a silver car parked next to a golden bicycle"

    print(f"\nTest prompt: {test_prompt}")
    print(f"This prompt failed in V7 (bicycle wasn't golden)")
    print(f"Let's see if V8 with CLIP guidance fixes it!\n")

    # Initialize V8 (this will download SDXL ~13GB if not cached)
    print("Initializing DynaPrompt V8...")
    print("Note: First run will download SDXL model (~13GB)")
    print("This may take a few minutes...\n")

    sampler = DynaPromptV8CLIP(
        sdxl_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        clip_model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )

    # Generate with CLIP guidance
    print("\nGenerating image with CLIP guidance...")

    image, metrics = sampler.sample_with_clip_guidance(
        prompt=test_prompt,
        critical_attributes=["silver car", "golden bicycle"],  # Explicit attributes
        num_inference_steps=30,  # Fewer steps for faster testing
        clip_guidance_scale=150.0,
        clip_threshold=0.30,  # Raised threshold for stronger matches
        clip_guidance_steps=3,  # Increased from 2 to 3 optimization steps per denoising step
        height=512,  # Reduce resolution to save memory
        width=512,
        seed=42,
        verbose=True,
    )

    # Save result
    output_path = "data/images/v8_simple_test/silver_car_golden_bicycle.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sampler.save_image(image, output_path, verbose=True)

    # Print results
    print(f"\n{'='*80}")
    print(f"✓ Test Complete!")
    print(f"{'='*80}")
    print(f"\nMetrics:")
    print(f"  CLIP guidance applied: {metrics['guidance_applied_steps']}/{metrics['total_steps']} steps")
    print(f"  Average final CLIP score: {metrics['avg_final_clip_score']:.3f}")
    print(f"  Individual scores:")
    for attr, score in metrics['final_clip_scores'].items():
        print(f"    - '{attr}': {score:.3f}")

    print(f"\nImage saved to: {output_path}")
    print(f"\nCompare this with V7 result:")
    print(f"  V7: data/images/v7_cleaned_eval/hard_p00_a_silver_car_parked_next_to_a_golden_bicycle.png")
    print(f"  V8: {output_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
