"""
Test DynaPrompt V8 Simple (SDXL + CLIP validation with adaptive CFG).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynaprompt.dynaprompt_v8_simple import DynaPromptV8Simple


def main():
    print("="*80)
    print("DynaPrompt V8 Simple Test")
    print("Testing: SDXL + CLIP Validation (Adaptive CFG)")
    print("="*80)

    # Test prompt (the problematic one from V7)
    test_prompt = "a silver car parked next to a golden bicycle"

    print(f"\nTest prompt: {test_prompt}")
    print(f"This prompt failed in V7 (bicycle wasn't golden)")
    print(f"Let's see if V8 Simple with adaptive CFG fixes it!\n")

    # Initialize V8 Simple
    print("Initializing DynaPrompt V8 Simple...")
    sampler = DynaPromptV8Simple(
        sdxl_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        clip_model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )

    # Generate with CLIP validation
    print("\nGenerating with CLIP validation...")

    image, metrics = sampler.sample_with_validation(
        prompt=test_prompt,
        critical_attributes=["silver car", "golden bicycle"],
        num_inference_steps=30,
        base_guidance_scale=7.5,
        clip_threshold=0.30,
        max_cfg_scale=12.0,
        max_retries=2,  # Try up to 3 different CFG scales
        height=768,  # Increased resolution
        width=768,
        seed=42,
        verbose=True,
    )

    # Save result
    output_path = "data/images/v8_simple_test/silver_car_golden_bicycle_simple.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sampler.save_image(image, output_path)

    # Print results
    print(f"\n{'='*80}")
    print(f"✓ Test Complete!")
    print(f"{'='*80}")
    print(f"\nMetrics:")
    print(f"  CFG scale used: {metrics['cfg_scale_used']:.1f}")
    print(f"  Attempts needed: {metrics['attempts']}")
    print(f"  Average CLIP score: {metrics['avg_final_clip_score']:.3f}")
    print(f"  Individual scores:")
    for attr, score in metrics['final_clip_scores'].items():
        status = "✓" if score >= 0.30 else "✗"
        print(f"    {status} '{attr}': {score:.3f}")

    print(f"\nImage saved to: {output_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
