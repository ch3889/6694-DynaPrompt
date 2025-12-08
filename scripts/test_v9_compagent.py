"""
Test DynaPrompt V9 (CompAgent-style with Ollama + qwen2.5).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynaprompt.dynaprompt_v9_compagent import DynaPromptV9CompAgent


def main():
    print("="*80)
    print("DynaPrompt V9 CompAgent Test")
    print("Using: Ollama (qwen2.5) + SDXL + CLIP")
    print("="*80)

    # Test prompt
    test_prompt = "a silver car parked next to a golden bicycle"

    print(f"\nTest prompt: {test_prompt}")
    print(f"This prompt failed in both V7 and V8")
    print(f"Let's see if CompAgent-style approach works!\n")

    # Initialize V9
    print("Initializing DynaPrompt V9...")
    sampler = DynaPromptV9CompAgent(
        sdxl_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        clip_model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        ollama_model="qwen2.5:7b",
    )

    # Generate
    print("\nGenerating with CompAgent approach...")

    image, metrics = sampler.sample_compositional(
        prompt=test_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        clip_threshold=0.30,
        max_retries=2,  # Try up to 3 CFG scales
        height=512,  # Reduced to avoid OOM
        width=512,
        seed=42,
        verbose=True,
    )

    # Save result
    output_path = "data/images/v9_test/silver_car_golden_bicycle.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sampler.save_image(image, output_path)

    # Print results
    print(f"\n{'='*80}")
    print(f"✓ Test Complete!")
    print(f"{'='*80}")
    print(f"\nMetrics:")
    print(f"  CFG scale used: {metrics['cfg_scale_used']:.1f}")
    print(f"  Attempts needed: {metrics['attempts']}")
    print(f"  Objects detected: {metrics['num_objects']}")
    print(f"  Average CLIP score: {metrics['avg_final_clip_score']:.3f}")
    print(f"\n  Individual scores:")
    for desc, score in metrics['final_clip_scores'].items():
        status = "✓" if score >= 0.30 else "✗"
        print(f"    {status} '{desc}': {score:.3f}")

    print(f"\nImage saved to: {output_path}")

    print(f"\nCompare with previous versions:")
    print(f"  V7: data/images/v7_cleaned_eval/hard_p00_a_silver_car_parked_next_to_a_golden_bicycle.png")
    print(f"  V8: data/images/v8_eval/hard_p00_a_silver_car_parked_next_to_a_golden_bicycle.png")
    print(f"  V9: {output_path}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
