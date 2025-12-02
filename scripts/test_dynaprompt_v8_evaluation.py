"""
DynaPrompt V8 Evaluation: 30 prompts (same as V7 baseline).

Uses SDXL + CLIP Guidance to compare against V7 baseline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
import json
from datetime import datetime
from dynaprompt.dynaprompt_v8_dit_clip import DynaPromptV8CLIP


# Same prompt sets as V7 evaluation
PROMPT_SETS = {
    "easy": [
        "a red car next to a blue truck",
        "a cat sitting on a wooden chair",
        "a dog playing with a yellow ball",
        "a white bird perched on a tree branch",
        "a brown horse standing in a green field",
        "a black bear eating honey from a jar",
        "a orange cat sleeping on a soft pillow",
        "a grey elephant drinking water from a pond",
        "a pink flower growing next to a stone",
        "a blue butterfly landing on a red rose",
    ],
    "medium": [
        "a silver laptop on a wooden desk",
        "a golden trophy next to a red ribbon",
        "a green bicycle leaning against a white wall",
        "a purple book resting on a glass table",
        "a yellow candle burning next to a mirror",
        "a brown guitar standing beside a window",
        "a black camera sitting on a leather bag",
        "a red umbrella leaning against a door",
        "a blue backpack lying on a bench",
        "a orange lamp standing on a nightstand",
    ],
    "hard": [
        "a silver car parked next to a golden bicycle",
        "a purple elephant standing next to a yellow giraffe",
        "a crystal vase containing rainbow flowers",
        "a turquoise dragon flying beside a pink unicorn",
        "a metallic robot holding a delicate butterfly",
        "a transparent glass sphere containing a miniature forest",
        "a neon green snake coiled around a blue gem",
        "a golden phoenix perched on a silver fountain",
        "a rainbow butterfly resting on a black rose",
        "a marble statue wearing a silk scarf",
    ]
}


def test_prompt(sampler, prompt, seed, output_dir, difficulty, prompt_idx, args):
    """Test a single prompt with V8."""
    # Generate unique seed if None
    if seed is None:
        seed = torch.randint(0, 1000000, (1,)).item()

    print(f"\n{'='*80}")
    print(f"[{difficulty.upper()}] Prompt {prompt_idx + 1}/10: {prompt}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    # Auto-detect critical attributes from prompt
    critical_attributes = sampler._extract_critical_attributes(prompt)

    # Generate
    start_time = datetime.now()

    image, metrics = sampler.sample_with_clip_guidance(
        prompt=prompt,
        critical_attributes=critical_attributes,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        clip_guidance_scale=args.clip_scale,
        clip_threshold=args.clip_threshold,
        clip_guidance_steps=args.clip_guidance_steps,
        height=args.height,
        width=args.width,
        seed=seed,
        verbose=False,  # Disable verbose for batch processing
    )

    generation_time = (datetime.now() - start_time).total_seconds()

    # Save image
    filename = f"{difficulty}_p{prompt_idx:02d}_{prompt.replace(' ', '_')[:50]}.png"
    filepath = output_dir / filename
    sampler.save_image(image, str(filepath))

    print(f"✓ Generated in {generation_time:.2f}s")
    print(f"  CLIP guidance applied: {metrics['guidance_applied_steps']}/{metrics['total_steps']} steps")
    print(f"  Average CLIP score: {metrics['avg_final_clip_score']:.3f}")
    print(f"  Saved: {filepath}")

    return {
        "filepath": str(filepath),
        "seed": seed,
        "generation_time": generation_time,
        "difficulty": difficulty,
        "prompt": prompt,
        "metrics": {
            "guidance_applied_steps": metrics['guidance_applied_steps'],
            "total_steps": metrics['total_steps'],
            "avg_final_clip_score": metrics['avg_final_clip_score'],
            "final_clip_scores": metrics['final_clip_scores'],
        }
    }


def main():
    parser = argparse.ArgumentParser(description="DynaPrompt V8 Evaluation")
    parser.add_argument("--difficulty", type=str, default="all", choices=["easy", "medium", "hard", "all"],
                        help="Which difficulty to test")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--clip_scale", type=float, default=150.0, help="CLIP guidance scale")
    parser.add_argument("--clip_threshold", type=float, default=0.25, help="CLIP score threshold")
    parser.add_argument("--clip_guidance_steps", type=int, default=0, help="Gradient steps per denoising step")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None for random)")
    parser.add_argument("--outdir", type=str, default="data/images/v8_eval", help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DynaPrompt V8 Evaluation (SDXL + CLIP Guidance)")
    print("="*80)
    print(f"Configuration:")
    print(f"  Inference steps: {args.steps}")
    print(f"  CFG scale: {args.cfg}")
    print(f"  CLIP guidance scale: {args.clip_scale}")
    print(f"  CLIP threshold: {args.clip_threshold}")
    print(f"  CLIP gradient steps: {args.clip_guidance_steps}")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  Seed: {'Different per prompt' if args.seed is None else args.seed}")
    print(f"  Output: {output_dir}")
    print("="*80 + "\n")

    # Initialize V8 sampler
    print("Initializing DynaPrompt V8 (SDXL + CLIP)...")
    sampler = DynaPromptV8CLIP(
        sdxl_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        clip_model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
    print("✓ V8 sampler ready\n")

    # Determine which difficulties to test
    if args.difficulty == "all":
        difficulties_to_test = ["easy", "medium", "hard"]
    else:
        difficulties_to_test = [args.difficulty]

    # Run evaluation
    all_results = []
    total_time = 0.0
    total_images = 0

    for difficulty in difficulties_to_test:
        prompts = PROMPT_SETS[difficulty]

        print(f"\n{'='*80}")
        print(f"Testing {difficulty.upper()} prompts ({len(prompts)} total)")
        print(f"{'='*80}")

        for i, prompt in enumerate(prompts):
            result = test_prompt(
                sampler=sampler,
                prompt=prompt,
                seed=args.seed,
                output_dir=output_dir,
                difficulty=difficulty,
                prompt_idx=i,
                args=args,
            )
            all_results.append(result)
            total_time += result["generation_time"]
            total_images += 1

    # Save results metadata
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "steps": args.steps,
            "cfg_scale": args.cfg,
            "clip_guidance_scale": args.clip_scale,
            "clip_threshold": args.clip_threshold,
            "clip_guidance_steps": args.clip_guidance_steps,
            "height": args.height,
            "width": args.width,
            "seed": args.seed,
        },
        "summary": {
            "total_images": total_images,
            "total_time_seconds": total_time,
            "average_time_per_image": total_time / total_images if total_images > 0 else 0,
        },
        "results": all_results
    }

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("✓ Evaluation Complete!")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  Total images: {total_images}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per image: {total_time / total_images:.2f}s")
    print(f"\nResults saved to: {results_path}")

    print(f"\nBreakdown by difficulty:")
    for difficulty in difficulties_to_test:
        count = sum(1 for r in all_results if r["difficulty"] == difficulty)
        avg_clip = sum(r["metrics"]["avg_final_clip_score"] for r in all_results if r["difficulty"] == difficulty) / count
        print(f"  {difficulty.upper()}: {count} images, avg CLIP score: {avg_clip:.3f}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
