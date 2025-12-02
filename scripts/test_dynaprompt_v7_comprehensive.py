"""
Comprehensive evaluation of DynaPrompt V7 with adaptive boosting.

Tests:
1. Easy prompts (3 prompts × 10 seeds = 30 images)
2. Medium prompts (3 prompts × 10 seeds = 30 images)
3. Hard prompts (3 prompts × 10 seeds = 30 images)

Total: 90 images across different difficulty levels
"""

import sys
import os
import torch
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add paths
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt/models/stable_diffusion_compvis')
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt')

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from dynaprompt.dynaprompt_v7 import DynaPromptV7Sampler
from PIL import Image
import numpy as np


# Test prompt sets (same as V6 evaluation for comparison)
PROMPT_SETS = {
    "easy": [
        "a red car next to a blue truck",
        "a cat sitting on a wooden chair",
        "a dog playing with a yellow ball",
    ],
    "medium": [
        "a silver laptop on a wooden desk",
        "a golden trophy next to a red ribbon",
        "a green bicycle leaning against a white wall",
    ],
    "hard": [
        "a silver car parked next to a golden bicycle",
        "a purple elephant standing next to a yellow giraffe",
        "a crystal vase containing rainbow flowers",
    ]
}

# Use different seeds for diversity
SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 5926, 8192, 16384]


def load_model():
    """Load Stable Diffusion model."""
    SD_PATH = Path("/home/cursedfox/6694-DynaPrompt/models/stable_diffusion_compvis")
    config_path = SD_PATH / "configs" / "stable-diffusion" / "v1-inference.yaml"
    ckpt_path = SD_PATH / "v1-5-pruned-emaonly.ckpt"

    print(f"Loading model from {ckpt_path}")

    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()

    return model


def test_prompt(model, dynaprompt_sampler, prompt, seed, output_dir, difficulty, prompt_idx):
    """Test a single prompt with V7."""
    print(f"\n{'='*80}")
    print(f"[{difficulty.upper()}] Prompt {prompt_idx + 1}: {prompt}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Generate with V7
    start_time = datetime.now()

    with torch.no_grad():
        shape = [1, 4, 64, 64]
        samples, intermediates = dynaprompt_sampler.sample_with_dynaprompt(
            prompt=prompt,
            shape=shape,
            steps=50,
            unconditional_guidance_scale=7.5,
            verbose=False  # Quiet mode for batch processing
        )

    generation_time = (datetime.now() - start_time).total_seconds()

    # Decode
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    # Save
    for i, x_sample in enumerate(x_samples):
        x_sample = 255. * x_sample.cpu().numpy().transpose(1, 2, 0)
        img = Image.fromarray(x_sample.astype(np.uint8))

        # Create filename
        prompt_sanitized = prompt.replace(" ", "_").replace(",", "")[:40]
        filename = f"{difficulty}_p{prompt_idx:02d}_{prompt_sanitized}_seed{seed}.png"
        filepath = output_dir / filename

        img.save(filepath)
        print(f"✓ Saved: {filepath.name} ({generation_time:.1f}s)")

    return {
        "filepath": str(filepath),
        "seed": seed,
        "generation_time": generation_time,
        "difficulty": difficulty,
        "prompt": prompt
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive V7 evaluation")
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of different seeds per prompt")
    parser.add_argument("--output", type=str, default="data/images/v7_comprehensive_eval")
    parser.add_argument("--boost_factor", type=float, default=5.0, help="Base boost factor (adaptive up to 15x)")
    parser.add_argument("--check_step", type=int, default=5, help="Early detection step")
    parser.add_argument("--max_retries", type=int, default=3, help="Max seed retries in Phase 1")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DynaPrompt V7 Comprehensive Evaluation")
    print("="*80)
    print(f"Difficulty levels: {args.difficulty}")
    print(f"Seeds per prompt: {args.num_seeds}")
    print(f"Boost factor: {args.boost_factor} (adaptive up to 15x)")
    print(f"Check step: {args.check_step}")
    print(f"Max retries: {args.max_retries}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")

    # Load model
    model = load_model()
    ddim_sampler = DDIMSampler(model)

    # Create V7 sampler with improved settings
    print("Initializing DynaPrompt V7 sampler...")
    dynaprompt_sampler = DynaPromptV7Sampler(
        ddim_sampler=ddim_sampler,
        model=model,
        tokenizer=model.cond_stage_model.tokenizer,
        check_step=args.check_step,
        attention_threshold=0.03,
        max_retries=args.max_retries,
        boost_factor=args.boost_factor,
        start_step_ratio=0.0,
        end_step_ratio=0.5,
    )
    print("✓ V7 sampler ready\n")

    # Determine which difficulties to test
    if args.difficulty == "all":
        difficulties_to_test = ["easy", "medium", "hard"]
    else:
        difficulties_to_test = [args.difficulty]

    # Track results
    all_results = []
    total_images = 0
    total_time = 0

    # Test each difficulty level
    for difficulty in difficulties_to_test:
        print(f"\n{'#'*80}")
        print(f"# Testing {difficulty.upper()} prompts")
        print(f"# {len(PROMPT_SETS[difficulty])} prompts × {args.num_seeds} seeds = {len(PROMPT_SETS[difficulty]) * args.num_seeds} images")
        print(f"{'#'*80}\n")

        difficulty_results = []

        for prompt_idx, prompt in enumerate(PROMPT_SETS[difficulty]):
            print(f"\n--- Prompt {prompt_idx + 1}/{len(PROMPT_SETS[difficulty])}: {prompt} ---")

            for seed_idx, seed in enumerate(SEEDS[:args.num_seeds]):
                print(f"  Seed {seed_idx + 1}/{args.num_seeds}: {seed}")

                result = test_prompt(
                    model=model,
                    dynaprompt_sampler=dynaprompt_sampler,
                    prompt=prompt,
                    seed=seed,
                    output_dir=output_dir,
                    difficulty=difficulty,
                    prompt_idx=prompt_idx
                )

                difficulty_results.append(result)
                all_results.append(result)
                total_images += 1
                total_time += result["generation_time"]

        # Print difficulty summary
        avg_time = sum(r["generation_time"] for r in difficulty_results) / len(difficulty_results)
        print(f"\n{difficulty.upper()} Summary:")
        print(f"  Images generated: {len(difficulty_results)}")
        print(f"  Average time: {avg_time:.1f}s per image")

    # Save results metadata
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "boost_factor": args.boost_factor,
            "check_step": args.check_step,
            "max_retries": args.max_retries,
            "attention_threshold": 0.03,
            "adaptive_boosting": True,
        },
        "summary": {
            "total_images": total_images,
            "total_time_seconds": total_time,
            "average_time_per_image": total_time / total_images if total_images > 0 else 0,
        },
        "results": all_results
    }

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    # Print final summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal images generated: {total_images}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per image: {total_time/total_images:.1f}s")
    print(f"\nOutput directory: {output_dir}")
    print(f"Results metadata: {results_file}")

    print(f"\nImages by difficulty:")
    for difficulty in difficulties_to_test:
        count = sum(1 for r in all_results if r["difficulty"] == difficulty)
        print(f"  {difficulty.upper()}: {count} images")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
