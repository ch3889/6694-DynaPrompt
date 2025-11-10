"""
Comprehensive evaluation of DynaPrompt V6 across diverse prompts.

Tests:
1. Easy prompts (common objects, frequent pairings)
2. Medium prompts (less common but plausible)
3. Hard prompts (rare combinations, semantic conflicts)
4. CFG scale variations on difficult prompt
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt/models/stable_diffusion_compvis')
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt')

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from dynaprompt.dynaprompt_v6 import DynaPromptV6Sampler
from PIL import Image
import numpy as np


# Test prompt sets
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
        "a silver car parked next to a golden bicycle",  # Our problematic one
        "a purple elephant standing next to a yellow giraffe",
        "a crystal vase containing rainbow flowers",
    ]
}


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


def test_prompt(model, ddim_sampler, dynaprompt_sampler, prompt, seed, output_dir, test_name):
    """Test a single prompt with V6."""
    print(f"\n{'='*80}")
    print(f"Testing: {prompt}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Generate with V6
    with torch.no_grad():
        shape = [1, 4, 64, 64]
        samples, intermediates = dynaprompt_sampler.sample_with_dynaprompt(
            prompt=prompt,
            shape=shape,
            steps=50,
            unconditional_guidance_scale=7.5,
            verbose=True
        )

    # Decode
    print("\nDecoding latents to image...")
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    # Save
    for i, x_sample in enumerate(x_samples):
        x_sample = 255. * x_sample.cpu().numpy().transpose(1, 2, 0)
        img = Image.fromarray(x_sample.astype(np.uint8))

        # Create filename from prompt (sanitized)
        prompt_sanitized = prompt.replace(" ", "_").replace(",", "")[:50]
        filename = f"{test_name}_{prompt_sanitized}_seed{seed}.png"
        filepath = output_dir / filename

        img.save(filepath)
        print(f"✓ Saved: {filepath}")

    return filepath


def test_cfg_variations(model, ddim_sampler, prompt, seed, output_dir):
    """Test different CFG scales on a difficult prompt."""
    cfg_scales = [7.5, 9.5, 11.5, 13.5]

    print(f"\n{'='*80}")
    print(f"CFG Scale Variation Test")
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")
    print(f"Testing CFG scales: {cfg_scales}")
    print(f"{'='*80}\n")

    results = {}

    for cfg_scale in cfg_scales:
        print(f"\n--- Testing CFG scale: {cfg_scale} ---")

        # Reset seed for fair comparison
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Create V6 sampler with this CFG
        dynaprompt_sampler = DynaPromptV6Sampler(
            ddim_sampler=ddim_sampler,
            model=model,
            tokenizer=model.cond_stage_model.tokenizer,
            check_step=15,
            attention_threshold=0.05,
            max_retries=2,
            boost_factor=2.5,
            start_step_ratio=0.0,
            end_step_ratio=0.4
        )

        # Generate
        with torch.no_grad():
            shape = [1, 4, 64, 64]
            samples, intermediates = dynaprompt_sampler.sample_with_dynaprompt(
                prompt=prompt,
                shape=shape,
                steps=50,
                unconditional_guidance_scale=cfg_scale,
                verbose=False
            )

        # Decode
        with torch.no_grad():
            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        # Save
        for i, x_sample in enumerate(x_samples):
            x_sample = 255. * x_sample.cpu().numpy().transpose(1, 2, 0)
            img = Image.fromarray(x_sample.astype(np.uint8))

            filename = f"cfg_{cfg_scale}_bicycle_car_seed{seed}.png"
            filepath = output_dir / filename
            img.save(filepath)

            results[cfg_scale] = filepath
            print(f"✓ Saved CFG {cfg_scale}: {filepath}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--test_cfg", action="store_true", help="Test CFG scale variations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--output", type=str, default="data/images/v6_evaluation")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DynaPrompt V6 Comprehensive Evaluation")
    print("="*80)
    print(f"Difficulty levels: {args.difficulty}")
    print(f"Test CFG variations: {args.test_cfg}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")

    # Load model
    model = load_model()
    ddim_sampler = DDIMSampler(model)

    # Create V6 sampler
    dynaprompt_sampler = DynaPromptV6Sampler(
        ddim_sampler=ddim_sampler,
        model=model,
        tokenizer=model.cond_stage_model.tokenizer,
        check_step=15,
        attention_threshold=0.05,
        max_retries=args.max_retries,
        boost_factor=2.5,
        start_step_ratio=0.0,
        end_step_ratio=0.4
    )

    # Test prompts
    results = {}

    if args.difficulty == "all":
        difficulties_to_test = ["easy", "medium", "hard"]
    else:
        difficulties_to_test = [args.difficulty]

    for difficulty in difficulties_to_test:
        print(f"\n{'#'*80}")
        print(f"# Testing {difficulty.upper()} prompts")
        print(f"{'#'*80}\n")

        results[difficulty] = {}

        for prompt in PROMPT_SETS[difficulty]:
            filepath = test_prompt(
                model=model,
                ddim_sampler=ddim_sampler,
                dynaprompt_sampler=dynaprompt_sampler,
                prompt=prompt,
                seed=args.seed,
                output_dir=output_dir,
                test_name=difficulty
            )
            results[difficulty][prompt] = filepath

    # Test CFG variations if requested
    if args.test_cfg:
        print(f"\n{'#'*80}")
        print(f"# Testing CFG Scale Variations")
        print(f"{'#'*80}\n")

        cfg_results = test_cfg_variations(
            model=model,
            ddim_sampler=ddim_sampler,
            prompt="a silver car parked next to a golden bicycle",
            seed=args.seed,
            output_dir=output_dir
        )
        results["cfg_variations"] = cfg_results

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated {sum(len(v) for v in results.values() if isinstance(v, dict))} images")
    print(f"Output directory: {output_dir}")
    print("\nResults by difficulty:")

    for difficulty in difficulties_to_test:
        if difficulty in results:
            print(f"\n{difficulty.upper()}:")
            for prompt in results[difficulty]:
                print(f"  ✓ {prompt}")

    if "cfg_variations" in results:
        print(f"\nCFG VARIATIONS:")
        for cfg_scale, filepath in results["cfg_variations"].items():
            print(f"  ✓ CFG {cfg_scale}: {filepath.name}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
