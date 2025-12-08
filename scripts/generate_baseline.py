#!/usr/bin/env python3
"""
Generate baseline images using vanilla Stable Diffusion.
This script generates images from test prompts without any dynamic guidance.
"""

import os
import sys
from pathlib import Path

# Add the stable diffusion directory to path
SD_PATH = Path(__file__).parent.parent / "models" / "stable_diffusion_compvis"
sys.path.insert(0, str(SD_PATH))

import torch
import argparse
from PIL import Image
from datetime import datetime


def load_prompts(prompt_file):
    """Load prompts from file, skipping comments and empty lines."""
    prompts = []
    with open(prompt_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


def generate_images(prompts, output_dir, ckpt_path, n_samples=1, steps=50, seed=42):
    """Generate images for each prompt using vanilla SD."""

    print(f"Generating {len(prompts)} prompts with {n_samples} samples each...")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Steps: {steps}, Seed: {seed}")
    print("-" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Generation timestamp: {datetime.now()}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Steps: {steps}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Number of prompts: {len(prompts)}\n")
        f.write(f"Samples per prompt: {n_samples}\n")
        f.write("\n" + "=" * 80 + "\n\n")

    # Generate images for each prompt
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Generating: {prompt}")

        # Create safe filename
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
        safe_prompt = safe_prompt[:80]  # Limit length
        prompt_dir = os.path.join(output_dir, f"{idx:03d}_{safe_prompt}")
        os.makedirs(prompt_dir, exist_ok=True)

        # Save prompt text
        with open(os.path.join(prompt_dir, "prompt.txt"), 'w') as f:
            f.write(prompt)

        # Build command to run txt2img
        cmd = [
            "python",
            str(SD_PATH / "scripts" / "txt2img.py"),
            "--prompt", f'"{prompt}"',
            "--ckpt", ckpt_path,
            "--outdir", prompt_dir,
            "--n_samples", str(n_samples),
            "--H", "512",
            "--W", "512",
            "--ddim_steps", str(steps),
            "--seed", str(seed),
        ]

        # Execute
        import subprocess
        result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✓ Generated successfully")
        else:
            print(f"  ✗ Generation failed")
            print(f"  Error: {result.stderr[:200]}")

        # Log to metadata
        with open(metadata_file, 'a') as f:
            f.write(f"Prompt {idx}: {prompt}\n")
            f.write(f"Output: {prompt_dir}\n")
            f.write(f"Status: {'Success' if result.returncode == 0 else 'Failed'}\n\n")

    print("\n" + "=" * 80)
    print(f"✓ Generation complete! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate baseline images from test prompts")
    parser.add_argument("--prompts", type=str,
                       default="data/test_prompts.txt",
                       help="Path to prompts file")
    parser.add_argument("--output", type=str,
                       default="data/images/baseline",
                       help="Output directory for generated images")
    parser.add_argument("--ckpt", type=str,
                       default="models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt",
                       help="Path to SD checkpoint")
    parser.add_argument("--n_samples", type=int, default=1,
                       help="Number of samples per prompt")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of DDIM steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Make paths absolute
    project_root = Path(__file__).parent.parent
    prompt_file = project_root / args.prompts
    output_dir = project_root / args.output
    ckpt_path = project_root / args.ckpt

    # Validate inputs
    if not prompt_file.exists():
        print(f"Error: Prompt file not found: {prompt_file}")
        return

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return

    # Load prompts
    prompts = load_prompts(prompt_file)
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    # Generate images
    generate_images(
        prompts=prompts,
        output_dir=str(output_dir),
        ckpt_path=str(ckpt_path),
        n_samples=args.n_samples,
        steps=args.steps,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
