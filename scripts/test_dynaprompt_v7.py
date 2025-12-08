"""
Test script for DynaPrompt V7 (Early Detection + Adaptive Boosting).
"""

import sys
import os
import argparse
import torch
from pathlib import Path

# Add paths
from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SD_PATH = WORKSPACE_ROOT / 'models' / 'stable_diffusion_compvis'
sys.path.insert(0, str(SD_PATH / 'stable-diffusion'))
sys.path.insert(0, str(WORKSPACE_ROOT))

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from dynaprompt.dynaprompt_v7 import DynaPromptV7Sampler
from PIL import Image
import numpy as np


def load_model():
    """Load Stable Diffusion model."""
    config_path = SD_PATH / "stable-diffusion" / "configs" / "stable-diffusion" / "v1-inference.yaml"
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


def main():
    parser = argparse.ArgumentParser(description="Test DynaPrompt V7")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--outdir", type=str, default="data/images/dynaprompt_v7_test", help="Output directory")
    parser.add_argument("--check_step", type=int, default=3, help="Step to check composition (default: 3)")
    parser.add_argument("--max_retries", type=int, default=15, help="Max seed retries (default: 15)")
    parser.add_argument("--threshold", type=float, default=0.03, help="Attention threshold (default: 0.03)")
    parser.add_argument("--boost_factor", type=float, default=7.5, help="Base boost factor for Phase 2 (default: 7.5, adaptive up to 22.5x)")

    args = parser.parse_args()

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DynaPrompt V7 Test")
    print("="*80)
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print(f"CFG scale: {args.cfg}")
    print(f"Check step: {args.check_step}")
    print(f"Max retries: {args.max_retries}")
    print(f"Attention threshold: {args.threshold}")
    print(f"Boost factor: {args.boost_factor} (adaptive up to {args.boost_factor * 3}x)")
    print(f"Output: {outdir}")
    print("="*80 + "\n")

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Load model
    model = load_model()
    ddim_sampler = DDIMSampler(model)

    # Create V7 sampler
    dynaprompt_v7 = DynaPromptV7Sampler(
        ddim_sampler=ddim_sampler,
        model=model,
        tokenizer=model.cond_stage_model.tokenizer,
        check_step=args.check_step,
        attention_threshold=args.threshold,
        max_retries=args.max_retries,
        boost_factor=args.boost_factor,
        start_step_ratio=0.0,
        end_step_ratio=0.5,
    )

    # Generate
    print("\nGenerating with DynaPrompt V7...\n")

    with torch.no_grad():
        shape = [1, 4, 64, 64]
        samples, intermediates = dynaprompt_v7.sample_with_dynaprompt(
            prompt=args.prompt,
            shape=shape,
            steps=args.steps,
            unconditional_guidance_scale=args.cfg,
            verbose=True,
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

        filename = f"dynaprompt_v7_sample_{i:04d}.png"
        filepath = outdir / filename

        img.save(filepath)
        print(f"✓ Saved: {filepath}")

    print(f"\n{'='*80}")
    print("✓ DynaPrompt V7 test complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
