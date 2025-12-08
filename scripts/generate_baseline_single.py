"""
Generate a single baseline image with specific seed for comparison
"""

import argparse
import torch
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
SD_PATH = PROJECT_ROOT / "models" / "stable_diffusion_compvis"
sys.path.insert(0, str(SD_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt, verbose=False):
    """Load Stable Diffusion model from checkpoint."""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to generate"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/images/baseline_comparison",
        help="Output directory"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of DDIM steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("="*80)
    print("Baseline Generation")
    print("="*80)
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print("="*80 + "\n")

    # Load model
    config_path = SD_PATH / "configs" / "stable-diffusion" / "v1-inference.yaml"
    ckpt_path = SD_PATH / "v1-5-pruned-emaonly.ckpt"

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)

    # Create sampler
    sampler = DDIMSampler(model)

    # Encode prompt
    with torch.no_grad():
        uc = model.get_learned_conditioning([""])
        c = model.get_learned_conditioning([args.prompt])

    # Sample
    print("Sampling...")
    shape = [4, 64, 64]  # Latent shape for 512x512 image

    with torch.no_grad():
        samples, _ = sampler.sample(
            S=args.steps,
            conditioning=c,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning=uc,
            eta=0.0
        )

    # Decode to image
    print("Decoding...")
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    # Save image
    os.makedirs(args.outdir, exist_ok=True)
    for i, x_sample in enumerate(x_samples):
        x_sample = 255. * x_sample.cpu().permute(1, 2, 0).numpy()
        img = Image.fromarray(x_sample.astype(np.uint8))

        output_path = os.path.join(args.outdir, f"baseline_seed{args.seed}_{i:04d}.png")
        img.save(output_path)
        print(f"✓ Saved: {output_path}")

    # Save prompt and parameters
    with open(os.path.join(args.outdir, "prompt.txt"), "w") as f:
        f.write(args.prompt)

    with open(os.path.join(args.outdir, "parameters.txt"), "w") as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Steps: {args.steps}\n")

    print("\n" + "="*80)
    print("✓ Baseline generation complete!")
    print(f"Output: {args.outdir}")
    print("="*80)


if __name__ == "__main__":
    main()
