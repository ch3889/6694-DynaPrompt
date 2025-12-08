"""
Test DynaPrompt V5 with early detection and adaptive restart
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
from dynaprompt.dynaprompt_v5 import DynaPromptV5Sampler


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
        default="a silver car parked next to a golden bicycle",
        help="Prompt to test"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/images/dynaprompt_v5_test",
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
        help="Random seed (default: 42, the problematic seed)"
    )
    parser.add_argument(
        "--check_step",
        type=int,
        default=15,
        help="Step at which to check composition (default: 15)"
    )
    parser.add_argument(
        "--attention_threshold",
        type=float,
        default=0.05,
        help="Minimum attention for object presence (default: 0.05)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum restart attempts (default: 3)"
    )
    parser.add_argument(
        "--critical_tokens",
        type=str,
        default=None,
        help="Comma-separated critical tokens (auto-detected if not provided)"
    )
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("="*80)
    print("DynaPrompt V5 Test (Early Detection + Adaptive Restart)")
    print("="*80)
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print(f"Check step: {args.check_step}")
    print(f"Attention threshold: {args.attention_threshold}")
    print(f"Max retries: {args.max_retries}")
    print("="*80 + "\n")

    # Load model
    config_path = SD_PATH / "configs" / "stable-diffusion" / "v1-inference.yaml"
    ckpt_path = SD_PATH / "v1-5-pruned-emaonly.ckpt"

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)

    # Get tokenizer
    tokenizer = model.cond_stage_model.tokenizer

    # Create samplers
    ddim_sampler = DDIMSampler(model)

    # Parse critical tokens if provided
    critical_tokens = None
    if args.critical_tokens:
        critical_tokens = [t.strip() for t in args.critical_tokens.split(',')]

    dynaprompt_v5_sampler = DynaPromptV5Sampler(
        ddim_sampler=ddim_sampler,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        check_step=args.check_step,
        attention_threshold=args.attention_threshold,
        max_retries=args.max_retries,
        noise_perturbation=0.3
    )

    # Sample
    print("\n" + "="*80)
    print("Starting DynaPrompt V5 sampling...")
    print("="*80)

    shape = [4, 64, 64]  # Latent shape for 512x512 image
    batch_size = 1

    samples, intermediates = dynaprompt_v5_sampler.sample_with_dynaprompt(
        prompt=args.prompt,
        shape=(batch_size, *shape),
        steps=args.steps,
        unconditional_guidance_scale=7.5,
        critical_tokens=critical_tokens,
        verbose=True
    )

    # Decode to image
    print("\nDecoding latents to image...")
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    # Save image
    os.makedirs(args.outdir, exist_ok=True)
    for i, x_sample in enumerate(x_samples):
        x_sample = 255. * x_sample.cpu().permute(1, 2, 0).numpy()
        img = Image.fromarray(x_sample.astype(np.uint8))

        output_path = os.path.join(args.outdir, f"dynaprompt_v5_sample_{i:04d}.png")
        img.save(output_path)
        print(f"✓ Saved: {output_path}")

    # Save prompt
    with open(os.path.join(args.outdir, "prompt.txt"), "w") as f:
        f.write(args.prompt)

    # Save parameters
    with open(os.path.join(args.outdir, "parameters.txt"), "w") as f:
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Check step: {args.check_step}\n")
        f.write(f"Attention threshold: {args.attention_threshold}\n")
        f.write(f"Max retries: {args.max_retries}\n")

    print("\n" + "="*80)
    print("✓ DynaPrompt V5 test complete!")
    print(f"Output: {args.outdir}")
    print("="*80)


if __name__ == "__main__":
    main()
