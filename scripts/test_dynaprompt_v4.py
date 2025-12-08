"""
Test DynaPrompt V4 with gradient-based latent refinement

This script tests V4 which uses gradient descent to actually steer the
denoising trajectory and create missing objects.
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
from dynaprompt.dynaprompt_v4_fixed import DynaPromptV4Sampler


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
        default="data/images/dynaprompt_v4_test",
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
        "--refinement_steps",
        type=int,
        default=20,
        help="Number of gradient steps per refinement (default: 20)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=20.0,
        help="Gradient descent learning rate (default: 20.0)"
    )
    parser.add_argument(
        "--feedback_interval",
        type=int,
        default=3,
        help="Analyze attention every N steps"
    )
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print("="*80)
    print("DynaPrompt V4 Test (Gradient-Based Refinement)")
    print("="*80)
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print(f"Refinement steps: {args.refinement_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Feedback interval: {args.feedback_interval}")
    print(f"Active phase: Steps 0-20 (structure formation with gradient steering)")
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

    dynaprompt_v4_sampler = DynaPromptV4Sampler(
        ddim_sampler=ddim_sampler,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        feedback_interval=args.feedback_interval,
        refinement_steps=args.refinement_steps,
        learning_rate=args.learning_rate,
        attention_threshold=0.3,
        start_step_ratio=0.0,  # Start immediately at step 0
        end_step_ratio=0.4     # End at step 20 (focus on structure phase)
    )

    # Sample
    print("\n" + "="*80)
    print("Starting DynaPrompt V4 sampling...")
    print("="*80)

    shape = [4, 64, 64]  # Latent shape for 512x512 image
    batch_size = 1

    with torch.no_grad():
        samples, intermediates = dynaprompt_v4_sampler.sample_with_dynaprompt(
            prompt=args.prompt,
            shape=(batch_size, *shape),
            steps=args.steps,
            unconditional_guidance_scale=7.5
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

        output_path = os.path.join(args.outdir, f"dynaprompt_v4_sample_{i:04d}.png")
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
        f.write(f"Refinement steps: {args.refinement_steps}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Feedback interval: {args.feedback_interval}\n")
        f.write(f"Active steps: 0-20 (gradient-based refinement)\n")

    print("\n" + "="*80)
    print("✓ DynaPrompt V4 test complete!")
    print(f"Output: {args.outdir}")
    print("\nTo compare with baseline, generate baseline with:")
    print(f"  python scripts/generate_baseline_single.py --prompt \"{args.prompt}\" --seed {args.seed}")
    print("="*80)


if __name__ == "__main__":
    main()
