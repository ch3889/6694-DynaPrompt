"""
Text-to-Image with Stable Diffusion 2 + DiT-like architecture.
Uses diffusers library for proper text conditioning.
"""

import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--cfg', type=float, default=7.5)
    parser.add_argument('--output', type=str, default='outputs/sd_dit_output.png')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    args = parser.parse_args()

    print("Loading Stable Diffusion model from HuggingFace...")
    print("Note: First run will download ~5GB model weights")
    
    # Load SD pipeline (v1.5 or v2.1)
    model_id = "runwayml/stable-diffusion-v1-5"
    # Alternative: "stabilityai/stable-diffusion-2-1"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "mps":
        pipe = pipe.to(device)
        # MPS optimization
        torch.mps.set_per_process_memory_fraction(0.8)
    else:
        pipe = pipe.to(device)
    
    print(f"\nGenerating image for: '{args.prompt}'")
    print(f"Steps: {args.steps}, CFG: {args.cfg}, Size: {args.height}x{args.width}\n")
    
    # Generate
    torch.manual_seed(args.seed)
    
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.height,
        width=args.width,
        generator=torch.Generator(device=device).manual_seed(args.seed)
    ).images[0]
    
    # Save output
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else 'outputs', exist_ok=True)
    image.save(args.output)
    print(f"\n✓ Image saved to: {args.output}")


if __name__ == '__main__':
    main()
