"""
Test DynaPrompt V10 (V7 + CLIP Validation).

This tests Phase 1 of the U-Net Enhancement Plan:
- V7's attention boosting
- CLIP validation
- Adaptive boost increase for failing attributes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

# SD imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'stable_diffusion_compvis'))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from dynaprompt.dynaprompt_v10_clip_validation import DynaPromptV10CLIPValidation


def load_model_from_config(config, ckpt, device="cuda", verbose=False):
    """Load Stable Diffusion model."""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def main():
    print("="*80)
    print("DynaPrompt V10 Test: V7 + CLIP Validation")
    print("Phase 1 of U-Net Enhancement Plan")
    print("="*80)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = "models/stable_diffusion_compvis/configs/stable-diffusion/v1-inference.yaml"
    ckpt_path = "models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt"

    # Test prompt (the one V7 partially failed on)
    test_prompt = "a silver car parked next to a golden bicycle"
    critical_attributes = ["silver car", "golden bicycle"]

    print(f"\nTest prompt: {test_prompt}")
    print(f"Critical attributes: {critical_attributes}")
    print(f"\nV7 result: Generated both objects, but bicycle was silver/gray instead of golden")
    print(f"V10 goal: Validate attributes and retry with stronger boost if needed\n")

    # Load model
    print("\n" + "="*80)
    print("Loading Stable Diffusion 1.5...")
    print("="*80)
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, device=device)

    # Initialize samplers
    ddim_sampler = DDIMSampler(model)
    tokenizer = model.cond_stage_model.tokenizer

    # Initialize V10
    print("\nInitializing DynaPrompt V10...")
    sampler = DynaPromptV10CLIPValidation(
        ddim_sampler=ddim_sampler,
        model=model,
        tokenizer=tokenizer,
        device=device,
        clip_model_id="openai/clip-vit-large-patch14",
        check_step=3,           # Early detection
        attention_threshold=0.05,
        max_retries=15,         # Seed retries
        boost_factor=7.5,       # Initial boost
        start_step_ratio=0.0,
        end_step_ratio=0.5,
    )

    # Generate with CLIP validation
    print("\n" + "="*80)
    print("Generating with CLIP validation...")
    print("="*80)

    shape = [1, 4, 64, 64]  # SD 1.5 latent shape for 512x512
    steps = 50

    samples, intermediates, metrics = sampler.sample_with_clip_validation(
        prompt=test_prompt,
        shape=shape,
        critical_attributes=critical_attributes,
        steps=steps,
        unconditional_guidance_scale=7.5,
        clip_threshold=0.25,
        max_validation_retries=3,  # Try up to 4 times (1 initial + 3 retries)
        boost_increase_factor=2.0,  # 7.5x → 15x → 30x
        verbose=True,
    )

    # Decode latents to image
    print("\nDecoding latents to image...")
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

    # Save image
    output_dir = Path("data/images/v10_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "silver_car_golden_bicycle.png"

    image_array = (x_samples[0] * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(output_path)

    # Print final results
    print("\n" + "="*80)
    print("✓ Test Complete!")
    print("="*80)
    print(f"\nMetrics:")
    print(f"  Total attempts: {metrics['attempts']}")
    print(f"  Validation passed: {metrics['validation_passed']}")
    print(f"  Boost factors tried: {[f'{b:.1f}x' for b in metrics['boost_factors_tried']]}")
    print(f"\n  Final CLIP scores:")
    for attr, score in metrics['final_clip_scores'].items():
        status = "✓" if score >= 0.25 else "✗"
        print(f"    {status} '{attr}': {score:.3f}")

    avg_score = np.mean(list(metrics['final_clip_scores'].values()))
    print(f"\n  Average CLIP score: {avg_score:.3f}")

    print(f"\nImage saved to: {output_path}")

    print(f"\nComparison with previous versions:")
    print(f"  V7: data/images/v7_cleaned_eval/hard_p00_a_silver_car_parked_next_to_a_golden_bicycle.png")
    print(f"      → Both objects present, but bicycle color wrong (silver/gray instead of golden)")
    print(f"  V8: data/images/v8_eval/hard_p00_a_silver_car_parked_next_to_a_golden_bicycle.png")
    print(f"      → No bicycle, only car with golden accents")
    print(f"  V10: {output_path}")
    print(f"       → Should have both objects with correct colors!")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
