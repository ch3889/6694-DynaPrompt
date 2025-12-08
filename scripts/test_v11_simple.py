"""
Test DynaPrompt V11 Simple (V7 + Smart Retry Strategy).

This tests a simpler but more effective approach than V10:
- Try multiple seeds (different latent initializations)
- Pick the best result via CLIP scores
- No attention boosting increase (V10 showed it doesn't help)
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

from dynaprompt.dynaprompt_v11_simple import DynaPromptV11Simple


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
    print("DynaPrompt V11 Simple Test: Smart Retry Strategy")
    print("="*80)

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = "models/stable_diffusion_compvis/configs/stable-diffusion/v1-inference.yaml"
    ckpt_path = "models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt"

    # Test prompt
    test_prompt = "a silver car parked next to a golden bicycle"
    critical_attributes = ["silver car", "golden bicycle"]

    print(f"\nTest prompt: {test_prompt}")
    print(f"Critical attributes: {critical_attributes}")
    print(f"\nStrategy: Try 5 different seeds, pick the one with best CLIP scores")
    print(f"Hypothesis: Some seeds naturally produce better attribute binding\n")

    # Load model
    print("\n" + "="*80)
    print("Loading Stable Diffusion 1.5...")
    print("="*80)
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, device=device)

    # Initialize samplers
    ddim_sampler = DDIMSampler(model)
    tokenizer = model.cond_stage_model.tokenizer

    # Initialize V11 Simple
    print("\nInitializing DynaPrompt V11 Simple...")
    sampler = DynaPromptV11Simple(
        ddim_sampler=ddim_sampler,
        model=model,
        tokenizer=tokenizer,
        device=device,
        clip_model_id="openai/clip-vit-large-patch14",
        check_step=3,
        attention_threshold=0.05,
        max_retries=15,
        boost_factor=7.5,
        start_step_ratio=0.0,
        end_step_ratio=0.5,
    )

    # Generate with smart retry
    print("\n" + "="*80)
    print("Generating with smart retry...")
    print("="*80)

    shape = [1, 4, 64, 64]
    steps = 50

    samples, intermediates, metrics = sampler.sample_with_smart_retry(
        prompt=test_prompt,
        shape=shape,
        critical_attributes=critical_attributes,
        steps=steps,
        unconditional_guidance_scale=7.5,
        clip_threshold=0.25,
        num_seed_trials=5,  # Try 5 different seeds
        verbose=True,
    )

    # Decode latents to image
    print("\nDecoding best result to image...")
    with torch.no_grad():
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()

    # Save image
    output_dir = Path("data/images/v11_simple_test")
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
    print(f"  Total trials: {metrics['num_trials']}")
    print(f"  Validation passed: {metrics['validation_passed']}")
    print(f"  Best average CLIP score: {metrics['best_avg_score']:.3f}")
    print(f"\n  Best individual scores:")
    for attr, score in metrics['best_scores'].items():
        status = "✓" if score >= 0.25 else "✗"
        print(f"    {status} '{attr}': {score:.3f}")

    print(f"\nImage saved to: {output_path}")

    # Compare scores across all trials
    print(f"\n{'='*80}")
    print(f"Score Distribution Across {metrics['num_trials']} Trials:")
    print(f"{'='*80}")
    for i, attempt in enumerate(metrics['all_attempts']):
        print(f"Trial {i+1}: avg={attempt['avg_score']:.3f} ", end="")
        for attr, score in attempt['scores'].items():
            print(f"[{attr[:15]}:{score:.2f}] ", end="")
        print()

    print(f"\nComparison with previous versions:")
    print(f"  V7:  Both objects present, bicycle wrong color")
    print(f"  V10: Tried increasing boost (7.5x→60x), no improvement")
    print(f"  V11: Try different seeds, pick best → Should see variation!")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
