"""
Generate DynaPrompt images for the worst baseline prompts and compare
"""

import json
import torch
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent
SD_PATH = PROJECT_ROOT / "models" / "stable_diffusion_compvis"
sys.path.insert(0, str(SD_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from dynaprompt.dynaprompt_sampler import DynaPromptSampler
import clip


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


def compute_clip_score(image_path, prompt, clip_model, preprocess, device):
    """Compute CLIP similarity between image and prompt."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()

    return similarity


def main():
    # Load baseline CLIP scores
    baseline_scores_path = PROJECT_ROOT / "data" / "images" / "baseline" / "clip_scores.json"
    with open(baseline_scores_path, 'r') as f:
        baseline_data = json.load(f)

    # Get worst 3 prompts
    sorted_prompts = sorted(baseline_data, key=lambda x: x['avg_similarity'])
    worst_prompts = sorted_prompts[:3]

    print("="*80)
    print("DynaPrompt Comparison: Worst Baseline Prompts")
    print("="*80)
    print("\nWorst baseline prompts:")
    for i, result in enumerate(worst_prompts, 1):
        print(f"  {i}. [{result['avg_similarity']:.3f}] {result['prompt']}")
    print("="*80 + "\n")

    # Load models
    print("Loading Stable Diffusion model...")
    config_path = SD_PATH / "configs" / "stable-diffusion" / "v1-inference.yaml"
    ckpt_path = SD_PATH / "v1-5-pruned-emaonly.ckpt"

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path)

    # Get tokenizer and text encoder
    tokenizer = model.cond_stage_model.tokenizer
    text_encoder = model.cond_stage_model.transformer

    # Create DynaPrompt sampler
    ddim_sampler = DDIMSampler(model)
    dynaprompt_sampler = DynaPromptSampler(
        ddim_sampler=ddim_sampler,
        model=model,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device="cuda",
        feedback_interval=10,
        boost_factor=1.5,
        use_adaptive=True
    )

    # Load CLIP for evaluation
    print("Loading CLIP for evaluation...")
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

    # Generate DynaPrompt images
    results = []
    shape = [4, 64, 64]
    batch_size = 1
    seed = 42

    output_dir = PROJECT_ROOT / "data" / "images" / "dynaprompt"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(worst_prompts, 1):
        prompt = result['prompt']
        baseline_score = result['avg_similarity']

        print(f"\n{'='*80}")
        print(f"[{idx}/3] Generating: {prompt}")
        print(f"Baseline CLIP score: {baseline_score:.3f}")
        print(f"{'='*80}\n")

        # Set seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Generate with DynaPrompt
        with torch.no_grad():
            samples, intermediates = dynaprompt_sampler.sample_with_dynaprompt(
                prompt=prompt,
                shape=(batch_size, *shape),
                steps=50,
                unconditional_guidance_scale=7.5
            )

        # Decode
        with torch.no_grad():
            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        # Save image
        prompt_dir = output_dir / f"{idx:03d}"
        prompt_dir.mkdir(exist_ok=True)

        for i, x_sample in enumerate(x_samples):
            x_sample = 255. * x_sample.cpu().permute(1, 2, 0).numpy()
            img = Image.fromarray(x_sample.astype(np.uint8))

            img_path = prompt_dir / f"sample_{i:04d}.png"
            img.save(img_path)

            # Compute CLIP score
            dynaprompt_score = compute_clip_score(
                img_path, prompt, clip_model, preprocess, "cuda"
            )

            print(f"\n✓ Saved: {img_path}")
            print(f"  DynaPrompt CLIP score: {dynaprompt_score:.3f}")
            print(f"  Improvement: {(dynaprompt_score - baseline_score):.3f} ({((dynaprompt_score/baseline_score - 1)*100):.1f}%)")

            results.append({
                'prompt': prompt,
                'baseline_score': baseline_score,
                'dynaprompt_score': dynaprompt_score,
                'improvement': dynaprompt_score - baseline_score,
                'improvement_pct': (dynaprompt_score/baseline_score - 1) * 100,
                'image_path': str(img_path)
            })

        # Save prompt
        with open(prompt_dir / "prompt.txt", "w") as f:
            f.write(prompt)

    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'avg_baseline': np.mean([r['baseline_score'] for r in results]),
                'avg_dynaprompt': np.mean([r['dynaprompt_score'] for r in results]),
                'avg_improvement': np.mean([r['improvement'] for r in results]),
                'avg_improvement_pct': np.mean([r['improvement_pct'] for r in results])
            }
        }, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nAverage baseline CLIP score:    {np.mean([r['baseline_score'] for r in results]):.3f}")
    print(f"Average DynaPrompt CLIP score:  {np.mean([r['dynaprompt_score'] for r in results]):.3f}")
    print(f"Average improvement:            {np.mean([r['improvement'] for r in results]):.3f} ({np.mean([r['improvement_pct'] for r in results]):.1f}%)")

    print("\nPer-prompt results:")
    for r in results:
        print(f"\n  Prompt: {r['prompt']}")
        print(f"    Baseline:    {r['baseline_score']:.3f}")
        print(f"    DynaPrompt:  {r['dynaprompt_score']:.3f}")
        print(f"    Improvement: +{r['improvement']:.3f} ({r['improvement_pct']:+.1f}%)")

    print(f"\n✓ Results saved to: {results_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
