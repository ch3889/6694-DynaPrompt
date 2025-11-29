"""
Test script for Hybrid DynaPrompt (zk2295 + ch3889)

Compares three generation modes:
1. Baseline (no feedback)
2. zk2295 only (embedding feedback)
3. Hybrid (embedding + attention feedback)
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix pytorch_lightning compatibility issue
try:
    import pytorch_lightning
except ImportError:
    pass
else:
    if not hasattr(pytorch_lightning, 'utilities') or not hasattr(pytorch_lightning.utilities, 'distributed'):
        import pytorch_lightning.utilities
        class _DistributedShim:
            @staticmethod
            def rank_zero_only(fn):
                return fn
        pytorch_lightning.utilities.distributed = _DistributedShim()

from dynaprompt.hybrid import HybridDynaPrompt
from dynaprompt.wrapper import DynaPromptPipeline
from dynaprompt.sd_loader import load_sd_model
from torchvision.utils import save_image, make_grid
import json
from datetime import datetime


def compare_methods(prompt, seed=42, steps=50):
    """
    Compare baseline and hybrid approaches
    
    Args:
        prompt: Text prompt to test
        seed: Random seed for reproducibility
        steps: Number of denoising steps
        
    Returns:
        dict with results from both methods
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"COMPARING BASELINE VS HYBRID FOR: {prompt}")
    print(f"Device: {device}, Seed: {seed}, Steps: {steps}")
    print(f"{'='*80}\n")
    
    results = {}
    
    # === Method 1: Baseline (No Feedback) ===
    print("\n" + "="*60)
    print("METHOD 1: BASELINE (No Feedback)")
    print("="*60)
    
    # Try to find checkpoint (different paths on different systems)
    import os
    possible_paths = [
        'models/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/v1-5-pruned-emaonly.ckpt',
        'models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt'
    ]
    ckpt_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ckpt_path = path
            break
    
    sd = load_sd_model(ckpt_path=ckpt_path, device=device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Generate without feedback
    c = sd.encode_text([prompt])
    uc = sd.encode_text([""])
    sampler = sd.create_sampler('ddim')
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)
    
    shape = [1, 4, 512 // 8, 512 // 8]
    latents = torch.randn(shape, device=device)
    
    import numpy as np
    from tqdm import tqdm
    timesteps = sampler.ddim_timesteps
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    
    for i, step in tqdm(enumerate(time_range), total=total_steps, desc="Baseline"):
        index = total_steps - i - 1
        ts = torch.full((1,), step, device=device, dtype=torch.long)
        latents = sampler.p_sample_ddim(
            x=latents, c=c, t=ts, index=index,
            unconditional_guidance_scale=7.5,
            unconditional_conditioning=uc
        )[0]
    
    # Decode
    with torch.no_grad():
        latents_scaled = 1 / 0.18215 * latents
        baseline_image = sd.model.first_stage_model.decode(latents_scaled)
        baseline_image = torch.clamp((baseline_image + 1.0) / 2.0, min=0.0, max=1.0)
    
    # Compute metrics
    from dynaprompt.core import DynaPrompt
    evaluator = DynaPrompt(device=device)
    baseline_clip = evaluator.compute_clipscore(baseline_image, prompt)
    baseline_comp = evaluator.compute_compositional_accuracy(
        baseline_image, prompt
    )
    
    results['baseline'] = {
        'image': baseline_image,
        'clipscore': baseline_clip,
        'compositional_accuracy': baseline_comp,
        'method': 'Baseline (No Feedback)'
    }
    
    print(f"\n✓ Baseline Complete")
    print(f"  CLIP Score: {baseline_clip:.4f}")
    print(f"  Compositional Accuracy: {baseline_comp:.4f}")
    
    # === Method 2: Hybrid (Embedding + Attention) ===
    print("\n" + "="*60)
    print("METHOD 2: HYBRID (Embedding + Attention Feedback)")
    print("="*60)
    
    hybrid_pipeline = HybridDynaPrompt(ckpt_path=ckpt_path, device=device)
    hybrid_result = hybrid_pipeline.generate(
        prompt=prompt,
        steps=steps,
        cfg_scale=7.5,
        seed=seed,
        embedding_feedback=True,
        attention_feedback=True
    )
    
    results['hybrid'] = {
        'image': hybrid_result['image'],
        'clipscore': hybrid_result['final_clipscore'],
        'compositional_accuracy': hybrid_result['compositional_accuracy'],
        'metrics_history': hybrid_result['metrics_history'],
        'weak_tokens_history': hybrid_result['weak_tokens_history'],
        'method': 'Hybrid (zk2295 + ch3889)'
    }
    
    print(f"\n✓ Hybrid Complete")
    print(f"  CLIP Score: {hybrid_result['final_clipscore']:.4f}")
    print(f"  Compositional Accuracy: {hybrid_result['compositional_accuracy']:.4f}")
    
    hybrid_pipeline.cleanup()
    
    return results


def visualize_comparison(results, prompt, save_dir='outputs/hybrid_comparison'):
    """
    Create side-by-side comparison visualization
    
    Args:
        results: Results dict from compare_methods
        prompt: Text prompt
        save_dir: Directory to save comparison images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare images for grid (ensure correct shape and type)
    images = []
    for method_name in ['baseline', 'hybrid']:
        img = results[method_name]['image']
        
        # Ensure image is in correct shape: (C, H, W)
        if img.dim() == 4:  # (B, C, H, W)
            img = img.squeeze(0)  # Remove batch dimension
        
        # Ensure it's a float tensor in [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
        
        images.append(img)
    
    # Stack images for grid
    grid = make_grid(images, nrow=2, padding=10, pad_value=1.0)
    
    # Save grid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = prompt.replace(' ', '_')[:50]
    grid_path = os.path.join(save_dir, f'{safe_prompt}_{timestamp}_grid.png')
    save_image(grid, grid_path)
    
    # Save individual images
    for method_name, result in results.items():
        img_path = os.path.join(save_dir, f'{safe_prompt}_{timestamp}_{method_name}.png')
        save_image(result['image'], img_path)
    
    # Save metrics
    metrics = {
        'prompt': prompt,
        'timestamp': timestamp,
        'baseline': {
            'clipscore': float(results['baseline']['clipscore']),
            'compositional_accuracy': float(results['baseline']['compositional_accuracy'])
        },
        'hybrid': {
            'clipscore': float(results['hybrid']['clipscore']),
            'compositional_accuracy': float(results['hybrid']['compositional_accuracy']),
            'improvement_vs_baseline': {
                'clipscore': float(results['hybrid']['clipscore'] - results['baseline']['clipscore']),
                'compositional': float(results['hybrid']['compositional_accuracy'] - results['baseline']['compositional_accuracy'])
            }
        }
    }
    
    metrics_path = os.path.join(save_dir, f'{safe_prompt}_{timestamp}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS SAVED")
    print(f"{'='*80}")
    print(f"Grid: {grid_path}")
    print(f"Metrics: {metrics_path}")
    print(f"\nIMPROVEMENTS:")
    print(f"  Hybrid vs Baseline:")
    print(f"    CLIP Score: {metrics['hybrid']['improvement_vs_baseline']['clipscore']:+.4f}")
    print(f"    Compositional Accuracy: {metrics['hybrid']['improvement_vs_baseline']['compositional']:+.4f}")
    print(f"{'='*80}\n")


def main():
    """Run comprehensive comparison tests"""
    
    # Challenging compositional prompts
    test_prompts = [
        "a silver car parked next to a golden bicycle",
        "a red cube and a blue sphere on a wooden table",
        "a golden retriever playing with a red ball in a snowy park",
        "a tiny red bicycle next to a giant blue umbrella",
        "a purple elephant wearing a pink hat"
    ]
    
    print("\n" + "="*80)
    print("HYBRID DYNAPROMPT COMPREHENSIVE EVALUATION")
    print("Comparing: Baseline vs Hybrid")
    print("="*80)
    
    all_results = {}
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST {i}/{len(test_prompts)}")
        print(f"{'#'*80}")
        
        results = compare_methods(prompt, seed=42, steps=50)
        visualize_comparison(results, prompt)
        all_results[prompt] = results
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary statistics
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for prompt, results in all_results.items():
        print(f"\n{prompt}")
        print(f"  Baseline: CLIP={results['baseline']['clipscore']:.4f}, "
              f"Comp={results['baseline']['compositional_accuracy']:.4f}")
        print(f"  Hybrid:   CLIP={results['hybrid']['clipscore']:.4f}, "
              f"Comp={results['hybrid']['compositional_accuracy']:.4f}")
        
        clip_improve = results['hybrid']['clipscore'] - results['baseline']['clipscore']
        comp_improve = results['hybrid']['compositional_accuracy'] - results['baseline']['compositional_accuracy']
        
        print(f"  Improvements: CLIP={clip_improve:+.4f}, "
              f"Compositional={comp_improve:+.4f}")
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
