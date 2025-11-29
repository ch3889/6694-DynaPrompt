"""
Baseline vs Hybrid DynaPrompt Comparison
Generates side-by-side comparison with quantitative metrics
"""

import torch
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynaprompt.hybrid import HybridDynaPrompt
from dynaprompt.sd_loader import load_sd_model
from dynaprompt.core import DynaPrompt
from torchvision.utils import save_image, make_grid
import numpy as np
from tqdm import tqdm


def generate_baseline(sd_model, prompt, steps=50, seed=42):
    """Generate image without any feedback"""
    device = sd_model.device
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\nGenerating baseline (no feedback)...")
    
    # Encode prompt
    c = sd_model.encode_text([prompt])
    uc = sd_model.encode_text([""])
    
    # Create sampler
    sampler = sd_model.create_sampler('ddim')
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)
    
    # Initialize latent
    shape = [1, 4, 512 // 8, 512 // 8]
    latents = torch.randn(shape, device=device)
    
    # Denoising loop
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
        image = sd_model.model.first_stage_model.decode(latents_scaled)
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    
    return image


def compare_baseline_vs_hybrid(prompt, steps=30, seed=42, output_dir='outputs/baseline_vs_hybrid'):
    """
    Generate and compare baseline vs hybrid
    
    Returns:
        dict with images, metrics, and comparison
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"BASELINE VS HYBRID COMPARISON")
    print(f"{'='*80}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {steps}, Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Find checkpoint
    possible_paths = [
        'models/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/v1-5-pruned-emaonly.ckpt',
        'models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt'
    ]
    ckpt_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ckpt_path = path
            break
    
    results = {}
    
    # === BASELINE ===
    print("\n" + "="*60)
    print("GENERATING BASELINE")
    print("="*60)
    
    sd_model = load_sd_model(ckpt_path=ckpt_path, device=device)
    baseline_image = generate_baseline(sd_model, prompt, steps=steps, seed=seed)
    
    # Compute baseline metrics
    evaluator = DynaPrompt(device=device)
    baseline_clip = evaluator.compute_clipscore(baseline_image, prompt)
    baseline_comp = evaluator.compute_compositional_accuracy(baseline_image, prompt)
    
    results['baseline'] = {
        'image': baseline_image,
        'clipscore': float(baseline_clip),
        'compositional_accuracy': float(baseline_comp)
    }
    
    print(f"\n✓ Baseline Complete")
    print(f"  CLIP Score: {baseline_clip:.4f}")
    print(f"  Compositional Accuracy: {baseline_comp:.4f}")
    
    # Clear memory
    del sd_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # === HYBRID ===
    print("\n" + "="*60)
    print("GENERATING HYBRID (Embedding + Attention)")
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
        'clipscore': float(hybrid_result['final_clipscore']),
        'compositional_accuracy': float(hybrid_result['compositional_accuracy']),
        'metrics_history': hybrid_result['metrics_history'],
        'adaptive_stats': hybrid_result['adaptive_stats'],
        'generation_time': hybrid_result['generation_time']
    }
    
    print(f"\n✓ Hybrid Complete")
    print(f"  CLIP Score: {hybrid_result['final_clipscore']:.4f}")
    print(f"  Compositional Accuracy: {hybrid_result['compositional_accuracy']:.4f}")
    print(f"  Generation Time: {hybrid_result['generation_time']:.1f}s")
    
    # === COMPARISON ===
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON")
    print("="*60)
    
    clip_improvement = results['hybrid']['clipscore'] - results['baseline']['clipscore']
    comp_improvement = results['hybrid']['compositional_accuracy'] - results['baseline']['compositional_accuracy']
    
    print(f"\nCLIP Score:")
    print(f"  Baseline: {results['baseline']['clipscore']:.4f}")
    print(f"  Hybrid:   {results['hybrid']['clipscore']:.4f}")
    print(f"  Δ Change: {clip_improvement:+.4f} ({clip_improvement/results['baseline']['clipscore']*100:+.2f}%)")
    
    print(f"\nCompositional Accuracy:")
    print(f"  Baseline: {results['baseline']['compositional_accuracy']:.4f}")
    print(f"  Hybrid:   {results['hybrid']['compositional_accuracy']:.4f}")
    print(f"  Δ Change: {comp_improvement:+.4f} ({comp_improvement/results['baseline']['compositional_accuracy']*100:+.2f}%)")
    
    print(f"\nAdaptive Reweighting:")
    print(f"  Final Alpha: {results['hybrid']['adaptive_stats']['current_alpha']:.4f}")
    print(f"  Final Boost: {results['hybrid']['adaptive_stats']['current_boost']:.2f}")
    print(f"  Avg Alpha:   {results['hybrid']['adaptive_stats']['avg_alpha']:.4f}")
    print(f"  Avg Boost:   {results['hybrid']['adaptive_stats']['avg_boost']:.2f}")
    
    # === SAVE RESULTS ===
    os.makedirs(output_dir, exist_ok=True)
    
    # Save side-by-side comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = prompt.replace(' ', '_')[:50]
    
    # Prepare images for grid
    baseline_img = results['baseline']['image']
    hybrid_img = results['hybrid']['image']
    
    if baseline_img.dim() == 4:
        baseline_img = baseline_img.squeeze(0)
    if hybrid_img.dim() == 4:
        hybrid_img = hybrid_img.squeeze(0)
    
    # Create comparison grid
    grid = make_grid([baseline_img, hybrid_img], nrow=2, padding=10, pad_value=1.0)
    grid_path = os.path.join(output_dir, f'{safe_prompt}_{timestamp}_comparison.png')
    save_image(grid, grid_path)
    
    # Save individual images
    baseline_path = os.path.join(output_dir, f'{safe_prompt}_{timestamp}_baseline.png')
    hybrid_path = os.path.join(output_dir, f'{safe_prompt}_{timestamp}_hybrid.png')
    save_image(baseline_img, baseline_path)
    save_image(hybrid_img, hybrid_path)
    
    # Save metrics JSON
    metrics_data = {
        'prompt': prompt,
        'timestamp': timestamp,
        'steps': steps,
        'seed': seed,
        'baseline': {
            'clipscore': results['baseline']['clipscore'],
            'compositional_accuracy': results['baseline']['compositional_accuracy']
        },
        'hybrid': {
            'clipscore': results['hybrid']['clipscore'],
            'compositional_accuracy': results['hybrid']['compositional_accuracy'],
            'generation_time': results['hybrid']['generation_time'],
            'adaptive_stats': results['hybrid']['adaptive_stats']
        },
        'improvements': {
            'clipscore_delta': clip_improvement,
            'clipscore_percent': clip_improvement / results['baseline']['clipscore'] * 100,
            'comp_accuracy_delta': comp_improvement,
            'comp_accuracy_percent': comp_improvement / results['baseline']['compositional_accuracy'] * 100
        }
    }
    
    metrics_path = os.path.join(output_dir, f'{safe_prompt}_{timestamp}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS SAVED")
    print(f"{'='*60}")
    print(f"Comparison: {grid_path}")
    print(f"Baseline:   {baseline_path}")
    print(f"Hybrid:     {hybrid_path}")
    print(f"Metrics:    {metrics_path}")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Run comparisons on test prompts"""
    
    # Test prompts - challenging compositional cases
    test_prompts = [
        "a red cube and a blue sphere",
        "a golden retriever playing with a red ball"
    ]
    
    print("\n" + "="*80)
    print("BASELINE VS HYBRID EVALUATION")
    print("="*80)
    
    all_results = {}
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST {i}/{len(test_prompts)}")
        print(f"{'#'*80}")
        
        results = compare_baseline_vs_hybrid(prompt, steps=30, seed=42)
        all_results[prompt] = results
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    avg_clip_baseline = np.mean([r['baseline']['clipscore'] for r in all_results.values()])
    avg_clip_hybrid = np.mean([r['hybrid']['clipscore'] for r in all_results.values()])
    avg_comp_baseline = np.mean([r['baseline']['compositional_accuracy'] for r in all_results.values()])
    avg_comp_hybrid = np.mean([r['hybrid']['compositional_accuracy'] for r in all_results.values()])
    
    print(f"\nAverage CLIP Score:")
    print(f"  Baseline: {avg_clip_baseline:.4f}")
    print(f"  Hybrid:   {avg_clip_hybrid:.4f}")
    print(f"  Improvement: {avg_clip_hybrid - avg_clip_baseline:+.4f} ({(avg_clip_hybrid - avg_clip_baseline)/avg_clip_baseline*100:+.2f}%)")
    
    print(f"\nAverage Compositional Accuracy:")
    print(f"  Baseline: {avg_comp_baseline:.4f}")
    print(f"  Hybrid:   {avg_comp_hybrid:.4f}")
    print(f"  Improvement: {avg_comp_hybrid - avg_comp_baseline:+.4f} ({(avg_comp_hybrid - avg_comp_baseline)/avg_comp_baseline*100:+.2f}%)")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
