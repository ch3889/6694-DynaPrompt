"""
Real Adaptive Parameter Experiments
Uses actual HybridDynaPrompt to test Method 1 parameter selection
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynaprompt.hybrid import HybridDynaPrompt

# Test prompts from DrawBench (with approximate baseline CLIP scores)
TEST_PROMPTS = [
    {"prompt": "a blue cube on top of a red sphere", "baseline_clip_est": 58.2},
    {"prompt": "a golden bicycle next to a silver car", "baseline_clip_est": 67.3},
    {"prompt": "a cat wearing a red hat", "baseline_clip_est": 41.7},
    {"prompt": "three red apples on a wooden table", "baseline_clip_est": 52.8},
    {"prompt": "a small dog sitting under a large tree", "baseline_clip_est": 63.1},
    {"prompt": "colorful balloons floating in the sky", "baseline_clip_est": 36.4},
    {"prompt": "a white vase with pink flowers", "baseline_clip_est": 69.2},
    {"prompt": "a person riding a horse", "baseline_clip_est": 48.9},
    {"prompt": "a green frog on a lily pad", "baseline_clip_est": 44.3},
    {"prompt": "a castle on a mountain peak", "baseline_clip_est": 59.7},
]


def select_parameters_method1(baseline_clip):
    """
    Method 1: Baseline Quality Assessment + Decision Rules
    """
    if baseline_clip < 35:
        tier = 'very_weak'
        params = {'alpha': 0.10, 'boost_factor': 1.5, 'frequency': 3}
    elif baseline_clip < 45:
        tier = 'weak'
        params = {'alpha': 0.07, 'boost_factor': 1.3, 'frequency': 4}
    elif baseline_clip < 55:
        tier = 'medium'
        params = {'alpha': 0.05, 'boost_factor': 1.2, 'frequency': 5}
    elif baseline_clip < 65:
        tier = 'strong'
        params = {'alpha': 0.03, 'boost_factor': 1.1, 'frequency': 6}
    else:
        tier = 'very_strong'
        params = {'alpha': 0.01, 'boost_factor': 1.05, 'frequency': 8}
    
    return tier, params


def run_method1_experiments(test_prompts, device='cuda'):
    """
    Run Method 1 experiments with actual hybrid generation
    """
    print("="*60)
    print("Method 1: Baseline Assessment + Decision Rules")
    print("="*60)
    
    # Initialize hybrid model
    hybrid = HybridDynaPrompt(device=device)
    
    results = []
    
    for test_case in test_prompts:
        prompt = test_case['prompt']
        baseline_clip_est = test_case['baseline_clip_est']
        
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"="*60)
        
        # Select parameters using Method 1
        tier, params = select_parameters_method1(baseline_clip_est)
        
        print(f"Estimated Baseline CLIP: {baseline_clip_est:.1f}")
        print(f"Quality Tier: {tier}")
        print(f"Selected Params: alpha={params['alpha']:.3f}, boost={params['boost_factor']:.2f}, freq={params['frequency']}")
        
        # Modify config temporarily
        original_alpha = hybrid.config['prompt_update']['update_alpha']
        original_boost = hybrid.config['attention']['boost_factor']
        original_freq = hybrid.config['feedback']['feedback_frequency']
        
        hybrid.config['prompt_update']['update_alpha'] = params['alpha']
        hybrid.config['attention']['boost_factor'] = params['boost_factor']
        hybrid.config['feedback']['feedback_frequency'] = params['frequency']
        
        # Also update attention modifier and reweighter
        hybrid.attention_modifier.boost_factor = params['boost_factor']
        hybrid.reweighter.alpha = params['alpha']
        hybrid.reweighter.boost_factor = params['boost_factor']
        
        # Generate with adaptive parameters
        print(f"\nGenerating with adaptive parameters...")
        result = hybrid.generate(
            prompt=prompt,
            steps=50,
            cfg_scale=7.5,
            seed=42,
            embedding_feedback=True,
            attention_feedback=True
        )
        
        hybrid_clip = result['final_clipscore']
        comp_acc = result['compositional_accuracy']
        improvement = hybrid_clip - baseline_clip_est
        
        print(f"\nResults:")
        print(f"  Hybrid CLIP: {hybrid_clip:.2f} (Δ{improvement:+.2f})")
        print(f"  Compositional Accuracy: {comp_acc:.4f}")
        print(f"  Generation Time: {result['generation_time']:.1f}s")
        
        # Restore original config
        hybrid.config['prompt_update']['update_alpha'] = original_alpha
        hybrid.config['attention']['boost_factor'] = original_boost
        hybrid.config['feedback']['feedback_frequency'] = original_freq
        hybrid.attention_modifier.boost_factor = original_boost
        hybrid.reweighter.alpha = original_alpha
        hybrid.reweighter.boost_factor = original_boost
        
        # Store results
        results.append({
            'prompt': prompt,
            'baseline_clip_est': float(baseline_clip_est),
            'quality_tier': tier,
            'selected_params': {
                'alpha': float(params['alpha']),
                'boost_factor': float(params['boost_factor']),
                'frequency': int(params['frequency'])
            },
            'hybrid_clip': float(hybrid_clip),
            'compositional_accuracy': float(comp_acc),
            'improvement': float(improvement),
            'generation_time': float(result['generation_time'])
        })
    
    # Cleanup
    hybrid.cleanup()
    
    # Summary statistics
    avg_improvement = np.mean([r['improvement'] for r in results])
    wins = sum(1 for r in results if r['improvement'] > 0.2)
    neutral = sum(1 for r in results if -0.2 <= r['improvement'] <= 0.2)
    losses = sum(1 for r in results if r['improvement'] < -0.2)
    
    summary = {
        'method': 'Method 1: Baseline Assessment + Rules',
        'num_prompts': len(results),
        'avg_improvement': float(avg_improvement),
        'wins': wins,
        'neutral': neutral,
        'losses': losses
    }
    
    print(f"\n{'='*60}")
    print(f"METHOD 1 SUMMARY")
    print(f"{'='*60}")
    print(f"Average Improvement: {avg_improvement:+.2f}")
    print(f"Wins/Neutral/Losses: {wins}/{neutral}/{losses}")
    print(f"{'='*60}")
    
    return {'results': results, 'summary': summary}


def run_fixed_baseline(test_prompts, device='cuda'):
    """
    Run fixed parameters baseline (alpha=0.07, boost=1.3, freq=4)
    """
    print("\n" + "="*60)
    print("Fixed Parameters Baseline (alpha=0.07, boost=1.3, freq=4)")
    print("="*60)
    
    # Initialize hybrid model
    hybrid = HybridDynaPrompt(device=device)
    
    results = []
    
    for test_case in test_prompts:
        prompt = test_case['prompt']
        baseline_clip_est = test_case['baseline_clip_est']
        
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"="*60)
        print(f"Estimated Baseline CLIP: {baseline_clip_est:.1f}")
        print(f"Using fixed params: alpha=0.07, boost=1.3, freq=4")
        
        # Generate with fixed parameters (already in config by default)
        result = hybrid.generate(
            prompt=prompt,
            steps=50,
            cfg_scale=7.5,
            seed=42,
            embedding_feedback=True,
            attention_feedback=True
        )
        
        hybrid_clip = result['final_clipscore']
        comp_acc = result['compositional_accuracy']
        improvement = hybrid_clip - baseline_clip_est
        
        print(f"\nResults:")
        print(f"  Hybrid CLIP: {hybrid_clip:.2f} (Δ{improvement:+.2f})")
        print(f"  Compositional Accuracy: {comp_acc:.4f}")
        
        # Store results
        results.append({
            'prompt': prompt,
            'baseline_clip_est': float(baseline_clip_est),
            'hybrid_clip': float(hybrid_clip),
            'compositional_accuracy': float(comp_acc),
            'improvement': float(improvement)
        })
    
    # Cleanup
    hybrid.cleanup()
    
    # Summary statistics
    avg_improvement = np.mean([r['improvement'] for r in results])
    wins = sum(1 for r in results if r['improvement'] > 0.2)
    neutral = sum(1 for r in results if -0.2 <= r['improvement'] <= 0.2)
    losses = sum(1 for r in results if r['improvement'] < -0.2)
    
    summary = {
        'method': 'Fixed Parameters (alpha=0.07, boost=1.3, freq=4)',
        'num_prompts': len(results),
        'avg_improvement': float(avg_improvement),
        'wins': wins,
        'neutral': neutral,
        'losses': losses
    }
    
    print(f"\n{'='*60}")
    print(f"FIXED PARAMETERS SUMMARY")
    print(f"{'='*60}")
    print(f"Average Improvement: {avg_improvement:+.2f}")
    print(f"Wins/Neutral/Losses: {wins}/{neutral}/{losses}")
    print(f"{'='*60}")
    
    return {'results': results, 'summary': summary}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*60)
    print("ADAPTIVE PARAMETER SELECTION EXPERIMENTS")
    print("="*60)
    print(f"Device: {device}")
    print(f"Test Prompts: {len(TEST_PROMPTS)}")
    print("="*60)
    
    # Run experiments
    fixed_results = run_fixed_baseline(TEST_PROMPTS, device)
    method1_results = run_method1_experiments(TEST_PROMPTS, device)
    
    # Save results
    output_path = Path('outputs/adaptive_results_real.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'fixed': fixed_results,
        'method1': method1_results,
        'note': 'Real experimental results from HybridDynaPrompt generation'
    }
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Method':<50} {'Avg Δ':<8} {'W/N/L'}")
    print("-"*60)
    print(f"{fixed_results['summary']['method']:<50} {fixed_results['summary']['avg_improvement']:+.2f}   "
          f"{fixed_results['summary']['wins']}/{fixed_results['summary']['neutral']}/{fixed_results['summary']['losses']}")
    print(f"{method1_results['summary']['method']:<50} {method1_results['summary']['avg_improvement']:+.2f}   "
          f"{method1_results['summary']['wins']}/{method1_results['summary']['neutral']}/{method1_results['summary']['losses']}")
    print("="*60)


if __name__ == '__main__':
    main()
