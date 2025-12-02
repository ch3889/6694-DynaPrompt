"""
Robust Method 1 experiment runner with checkpointing and error recovery
Runs Fixed params vs Method 1 adaptive params on 10 DrawBench prompts
"""
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dynaprompt.hybrid import HybridDynaPrompt

# Test prompts with estimated baseline CLIPs
TEST_PROMPTS = [
    "a blue cube on top of a red sphere",
    "a golden bicycle next to a silver car", 
    "a cat wearing a red hat",
    "three red apples on a wooden table",
    "a small dog sitting under a large tree",
    "colorful balloons floating in the sky",
    "a white vase with pink flowers",
    "a person riding a horse",
    "a green frog on a lily pad",
    "a castle on a mountain peak"
]

# Method 1 decision rules
def select_adaptive_params(baseline_clip):
    """Select parameters based on baseline quality tier"""
    if baseline_clip < 35:
        return {'alpha': 0.10, 'boost_factor': 1.5, 'frequency': 3, 'tier': 'very_weak'}
    elif baseline_clip < 45:
        return {'alpha': 0.07, 'boost_factor': 1.3, 'frequency': 4, 'tier': 'weak'}
    elif baseline_clip < 55:
        return {'alpha': 0.05, 'boost_factor': 1.2, 'frequency': 5, 'tier': 'medium'}
    elif baseline_clip < 65:
        return {'alpha': 0.03, 'boost_factor': 1.1, 'frequency': 6, 'tier': 'strong'}
    else:
        return {'alpha': 0.01, 'boost_factor': 1.05, 'frequency': 8, 'tier': 'very_strong'}

def save_checkpoint(results, checkpoint_path):
    """Save intermediate results"""
    with open(checkpoint_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """Load previous results if they exist"""
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'fixed_baseline': [], 'method1_adaptive': []}

def main():
    print("="*60)
    print("Method 1 Robust Experiment Runner")
    print("="*60)
    
    # Setup paths
    checkpoint_path = Path("outputs/method1_checkpoint.json")
    final_path = Path("outputs/adaptive_results_real.json")
    checkpoint_path.parent.mkdir(exist_ok=True)
    
    # Load checkpoint if exists
    results = load_checkpoint(checkpoint_path)
    completed_fixed = len(results['fixed_baseline'])
    completed_method1 = len(results['method1_adaptive'])
    
    print(f"\n📊 Progress from checkpoint:")
    print(f"   Fixed: {completed_fixed}/{len(TEST_PROMPTS)}")
    print(f"   Method 1: {completed_method1}/{len(TEST_PROMPTS)}")
    
    # Initialize model
    print("\n🔧 Initializing HybridDynaPrompt...")
    model = HybridDynaPrompt()
    print("✅ Model loaded")
    
    # Fixed parameters
    FIXED_ALPHA = 0.07
    FIXED_BOOST = 1.3
    FIXED_FREQ = 4
    
    # Run Fixed baseline experiments (if not complete)
    if completed_fixed < len(TEST_PROMPTS):
        print(f"\n{'='*60}")
        print(f"FIXED BASELINE (α={FIXED_ALPHA}, β={FIXED_BOOST}, f={FIXED_FREQ})")
        print(f"{'='*60}")
        
        # Configure fixed params
        model.config.alpha = FIXED_ALPHA
        model.config.boost_factor = FIXED_BOOST
        model.config.feedback_frequency = FIXED_FREQ
        
        for i in range(completed_fixed, len(TEST_PROMPTS)):
            prompt = TEST_PROMPTS[i]
            print(f"\n[{i+1}/{len(TEST_PROMPTS)}] Prompt: \"{prompt}\"")
            print(f"   Params: α={FIXED_ALPHA}, β={FIXED_BOOST}, f={FIXED_FREQ}")
            
            try:
                start_time = time.time()
                result = model.generate(
                    prompt=prompt,
                    steps=50,
                    cfg_scale=7.5,
                    seed=42,
                    embedding_feedback=True,
                    attention_feedback=True
                )
                gen_time = time.time() - start_time
                
                # Save result
                results['fixed_baseline'].append({
                    'prompt': prompt,
                    'final_clipscore': float(result['final_clipscore']),
                    'compositional_accuracy': float(result.get('compositional_accuracy', 0.0)),
                    'generation_time': gen_time
                })
                
                print(f"   ✅ CLIP: {result['final_clipscore']:.2f}, Time: {gen_time:.1f}s")
                
                # Save checkpoint after each image
                save_checkpoint(results, checkpoint_path)
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                print(f"   Skipping and saving checkpoint...")
                save_checkpoint(results, checkpoint_path)
                continue
    
    # Run Method 1 adaptive experiments (if not complete)
    if completed_method1 < len(TEST_PROMPTS):
        print(f"\n{'='*60}")
        print(f"METHOD 1 ADAPTIVE")
        print(f"{'='*60}")
        
        for i in range(completed_method1, len(TEST_PROMPTS)):
            prompt = TEST_PROMPTS[i]
            print(f"\n[{i+1}/{len(TEST_PROMPTS)}] Prompt: \"{prompt}\"")
            
            try:
                # Step 1: Run baseline for 10 steps to assess quality
                print(f"   🔍 Assessing baseline quality (10 steps)...")
                model.config.alpha = 0.0  # No feedback for baseline
                model.config.boost_factor = 1.0
                
                baseline_result = model.generate(
                    prompt=prompt,
                    steps=10,
                    cfg_scale=7.5,
                    seed=42,
                    embedding_feedback=False,
                    attention_feedback=False
                )
                baseline_clip = baseline_result['final_clipscore']
                
                # Step 2: Select adaptive parameters
                params = select_adaptive_params(baseline_clip)
                print(f"   📊 Baseline CLIP: {baseline_clip:.1f} → Tier: {params['tier']}")
                print(f"   🎯 Selected: α={params['alpha']:.2f}, β={params['boost_factor']:.2f}, f={params['frequency']}")
                
                # Step 3: Run full generation with adaptive params
                model.config.alpha = params['alpha']
                model.config.boost_factor = params['boost_factor']
                model.config.feedback_frequency = params['frequency']
                
                start_time = time.time()
                result = model.generate(
                    prompt=prompt,
                    steps=50,
                    cfg_scale=7.5,
                    seed=42,
                    embedding_feedback=True,
                    attention_feedback=True
                )
                gen_time = time.time() - start_time
                
                # Save result
                results['method1_adaptive'].append({
                    'prompt': prompt,
                    'baseline_clip': float(baseline_clip),
                    'tier': params['tier'],
                    'alpha': params['alpha'],
                    'boost_factor': params['boost_factor'],
                    'frequency': params['frequency'],
                    'final_clipscore': float(result['final_clipscore']),
                    'compositional_accuracy': float(result.get('compositional_accuracy', 0.0)),
                    'generation_time': gen_time
                })
                
                improvement = result['final_clipscore'] - baseline_clip
                print(f"   ✅ Final CLIP: {result['final_clipscore']:.2f} ({improvement:+.2f}), Time: {gen_time:.1f}s")
                
                # Save checkpoint after each image
                save_checkpoint(results, checkpoint_path)
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                print(f"   Skipping and saving checkpoint...")
                save_checkpoint(results, checkpoint_path)
                continue
    
    # Save final results
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Final results saved: {final_path}")
    
    # Calculate summary
    if len(results['fixed_baseline']) == len(results['method1_adaptive']) == len(TEST_PROMPTS):
        fixed_avg = sum(r['final_clipscore'] for r in results['fixed_baseline']) / len(TEST_PROMPTS)
        method1_avg = sum(r['final_clipscore'] for r in results['method1_adaptive']) / len(TEST_PROMPTS)
        improvement = ((method1_avg - fixed_avg) / fixed_avg) * 100
        
        print(f"\n📊 Summary:")
        print(f"   Fixed avg CLIP: {fixed_avg:.2f}")
        print(f"   Method 1 avg CLIP: {method1_avg:.2f}")
        print(f"   Average improvement: {improvement:+.2f}%")
    else:
        print(f"\n⚠️  Incomplete results:")
        print(f"   Fixed: {len(results['fixed_baseline'])}/{len(TEST_PROMPTS)}")
        print(f"   Method 1: {len(results['method1_adaptive'])}/{len(TEST_PROMPTS)}")

if __name__ == "__main__":
    main()
