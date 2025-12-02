"""
Run Adaptive Parameter Selection Experiments

This script evaluates Method 1 (baseline assessment + rules) and Method 4 (meta-learning)
on a test set of DrawBench prompts to get real results for the presentation/report.

Usage:
    python scripts/run_adaptive_experiments.py --test-size 10 --output results/adaptive_results.json
"""

import argparse
import json
import torch
import clip
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.adaptive_parameter_methods import BaselineQualityAssessor, MetaLearningPredictor
from dynaprompt.hybrid import HybridDynaPrompt


# DrawBench test prompts spanning different quality tiers
DRAWBENCH_TEST_PROMPTS = [
    # Very Weak / Weak (CLIP < 45)
    "a cat wearing a red hat",
    "a green frog on a lily pad",
    "colorful balloons floating in the sky",
    
    # Medium (CLIP 45-55)
    "a person riding a horse",
    "three red apples on a wooden table",
    "a blue bird perched on a branch",
    
    # Strong (CLIP 55-65)
    "a blue cube on top of a red sphere",
    "a small dog sitting under a large tree",
    "a castle on a mountain peak",
    
    # Very Strong (CLIP > 65)
    "a golden bicycle next to a silver car",
    "a white vase with pink flowers",
    "a lighthouse on a rocky coast"
]


def setup_models(device='cuda'):
    """Initialize all required models"""
    print(f"Setting up models on {device}...")
    
    # Load CLIP
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load Stable Diffusion (baseline and hybrid)
    print("Loading Stable Diffusion models...")
    from dynaprompt.sd_loader import load_sd_model
    baseline_model = load_sd_model(device=device)
    
    # Load hybrid model
    hybrid_model = HybridDynaPrompt(device=device)
    
    return clip_model, baseline_model, hybrid_model


def generate_baseline(model, prompt, num_steps=50, seed=42):
    """Generate baseline image and compute CLIP score"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    result = model.sample(
        prompt=prompt,
        num_steps=num_steps,
        guidance_scale=7.5,
        seed=seed
    )
    
    return result


def generate_hybrid(model, prompt, alpha, boost_factor, frequency, num_steps=50, seed=42):
    """Generate hybrid image with specified parameters"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    result = model.generate(
        prompt=prompt,
        num_steps=num_steps,
        alpha=alpha,
        boost_factor=boost_factor,
        feedback_frequency=frequency,
        guidance_scale=7.5,
        seed=seed
    )
    
    return result


def compute_clip_score(clip_model, image, prompt, device='cuda'):
    """Compute CLIP score between image and text"""
    from PIL import Image
    import torchvision.transforms as transforms
    
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Encode
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_input = clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(text_input)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (image_features @ text_features.T).item()
        clip_score = (similarity + 1) * 50  # Map [-1,1] to [0,100]
    
    return clip_score


def run_method1_experiments(
    baseline_model,
    hybrid_model,
    clip_model,
    test_prompts,
    device='cuda'
):
    """Run Method 1: Baseline Assessment + Decision Rules"""
    print("\n" + "="*60)
    print("Running Method 1: Baseline Assessment + Decision Rules")
    print("="*60)
    
    # Initialize assessor
    assessor = BaselineQualityAssessor(clip_model, device=device)
    
    results = []
    
    for prompt in tqdm(test_prompts, desc="Method 1"):
        print(f"\nProcessing: {prompt}")
        
        # 1. Generate baseline
        baseline_result = generate_baseline(baseline_model, prompt)
        baseline_clip = compute_clip_score(
            clip_model, 
            baseline_result['image'], 
            prompt, 
            device
        )
        
        # 2. Assess baseline quality and select parameters
        assessment = assessor.assess_baseline_quality(
            baseline_model, 
            prompt,
            num_assessment_steps=10
        )
        
        params = assessment['recommended_params']
        tier = assessment['quality_tier']
        
        print(f"  Baseline CLIP: {baseline_clip:.2f}")
        print(f"  Quality Tier: {tier}")
        print(f"  Selected Params: alpha={params['alpha']:.3f}, "
              f"boost={params['boost_factor']:.2f}, freq={params['frequency']}")
        
        # 3. Generate with adaptive parameters
        hybrid_result = generate_hybrid(
            hybrid_model,
            prompt,
            alpha=params['alpha'],
            boost_factor=params['boost_factor'],
            frequency=params['frequency']
        )
        
        hybrid_clip = compute_clip_score(
            clip_model,
            hybrid_result['image'],
            prompt,
            device
        )
        
        improvement = hybrid_clip - baseline_clip
        
        print(f"  Hybrid CLIP: {hybrid_clip:.2f} (Δ{improvement:+.2f})")
        
        # Store results
        results.append({
            'prompt': prompt,
            'baseline_clip': float(baseline_clip),
            'quality_tier': tier,
            'selected_params': {
                'alpha': float(params['alpha']),
                'boost_factor': float(params['boost_factor']),
                'frequency': int(params['frequency'])
            },
            'hybrid_clip': float(hybrid_clip),
            'improvement': float(improvement)
        })
    
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
    print(f"Method 1 Summary:")
    print(f"  Average Improvement: {avg_improvement:+.2f}%")
    print(f"  Wins/Neutral/Losses: {wins}/{neutral}/{losses}")
    print(f"{'='*60}")
    
    return {'results': results, 'summary': summary}


def run_method4_experiments(
    baseline_model,
    hybrid_model,
    clip_model,
    test_prompts,
    training_prompts,
    device='cuda'
):
    """Run Method 4: Meta-Learning Predictor"""
    print("\n" + "="*60)
    print("Running Method 4: Meta-Learning Predictor")
    print("="*60)
    
    # Initialize predictor
    predictor = MetaLearningPredictor(clip_model, device=device)
    
    # Check if we have a trained model
    model_path = 'outputs/meta_predictor.pt'
    
    if Path(model_path).exists():
        print(f"Loading trained predictor from {model_path}")
        predictor.load(model_path)
    else:
        print("No trained model found. Training new predictor...")
        print(f"This will take ~2 hours for {len(training_prompts)} training prompts")
        
        # Collect training data
        training_data = predictor.collect_training_data(
            baseline_model,
            hybrid_model,
            training_prompts,
            num_samples_per_prompt=10
        )
        
        # Train predictor
        predictor.train(
            training_data,
            num_epochs=100,
            learning_rate=0.001,
            batch_size=16
        )
        
        # Save trained model
        Path('outputs').mkdir(exist_ok=True)
        predictor.save(model_path)
    
    # Run inference on test set
    results = []
    
    for prompt in tqdm(test_prompts, desc="Method 4"):
        print(f"\nProcessing: {prompt}")
        
        # 1. Generate baseline
        baseline_result = generate_baseline(baseline_model, prompt)
        baseline_clip = compute_clip_score(
            clip_model,
            baseline_result['image'],
            prompt,
            device
        )
        
        # 2. Predict optimal parameters
        predicted_params = predictor.predict_parameters(prompt, baseline_clip)
        
        print(f"  Baseline CLIP: {baseline_clip:.2f}")
        print(f"  Predicted Params: alpha={predicted_params['alpha']:.3f}, "
              f"boost={predicted_params['boost_factor']:.2f}, "
              f"freq={predicted_params['frequency']}")
        
        # 3. Generate with predicted parameters
        hybrid_result = generate_hybrid(
            hybrid_model,
            prompt,
            alpha=predicted_params['alpha'],
            boost_factor=predicted_params['boost_factor'],
            frequency=predicted_params['frequency']
        )
        
        hybrid_clip = compute_clip_score(
            clip_model,
            hybrid_result['image'],
            prompt,
            device
        )
        
        improvement = hybrid_clip - baseline_clip
        
        print(f"  Hybrid CLIP: {hybrid_clip:.2f} (Δ{improvement:+.2f})")
        
        # Store results
        results.append({
            'prompt': prompt,
            'baseline_clip': float(baseline_clip),
            'predicted_params': {
                'alpha': float(predicted_params['alpha']),
                'boost_factor': float(predicted_params['boost_factor']),
                'frequency': int(predicted_params['frequency'])
            },
            'hybrid_clip': float(hybrid_clip),
            'improvement': float(improvement)
        })
    
    # Summary statistics
    avg_improvement = np.mean([r['improvement'] for r in results])
    wins = sum(1 for r in results if r['improvement'] > 0.2)
    neutral = sum(1 for r in results if -0.2 <= r['improvement'] <= 0.2)
    losses = sum(1 for r in results if r['improvement'] < -0.2)
    
    summary = {
        'method': 'Method 4: Meta-Learning Predictor',
        'num_prompts': len(results),
        'avg_improvement': float(avg_improvement),
        'wins': wins,
        'neutral': neutral,
        'losses': losses
    }
    
    print(f"\n{'='*60}")
    print(f"Method 4 Summary:")
    print(f"  Average Improvement: {avg_improvement:+.2f}%")
    print(f"  Wins/Neutral/Losses: {wins}/{neutral}/{losses}")
    print(f"{'='*60}")
    
    return {'results': results, 'summary': summary}


def run_fixed_baseline_experiments(
    baseline_model,
    hybrid_model,
    clip_model,
    test_prompts,
    device='cuda'
):
    """Run fixed parameters baseline (alpha=0.07, boost=1.3, freq=4)"""
    print("\n" + "="*60)
    print("Running Fixed Parameters Baseline (alpha=0.07, boost=1.3, freq=4)")
    print("="*60)
    
    results = []
    
    for prompt in tqdm(test_prompts, desc="Fixed Baseline"):
        print(f"\nProcessing: {prompt}")
        
        # 1. Generate baseline
        baseline_result = generate_baseline(baseline_model, prompt)
        baseline_clip = compute_clip_score(
            clip_model,
            baseline_result['image'],
            prompt,
            device
        )
        
        # 2. Generate with fixed parameters
        hybrid_result = generate_hybrid(
            hybrid_model,
            prompt,
            alpha=0.07,
            boost_factor=1.3,
            frequency=4
        )
        
        hybrid_clip = compute_clip_score(
            clip_model,
            hybrid_result['image'],
            prompt,
            device
        )
        
        improvement = hybrid_clip - baseline_clip
        
        print(f"  Baseline CLIP: {baseline_clip:.2f}")
        print(f"  Hybrid CLIP: {hybrid_clip:.2f} (Δ{improvement:+.2f})")
        
        # Store results
        results.append({
            'prompt': prompt,
            'baseline_clip': float(baseline_clip),
            'hybrid_clip': float(hybrid_clip),
            'improvement': float(improvement)
        })
    
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
    print(f"Fixed Parameters Summary:")
    print(f"  Average Improvement: {avg_improvement:+.2f}%")
    print(f"  Wins/Neutral/Losses: {wins}/{neutral}/{losses}")
    print(f"{'='*60}")
    
    return {'results': results, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description='Run adaptive parameter selection experiments')
    parser.add_argument('--test-size', type=int, default=12, help='Number of test prompts')
    parser.add_argument('--training-size', type=int, default=30, help='Number of training prompts for Method 4')
    parser.add_argument('--output', type=str, default='outputs/adaptive_results.json', help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-method4', action='store_true', help='Skip Method 4 (saves time)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Adaptive Parameter Selection Experiments")
    print("="*60)
    print(f"Test size: {args.test_size}")
    print(f"Training size: {args.training_size}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Setup models
    clip_model, baseline_model, hybrid_model = setup_models(args.device)
    
    # Select test prompts
    test_prompts = DRAWBENCH_TEST_PROMPTS[:args.test_size]
    
    # For Method 4 training, use different prompts from DrawBench
    # (In practice, you'd load from a DrawBench JSON file)
    training_prompts = [
        "a red cube and a blue sphere",
        "two cats playing with a ball",
        # Add more training prompts here...
    ][:args.training_size]
    
    # Run experiments
    all_results = {}
    
    # 1. Fixed parameters baseline
    fixed_results = run_fixed_baseline_experiments(
        baseline_model,
        hybrid_model,
        clip_model,
        test_prompts,
        args.device
    )
    all_results['fixed'] = fixed_results
    
    # 2. Method 1
    method1_results = run_method1_experiments(
        baseline_model,
        hybrid_model,
        clip_model,
        test_prompts,
        args.device
    )
    all_results['method1'] = method1_results
    
    # 3. Method 4 (optional, takes longer)
    if not args.skip_method4:
        method4_results = run_method4_experiments(
            baseline_model,
            hybrid_model,
            clip_model,
            test_prompts,
            training_prompts,
            args.device
        )
        all_results['method4'] = method4_results
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Method':<40} {'Avg Δ':<10} {'W/N/L':<10}")
    print("-"*60)
    print(f"{all_results['fixed']['summary']['method']:<40} "
          f"{all_results['fixed']['summary']['avg_improvement']:+.2f}% "
          f"{all_results['fixed']['summary']['wins']}/"
          f"{all_results['fixed']['summary']['neutral']}/"
          f"{all_results['fixed']['summary']['losses']}")
    print(f"{all_results['method1']['summary']['method']:<40} "
          f"{all_results['method1']['summary']['avg_improvement']:+.2f}% "
          f"{all_results['method1']['summary']['wins']}/"
          f"{all_results['method1']['summary']['neutral']}/"
          f"{all_results['method1']['summary']['losses']}")
    if 'method4' in all_results:
        print(f"{all_results['method4']['summary']['method']:<40} "
              f"{all_results['method4']['summary']['avg_improvement']:+.2f}% "
              f"{all_results['method4']['summary']['wins']}/"
              f"{all_results['method4']['summary']['neutral']}/"
              f"{all_results['method4']['summary']['losses']}")
    print("="*60)


if __name__ == '__main__':
    main()
