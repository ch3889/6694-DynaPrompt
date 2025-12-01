"""
Evaluate DynaPrompt on DrawBench benchmark
Compares Baseline vs Hybrid methods across multiple compositional categories
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynaprompt.wrapper import StableDiffusionWrapper
from dynaprompt.sd_loader import load_stable_diffusion

def extract_concepts(prompt):
    """Extract concepts from prompt for compositional accuracy evaluation"""
    # Remove articles and conjunctions
    stop_words = ['a', 'an', 'the', 'and', 'or', 'with', 'of', 'in', 'on', 'at', 'to', 'for']
    
    # Tokenize
    words = prompt.lower().split()
    
    # Remove stop words
    concepts = [w.strip('.,!?') for w in words if w not in stop_words]
    
    # Also create bigrams and trigrams for multi-word concepts
    bigrams = [f"{concepts[i]} {concepts[i+1]}" for i in range(len(concepts)-1)]
    trigrams = [f"{concepts[i]} {concepts[i+1]} {concepts[i+2]}" for i in range(len(concepts)-2)]
    
    # Combine all
    all_concepts = concepts + bigrams + trigrams
    
    # Remove duplicates while preserving order
    seen = set()
    unique_concepts = []
    for c in all_concepts:
        if c not in seen and len(c) > 1:  # Skip single characters
            seen.add(c)
            unique_concepts.append(c)
    
    return unique_concepts

def compute_clip_score(image, text, clip_model, clip_preprocess, device):
    """Compute CLIP similarity between image and text"""
    import clip
    
    # Preprocess image
    if isinstance(image, Image.Image):
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
    else:
        image_input = image
    
    # Tokenize text
    text_input = clip.tokenize([text]).to(device)
    
    # Compute features
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (image_features @ text_features.T).item()
        
        # Scale to 0-100
        score = (similarity + 1) * 50
    
    return score

def compute_compositional_accuracy(image, prompt, clip_model, clip_preprocess, device, threshold=20):
    """
    Compute compositional accuracy: fraction of concepts with CLIP score > threshold
    """
    concepts = extract_concepts(prompt)
    
    if len(concepts) == 0:
        return 0.0
    
    detected = 0
    for concept in concepts:
        score = compute_clip_score(image, concept, clip_model, clip_preprocess, device)
        if score > threshold:
            detected += 1
    
    return detected / len(concepts)

def evaluate_drawbench(
    prompts_file="data/drawbench_prompts.json",
    methods=["baseline", "hybrid"],
    output_dir="outputs/drawbench",
    subset_categories=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    device="cuda"
):
    """
    Evaluate methods on DrawBench benchmark
    
    Args:
        prompts_file: Path to DrawBench prompts JSON
        methods: List of methods to evaluate (baseline, hybrid)
        output_dir: Where to save generated images and results
        subset_categories: If provided, only evaluate these categories
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        device: Device to run on (cuda/cpu)
    """
    
    # Load prompts
    print(f"Loading prompts from {prompts_file}...")
    with open(prompts_file) as f:
        drawbench = json.load(f)
    
    # Filter by category if requested
    if subset_categories:
        drawbench = {
            k: v for k, v in drawbench.items() 
            if k in subset_categories
        }
        print(f"Evaluating subset: {subset_categories}")
    
    # Flatten prompts
    all_prompts = []
    for category, prompts in drawbench.items():
        for prompt in prompts:
            all_prompts.append({"category": category, "prompt": prompt})
    
    print(f"\n{'='*80}")
    print(f"DRAWBENCH EVALUATION")
    print(f"{'='*80}")
    print(f"Total prompts: {len(all_prompts)}")
    print(f"Categories: {len(drawbench)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load CLIP for evaluation
    print("Loading CLIP model for evaluation...")
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # Initialize Stable Diffusion models
    print("\nInitializing Stable Diffusion models...")
    models = {}
    
    if "baseline" in methods:
        print("  Loading Baseline (vanilla SD)...")
        baseline_pipe = load_stable_diffusion(device=device)
        models["baseline"] = {
            "pipe": baseline_pipe,
            "use_dynaprompt": False
        }
    
    if "hybrid" in methods:
        print("  Loading Hybrid (DynaPrompt)...")
        from dynaprompt.hybrid import apply_hybrid_dynaprompt
        from configs.dynaprompt_config import load_config
        
        config = load_config("configs/dynaprompt_config.yaml")
        hybrid_pipe = load_stable_diffusion(device=device)
        
        # Apply hybrid modifications
        apply_hybrid_dynaprompt(
            pipe=hybrid_pipe,
            config=config,
            device=device
        )
        
        models["hybrid"] = {
            "pipe": hybrid_pipe,
            "use_dynaprompt": True,
            "config": config
        }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluation loop
    results = {method: [] for method in methods}
    
    print("\nStarting evaluation...\n")
    
    for item in tqdm(all_prompts, desc="Evaluating prompts"):
        prompt = item["prompt"]
        category = item["category"]
        
        for method_name, model_info in models.items():
            try:
                # Generate image
                generator = torch.Generator(device=device).manual_seed(seed)
                
                output = model_info["pipe"](
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator
                )
                
                image = output.images[0]
                
                # Save image
                safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)[:50]
                img_dir = output_path / method_name / category
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"{safe_prompt}.png"
                image.save(img_path)
                
                # Compute metrics
                comp_acc = compute_compositional_accuracy(
                    image, prompt, clip_model, clip_preprocess, device
                )
                clip_score = compute_clip_score(
                    image, prompt, clip_model, clip_preprocess, device
                )
                
                results[method_name].append({
                    "prompt": prompt,
                    "category": category,
                    "compositional_accuracy": float(comp_acc),
                    "clip_score": float(clip_score),
                    "image_path": str(img_path)
                })
                
            except Exception as e:
                print(f"\n⚠️  Error on prompt '{prompt}' with {method_name}: {e}")
                continue
    
    # Aggregate results
    print("\n\nComputing summary statistics...")
    summary = {}
    
    for method_name, method_results in results.items():
        summary[method_name] = {}
        
        # Overall statistics
        comp_accs = [r["compositional_accuracy"] for r in method_results]
        clip_scores = [r["clip_score"] for r in method_results]
        
        summary[method_name]["overall"] = {
            "comp_acc_mean": float(np.mean(comp_accs)),
            "comp_acc_std": float(np.std(comp_accs)),
            "clip_score_mean": float(np.mean(clip_scores)),
            "clip_score_std": float(np.std(clip_scores)),
            "count": len(method_results)
        }
        
        # Per category statistics
        for category in drawbench.keys():
            cat_results = [r for r in method_results if r["category"] == category]
            if cat_results:
                cat_comp = [r["compositional_accuracy"] for r in cat_results]
                cat_clip = [r["clip_score"] for r in cat_results]
                
                summary[method_name][category] = {
                    "comp_acc_mean": float(np.mean(cat_comp)),
                    "comp_acc_std": float(np.std(cat_comp)),
                    "clip_score_mean": float(np.mean(cat_clip)),
                    "clip_score_std": float(np.std(cat_clip)),
                    "count": len(cat_results)
                }
    
    # Save results
    results_file = output_path / "results_detailed.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved detailed results to {results_file}")
    
    summary_file = output_path / "results_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_file}")
    
    # Print summary
    print_summary(summary, methods)
    
    # Compare methods if both evaluated
    if len(methods) == 2:
        compare_methods(summary, methods[0], methods[1])
    
    return results, summary

def print_summary(summary, methods):
    """Print formatted summary of results"""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for method_name in methods:
        if method_name not in summary:
            continue
            
        method_summary = summary[method_name]
        
        print(f"\n{method_name.upper()}:")
        print(f"  Overall:")
        print(f"    Compositional Accuracy: {method_summary['overall']['comp_acc_mean']:.4f} ± {method_summary['overall']['comp_acc_std']:.4f}")
        print(f"    CLIP Score: {method_summary['overall']['clip_score_mean']:.2f} ± {method_summary['overall']['clip_score_std']:.2f}")
        print(f"    Prompts evaluated: {method_summary['overall']['count']}")
        
        print(f"\n  Per Category:")
        for category, cat_stats in sorted(method_summary.items()):
            if category != "overall":
                print(f"    {category:20s}: Comp={cat_stats['comp_acc_mean']:.4f} (±{cat_stats['comp_acc_std']:.3f}), "
                      f"CLIP={cat_stats['clip_score_mean']:.2f} (±{cat_stats['clip_score_std']:.2f}), "
                      f"n={cat_stats['count']}")

def compare_methods(summary, method1, method2):
    """Compare two methods and print differences"""
    print("\n" + "="*80)
    print(f"COMPARISON: {method1.upper()} vs {method2.upper()}")
    print("="*80)
    
    # Overall comparison
    m1_comp = summary[method1]["overall"]["comp_acc_mean"]
    m2_comp = summary[method2]["overall"]["comp_acc_mean"]
    comp_delta = ((m2_comp / m1_comp) - 1) * 100
    
    m1_clip = summary[method1]["overall"]["clip_score_mean"]
    m2_clip = summary[method2]["overall"]["clip_score_mean"]
    clip_delta = ((m2_clip / m1_clip) - 1) * 100
    
    print(f"\nOverall:")
    print(f"  Compositional Accuracy: {m1_comp:.4f} → {m2_comp:.4f} ({comp_delta:+.2f}%)")
    print(f"  CLIP Score: {m1_clip:.2f} → {m2_clip:.2f} ({clip_delta:+.2f}%)")
    
    # Per category comparison
    print(f"\nPer Category:")
    
    categories = [k for k in summary[method1].keys() if k != "overall"]
    for category in sorted(categories):
        if category in summary[method2]:
            m1_cat_comp = summary[method1][category]["comp_acc_mean"]
            m2_cat_comp = summary[method2][category]["comp_acc_mean"]
            cat_comp_delta = ((m2_cat_comp / m1_cat_comp) - 1) * 100
            
            m1_cat_clip = summary[method1][category]["clip_score_mean"]
            m2_cat_clip = summary[method2][category]["clip_score_mean"]
            cat_clip_delta = ((m2_cat_clip / m1_cat_clip) - 1) * 100
            
            # Highlight significant changes
            comp_flag = "🔼" if cat_comp_delta > 5 else "🔽" if cat_comp_delta < -5 else "  "
            clip_flag = "🔼" if cat_clip_delta > 2 else "🔽" if cat_clip_delta < -2 else "  "
            
            print(f"  {category:20s}: Comp {comp_flag} {cat_comp_delta:+6.2f}%  |  CLIP {clip_flag} {cat_clip_delta:+6.2f}%")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Evaluate DynaPrompt on DrawBench")
    parser.add_argument("--prompts", type=str, default="data/drawbench_prompts.json",
                        help="Path to DrawBench prompts JSON")
    parser.add_argument("--methods", nargs="+", default=["baseline", "hybrid"],
                        choices=["baseline", "hybrid"],
                        help="Methods to evaluate")
    parser.add_argument("--output", type=str, default="outputs/drawbench",
                        help="Output directory for results")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Subset of categories to evaluate (default: all)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Run evaluation
    results, summary = evaluate_drawbench(
        prompts_file=args.prompts,
        methods=args.methods,
        output_dir=args.output,
        subset_categories=args.categories,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        device=args.device
    )

if __name__ == "__main__":
    main()
