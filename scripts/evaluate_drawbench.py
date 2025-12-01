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
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynaprompt.hybrid import HybridDynaPrompt
from dynaprompt.sd_loader import load_sd_model
from dynaprompt.core import DynaPrompt

def generate_baseline(sd_model, prompt, steps=50, seed=42):
    """Generate image without any feedback"""
    device = sd_model.device
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
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
    
    for i, step in enumerate(time_range):
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
    
    # Find checkpoint
    possible_paths = [
        'models/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/v1-5-pruned-emaonly.ckpt',
        'models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt'
    ]
    ckpt_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ckpt_path = path
            print(f"  Found checkpoint: {path}")
            break
    
    if ckpt_path is None:
        print("  Warning: No checkpoint found, will use default")
    
    models = {}
    
    if "baseline" in methods:
        print("  Loading Baseline (vanilla SD)...")
        models["baseline"] = {
            "type": "baseline",
            "ckpt_path": ckpt_path
        }
    
    if "hybrid" in methods:
        print("  Loading Hybrid (DynaPrompt)...")
        models["hybrid"] = {
            "type": "hybrid",
            "ckpt_path": ckpt_path
        }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluation loop - Process one method at a time to conserve memory
    results = {method: [] for method in methods}
    
    print("\nStarting evaluation...\n")
    
    # Process each method separately to avoid OOM
    for method_name, model_info in models.items():
        print(f"\n{'='*60}")
        print(f"Processing {method_name.upper()} method ({len(all_prompts)} prompts)")
        print(f"{'='*60}\n")
        
        # Load model once for this method
        current_model = None
        if method_name == "baseline":
            print("Loading Baseline SD model...")
            current_model = load_sd_model(ckpt_path=model_info["ckpt_path"], device=device)
        elif method_name == "hybrid":
            print("Loading Hybrid DynaPrompt model...")
            current_model = HybridDynaPrompt(ckpt_path=model_info["ckpt_path"], device=device)
        
        # Generate all images for this method
        prompt_count = 0
        for item in tqdm(all_prompts, desc=f"{method_name}"):
            prompt = item["prompt"]
            category = item["category"]
            prompt_count += 1
            
            try:
                # Generate image based on method
                if method_name == "baseline":
                    image_tensor = generate_baseline(current_model, prompt, num_inference_steps, seed)
                    
                    # Convert tensor to PIL
                    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                    
                elif method_name == "hybrid":
                    hybrid_result = current_model.generate(
                        prompt=prompt,
                        steps=num_inference_steps,
                        cfg_scale=guidance_scale,
                        seed=seed,
                        embedding_feedback=True,
                        attention_feedback=True
                    )
                    
                    # Convert tensor to PIL and move to CPU immediately
                    image_tensor = hybrid_result['image']
                    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                    
                    # Cleanup ALL result tensors aggressively
                    for key in list(hybrid_result.keys()):
                        if key != 'image':  # Already processed
                            del hybrid_result[key]
                    del hybrid_result
                    del image_tensor
                
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
                
                # Aggressive cleanup after each image
                del image
                del image_np
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Every 5 prompts, run full garbage collection
                if prompt_count % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n⚠️  Error on prompt '{prompt}' with {method_name}: {e}")
                continue
        
        # Cleanup model after finishing all prompts for this method
        print(f"\nFinished {method_name}, cleaning up model...")
        del current_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Memory freed\n")
    
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
