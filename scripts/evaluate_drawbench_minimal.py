"""
ULTRA LOW MEMORY VERSION: Evaluate DynaPrompt on DrawBench
Process 1 image at a time, save after each, reload model every image for hybrid
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynaprompt.hybrid import HybridDynaPrompt
from dynaprompt.sd_loader import load_sd_model

def generate_baseline(sd_model, prompt, steps=50, seed=42):
    """Generate image without any feedback"""
    device = sd_model.device
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    c = sd_model.encode_text([prompt])
    uc = sd_model.encode_text([""])
    
    sampler = sd_model.create_sampler('ddim')
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)
    
    shape = [1, 4, 512 // 8, 512 // 8]
    latents = torch.randn(shape, device=device)
    
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
    
    with torch.no_grad():
        latents_scaled = 1 / 0.18215 * latents
        image = sd_model.model.first_stage_model.decode(latents_scaled)
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    
    return image

def extract_concepts(prompt):
    """Extract concepts from prompt"""
    stop_words = ['a', 'an', 'the', 'and', 'or', 'with', 'of', 'in', 'on', 'at', 'to', 'for']
    words = prompt.lower().split()
    concepts = [w.strip('.,!?') for w in words if w not in stop_words]
    bigrams = [f"{concepts[i]} {concepts[i+1]}" for i in range(len(concepts)-1)]
    trigrams = [f"{concepts[i]} {concepts[i+1]} {concepts[i+2]}" for i in range(len(concepts)-2)]
    
    all_concepts = concepts + bigrams + trigrams
    seen = set()
    unique_concepts = []
    for c in all_concepts:
        if c not in seen and len(c) > 1:
            seen.add(c)
            unique_concepts.append(c)
    
    return unique_concepts

def compute_clip_score(image, text, clip_model, clip_preprocess, device):
    """Compute CLIP similarity"""
    import clip
    
    if isinstance(image, Image.Image):
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
    else:
        image_input = image
    
    text_input = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).item()
        score = (similarity + 1) * 50
    
    return score

def compute_compositional_accuracy(image, prompt, clip_model, clip_preprocess, device):
    """Compute fraction of concepts present"""
    concepts = extract_concepts(prompt)
    
    if len(concepts) == 0:
        return 1.0
    
    present_count = 0
    for concept in concepts:
        score = compute_clip_score(image, concept, clip_model, clip_preprocess, device)
        if score > 20.0:
            present_count += 1
    
    return present_count / len(concepts)


def main():
    parser = argparse.ArgumentParser(description='Evaluate DynaPrompt on DrawBench (Minimal Memory)')
    parser.add_argument('--categories', nargs='+', default=['Colors', 'Positional', 'Counting', 'Descriptions', 'Conflicting'])
    parser.add_argument('--methods', nargs='+', default=['baseline', 'hybrid'])
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/drawbench_phase1')
    
    args = parser.parse_args()
    
    prompts_file = 'data/drawbench_prompts.json'
    print(f"Loading prompts from {prompts_file}...")
    with open(prompts_file, 'r') as f:
        all_data = json.load(f)
    
    all_prompts = []
    for cat in args.categories:
        if cat in all_data:
            for prompt in all_data[cat]:
                all_prompts.append({"prompt": prompt, "category": cat})
    
    print(f"\n{'='*80}")
    print("DRAWBENCH EVALUATION (MINIMAL MEMORY MODE)")
    print(f"{'='*80}")
    print(f"Total prompts: {len(all_prompts)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Mode: Process 1 image at a time, reload hybrid model each image")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*80}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load CLIP once
    print("Loading CLIP model...")
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
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
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Results file for incremental saving
    results_file = output_path / "results_detailed.json"
    
    # Load existing results if any
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded existing results: {sum(len(v) for v in results.values())} images")
    else:
        results = {method: [] for method in args.methods}
    
    # Process each method
    for method_name in args.methods:
        print(f"\n{'='*60}")
        print(f"Processing {method_name.upper()} method")
        print(f"{'='*60}\n")
        
        # For baseline: load model once
        baseline_model = None
        if method_name == "baseline":
            print("Loading Baseline SD model...")
            baseline_model = load_sd_model(ckpt_path=ckpt_path, device=device)
        
        for idx, item in enumerate(tqdm(all_prompts, desc=f"{method_name}")):
            prompt = item["prompt"]
            category = item["category"]
            
            # Check if already processed
            already_done = any(r['prompt'] == prompt for r in results[method_name])
            if already_done:
                continue
            
            try:
                # Generate image
                if method_name == "baseline":
                    image_tensor = generate_baseline(baseline_model, prompt, args.steps, args.seed)
                    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                    del image_tensor
                    
                elif method_name == "hybrid":
                    # Load model for THIS image only
                    hybrid_model = HybridDynaPrompt(ckpt_path=ckpt_path, device=device)
                    
                    hybrid_result = hybrid_model.generate(
                        prompt=prompt,
                        steps=args.steps,
                        cfg_scale=args.guidance,
                        seed=args.seed,
                        embedding_feedback=True,
                        attention_feedback=True
                    )
                    
                    image_tensor = hybrid_result['image']
                    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                    
                    # Cleanup ALL hybrid result data
                    for key in list(hybrid_result.keys()):
                        del hybrid_result[key]
                    del hybrid_result
                    del image_tensor
                    del hybrid_model
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Save image
                safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)[:50]
                img_dir = output_path / method_name / category
                img_dir.mkdir(parents=True, exist_ok=True)
                img_path = img_dir / f"{safe_prompt}.png"
                image.save(img_path)
                
                # Compute metrics
                comp_acc = compute_compositional_accuracy(image, prompt, clip_model, clip_preprocess, device)
                clip_score = compute_clip_score(image, prompt, clip_model, clip_preprocess, device)
                
                # Save result immediately
                results[method_name].append({
                    "prompt": prompt,
                    "category": category,
                    "compositional_accuracy": float(comp_acc),
                    "clip_score": float(clip_score),
                    "image_path": str(img_path)
                })
                
                # Write to file immediately
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Cleanup
                del image
                del image_np
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n⚠️  Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cleanup baseline model after all prompts
        if baseline_model is not None:
            del baseline_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Compute summary
    print("\n\nComputing summary statistics...")
    summary = {}
    
    for method_name, method_results in results.items():
        if not method_results:
            continue
            
        summary[method_name] = {}
        
        comp_accs = [r["compositional_accuracy"] for r in method_results]
        clip_scores = [r["clip_score"] for r in method_results]
        
        summary[method_name]["overall"] = {
            "avg_compositional_accuracy": np.mean(comp_accs),
            "avg_clip_score": np.mean(clip_scores),
            "std_compositional_accuracy": np.std(comp_accs),
            "std_clip_score": np.std(clip_scores),
            "num_samples": len(method_results)
        }
        
        summary[method_name]["per_category"] = {}
        for cat in args.categories:
            cat_results = [r for r in method_results if r["category"] == cat]
            if cat_results:
                cat_comp = [r["compositional_accuracy"] for r in cat_results]
                cat_clip = [r["clip_score"] for r in cat_results]
                summary[method_name]["per_category"][cat] = {
                    "avg_compositional_accuracy": np.mean(cat_comp),
                    "avg_clip_score": np.mean(cat_clip),
                    "num_samples": len(cat_results)
                }
    
    summary_path = output_path / "results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for method_name in args.methods:
        if method_name not in summary:
            continue
        print(f"\n{method_name.upper()}:")
        overall = summary[method_name]["overall"]
        print(f"  Overall:")
        print(f"    Compositional Accuracy: {overall['avg_compositional_accuracy']:.4f} ± {overall['std_compositional_accuracy']:.4f}")
        print(f"    CLIP Score: {overall['avg_clip_score']:.2f} ± {overall['std_clip_score']:.2f}")
        print(f"  Per-Category:")
        for cat, stats in summary[method_name]["per_category"].items():
            print(f"    {cat:20s}: Comp {stats['avg_compositional_accuracy']:.4f}, CLIP {stats['avg_clip_score']:.2f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
