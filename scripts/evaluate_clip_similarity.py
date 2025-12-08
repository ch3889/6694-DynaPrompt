#!/usr/bin/env python3
"""
Evaluate CLIP similarity between generated images and their prompts.
This helps us understand which prompts have low alignment (semantic drift).
"""

import os
import sys
import torch
import clip
from PIL import Image
from pathlib import Path
import argparse
import json


def load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load CLIP model."""
    print(f"Loading CLIP model on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def compute_similarity(image_path, text, model, preprocess, device):
    """Compute CLIP similarity between an image and text."""
    # Load and preprocess image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Tokenize text
    text_tokens = clip.tokenize([text]).to(device)

    # Compute features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = (image_features @ text_features.T).item()

    return similarity


def evaluate_directory(base_dir, model, preprocess, device):
    """Evaluate all images in a directory structure."""
    results = []
    base_path = Path(base_dir)

    # Find all prompt directories
    prompt_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    print(f"\nEvaluating {len(prompt_dirs)} prompts...")
    print("=" * 80)

    for prompt_dir in prompt_dirs:
        # Read prompt
        prompt_file = prompt_dir / "prompt.txt"
        if not prompt_file.exists():
            continue

        with open(prompt_file, 'r') as f:
            prompt_text = f.read().strip()

        # Find generated images
        image_dir = prompt_dir / "samples"
        if not image_dir.exists():
            # Try alternative location
            image_dir = prompt_dir / "txt2img-samples" / "samples"

        if not image_dir.exists():
            print(f"⚠ No images found for: {prompt_dir.name}")
            continue

        # Get all PNG/JPG files
        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

        if not image_files:
            print(f"⚠ No image files in: {image_dir}")
            continue

        # Evaluate each image
        prompt_scores = []
        for img_path in image_files:
            try:
                score = compute_similarity(img_path, prompt_text, model, preprocess, device)
                prompt_scores.append(score)
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
                continue

        if prompt_scores:
            avg_score = sum(prompt_scores) / len(prompt_scores)
            max_score = max(prompt_scores)
            min_score = min(prompt_scores)

            result = {
                "prompt": prompt_text,
                "prompt_dir": str(prompt_dir.name),
                "num_images": len(prompt_scores),
                "avg_similarity": avg_score,
                "max_similarity": max_score,
                "min_similarity": min_score,
                "all_scores": prompt_scores
            }
            results.append(result)

            # Print result
            emoji = "✓" if avg_score > 0.30 else ("⚠" if avg_score > 0.25 else "✗")
            print(f"{emoji} {avg_score:.3f} | {prompt_text[:70]}")

    return results


def print_summary(results):
    """Print summary statistics."""
    if not results:
        print("\n⚠ No results to summarize")
        return

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_scores = [r["avg_similarity"] for r in results]
    avg_overall = sum(all_scores) / len(all_scores)

    print(f"\nTotal prompts evaluated: {len(results)}")
    print(f"Overall average CLIP score: {avg_overall:.3f}")
    print(f"Best score: {max(all_scores):.3f}")
    print(f"Worst score: {min(all_scores):.3f}")

    # Show top 3 and bottom 3
    sorted_results = sorted(results, key=lambda x: x["avg_similarity"], reverse=True)

    print(f"\n🏆 TOP 3 (Best Alignment):")
    for i, r in enumerate(sorted_results[:3], 1):
        print(f"  {i}. [{r['avg_similarity']:.3f}] {r['prompt'][:60]}")

    print(f"\n⚠️  BOTTOM 3 (Worst Alignment - Need DynaPrompt!):")
    for i, r in enumerate(sorted_results[-3:], 1):
        print(f"  {i}. [{r['avg_similarity']:.3f}] {r['prompt'][:60]}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP similarity for generated images")
    parser.add_argument("--images_dir", type=str,
                       default="data/images/baseline",
                       help="Directory containing generated images")
    parser.add_argument("--output", type=str,
                       default="data/images/baseline/clip_scores.json",
                       help="Output JSON file for scores")

    args = parser.parse_args()

    # Make paths absolute
    project_root = Path(__file__).parent.parent
    images_dir = project_root / args.images_dir
    output_file = project_root / args.output

    if not images_dir.exists():
        print(f"Error: Directory not found: {images_dir}")
        return

    # Load CLIP
    model, preprocess, device = load_clip_model()

    # Evaluate
    results = evaluate_directory(str(images_dir), model, preprocess, device)

    # Print summary
    print_summary(results)

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
