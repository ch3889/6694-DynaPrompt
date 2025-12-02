"""
DrawBench Evaluation using DynaPrompt V3 (Pure Attention Boosting)

NO seed retries - just direct attention weight modification.
This is the core DynaPrompt mechanism:
- Modifies: Attention weights (how U-Net focuses on different words)
- Where: Inside SD's brain (U-Net cross-attention layers)

Tests multiple boost factors for paper comparison:
- Baseline (1.0x) - No boosting
- Low (2.5x)
- Medium (5.0x)
- High (7.5x)
- Very High (10.0x)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'stable_diffusion_compvis'))

import torch
import json
import time
from datetime import datetime
from PIL import Image
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from dynaprompt.dynaprompt_v3 import DynaPromptV3Sampler

# CLIP for evaluation
from transformers import CLIPProcessor, CLIPModel


def load_model_from_config(config, ckpt, device="cuda"):
    """Load Stable Diffusion model."""
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


# DrawBench Phase 1: 50 Color & Composition Prompts
DRAWBENCH_PROMPTS = [
    # === COLOR BINDING (20 prompts) ===
    {"id": "color_01", "prompt": "a red cube on top of a blue cube", "attributes": ["red cube", "blue cube"]},
    {"id": "color_02", "prompt": "a green apple and a red apple", "attributes": ["green apple", "red apple"]},
    {"id": "color_03", "prompt": "a yellow car and a blue car", "attributes": ["yellow car", "blue car"]},
    {"id": "color_04", "prompt": "a purple cat sitting on an orange couch", "attributes": ["purple cat", "orange couch"]},
    {"id": "color_05", "prompt": "a pink elephant next to a green tree", "attributes": ["pink elephant", "green tree"]},
    {"id": "color_06", "prompt": "a silver robot holding a golden ball", "attributes": ["silver robot", "golden ball"]},
    {"id": "color_07", "prompt": "a white dog wearing a red collar", "attributes": ["white dog", "red collar"]},
    {"id": "color_08", "prompt": "a black cat with blue eyes", "attributes": ["black cat", "blue eyes"]},
    {"id": "color_09", "prompt": "a brown horse next to a white fence", "attributes": ["brown horse", "white fence"]},
    {"id": "color_10", "prompt": "a red bird on a yellow branch", "attributes": ["red bird", "yellow branch"]},
    {"id": "color_11", "prompt": "a blue butterfly on a pink flower", "attributes": ["blue butterfly", "pink flower"]},
    {"id": "color_12", "prompt": "a green frog on a brown log", "attributes": ["green frog", "brown log"]},
    {"id": "color_13", "prompt": "a purple umbrella next to a yellow raincoat", "attributes": ["purple umbrella", "yellow raincoat"]},
    {"id": "color_14", "prompt": "a orange basketball and a white basketball", "attributes": ["orange basketball", "white basketball"]},
    {"id": "color_15", "prompt": "a red strawberry on a white plate", "attributes": ["red strawberry", "white plate"]},
    {"id": "color_16", "prompt": "a golden crown on a red pillow", "attributes": ["golden crown", "red pillow"]},
    {"id": "color_17", "prompt": "a silver car parked next to a golden bicycle", "attributes": ["silver car", "golden bicycle"]},
    {"id": "color_18", "prompt": "a cyan teapot and a magenta cup", "attributes": ["cyan teapot", "magenta cup"]},
    {"id": "color_19", "prompt": "a navy blue boat on a turquoise sea", "attributes": ["navy blue boat", "turquoise sea"]},
    {"id": "color_20", "prompt": "a lime green lizard on a coral rock", "attributes": ["lime green lizard", "coral rock"]},

    # === MULTI-OBJECT COMPOSITION (15 prompts) ===
    {"id": "comp_01", "prompt": "a cat and a dog sitting together", "attributes": ["cat", "dog"]},
    {"id": "comp_02", "prompt": "a book on a table next to a lamp", "attributes": ["book", "table", "lamp"]},
    {"id": "comp_03", "prompt": "a cup of coffee and a croissant on a plate", "attributes": ["cup of coffee", "croissant"]},
    {"id": "comp_04", "prompt": "a bicycle leaning against a brick wall", "attributes": ["bicycle", "brick wall"]},
    {"id": "comp_05", "prompt": "a bird perched on a tree branch", "attributes": ["bird", "tree branch"]},
    {"id": "comp_06", "prompt": "a laptop on a wooden desk with a plant", "attributes": ["laptop", "wooden desk", "plant"]},
    {"id": "comp_07", "prompt": "a guitar standing next to an amplifier", "attributes": ["guitar", "amplifier"]},
    {"id": "comp_08", "prompt": "a teddy bear sitting on a bed", "attributes": ["teddy bear", "bed"]},
    {"id": "comp_09", "prompt": "a camera on a tripod", "attributes": ["camera", "tripod"]},
    {"id": "comp_10", "prompt": "a bottle of wine and two glasses on a table", "attributes": ["bottle of wine", "glasses", "table"]},
    {"id": "comp_11", "prompt": "a soccer ball and a basketball on grass", "attributes": ["soccer ball", "basketball", "grass"]},
    {"id": "comp_12", "prompt": "a piano in a room with a chandelier", "attributes": ["piano", "chandelier"]},
    {"id": "comp_13", "prompt": "a hat on a coat rack next to an umbrella", "attributes": ["hat", "coat rack", "umbrella"]},
    {"id": "comp_14", "prompt": "a pair of sunglasses on a beach towel", "attributes": ["sunglasses", "beach towel"]},
    {"id": "comp_15", "prompt": "a clock on a wall above a fireplace", "attributes": ["clock", "wall", "fireplace"]},

    # === SPATIAL RELATIONSHIPS (15 prompts) ===
    {"id": "spatial_01", "prompt": "a cat sitting under a table", "attributes": ["cat under table"]},
    {"id": "spatial_02", "prompt": "a dog jumping over a fence", "attributes": ["dog over fence"]},
    {"id": "spatial_03", "prompt": "a bird flying above the clouds", "attributes": ["bird above clouds"]},
    {"id": "spatial_04", "prompt": "a fish swimming below a boat", "attributes": ["fish below boat"]},
    {"id": "spatial_05", "prompt": "a ball rolling towards a goal", "attributes": ["ball", "goal"]},
    {"id": "spatial_06", "prompt": "a child standing behind a tree", "attributes": ["child behind tree"]},
    {"id": "spatial_07", "prompt": "a car driving between two buildings", "attributes": ["car between buildings"]},
    {"id": "spatial_08", "prompt": "a plane flying through clouds", "attributes": ["plane through clouds"]},
    {"id": "spatial_09", "prompt": "a person walking along a beach", "attributes": ["person", "beach"]},
    {"id": "spatial_10", "prompt": "a boat floating on a river", "attributes": ["boat on river"]},
    {"id": "spatial_11", "prompt": "a mountain rising behind a lake", "attributes": ["mountain behind lake"]},
    {"id": "spatial_12", "prompt": "a rainbow arching over a waterfall", "attributes": ["rainbow over waterfall"]},
    {"id": "spatial_13", "prompt": "a moon shining above a castle", "attributes": ["moon above castle"]},
    {"id": "spatial_14", "prompt": "a bridge crossing over a canyon", "attributes": ["bridge over canyon"]},
    {"id": "spatial_15", "prompt": "a tunnel going through a mountain", "attributes": ["tunnel through mountain"]},
]


def compute_clip_score(clip_model, clip_processor, image, text, device):
    """Compute CLIP similarity score."""
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = clip_model(**inputs)
        similarity = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds, dim=1)

    return similarity.item()


# Boost factors to compare
BOOST_FACTORS = [1.0, 2.5, 5.0, 7.5, 10.0]
BOOST_NAMES = ["baseline", "low", "medium", "high", "very_high"]


def run_evaluation():
    print("=" * 80)
    print("DrawBench Evaluation: DynaPrompt V3 (Pure Attention Boosting)")
    print("=" * 80)
    print(f"\nTotal prompts: {len(DRAWBENCH_PROMPTS)}")
    print(f"Boost factors to test: {BOOST_FACTORS}")
    print(f"Method: Direct attention weight modification (NO seed retries)")
    print(f"\n{'=' * 80}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SD 1.5
    print("Loading Stable Diffusion 1.5...")
    config_path = "models/stable_diffusion_compvis/configs/stable-diffusion/v1-inference.yaml"
    ckpt_path = "models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt"

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, device=device)

    ddim_sampler = DDIMSampler(model)
    tokenizer = model.cond_stage_model.tokenizer

    # Load CLIP for evaluation
    print("Loading CLIP for evaluation...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()

    # Output directory
    output_dir = Path("data/drawbench_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Master results for all boost factors
    all_results = {
        "metadata": {
            "version": "V3 (Pure Attention Boosting)",
            "date": datetime.now().isoformat(),
            "num_prompts": len(DRAWBENCH_PROMPTS),
            "boost_factors": BOOST_FACTORS,
            "method": "Direct attention weight modification, no seed retries",
        },
        "by_boost_factor": {},
        "comparison_summary": {},
    }

    start_time = time.time()

    # Test each boost factor
    for boost_idx, (boost_factor, boost_name) in enumerate(zip(BOOST_FACTORS, BOOST_NAMES)):
        print("\n" + "=" * 80)
        print(f"BOOST FACTOR: {boost_factor}x ({boost_name})")
        print("=" * 80)

        # Create sampler with this boost factor
        sampler = DynaPromptV3Sampler(
            ddim_sampler=ddim_sampler,
            model=model,
            tokenizer=tokenizer,
            device=device,
            boost_factor=boost_factor,
            start_step_ratio=0.0,
            end_step_ratio=0.5,
        )

        # Images directory for this boost factor
        images_dir = output_dir / f"images_boost_{boost_name}"
        images_dir.mkdir(exist_ok=True)

        results = {"prompts": [], "summary": {}}
        category_scores = {"color": [], "comp": [], "spatial": []}

        for i, prompt_data in enumerate(DRAWBENCH_PROMPTS):
            prompt_id = prompt_data["id"]
            prompt = prompt_data["prompt"]
            attributes = prompt_data["attributes"]
            category = prompt_id.split("_")[0]

            print(f"\n[{i+1}/{len(DRAWBENCH_PROMPTS)}] {prompt_id}")
            print(f"  Prompt: {prompt}")

            try:
                # Generate with V3 (single pass, attention boosting)
                shape = [1, 4, 64, 64]
                samples, _ = sampler.sample_with_dynaprompt(
                    prompt=prompt,
                    shape=shape,
                    steps=50,
                    unconditional_guidance_scale=7.5,
                    verbose=False,
                )

                # Decode to image
                with torch.no_grad():
                    decoded = model.decode_first_stage(samples)
                    decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                    decoded = decoded[0].permute(1, 2, 0).cpu().numpy()
                    decoded = (decoded * 255).astype("uint8")
                    pil_image = Image.fromarray(decoded)

                # Save image
                image_path = images_dir / f"{prompt_id}.png"
                pil_image.save(str(image_path))

                # CLIP evaluation
                scores = {}
                for attr in attributes:
                    score = compute_clip_score(clip_model, clip_processor, pil_image, attr, device)
                    scores[attr] = score

                avg_score = sum(scores.values()) / len(scores)
                passed = all(s >= 0.25 for s in scores.values())

                result = {
                    "id": prompt_id,
                    "prompt": prompt,
                    "attributes": attributes,
                    "scores": scores,
                    "avg_score": avg_score,
                    "passed": passed,
                    "image_path": str(image_path),
                }
                results["prompts"].append(result)
                category_scores[category].append(avg_score)

                status = "✓" if passed else "✗"
                print(f"  {status} avg: {avg_score:.3f}")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results["prompts"].append({"id": prompt_id, "prompt": prompt, "error": str(e)})
                category_scores[category].append(0.0)

        # Summary for this boost factor
        all_scores = [p["avg_score"] for p in results["prompts"] if "avg_score" in p]
        all_passed = [p["passed"] for p in results["prompts"] if "passed" in p]

        results["summary"] = {
            "boost_factor": boost_factor,
            "boost_name": boost_name,
            "total_prompts": len(DRAWBENCH_PROMPTS),
            "passed": sum(all_passed),
            "pass_rate": sum(all_passed) / len(all_passed) * 100 if all_passed else 0,
            "avg_clip_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "by_category": {
                "color_binding": sum(category_scores["color"]) / len(category_scores["color"]) if category_scores["color"] else 0,
                "multi_object": sum(category_scores["comp"]) / len(category_scores["comp"]) if category_scores["comp"] else 0,
                "spatial": sum(category_scores["spatial"]) / len(category_scores["spatial"]) if category_scores["spatial"] else 0,
            },
        }

        all_results["by_boost_factor"][boost_name] = results

        print(f"\n  Summary for {boost_factor}x:")
        print(f"    Pass rate: {results['summary']['pass_rate']:.1f}%")
        print(f"    Avg CLIP: {results['summary']['avg_clip_score']:.3f}")

    elapsed_time = time.time() - start_time

    # Comparison summary
    all_results["comparison_summary"] = {
        "boost_factors": BOOST_FACTORS,
        "boost_names": BOOST_NAMES,
        "pass_rates": [all_results["by_boost_factor"][name]["summary"]["pass_rate"] for name in BOOST_NAMES],
        "avg_clip_scores": [all_results["by_boost_factor"][name]["summary"]["avg_clip_score"] for name in BOOST_NAMES],
        "color_binding": [all_results["by_boost_factor"][name]["summary"]["by_category"]["color_binding"] for name in BOOST_NAMES],
        "multi_object": [all_results["by_boost_factor"][name]["summary"]["by_category"]["multi_object"] for name in BOOST_NAMES],
        "spatial": [all_results["by_boost_factor"][name]["summary"]["by_category"]["spatial"] for name in BOOST_NAMES],
        "total_time_minutes": elapsed_time / 60,
    }

    # Save
    with open(output_dir / "results_all_boosts.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final comparison
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE - COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Boost':<10} {'Pass Rate':<12} {'Avg CLIP':<10} {'Color':<10} {'Comp':<10} {'Spatial':<10}")
    print("-" * 62)
    for name, factor in zip(BOOST_NAMES, BOOST_FACTORS):
        s = all_results["by_boost_factor"][name]["summary"]
        print(f"{factor}x{'':<6} {s['pass_rate']:<12.1f} {s['avg_clip_score']:<10.3f} {s['by_category']['color_binding']:<10.3f} {s['by_category']['multi_object']:<10.3f} {s['by_category']['spatial']:<10.3f}")

    print(f"\nTotal time: {elapsed_time/60:.1f} minutes")
    print(f"Results saved: {output_dir / 'results_all_boosts.json'}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
