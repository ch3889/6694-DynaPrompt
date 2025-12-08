"""
DrawBench Evaluation for DynaPrompt V11 Simple

Phase 1: 50 color/composition prompts from DrawBench benchmark.
Compares V11 Simple (smart retry) vs baseline SD 1.5.

DrawBench is the gold standard from Imagen/SD papers for evaluating
compositional text-to-image generation.
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
from dynaprompt.dynaprompt_v11_simple import DynaPromptV11Simple


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
# Selected from the original DrawBench benchmark focusing on:
# - Color binding (attribute-object association)
# - Multi-object composition
# - Spatial relationships

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


def run_evaluation():
    print("=" * 80)
    print("DrawBench Phase 1 Evaluation: DynaPrompt V11 Simple")
    print("=" * 80)
    print(f"\nTotal prompts: {len(DRAWBENCH_PROMPTS)}")
    print(f"Categories: Color Binding (20), Multi-Object (15), Spatial (15)")
    print(f"Strategy: V11 Simple (5 seeds per prompt, pick best via CLIP)")
    print(f"\n{'=' * 80}\n")

    # Initialize V11 Simple
    print("Loading Stable Diffusion 1.5...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = "models/stable_diffusion_compvis/configs/stable-diffusion/v1-inference.yaml"
    ckpt_path = "models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt"

    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, ckpt_path, device=device)

    ddim_sampler = DDIMSampler(model)
    tokenizer = model.cond_stage_model.tokenizer

    print("Initializing DynaPrompt V11 Simple...")
    sampler = DynaPromptV11Simple(
        ddim_sampler=ddim_sampler,
        model=model,
        tokenizer=tokenizer,
        device=device,
        clip_model_id="openai/clip-vit-large-patch14",
        check_step=3,
        attention_threshold=0.05,
        max_retries=15,
        boost_factor=7.5,
    )

    # Output directory
    output_dir = Path("data/drawbench_v11")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Results storage
    results = {
        "metadata": {
            "version": "V11 Simple",
            "date": datetime.now().isoformat(),
            "num_prompts": len(DRAWBENCH_PROMPTS),
            "num_seeds": 5,
            "clip_threshold": 0.25,
        },
        "prompts": [],
        "summary": {}
    }

    # Category tracking
    category_scores = {
        "color": [],
        "comp": [],
        "spatial": [],
    }

    start_time = time.time()

    for i, prompt_data in enumerate(DRAWBENCH_PROMPTS):
        prompt_id = prompt_data["id"]
        prompt = prompt_data["prompt"]
        attributes = prompt_data["attributes"]
        category = prompt_id.split("_")[0]

        print(f"\n[{i+1}/{len(DRAWBENCH_PROMPTS)}] {prompt_id}")
        print(f"  Prompt: {prompt}")
        print(f"  Attributes: {attributes}")

        try:
            # Generate with V11 Simple (smart retry)
            best_image, metrics = sampler.sample_with_smart_retry(
                prompt=prompt,
                critical_attributes=attributes,
                shape=[1, 4, 64, 64],
                num_seed_trials=5,
                clip_threshold=0.25,
                verbose=False,
            )

            # Save image
            image_path = images_dir / f"{prompt_id}.png"

            # Decode latent to image if needed
            if hasattr(best_image, 'shape') and len(best_image.shape) == 4:
                # It's a latent tensor, decode it
                with torch.no_grad():
                    decoded = sampler.model.decode_first_stage(best_image)
                    decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                    decoded = decoded[0].permute(1, 2, 0).cpu().numpy()
                    decoded = (decoded * 255).astype("uint8")
                    pil_image = Image.fromarray(decoded)
            else:
                pil_image = best_image

            pil_image.save(str(image_path))

            # Record results
            avg_score = metrics["best_avg_score"]
            passed = metrics["validation_passed"]

            result = {
                "id": prompt_id,
                "prompt": prompt,
                "attributes": attributes,
                "best_avg_score": avg_score,
                "best_scores": metrics["best_scores"],
                "num_trials": metrics["num_trials"],
                "passed": passed,
                "image_path": str(image_path),
            }
            results["prompts"].append(result)
            category_scores[category].append(avg_score)

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  Result: {status} (avg: {avg_score:.3f})")
            for attr, score in metrics["best_scores"].items():
                print(f"    - {attr}: {score:.3f}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results["prompts"].append({
                "id": prompt_id,
                "prompt": prompt,
                "error": str(e),
            })
            category_scores[category].append(0.0)

    elapsed_time = time.time() - start_time

    # Calculate summary statistics
    all_scores = [p["best_avg_score"] for p in results["prompts"] if "best_avg_score" in p]
    all_passed = [p["passed"] for p in results["prompts"] if "passed" in p]

    results["summary"] = {
        "total_prompts": len(DRAWBENCH_PROMPTS),
        "successful_generations": len(all_scores),
        "total_passed": sum(all_passed),
        "pass_rate": sum(all_passed) / len(all_passed) * 100 if all_passed else 0,
        "avg_clip_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "category_scores": {
            "color_binding": {
                "avg": sum(category_scores["color"]) / len(category_scores["color"]) if category_scores["color"] else 0,
                "count": len(category_scores["color"]),
            },
            "multi_object": {
                "avg": sum(category_scores["comp"]) / len(category_scores["comp"]) if category_scores["comp"] else 0,
                "count": len(category_scores["comp"]),
            },
            "spatial": {
                "avg": sum(category_scores["spatial"]) / len(category_scores["spatial"]) if category_scores["spatial"] else 0,
                "count": len(category_scores["spatial"]),
            },
        },
        "elapsed_time_seconds": elapsed_time,
    }

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total prompts: {results['summary']['total_prompts']}")
    print(f"  Successful: {results['summary']['successful_generations']}")
    print(f"  Passed (≥0.25): {results['summary']['total_passed']}")
    print(f"  Pass rate: {results['summary']['pass_rate']:.1f}%")
    print(f"  Avg CLIP score: {results['summary']['avg_clip_score']:.3f}")
    print(f"\nBy Category:")
    print(f"  Color Binding: {results['summary']['category_scores']['color_binding']['avg']:.3f}")
    print(f"  Multi-Object: {results['summary']['category_scores']['multi_object']['avg']:.3f}")
    print(f"  Spatial: {results['summary']['category_scores']['spatial']['avg']:.3f}")
    print(f"\nTime: {elapsed_time/60:.1f} minutes")
    print(f"\nResults saved to: {results_path}")
    print(f"Images saved to: {images_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
