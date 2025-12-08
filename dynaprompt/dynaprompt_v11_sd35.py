"""
DynaPrompt V11 SD3.5: Smart Retry with Stable Diffusion 3.5

Upgrade from V11 Simple (SD 1.5) to SD 3.5 for better compositional generation.

Key differences from V11 Simple:
- Uses SD 3.5 instead of SD 1.5
- SD 3.5 has better prompt adherence and compositional understanding
- Same smart retry strategy (try multiple seeds, pick best)
- Expected to achieve MUCH better CLIP scores
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from pathlib import Path

from diffusers import StableDiffusion3Pipeline
from transformers import CLIPProcessor, CLIPModel


class DynaPromptV11SD35:
    """
    DynaPrompt V11 with Stable Diffusion 3.5.

    Strategy: Try multiple seeds, validate with CLIP, pick the best.
    """

    def __init__(
        self,
        sd35_model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers",
        clip_model_id: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize V11 with SD 3.

        Args:
            sd35_model_id: SD 3 model ID (default: medium 2B variant)
            clip_model_id: CLIP model for validation
            device: Device to use
            dtype: Data type for models
        """
        self.device = device
        self.dtype = dtype

        print(f"Loading Stable Diffusion 3: {sd35_model_id}")
        print("This may take a while (2B parameters)...")

        # Load SD 3 with memory optimizations
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            sd35_model_id,
            torch_dtype=dtype,
        )

        # Enable memory efficient attention and CPU offloading for 15GB GPU
        self.pipeline.enable_model_cpu_offload()
        # self.pipeline = self.pipeline.to(device)  # Not needed with CPU offload

        print(f"Loading CLIP model: {clip_model_id}")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
        self.clip_model.eval()

    def _compute_clip_score(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity score."""
        inputs = self.clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            similarity = torch.cosine_similarity(
                outputs.image_embeds,
                outputs.text_embeds,
                dim=1
            )

        return similarity.item()

    def generate_single(
        self,
        prompt: str,
        num_inference_steps: int = 28,  # SD3.5 default
        guidance_scale: float = 7.0,
        height: int = 1024,  # SD3.5 default
        width: int = 1024,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Image.Image:
        """
        Generate a single image with SD 3.5.

        Args:
            prompt: Text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            height: Image height
            width: Image width
            seed: Random seed
            verbose: Print progress

        Returns:
            Generated PIL image
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        if verbose:
            print(f"  Generating with seed {seed}...")

        output = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return output.images[0]

    def sample_with_smart_retry(
        self,
        prompt: str,
        critical_attributes: List[str],
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        height: int = 1024,
        width: int = 1024,
        clip_threshold: float = 0.25,
        num_seed_trials: int = 5,
        verbose: bool = True,
    ) -> tuple:
        """
        Generate with smart retry strategy (same as V11 Simple).

        Args:
            prompt: Text prompt
            critical_attributes: Attributes to validate
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            height: Image height
            width: Image width
            clip_threshold: CLIP score threshold
            num_seed_trials: Number of seeds to try
            verbose: Print progress

        Returns:
            (best_image, metrics)
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"DynaPrompt V11 with SD 3.5: Smart Retry Strategy")
            print(f"{'='*80}")
            print(f"Prompt: {prompt}")
            print(f"Critical attributes: {critical_attributes}")
            print(f"Strategy: Try {num_seed_trials} different seeds, pick best via CLIP")
            print(f"{'='*80}\n")

        best_image = None
        best_avg_score = -1.0
        best_scores = {}
        all_attempts = []

        for trial in range(num_seed_trials):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Trial {trial + 1}/{num_seed_trials}")
                print(f"{'='*60}")

            # Use different seed for each trial
            seed = torch.randint(0, 1000000, (1,)).item()

            if verbose:
                print(f"Using seed: {seed}")

            # Generate with SD 3.5
            image = self.generate_single(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                seed=seed,
                verbose=verbose,
            )

            # Validate with CLIP
            scores = {}

            if verbose:
                print(f"\nCLIP Validation:")

            for attr in critical_attributes:
                score = self._compute_clip_score(image, attr)
                scores[attr] = score

                if verbose:
                    status = "✓" if score >= clip_threshold else "✗"
                    print(f"  {status} '{attr}': {score:.3f}")

            avg_score = np.mean(list(scores.values()))

            if verbose:
                print(f"  Average: {avg_score:.3f}")

            # Track this attempt
            all_attempts.append({
                'seed': seed,
                'scores': scores,
                'avg_score': avg_score,
                'image': image,
            })

            # Update best if this is better
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_image = image
                best_scores = scores

                if verbose:
                    print(f"  → New best! (avg: {avg_score:.3f})")

            # Early exit if we found a passing result
            if all(score >= clip_threshold for score in scores.values()):
                if verbose:
                    print(f"\n✓ Found passing result on trial {trial + 1}!")
                break

        # Final summary
        if verbose:
            print(f"\n{'='*80}")
            print(f"Best Result (from {len(all_attempts)} trials):")
            print(f"{'='*80}")
            for attr, score in best_scores.items():
                status = "✓" if score >= clip_threshold else "✗"
                print(f"  {status} '{attr}': {score:.3f}")
            print(f"  Average: {best_avg_score:.3f}")
            passed = all(score >= clip_threshold for score in best_scores.values())
            print(f"\nValidation: {'✓ PASSED' if passed else '✗ FAILED'}")
            print(f"{'='*80}\n")

        metrics = {
            'num_trials': len(all_attempts),
            'best_avg_score': best_avg_score,
            'best_scores': best_scores,
            'validation_passed': all(score >= clip_threshold for score in best_scores.values()),
            'all_attempts': all_attempts,
        }

        return best_image, metrics

    def save_image(self, image: Image.Image, path: str):
        """Save image to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        print(f"Image saved to: {path}")
