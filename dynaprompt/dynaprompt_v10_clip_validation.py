"""
DynaPrompt V10: V7 + CLIP Validation with Adaptive Boost

Phase 1 Enhancement: Add CLIP validation to V7's attention boosting.

Approach:
1. Generate with V7's attention boosting
2. Validate attributes with CLIP
3. If attributes fail: increase boost_factor and retry
4. Self-correction loop (max 3 retries)

Expected improvement: 70-80% success rate (vs V7's ~50%)
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'stable_diffusion_compvis'))
sys.path.insert(0, str(Path(__file__).parent))

from dynaprompt_v7 import DynaPromptV7Sampler
from transformers import CLIPProcessor, CLIPModel


class DynaPromptV10CLIPValidation(DynaPromptV7Sampler):
    """
    DynaPrompt V10: V7 with CLIP validation and adaptive boost.

    New features over V7:
    - CLIP-based attribute validation
    - Adaptive boost increase for failing attributes (7.5x → 15x → 22.5x)
    - Self-correction loop (up to 3 retries)
    """

    def __init__(
        self,
        ddim_sampler,
        model,
        tokenizer,
        device="cuda",
        clip_model_id="openai/clip-vit-large-patch14",
        check_step=3,
        attention_threshold=0.05,
        max_retries=15,
        boost_factor=7.5,
        start_step_ratio=0.0,
        end_step_ratio=0.5,
    ):
        """
        Initialize DynaPrompt V10 sampler.

        Args:
            ddim_sampler: DDIM sampler instance
            model: Stable Diffusion model
            tokenizer: Text tokenizer
            device: Device to use
            clip_model_id: CLIP model to use for validation
            check_step: Step to check composition
            attention_threshold: Minimum attention score
            max_retries: Maximum seed retries before boosting
            boost_factor: Base attention boost multiplier
            start_step_ratio: When to start boosting
            end_step_ratio: When to stop boosting
        """
        super().__init__(
            ddim_sampler=ddim_sampler,
            model=model,
            tokenizer=tokenizer,
            device=device,
            check_step=check_step,
            attention_threshold=attention_threshold,
            max_retries=max_retries,
            boost_factor=boost_factor,
            start_step_ratio=start_step_ratio,
            end_step_ratio=end_step_ratio,
        )

        # Load CLIP model for validation
        print(f"Loading CLIP model: {clip_model_id}")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
        self.clip_model.eval()

    def _compute_clip_score(
        self,
        image: torch.Tensor,
        text: str
    ) -> float:
        """
        Compute CLIP similarity score between image and text.

        Args:
            image: Image tensor in [-1, 1] range, shape [C, H, W]
            text: Text description

        Returns:
            CLIP similarity score (0-1)
        """
        # Convert image from [-1, 1] to [0, 1]
        image_01 = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)

        # Convert to PIL
        image_np = image_01.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        # Process with CLIP
        inputs = self.clip_processor(
            text=[text],
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Cosine similarity between image and text embeddings
            similarity = torch.cosine_similarity(
                outputs.image_embeds,
                outputs.text_embeds,
                dim=1
            )

        return similarity.item()

    def _validate_attributes(
        self,
        image: torch.Tensor,
        critical_attributes: List[str],
        clip_threshold: float = 0.25,
        verbose: bool = True
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate that all critical attributes are present in the image.

        Args:
            image: Generated image tensor [C, H, W]
            critical_attributes: List of attribute descriptions to validate
            clip_threshold: Minimum CLIP score to consider attribute present
            verbose: Print validation details

        Returns:
            (all_valid, scores_dict)
        """
        scores = {}
        all_valid = True

        if verbose:
            print(f"\n{'='*60}")
            print("CLIP Validation:")
            print(f"{'='*60}")

        for attr in critical_attributes:
            score = self._compute_clip_score(image, attr)
            scores[attr] = score

            is_valid = score >= clip_threshold
            all_valid = all_valid and is_valid

            if verbose:
                status = "✓" if is_valid else "✗"
                print(f"  {status} '{attr}': {score:.3f} {'(PASS)' if is_valid else '(FAIL)'}")

        if verbose:
            avg_score = np.mean(list(scores.values()))
            print(f"\nAverage CLIP score: {avg_score:.3f}")
            print(f"Overall: {'✓ PASS' if all_valid else '✗ FAIL'}")
            print(f"{'='*60}\n")

        return all_valid, scores

    def sample_with_clip_validation(
        self,
        prompt: str,
        shape,
        critical_attributes: List[str],
        steps: int = 50,
        unconditional_guidance_scale: float = 7.5,
        clip_threshold: float = 0.25,
        max_validation_retries: int = 3,
        boost_increase_factor: float = 2.0,
        verbose: bool = True,
    ):
        """
        Generate image with CLIP validation and adaptive boost.

        Args:
            prompt: Text prompt
            shape: Latent shape
            critical_attributes: List of attributes to validate (e.g., ["silver car", "golden bicycle"])
            steps: Number of diffusion steps
            unconditional_guidance_scale: CFG scale
            clip_threshold: Minimum CLIP score for validation
            max_validation_retries: Maximum retries with increased boost
            boost_increase_factor: Multiplier for boost_factor on retry
            verbose: Print progress

        Returns:
            (samples, intermediates, metrics)
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"DynaPrompt V10: CLIP Validation + Adaptive Boost")
            print(f"{'='*80}")
            print(f"Prompt: {prompt}")
            print(f"Critical attributes: {critical_attributes}")
            print(f"Initial boost factor: {self.boost_factor:.1f}x")
            print(f"Max validation retries: {max_validation_retries}")
            print(f"{'='*80}\n")

        original_boost = self.boost_factor
        metrics = {
            'attempts': 0,
            'boost_factors_tried': [],
            'clip_scores_per_attempt': [],
            'final_clip_scores': {},
            'validation_passed': False,
        }

        for attempt in range(max_validation_retries + 1):
            metrics['attempts'] = attempt + 1
            metrics['boost_factors_tried'].append(self.boost_factor)

            if verbose:
                print(f"\n{'='*60}")
                print(f"Attempt {attempt + 1}/{max_validation_retries + 1} (boost: {self.boost_factor:.1f}x)")
                print(f"{'='*60}")

            # Generate with current boost factor
            samples, intermediates = super().sample_with_dynaprompt(
                prompt=prompt,
                shape=shape,
                steps=steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                critical_tokens=None,  # V7 will auto-detect
                verbose=verbose,
            )

            # Validate with CLIP
            image = samples[0]  # First sample
            all_valid, scores = self._validate_attributes(
                image,
                critical_attributes,
                clip_threshold,
                verbose
            )

            metrics['clip_scores_per_attempt'].append(scores)
            metrics['final_clip_scores'] = scores

            if all_valid:
                metrics['validation_passed'] = True
                if verbose:
                    print(f"\n✓ Validation passed on attempt {attempt + 1}!")
                    print(f"  Used boost factor: {self.boost_factor:.1f}x\n")
                break

            # If failed and more retries available, increase boost
            if attempt < max_validation_retries:
                # Identify failing attributes
                failing_attrs = [attr for attr, score in scores.items() if score < clip_threshold]

                if verbose:
                    print(f"\n✗ Validation failed. Failing attributes: {failing_attrs}")
                    print(f"  Increasing boost factor: {self.boost_factor:.1f}x → ", end="")

                self.boost_factor *= boost_increase_factor

                if verbose:
                    print(f"{self.boost_factor:.1f}x")
                    print(f"  Retrying...\n")

        # Restore original boost factor
        self.boost_factor = original_boost

        if verbose:
            print(f"\n{'='*80}")
            print(f"Generation Complete!")
            print(f"{'='*80}")
            print(f"Total attempts: {metrics['attempts']}")
            print(f"Validation passed: {metrics['validation_passed']}")
            print(f"Final boost used: {metrics['boost_factors_tried'][-1]:.1f}x")
            print(f"{'='*80}\n")

        return samples, intermediates, metrics
