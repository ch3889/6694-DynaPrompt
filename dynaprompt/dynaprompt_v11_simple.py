"""
DynaPrompt V11 Simple: V7 + Smart Retry Strategy

Based on V10 findings:
- Attention boosting alone doesn't help attribute binding
- Need to try different seeds (different latent initializations)
- CLIP validation to detect failures
- Strategic retry with seed variation

This is a simpler, more practical approach than full Attend-and-Excite.
Expected improvement: 60-70% success rate
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


class DynaPromptV11Simple(DynaPromptV7Sampler):
    """
    DynaPrompt V11 Simple: V7 + Smart Retry.

    Key insight from V10:
    - Boosting doesn't help → try different seeds instead
    - Some seeds naturally produce better attribute binding
    - Use CLIP to find the best seed

    Strategy:
    1. Try multiple seeds with V7
    2. Validate each with CLIP
    3. Return the best one
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
        """Initialize V11 Simple sampler."""
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

        # Load CLIP
        print(f"Loading CLIP model: {clip_model_id}")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
        self.clip_model.eval()

    def _compute_clip_score(self, image: torch.Tensor, text: str) -> float:
        """Compute CLIP similarity score."""
        image_01 = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)
        image_np = image_01.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        inputs = self.clip_processor(
            text=[text],
            images=pil_image,
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

    def sample_with_smart_retry(
        self,
        prompt: str,
        shape,
        critical_attributes: List[str],
        steps: int = 50,
        unconditional_guidance_scale: float = 7.5,
        clip_threshold: float = 0.25,
        num_seed_trials: int = 5,
        verbose: bool = True,
    ):
        """
        Generate with smart retry strategy.

        Args:
            prompt: Text prompt
            shape: Latent shape
            critical_attributes: Attributes to validate
            steps: Diffusion steps
            unconditional_guidance_scale: CFG scale
            clip_threshold: CLIP score threshold
            num_seed_trials: Number of different seeds to try
            verbose: Print progress

        Returns:
            (best_samples, intermediates, metrics)
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"DynaPrompt V11 Simple: Smart Retry Strategy")
            print(f"{'='*80}")
            print(f"Prompt: {prompt}")
            print(f"Critical attributes: {critical_attributes}")
            print(f"Strategy: Try {num_seed_trials} different seeds, pick best via CLIP")
            print(f"{'='*80}\n")

        best_samples = None
        best_intermediates = None
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
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            if verbose:
                print(f"Using seed: {seed}")

            # Generate with V7
            samples, intermediates = super().sample_with_dynaprompt(
                prompt=prompt,
                shape=shape,
                steps=steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                critical_tokens=None,
                verbose=verbose,
            )

            # Validate with CLIP
            image = samples[0]
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
                'samples': samples,
                'intermediates': intermediates,
            })

            # Update best if this is better
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_samples = samples
                best_intermediates = intermediates
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

        return best_samples, best_intermediates, metrics
