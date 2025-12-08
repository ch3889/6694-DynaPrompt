"""
DynaPrompt V8 Simple: SDXL + CLIP Validation (No Complex Guidance)

This version uses a simpler, more reliable approach:
1. Generate with SDXL (better than SD 1.5)
2. Validate with CLIP scores
3. If scores are low, increase CFG scale and retry
4. Much faster and more stable than perturbation-based guidance
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor
import numpy as np


class DynaPromptV8Simple:
    """
    DynaPrompt V8 Simple: SDXL + CLIP validation with adaptive CFG.

    Instead of complex perturbation-based guidance, we:
    1. Generate with increasing CFG scales until CLIP scores are acceptable
    2. Much faster and more stable
    """

    def __init__(
        self,
        sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        clip_model_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        print(f"Loading SDXL model: {sdxl_model_id}...")
        self.device = device
        self.dtype = dtype

        # Load SDXL
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            sdxl_model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        ).to(device)

        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )

        print(f"Loading CLIP model: {clip_model_id}...")
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

        for param in self.clip_model.parameters():
            param.requires_grad = False

        print("✓ DynaPrompt V8 Simple ready (SDXL + CLIP)")

    def _compute_clip_score(self, image: torch.Tensor, text: str) -> float:
        """Compute CLIP similarity between image and text."""
        # Convert to [0, 1]
        image_01 = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)

        # Convert to PIL
        from PIL import Image as PILImage
        image_np = image_01.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(image_np)

        # CLIP forward
        inputs = self.clip_processor(
            text=[text],
            images=pil_image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            similarity = F.cosine_similarity(
                outputs.image_embeds,
                outputs.text_embeds,
                dim=-1
            )

        return similarity.item()

    def _extract_critical_attributes(self, prompt: str) -> List[str]:
        """Extract critical attributes from prompt."""
        separators = [' next to ', ' and ', ' with ', ' on ', ' in ', ' beside ']

        phrases = [prompt]
        for sep in separators:
            new_phrases = []
            for phrase in phrases:
                new_phrases.extend(phrase.split(sep))
            phrases = new_phrases

        attributes = []
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if len(phrase.split()) >= 2:
                attributes.append(phrase)

        return attributes if attributes else [prompt]

    def sample_with_validation(
        self,
        prompt: str,
        critical_attributes: Optional[List[str]] = None,
        num_inference_steps: int = 30,
        base_guidance_scale: float = 7.5,
        clip_threshold: float = 0.30,
        max_cfg_scale: float = 12.0,
        max_retries: int = 3,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate with SDXL and validate with CLIP.

        If CLIP scores are low, increase CFG and retry.

        Args:
            prompt: Text prompt
            critical_attributes: Attributes to validate (auto-detected if None)
            num_inference_steps: Denoising steps
            base_guidance_scale: Starting CFG scale
            clip_threshold: Minimum acceptable CLIP score
            max_cfg_scale: Maximum CFG to try
            max_retries: How many times to retry with higher CFG
            height, width: Image resolution
            seed: Random seed
            verbose: Print progress

        Returns:
            (image, metrics)
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if critical_attributes is None:
            critical_attributes = self._extract_critical_attributes(prompt)

        if verbose:
            print(f"\n{'='*80}")
            print(f"DynaPrompt V8 Simple (SDXL + CLIP Validation)")
            print(f"Prompt: {prompt}")
            print(f"Critical attributes: {critical_attributes}")
            print(f"{'='*80}\n")

        best_image = None
        best_score = -float('inf')
        best_metrics = {}

        # Try with increasing CFG scales
        cfg_scales = [base_guidance_scale]
        if max_retries > 0:
            step = (max_cfg_scale - base_guidance_scale) / max_retries
            cfg_scales = [base_guidance_scale + i * step for i in range(max_retries + 1)]

        for attempt, cfg_scale in enumerate(cfg_scales):
            if verbose:
                print(f"Attempt {attempt + 1}/{len(cfg_scales)}: CFG scale = {cfg_scale:.1f}")

            # Generate
            output = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None,
            )

            image = output.images[0]

            # Convert PIL to tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
            image_tensor = (image_tensor * 2.0 - 1.0).to(self.dtype).to(self.device)

            # Validate with CLIP
            clip_scores = {}
            for attr in critical_attributes:
                score = self._compute_clip_score(image_tensor, attr)
                clip_scores[attr] = score
                if verbose:
                    status = "✓" if score >= clip_threshold else "✗"
                    print(f"  {status} '{attr}': {score:.3f}")

            avg_score = np.mean(list(clip_scores.values()))

            # Keep best
            if avg_score > best_score:
                best_score = avg_score
                best_image = image_tensor
                best_metrics = {
                    "cfg_scale_used": cfg_scale,
                    "attempts": attempt + 1,
                    "final_clip_scores": clip_scores,
                    "avg_final_clip_score": avg_score,
                }

            # Early stop if all scores pass
            all_pass = all(s >= clip_threshold for s in clip_scores.values())
            if all_pass:
                if verbose:
                    print(f"✓ All attributes pass! Stopping early.")
                break

        if verbose:
            print(f"\nBest result: CFG={best_metrics['cfg_scale_used']:.1f}, "
                  f"Avg CLIP={best_score:.3f}")

        return best_image, best_metrics

    def save_image(self, image_tensor: torch.Tensor, path: str):
        """Save image tensor to file."""
        from PIL import Image

        image_01 = (image_tensor + 1.0) / 2.0
        image_np = image_01.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        Image.fromarray(image_np).save(path)
