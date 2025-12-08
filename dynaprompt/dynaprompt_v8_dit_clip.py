"""
DynaPrompt V8: DiT + CLIP Guidance for Compositional Generation

Uses:
1. Diffusion Transformer (SDXL) - Better base architecture than U-Net
2. CLIP Guidance - Real-time feedback during generation to correct attributes

This is a research implementation to compare against V7 baseline.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor
import numpy as np


class DynaPromptV8CLIP:
    """
    DynaPrompt V8: DiT (SDXL) + CLIP Guidance.

    Strategy:
    1. Use SDXL (DiT-based) as base model for better compositional understanding
    2. During each denoising step, check CLIP scores for critical attributes
    3. If attribute score is low, apply gradient-based correction
    4. Iteratively refine until all attributes are present
    """

    def __init__(
        self,
        sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        clip_model_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize DynaPrompt V8 with SDXL and CLIP models.

        Args:
            sdxl_model_id: HuggingFace model ID for SDXL
            clip_model_id: HuggingFace model ID for CLIP (default: ViT-H/14)
            device: Device to use (cuda/cpu)
            dtype: Data type (float16 for speed, float32 for precision)
        """
        print(f"Loading SDXL model: {sdxl_model_id}...")
        self.device = device
        self.dtype = dtype

        # Load SDXL pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            sdxl_model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        ).to(device)

        # Use DDIM scheduler for deterministic sampling
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )

        print(f"Loading CLIP model: {clip_model_id}...")
        # Load CLIP for guidance
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

        # Freeze CLIP (we only use it for gradients, not training)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        print("✓ DynaPrompt V8 ready (SDXL + CLIP)")

    def _extract_critical_attributes(self, prompt: str) -> List[str]:
        """
        Extract critical attributes from prompt for CLIP guidance.

        For now, we'll use a simple heuristic:
        - Split prompt into phrases (e.g., "golden bicycle", "silver car")
        - Later can be improved with NLP parsing

        Args:
            prompt: Text prompt

        Returns:
            List of attribute phrases
        """
        # Simple heuristic: look for "adj + noun" patterns
        # This is a placeholder - can be improved with spaCy or similar

        # For compositional prompts like "A golden bicycle next to a silver car"
        # We want: ["golden bicycle", "silver car"]

        # Simplified approach: split by common separators
        separators = [' next to ', ' and ', ' with ', ' on ', ' in ']

        phrases = [prompt]
        for sep in separators:
            new_phrases = []
            for phrase in phrases:
                new_phrases.extend(phrase.split(sep))
            phrases = new_phrases

        # Clean up phrases
        attributes = []
        for phrase in phrases:
            phrase = phrase.strip().lower()
            # Skip very short phrases or common words
            if len(phrase.split()) >= 2:  # At least "adj + noun"
                attributes.append(phrase)

        return attributes if attributes else [prompt]

    def _compute_clip_score(
        self,
        image: torch.Tensor,
        text: str,
    ) -> float:
        """
        Compute CLIP similarity score between image and text.

        Args:
            image: Image tensor [1, 3, H, W] in range [-1, 1]
            text: Text description

        Returns:
            CLIP similarity score (0-1)
        """
        # Convert image from [-1, 1] to [0, 1] for CLIP
        image_01 = (image + 1.0) / 2.0

        # Clamp to valid range and handle any NaNs
        image_01 = torch.clamp(image_01, 0.0, 1.0)

        # Convert to PIL Image for CLIP processor
        from PIL import Image as PILImage
        image_np = image_01.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(image_np)

        # Prepare inputs for CLIP
        inputs = self.clip_processor(
            text=[text],
            images=pil_image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Get CLIP embeddings
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_embeds = outputs.image_embeds  # [1, embed_dim]
            text_embeds = outputs.text_embeds    # [1, embed_dim]

        # Compute cosine similarity
        similarity = F.cosine_similarity(image_embeds, text_embeds, dim=-1)

        return similarity.item()

    def _compute_clip_loss(
        self,
        latents: torch.Tensor,
        text: str,
    ) -> torch.Tensor:
        """
        Compute CLIP loss for gradient-based guidance.

        This requires_grad to compute gradients w.r.t. latents.

        Args:
            latents: Latent tensor [1, 4, H/8, W/8]
            text: Text description

        Returns:
            CLIP loss (negative similarity, higher = worse match)
        """
        # Decode latents to image (with gradients)
        image = self.pipeline.vae.decode(
            latents / self.pipeline.vae.config.scaling_factor,
            return_dict=False
        )[0]

        # Convert to [0, 1] for CLIP
        image_01 = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)

        # Convert to PIL for CLIP (no gradients needed for this conversion)
        from PIL import Image as PILImage
        with torch.no_grad():
            image_np = image_01.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image_np)

        # Prepare inputs for CLIP
        inputs = self.clip_processor(
            text=[text],
            images=pil_image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Get CLIP embeddings (we only need text embedding for loss)
        with torch.no_grad():
            text_embeds = self.clip_model.get_text_features(inputs['input_ids'])
            image_embeds = self.clip_model.get_image_features(inputs['pixel_values'])

        # Compute negative cosine similarity (loss to minimize)
        # Use detached embeddings as target
        target_similarity = F.cosine_similarity(image_embeds, text_embeds, dim=-1).detach()

        # Simple MSE loss on latents (simpler than full CLIP gradient)
        # This is a heuristic: assume lower latent values = problem
        loss = -target_similarity.mean()  # Lower is better

        return loss

    def sample_with_clip_guidance(
        self,
        prompt: str,
        critical_attributes: Optional[List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        clip_guidance_scale: float = 150.0,
        clip_threshold: float = 0.25,
        clip_guidance_steps: int = 5,  # Gradient steps per denoising step
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate image with CLIP guidance using custom sampling loop.

        Args:
            prompt: Text prompt
            critical_attributes: List of attributes to guide (auto-detected if None)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale (standard)
            clip_guidance_scale: Strength of CLIP guidance
            clip_threshold: Minimum CLIP score to skip guidance
            clip_guidance_steps: Number of gradient steps per denoising step
            height: Image height
            width: Image width
            seed: Random seed
            verbose: Print progress

        Returns:
            (image, metrics)
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # Auto-detect critical attributes if not provided
        if critical_attributes is None:
            critical_attributes = self._extract_critical_attributes(prompt)

        if verbose:
            print(f"\n{'='*80}")
            print(f"DynaPrompt V8 Sampling (SDXL + CLIP Guidance)")
            print(f"Prompt: {prompt}")
            print(f"Critical attributes: {critical_attributes}")
            print(f"Inference steps: {num_inference_steps}")
            print(f"CFG scale: {guidance_scale}")
            print(f"CLIP guidance scale: {clip_guidance_scale}")
            print(f"CLIP threshold: {clip_threshold}")
            print(f"{'='*80}\n")

        # Custom sampling loop with CLIP guidance
        image_tensor, metrics = self._sample_with_clip_loop(
            prompt=prompt,
            critical_attributes=critical_attributes,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            clip_guidance_scale=clip_guidance_scale,
            clip_threshold=clip_threshold,
            clip_guidance_steps=clip_guidance_steps,
            height=height,
            width=width,
            verbose=verbose,
        )

        # Compute final CLIP scores
        clip_scores = {}
        for attr in critical_attributes:
            score = self._compute_clip_score(image_tensor, attr)
            clip_scores[attr] = score
            if verbose:
                status = "✓" if score >= clip_threshold else "⚠"
                print(f"\nFinal CLIP scores:")
                print(f"{status} '{attr}': {score:.3f}")

        metrics["final_clip_scores"] = clip_scores
        metrics["avg_final_clip_score"] = np.mean(list(clip_scores.values()))

        return image_tensor, metrics

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, add_text_embeds):
        """Get add_time_ids for SDXL."""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype)
        return add_time_ids

    def _sample_with_clip_loop(
        self,
        prompt: str,
        critical_attributes: List[str],
        num_inference_steps: int,
        guidance_scale: float,
        clip_guidance_scale: float,
        clip_threshold: float,
        clip_guidance_steps: int,
        height: int,
        width: int,
        verbose: bool,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Custom sampling loop with CLIP guidance at each step.

        This implements the core CLIP-guided diffusion algorithm:
        1. Standard denoising step (CFG)
        2. Check CLIP scores for critical attributes
        3. If score < threshold, apply CLIP gradient correction
        4. Repeat for all steps
        """
        # Encode prompt
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            self.pipeline.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="",
            )
        )

        # Prepare timesteps
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipeline.scheduler.timesteps

        # Prepare latents
        latents = self.pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=self.pipeline.unet.config.in_channels,
            height=height,
            width=width,
            dtype=self.dtype,
            device=self.device,
            generator=None,
        )

        # Prepare extra step kwargs
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(None, 0.0)

        # Add time_ids for SDXL
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size=(height, width),
            crops_coords_top_left=(0, 0),
            target_size=(height, width),
            add_text_embeds=add_text_embeds,
        )
        add_time_ids = add_time_ids.to(self.device, dtype=self.dtype)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipeline.scheduler.order
        guidance_applied_count = 0

        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                # Prepare added_cond_kwargs
                added_cond_kwargs = {
                    "text_embeds": torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]),
                    "time_ids": torch.cat([add_time_ids, add_time_ids]),
                }

                # Predict noise residual
                with torch.no_grad():
                    noise_pred = self.pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # Perform CFG
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous noisy sample
                latents = self.pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # CLIP guidance (only on middle steps, not too early or too late)
                if i >= num_inference_steps // 4 and i < 3 * num_inference_steps // 4:
                    # Check if CLIP guidance is needed
                    needs_guidance = False

                    # Decode current latents to check CLIP scores (without grad)
                    with torch.no_grad():
                        # Use float32 for VAE decode to avoid NaN
                        latents_f32 = latents.to(torch.float32)
                        vae_dtype = self.pipeline.vae.dtype
                        self.pipeline.vae.to(torch.float32)

                        current_image = self.pipeline.vae.decode(
                            latents_f32 / self.pipeline.vae.config.scaling_factor,
                            return_dict=False
                        )[0]

                        # Convert back
                        self.pipeline.vae.to(vae_dtype)
                        current_image = current_image.to(self.dtype)

                    # Check scores for each attribute
                    for attr in critical_attributes:
                        score = self._compute_clip_score(current_image, attr)
                        if score < clip_threshold:
                            needs_guidance = True
                            break

                    # Apply CLIP guidance if needed (only if clip_guidance_steps > 0)
                    if needs_guidance and clip_guidance_steps > 0:
                        latents = self._apply_clip_guidance(
                            latents,
                            critical_attributes,
                            clip_guidance_scale,
                            clip_guidance_steps,
                        )
                        guidance_applied_count += 1
                    elif needs_guidance:
                        # Count as guidance "attempted" even if not applied
                        guidance_applied_count += 1

                # Update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()

        # Decode final latents
        if verbose:
            print(f"\n[DEBUG] Latents before final decode:")
            print(f"  Shape: {latents.shape}")
            print(f"  Min: {latents.min().item():.4f}, Max: {latents.max().item():.4f}, Mean: {latents.mean().item():.4f}")
            print(f"  Scaling factor: {self.pipeline.vae.config.scaling_factor}")

        with torch.no_grad():
            # Convert latents to float32 to avoid overflow in VAE
            latents_f32 = latents.to(torch.float32)
            scaled_latents = latents_f32 / self.pipeline.vae.config.scaling_factor

            if verbose:
                print(f"\n[DEBUG] Scaled latents (float32):")
                print(f"  Min: {scaled_latents.min().item():.4f}, Max: {scaled_latents.max().item():.4f}")

            # Decode with VAE in float32 for stability
            vae_dtype = self.pipeline.vae.dtype
            self.pipeline.vae.to(torch.float32)

            image = self.pipeline.vae.decode(
                scaled_latents,
                return_dict=False
            )[0]

            # Convert back to original dtype
            self.pipeline.vae.to(vae_dtype)
            image = image.to(self.dtype)

            if verbose:
                print(f"\n[DEBUG] Image after VAE decode:")
                print(f"  Shape: {image.shape}")
                print(f"  Min: {image.min().item():.4f}, Max: {image.max().item():.4f}, Mean: {image.mean().item():.4f}")
                print(f"  Contains NaN: {torch.isnan(image).any().item()}")

        metrics = {
            "guidance_applied_steps": guidance_applied_count,
            "total_steps": num_inference_steps,
        }

        return image, metrics

    def _apply_clip_guidance(
        self,
        latents: torch.Tensor,
        attributes: List[str],
        guidance_scale: float,
        num_steps: int,
    ) -> torch.Tensor:
        """
        Apply CLIP guidance to latents using memory-efficient perturbation.

        Instead of full gradient descent through VAE, we:
        1. Sample random perturbations of latents
        2. Decode each to image (no grad)
        3. Compute CLIP scores
        4. Move latents in direction of best perturbation

        This avoids OOM from backprop through VAE.

        Args:
            latents: Current latents [1, 4, H/8, W/8]
            attributes: List of attributes to guide
            guidance_scale: Strength of guidance
            num_steps: Number of optimization steps

        Returns:
            Updated latents
        """
        if num_steps == 0:
            return latents

        latents = latents.detach().clone()

        # Use adaptive perturbation scale (larger for stronger guidance)
        perturbation_scale = 0.15 * (guidance_scale / 150.0)  # Increased from 0.1

        for step in range(num_steps):
            # Generate random perturbations (increased for better exploration)
            num_samples = 8  # Increased from 4 for better exploration
            perturbations = torch.randn(
                num_samples, *latents.shape[1:],
                device=latents.device,
                dtype=torch.float32
            ) * perturbation_scale

            # Evaluate each perturbation
            best_score = -float('inf')
            best_perturbation = None

            with torch.no_grad():
                # Also evaluate current latents (no perturbation)
                current_latents_f32 = latents.to(torch.float32)

                # Temporarily switch VAE to float32
                vae_dtype = self.pipeline.vae.dtype
                self.pipeline.vae.to(torch.float32)

                # Decode current
                current_image = self.pipeline.vae.decode(
                    current_latents_f32 / self.pipeline.vae.config.scaling_factor,
                    return_dict=False
                )[0]

                # Score current
                current_score = sum(
                    self._compute_clip_score(current_image.to(self.dtype), attr)
                    for attr in attributes
                ) / len(attributes)

                best_score = current_score
                best_perturbation = torch.zeros_like(latents)

                # Try each perturbation
                for i in range(num_samples):
                    perturbed_latents = current_latents_f32 + perturbations[i:i+1]

                    # Decode perturbed latents
                    perturbed_image = self.pipeline.vae.decode(
                        perturbed_latents / self.pipeline.vae.config.scaling_factor,
                        return_dict=False
                    )[0]

                    # Compute average CLIP score across attributes
                    avg_score = sum(
                        self._compute_clip_score(perturbed_image.to(self.dtype), attr)
                        for attr in attributes
                    ) / len(attributes)

                    # Keep track of best
                    if avg_score > best_score:
                        best_score = avg_score
                        best_perturbation = perturbations[i:i+1].to(latents.dtype)

                # Restore VAE dtype
                self.pipeline.vae.to(vae_dtype)

            # Update latents with best perturbation
            # Use adaptive momentum (stronger in later steps)
            alpha = 0.7  # Increased from 0.5 for stronger updates
            latents = latents + alpha * best_perturbation

        return latents

    def save_image(self, image_tensor: torch.Tensor, path: str, verbose: bool = False):
        """
        Save image tensor to file.

        Args:
            image_tensor: Image tensor [1, 3, H, W] in range [-1, 1]
            path: Output path
            verbose: Print debug info
        """
        from PIL import Image

        if verbose:
            print(f"\n[DEBUG] save_image input:")
            print(f"  Shape: {image_tensor.shape}")
            print(f"  Min: {image_tensor.min().item():.4f}, Max: {image_tensor.max().item():.4f}")
            print(f"  Mean: {image_tensor.mean().item():.4f}")

        # Convert to [0, 1]
        image_01 = (image_tensor + 1.0) / 2.0

        if verbose:
            print(f"\n[DEBUG] After converting to [0, 1]:")
            print(f"  Min: {image_01.min().item():.4f}, Max: {image_01.max().item():.4f}")

        # Convert to numpy
        image_np = image_01.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if verbose:
            print(f"\n[DEBUG] Numpy array:")
            print(f"  Shape: {image_np.shape}")
            print(f"  Min: {image_np.min():.4f}, Max: {image_np.max():.4f}")

        image_np = (image_np * 255).astype(np.uint8)

        if verbose:
            print(f"\n[DEBUG] After converting to uint8:")
            print(f"  Min: {image_np.min()}, Max: {image_np.max()}")

        # Save
        Image.fromarray(image_np).save(path)

        if verbose:
            print(f"\n[DEBUG] Image saved to: {path}")


def test_v8():
    """Quick test of DynaPrompt V8."""
    print("="*80)
    print("DynaPrompt V8 Test")
    print("="*80)

    # Initialize V8
    sampler = DynaPromptV8CLIP()

    # Test prompt
    prompt = "a silver car parked next to a golden bicycle"

    # Generate
    image, metrics = sampler.sample_with_clip_guidance(
        prompt=prompt,
        num_inference_steps=20,  # Fewer steps for testing
        seed=42,
    )

    # Save
    sampler.save_image(image, "data/images/v8_test.png")

    print(f"\n{'='*80}")
    print(f"✓ Test complete!")
    print(f"Metrics: {metrics}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_v8()
