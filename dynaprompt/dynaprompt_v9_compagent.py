"""
DynaPrompt V9: CompAgent-style Compositional Generation

Uses LLM (Ollama + qwen2.5) for divide-and-conquer approach:
1. Decompose complex prompt into individual objects
2. Generate each object separately
3. Compose them together
4. Validate with CLIP and self-correct if needed

This addresses the fundamental compositional limitations we found in V8.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import json
import subprocess


class DynaPromptV9CompAgent:
    """
    DynaPrompt V9: CompAgent-style compositional generation.

    Key innovation: Generate objects separately, then compose.
    This avoids attribute binding failures in end-to-end generation.
    """

    def __init__(
        self,
        sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        clip_model_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        ollama_model: str = "qwen2.5:7b",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        print(f"Loading SDXL model: {sdxl_model_id}...")
        self.device = device
        self.dtype = dtype
        self.ollama_model = ollama_model

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

        print("✓ DynaPrompt V9 ready (CompAgent + SDXL + CLIP)")

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with qwen2.5."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.ollama_model, prompt],
                capture_output=True,
                text=True,
                timeout=60,  # Increased from 30s
            )
            return result.stdout.strip()
        except Exception as e:
            print(f"Warning: Ollama call failed: {e}")
            return None

    def _decompose_prompt(self, prompt: str, verbose: bool = True) -> List[Dict[str, str]]:
        """
        Use LLM to decompose prompt into individual objects.

        Falls back to rule-based decomposition if LLM fails.
        """
        if verbose:
            print(f"\n[LLM] Decomposing prompt...")

        llm_prompt = f"""Decompose this image generation prompt into individual objects.
For each object, extract its description including any attributes (color, size, etc.).
Return ONLY a JSON list with no additional text.

Prompt: "{prompt}"

Output format:
```json
[
    {{"object": "object name", "description": "full description with attributes"}},
    ...
]
```"""

        response = self._call_ollama(llm_prompt)

        if response:
            try:
                # Extract JSON from response (may have markdown code blocks)
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response

                objects = json.loads(json_str)

                if verbose:
                    print(f"[LLM] Found {len(objects)} objects:")
                    for obj in objects:
                        print(f"  - {obj['description']}")

                return objects

            except Exception as e:
                if verbose:
                    print(f"[LLM] JSON parsing failed: {e}")
                    print(f"[LLM] Falling back to rule-based decomposition")

        # Fallback: rule-based decomposition
        return self._decompose_prompt_fallback(prompt)

    def _decompose_prompt_fallback(self, prompt: str) -> List[Dict[str, str]]:
        """Simple rule-based decomposition as fallback."""
        separators = [' next to ', ' and ', ' with ', ' beside ', ' on ', ' in ']

        parts = [prompt]
        for sep in separators:
            new_parts = []
            for part in parts:
                if sep in part.lower():
                    new_parts.extend(part.split(sep))
                else:
                    new_parts.append(part)
            parts = new_parts

        objects = []
        for part in parts:
            part = part.strip()
            if len(part.split()) >= 2:  # At least 2 words
                objects.append({
                    "object": part.split()[0],  # First word as object name
                    "description": part
                })

        return objects if objects else [{"object": "full scene", "description": prompt}]

    def _generate_single_object(
        self,
        description: str,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        seed: Optional[int],
        verbose: bool = False,
    ) -> torch.Tensor:
        """Generate a single object image with memory optimization."""
        # Offload CLIP to CPU to free memory for generation
        if verbose:
            print(f"  [Memory] Offloading CLIP to CPU...")
        self.clip_model.to('cpu')
        torch.cuda.empty_cache()

        # Ensure SDXL is on GPU
        if verbose:
            print(f"  [Memory] Loading SDXL to GPU...")
        self.pipeline.to(self.device)

        output = self.pipeline(
            prompt=description,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None,
        )

        image = output.images[0]

        # Convert PIL to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        image_tensor = (image_tensor * 2.0 - 1.0).to(self.dtype).to(self.device)

        return image_tensor

    def _compute_clip_score(self, image: torch.Tensor, text: str, verbose: bool = False) -> float:
        """Compute CLIP similarity between image and text with memory optimization."""
        # Ensure CLIP is on GPU, offload SDXL if needed
        if self.clip_model.device.type != 'cuda':
            if verbose:
                print(f"  [Memory] Loading CLIP to GPU...")
            self.clip_model.to(self.device)

        image_01 = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)

        from PIL import Image as PILImage
        image_np = image_01.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = PILImage.fromarray(image_np)

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

    def sample_compositional(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        clip_threshold: float = 0.30,
        max_retries: int = 2,
        height: int = 768,
        width: int = 768,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Generate image using CompAgent-style divide-and-conquer.

        For now, we'll implement the "simple mode":
        - If prompt has multiple objects: generate full scene with higher CFG
        - Validate with CLIP
        - Retry with modified parameters if needed

        Future: Implement full composition with separate generation + layout.
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if verbose:
            print(f"\n{'='*80}")
            print(f"DynaPrompt V9 CompAgent")
            print(f"Prompt: {prompt}")
            print(f"{'='*80}")

        # Step 1: Decompose prompt
        objects = self._decompose_prompt(prompt, verbose=verbose)

        # Step 2: For now, generate full scene (future: generate separately)
        if verbose:
            print(f"\n[Generation] Creating full scene...")
            print(f"  Objects detected: {len(objects)}")
            print(f"  Using CFG scale: {guidance_scale}")

        best_image = None
        best_score = -float('inf')
        best_metrics = {}

        # Try with increasing CFG if CLIP scores are low
        cfg_scales = [guidance_scale]
        if max_retries > 0:
            cfg_scales = [guidance_scale + i * 1.5 for i in range(max_retries + 1)]

        for attempt, cfg in enumerate(cfg_scales):
            if verbose:
                print(f"\n[Attempt {attempt + 1}/{len(cfg_scales)}] CFG={cfg:.1f}")

            # Generate full scene
            image = self._generate_single_object(
                description=prompt,  # Use full prompt for now
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg,
                height=height,
                width=width,
                seed=seed,
                verbose=verbose,
            )

            # Validate with CLIP
            clip_scores = {}
            for obj in objects:
                desc = obj['description']
                score = self._compute_clip_score(image, desc, verbose=verbose)
                clip_scores[desc] = score

                if verbose:
                    status = "✓" if score >= clip_threshold else "✗"
                    print(f"  {status} '{desc}': {score:.3f}")

            avg_score = np.mean(list(clip_scores.values()))

            if avg_score > best_score:
                best_score = avg_score
                best_image = image
                best_metrics = {
                    "cfg_scale_used": cfg,
                    "attempts": attempt + 1,
                    "final_clip_scores": clip_scores,
                    "avg_final_clip_score": avg_score,
                    "num_objects": len(objects),
                    "objects": objects,
                }

            # Early stop if all pass
            all_pass = all(s >= clip_threshold for s in clip_scores.values())
            if all_pass:
                if verbose:
                    print(f"\n✓ All objects validated! Stopping early.")
                break

        if verbose:
            print(f"\n[Result] Best: CFG={best_metrics['cfg_scale_used']:.1f}, "
                  f"Avg CLIP={best_score:.3f}")

        return best_image, best_metrics

    def save_image(self, image_tensor: torch.Tensor, path: str):
        """Save image tensor to file."""
        from PIL import Image

        image_01 = (image_tensor + 1.0) / 2.0
        image_np = image_01.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        Image.fromarray(image_np).save(path)
