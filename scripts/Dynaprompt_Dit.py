#!/usr/bin/env python3
"""
DynaPrompt - DiT Image Generation
Dynamic Prompt Guidance for Diffusion Transformers

Team: Charles Hou (ch3889), Max Kim (zk2295), Swapnil Banerjee (sb5041)
Course: EECS 6694 Deep Learning
"""

import torch
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import clip
from PIL import Image
import functools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from diffusers import DiTPipeline, DPMSolverMultistepScheduler, PixArtAlphaPipeline
from IPython.display import display

# ============================================================================
# Step 1: Verify GPU
# ============================================================================

def verify_gpu():
    """Check GPU availability and memory"""
    print("="*70)
    print("STEP 1: GPU Verification")
    print("="*70)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()

# ============================================================================
# Step 2: Load Models
# ============================================================================

def load_dit_model():
    """Load DiT-XL/2-256 model (class-conditional)"""
    print("="*70)
    print("Loading DiT-XL/2-256 model...")
    print("="*70)
    print("First run downloads ~2GB model weights\n")

    pipe = DiTPipeline.from_pretrained(
        "facebook/DiT-XL-2-256",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    print(f"✓ DiT model loaded on {device}\n")
    return pipe, device


def load_pixart_model():
    """Load PixArt-α (DiT-based text-to-image)"""
    print("="*70)
    print("Loading PixArt-α (DiT-based text-to-image)...")
    print("="*70)
    print("Downloading ~1GB model weights\n")

    pixart_pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixart_pipe = pixart_pipe.to(device)

    print(f"✓ PixArt-α loaded on {device}\n")
    return pixart_pipe, device

# ============================================================================
# Step 3: DynaPrompt Implementation
# ============================================================================

class DynaPromptAutoReweighter:
    """
    Hybrid DynaPrompt: Combines paper's during-diffusion embedding modification with CLIP-guided detection.

    Paper's approach: Modify embeddings during diffusion based on CLIP scores of intermediate images
    Our addition: Use final image CLIP analysis to guide which tokens need boosting
    """

    def __init__(self, boost_base: float = 1.3, max_iterations: int = 3,
                 confidence_threshold: float = 0.23, clip_check_steps: List[int] = None):
        self.boost_base = boost_base  # Base boost factor from paper (1.3)
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.clip_check_steps = clip_check_steps or [10, 15, 20]  # Steps to check CLIP during diffusion
        self.ignore_words = {"a","an","the","of","in","on","at","to","for","with","by","from","is","are","was","were","and","or"}

        # Load CLIP for image analysis
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # For embedding modification
        self.boost_indices = []
        self.pipe = None

    def get_token_indices(self, pipe, prompt: str, words: List[str]) -> List[int]:
        """Get token indices for specific words in the prompt."""
        tokenizer = pipe.tokenizer
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        indices = []
        for word in words:
            word_lower = word.lower().strip()
            for i, token_id in enumerate(input_ids):
                token_text = tokenizer.decode([token_id]).strip().lower()
                if word_lower in token_text or token_text in word_lower:
                    if i not in indices:
                        indices.append(i)

        return indices

    def compute_dynamic_boost(self, clip_score: float, timestep: int, total_timesteps: int = 20) -> float:
        """
        Compute dynamic boost factor based on CLIP score and timestep.
        Formula from paper: 1 + boost_base * (total_timesteps - timestep) / total_timesteps
        When CLIP score is low, boost more; as timestep decreases, boost decreases.
        """
        # Normalize timestep (diffusion timesteps are often in range [0, 1000])
        normalized_step = min(timestep / 50, total_timesteps)  # Map to 0-20 range
        boost = 1 + self.boost_base * (total_timesteps - normalized_step) / total_timesteps

        # Additional scaling based on CLIP confidence (lower score = higher boost)
        if clip_score < self.confidence_threshold:
            confidence_factor = 1 + (self.confidence_threshold - clip_score) * 2
            boost *= confidence_factor

        return boost

    def create_embedding_modifier_callback(self, pipe, prompt: str, missed_words: List[str]):
        """
        Create callback that modifies embeddings during diffusion.
        Implements paper's algorithm: decode x_t, compute CLIP, update embeddings.
        """
        boost_indices = self.get_token_indices(pipe, prompt, missed_words)

        def callback(pipe_obj, step_idx: int, timestep: int, callback_kwargs: dict):
            # Only check at specified steps to avoid overhead
            if timestep not in self.clip_check_steps and step_idx % 5 != 0:
                return callback_kwargs

            # Get current latents
            latents = callback_kwargs.get("latents", None)
            if latents is None:
                return callback_kwargs

            # Decode intermediate image (x_t)
            try:
                with torch.no_grad():
                    # Scale latents back to image space
                    latents_scaled = latents / pipe.vae.config.scaling_factor
                    image = pipe.vae.decode(latents_scaled).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    image_pil = Image.fromarray((image * 255).astype(np.uint8))

                    # Compute CLIP scores for each missed word
                    image_input = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)

                    for word in missed_words:
                        text_input = clip.tokenize([f"a photo of {word}"]).to(self.device)

                        with torch.no_grad():
                            image_features = self.clip_model.encode_image(image_input)
                            text_features = self.clip_model.encode_text(text_input)

                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                            clip_score = (image_features @ text_features.T).item()

                        # If CLIP score is low, boost the embedding
                        if clip_score < self.confidence_threshold:
                            boost_factor = self.compute_dynamic_boost(clip_score, timestep)

                            # Modify prompt embeddings in callback_kwargs
                            if "prompt_embeds" in callback_kwargs:
                                embeds = callback_kwargs["prompt_embeds"]
                                for idx in boost_indices:
                                    if idx < embeds.shape[1]:
                                        embeds[0, idx] = embeds[0, idx] * boost_factor
                                callback_kwargs["prompt_embeds"] = embeds

            except Exception as e:
                # Silently continue if callback fails (VAE decode can fail at early steps)
                pass

            return callback_kwargs

        return callback

    def extract_critical_words(self, prompt: str) -> List[str]:
        """Extract nouns, adjectives, colors - words that MUST appear in image"""
        tokens = [w.strip('.,!?;:').lower() for w in prompt.split()]
        return [w for w in tokens if w and w not in self.ignore_words and len(w) > 2]

    def analyze_image_for_word(self, image: Image.Image, word: str) -> float:
        """
        Use CLIP to check if a word is present in the generated image.
        Returns confidence score (0-1).
        """
        # Prepare image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Create text queries
        text_queries = [
            f"a photo with {word}",
            f"a photo without {word}",
        ]
        text_tokens = clip.tokenize(text_queries).to(self.device)

        # Get CLIP scores
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Return probability that word IS present
        return similarity[0][0].item()

    def detect_missed_words(self, image: Image.Image, prompt: str) -> List[Tuple[str, float]]:
        """
        Analyze generated image with CLIP to detect which critical words are missing.
        Returns list of (word, confidence) for words below threshold.
        """
        critical_words = self.extract_critical_words(prompt)
        missed = []

        for word in critical_words:
            confidence = self.analyze_image_for_word(image, word)

            if confidence < self.confidence_threshold:
                missed.append((word, confidence))

        return missed

    def iterative_reweight(self, prompt: str, pipe, num_inference_steps: int = 50,
                          guidance_scale: float = 7.5, generator_seed: int = None) -> Tuple[Image.Image, str, List[Dict]]:
        """
        Hybrid DynaPrompt: Use CLIP to detect missed words, then generate with
        during-diffusion embedding modification (paper's approach).
        """
        print("\n🚀 Starting Hybrid DynaPrompt (CLIP detection + during-diffusion boosting)!")
        print(f"   Prompt: {prompt}")

        current_prompt = prompt
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(generator_seed) if generator_seed else None
        history = []

        for iteration in range(self.max_iterations):
            print(f"\n📸 Iteration {iteration + 1}/{self.max_iterations}")

            # Generate initial image to detect missed words
            print(f"   Generating test image...")
            test_image = pipe(
                prompt=current_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

            # Analyze with CLIP to find missed words
            print("   🔍 Analyzing with CLIP...")
            missed_analysis = self.detect_missed_words(test_image, prompt)
            missed_words = [word for word, conf in missed_analysis]

            # Record history
            history.append({
                'iteration': iteration + 1,
                'prompt': current_prompt,
                'missed_words': missed_analysis,
                'image': test_image
            })

            if not missed_words:
                print("   ✅ All words detected! Generation successful.")
                return test_image, current_prompt, history

            print(f"   ⚠️  Missed words detected:")
            for word, conf in missed_analysis:
                print(f"      - '{word}' confidence: {conf:.3f}")

            print(f"   🎯 Regenerating with during-diffusion embedding boost...")

            # Create callback for during-diffusion embedding modification
            callback = self.create_embedding_modifier_callback(pipe, prompt, missed_words)

            # Regenerate with dynamic embedding modification during diffusion
            try:
                image = pipe(
                    prompt=current_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=callback
                ).images[0]
            except Exception as e:
                print(f"   Warning: Callback generation failed ({str(e)}), using test image")
                image = test_image

        print("\n⏰ Max iterations reached")
        return image, current_prompt, history


# ============================================================================
# Step 4: Evaluator
# ============================================================================

class DynaPromptEvaluator:
    """Quantitative evaluation using CLIP metrics"""

    def __init__(self, clip_model, clip_preprocess, device):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device

    def compute_text_image_similarity(self, image: Image.Image, text: str) -> float:
        """Compute CLIP similarity between image and text"""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)

            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = (image_features @ text_features.T).item()

        return similarity

    def compute_concept_scores(self, image: Image.Image, concepts: List[str]) -> Dict[str, float]:
        """Compute CLIP score for each concept individually"""
        scores = {}
        for concept in concepts:
            score = self.compute_text_image_similarity(image, f"a photo of {concept}")
            scores[concept] = score
        return scores

    def evaluate_prompt(self, prompt: str, vanilla_image: Image.Image,
                       dynaprompt_image: Image.Image) -> Dict:
        """Compare vanilla vs DynaPrompt across multiple metrics"""

        # Overall prompt similarity
        vanilla_overall = self.compute_text_image_similarity(vanilla_image, prompt)
        dynaprompt_overall = self.compute_text_image_similarity(dynaprompt_image, prompt)

        # Extract critical concepts
        concepts = [w.strip('.,!?;:').lower() for w in prompt.split()
                   if w.lower() not in {"a","an","the","of","in","on","at","to","for","with","by","from","is","are","was","were","and","or"}
                   and len(w) > 2]

        # Per-concept scores
        vanilla_concepts = self.compute_concept_scores(vanilla_image, concepts)
        dynaprompt_concepts = self.compute_concept_scores(dynaprompt_image, concepts)

        # Average concept score
        vanilla_avg_concept = np.mean(list(vanilla_concepts.values()))
        dynaprompt_avg_concept = np.mean(list(dynaprompt_concepts.values()))

        # Count concepts above threshold
        threshold = 0.23
        vanilla_detected = sum(1 for s in vanilla_concepts.values() if s > threshold)
        dynaprompt_detected = sum(1 for s in dynaprompt_concepts.values() if s > threshold)

        return {
            'prompt': prompt,
            'vanilla_overall': vanilla_overall,
            'dynaprompt_overall': dynaprompt_overall,
            'improvement_overall': dynaprompt_overall - vanilla_overall,
            'vanilla_avg_concept': vanilla_avg_concept,
            'dynaprompt_avg_concept': dynaprompt_avg_concept,
            'improvement_concept': dynaprompt_avg_concept - vanilla_avg_concept,
            'vanilla_detected': vanilla_detected,
            'dynaprompt_detected': dynaprompt_detected,
            'total_concepts': len(concepts),
            'vanilla_concept_scores': vanilla_concepts,
            'dynaprompt_concept_scores': dynaprompt_concepts
        }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    
    # Step 1: Verify GPU
    verify_gpu()
    
    # Step 2: Load models
    pixart_pipe, device = load_pixart_model()
    
    # Step 3: Initialize DynaPrompt
    print("="*70)
    print("Initializing Hybrid DynaPrompt...")
    print("="*70)
    reweighter = DynaPromptAutoReweighter(
        boost_base=1.3,
        max_iterations=3,
        confidence_threshold=0.23,
        clip_check_steps=[10, 15, 20]
    )
    print("✅ DynaPrompt ready!\n")
    
    # Step 4: Generate test image
    test_prompt = "A white dog with brown spots sitting under an orange umbrella"
    print("="*70)
    print(f"Test Prompt: {test_prompt}")
    print("="*70)
    
    # Extract critical words
    critical = reweighter.extract_critical_words(test_prompt)
    print(f"Critical words: {critical}\n")
    
    # Generate with DynaPrompt
    final_image, final_prompt, history = reweighter.iterative_reweight(
        prompt=test_prompt,
        pipe=pixart_pipe,
        generator_seed=42,
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    # Summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    for step in history:
        print(f"\nIteration {step['iteration']}:")
        if step['missed_words']:
            print(f"  Missed: {[(w, f'{c:.3f}') for w, c in step['missed_words']]}")
        else:
            print(f"  ✓ All concepts present!")
    
    print(f"\n✓ Generation complete!")
    print(f"Final prompt: {final_prompt}")
    
    # Save image
    import os
    os.makedirs("outputs", exist_ok=True)
    final_image.save("outputs/dynaprompt_dit_result.png")
    print(f"✓ Saved to outputs/dynaprompt_dit_result.png")


if __name__ == "__main__":
    main()
