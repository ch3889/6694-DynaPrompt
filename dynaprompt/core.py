
import torch
from transformers import CLIPProcessor, CLIPModel
import time

# Lazy import for FID to avoid matplotlib circular import issues
_FID_MODULE = None

def _get_fid_module():
    """Lazy load FrechetInceptionDistance to avoid import errors"""
    global _FID_MODULE
    if _FID_MODULE is None:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            _FID_MODULE = FrechetInceptionDistance
        except ImportError as e:
            print(f"Warning: Could not import FrechetInceptionDistance: {e}")
            _FID_MODULE = False
    return _FID_MODULE

class DynaPrompt:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
        self.fid = None  # Lazy-loaded when needed

    def compute_clipscore(self, image, prompt):
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True, do_rescale=False).to(self.device)
        outputs = self.clip_model(**inputs)
        clipscore = outputs.logits_per_image[0][0].item()
        return clipscore
    
    def compute_per_token_alignment(self, image, prompt, sd_tokenizer=None):
        """
        Compute CLIP alignment score for each token/concept in the prompt.
        Identifies which concepts are underrepresented in the generated image.
        
        Args:
            image: Generated image tensor (1, 3, H, W) in [0, 1]
            prompt: Full text prompt (str)
            sd_tokenizer: Optional SD tokenizer to match token positions
            
        Returns:
            dict with 'token_scores', 'weak_tokens', 'strong_tokens'
        """
        import re
        
        # Split prompt into meaningful phrases and words
        # Remove common stop words that don't carry semantic meaning
        stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'to', 'is', 'are'}
        
        # Extract noun phrases and key concepts
        words = prompt.lower().split()
        
        # Also try phrase combinations for multi-word concepts
        concepts = []
        concept_positions = []
        
        # Add individual meaningful words
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word not in stop_words and len(clean_word) > 2:
                concepts.append(clean_word)
                concept_positions.append(i)
        
        # Add bigrams for compound concepts (e.g., "red ball", "snowy park")
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            # Skip if both words are stop words
            if words[i] not in stop_words or words[i+1] not in stop_words:
                concepts.append(bigram)
                concept_positions.append(i)
        
        # Add trigrams for complex phrases (e.g., "golden retriever playing")
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if any(w not in stop_words for w in [words[i], words[i+1], words[i+2]]):
                concepts.append(trigram)
                concept_positions.append(i)
        
        # Compute CLIP score for each concept
        token_scores = {}
        for concept in concepts:
            try:
                score = self.compute_clipscore(image, concept)
                token_scores[concept] = score
            except Exception as e:
                # Skip if CLIP fails for this concept
                token_scores[concept] = 0.0
        
        # Identify weak (underrepresented) and strong tokens
        if token_scores:
            scores_list = list(token_scores.values())
            mean_score = sum(scores_list) / len(scores_list)
            std_score = (sum((s - mean_score) ** 2 for s in scores_list) / len(scores_list)) ** 0.5
            
            # Threshold: tokens below mean - 0.5*std are considered weak
            threshold = mean_score - 0.5 * std_score
            
            weak_tokens = {k: v for k, v in token_scores.items() if v < threshold}
            strong_tokens = {k: v for k, v in token_scores.items() if v >= threshold}
        else:
            weak_tokens = {}
            strong_tokens = {}
            threshold = 0.0
        
        return {
            'token_scores': token_scores,
            'weak_tokens': weak_tokens,
            'strong_tokens': strong_tokens,
            'threshold': threshold,
            'concepts': concepts
        }

    def compute_fid(self, generated_images, real_images):
        """
        Compute FID score between generated and real images
        
        Args:
            generated_images: Tensor (B, C, H, W) in [0, 1]
            real_images: Tensor (B, C, H, W) in [0, 1]
            
        Returns:
            FID score (float) or None if FID not available
        """
        FID_CLASS = _get_fid_module()
        if FID_CLASS is False:
            print("Warning: FID computation not available (torchmetrics import failed)")
            return None
        
        # Lazy initialize FID metric
        if self.fid is None:
            self.fid = FID_CLASS(feature=64).to(self.device)
        
        self.fid.reset()
        self.fid.update(generated_images, real=False)
        self.fid.update(real_images, real=True)
        fid_score = self.fid.compute().item()
        return fid_score

    def compute_compositional_accuracy(self, image, prompt):
        """
        Compute compositional accuracy based on per-token alignment scores.
        This avoids the need for BLIP-2 (15GB download) by leveraging the
        per-token analysis already computed during generation.
        
        Args:
            image: Generated image tensor (1, 3, H, W) in [0, 1]
            prompt: Text prompt (str)
            
        Returns:
            Compositional accuracy score (0-1) based on token alignment
        """
        try:
            # Extract concepts using per-token analysis
            analysis_result = self.compute_per_token_alignment(image, prompt)
            concept_scores = analysis_result['token_scores']
            weak_tokens = analysis_result['weak_tokens']
            
            if not concept_scores:
                return 0.0
            
            # Calculate compositional completeness
            total_concepts = len(concept_scores)
            weak_concepts = len(weak_tokens)
            compositional_completeness = (total_concepts - weak_concepts) / total_concepts
            
            # Calculate average alignment score (normalized to 0-1 range)
            avg_alignment = sum(concept_scores.values()) / total_concepts
            normalized_alignment = avg_alignment / 30.0  # CLIP scores typically 0-30
            
            # Weighted combination: 70% completeness + 30% alignment strength
            # Completeness ensures all concepts present, alignment ensures quality
            compositional_accuracy = 0.7 * compositional_completeness + 0.3 * normalized_alignment
            
            return compositional_accuracy
            
        except Exception as e:
            print(f"Error computing compositional accuracy: {e}")
            return None

    def measure_generation_time(self, start_time, end_time):
        return end_time - start_time

    def update_prompt_embedding(self, prompt_embedding, feedback_gradient, alpha=0.1):
        """
        Update prompt embedding using gradient-like feedback
        
        Args:
            prompt_embedding: Current embedding (1, seq_len, dim)
            feedback_gradient: Gradient tensor same shape as embedding
            alpha: Update strength (smaller = more conservative)
        """
        # Normalize gradient to prevent explosion
        gradient_norm = torch.norm(feedback_gradient)
        if gradient_norm > 0:
            feedback_gradient = feedback_gradient / gradient_norm
        
        # Conservative update with small alpha
        updated_embedding = prompt_embedding + alpha * feedback_gradient
        return updated_embedding
    
    def selective_token_reweight(self, prompt_embedding, weak_tokens, prompt, tokenizer=None, boost_factor=1.5):
        """
        Selectively boost embeddings of underrepresented tokens.
        This is the core of DynaPrompt - only emphasize missing concepts.
        
        Args:
            prompt_embedding: Current embedding (1, seq_len, dim)
            weak_tokens: Dict of underrepresented concepts {concept: score}
            prompt: Original text prompt
            tokenizer: SD tokenizer to map concepts to token positions
            boost_factor: How much to amplify weak tokens (default 1.5x)
            
        Returns:
            Updated embedding with weak tokens boosted
        """
        if not weak_tokens:
            return prompt_embedding
        
        # Create a copy to modify
        updated_embedding = prompt_embedding.clone()
        
        # Get token positions for weak concepts
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        
        # For each weak token, find its position and boost its embedding
        boost_mask = torch.ones_like(updated_embedding)
        
        for weak_concept, score in weak_tokens.items():
            # Find positions of this concept in the prompt
            concept_words = weak_concept.split()
            
            # Search for concept in prompt
            for i in range(len(words)):
                # Check if concept matches at this position
                match = True
                for j, cword in enumerate(concept_words):
                    if i + j >= len(words) or words[i + j] != cword:
                        match = False
                        break
                
                if match:
                    # Boost tokens at positions i to i+len(concept_words)
                    # SD tokenizer typically adds BOS token, so offset by 1
                    for j in range(len(concept_words)):
                        token_idx = i + j + 1  # +1 for BOS token
                        if token_idx < updated_embedding.shape[1]:
                            # Stronger boost for weaker concepts
                            # score is typically 0-30 for CLIP, normalize
                            weakness = max(0, 20 - score) / 20  # 0 to 1, higher = weaker
                            adaptive_boost = 1.0 + boost_factor * weakness
                            boost_mask[0, token_idx, :] *= adaptive_boost
        
        # Apply boost mask
        updated_embedding = updated_embedding * boost_mask
        
        # Normalize to prevent explosion
        norm_before = torch.norm(prompt_embedding)
        norm_after = torch.norm(updated_embedding)
        if norm_after > 0:
            updated_embedding = updated_embedding * (norm_before / norm_after)
        
        return updated_embedding

    def feedback_loop(self, prompt, current_embedding, generated_image, step, use_per_token=True, alpha=None):
        """
        Main DynaPrompt feedback loop - computes semantic alignment and updates embeddings
        
        Args:
            prompt: Original text prompt (str)
            current_embedding: Current prompt embedding tensor (1, seq_len, dim)
            generated_image: Current generated image (1, 3, H, W) in [0, 1]
            step: Current denoising step (int)
            use_per_token: Whether to use per-token analysis (default True)
            alpha: Update strength (default 0.15, can be overridden for adaptive control)
            
        Returns:
            dict with 'updated_embedding', 'clip_score', 'embedding_shift', 'weak_tokens'
        """
        # Compute global CLIP score for semantic alignment
        clipscore = self.compute_clipscore(generated_image, prompt)
        
        # Per-token analysis to detect underrepresented concepts
        weak_tokens = {}
        token_analysis = None
        
        if use_per_token:
            token_analysis = self.compute_per_token_alignment(generated_image, prompt)
            weak_tokens = token_analysis['weak_tokens']
        
        # Compute text-image features for gradient-based feedback
        with torch.no_grad():
            # Get CLIP text embedding for the prompt
            inputs = self.clip_processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
            text_features = self.clip_model.get_text_features(**inputs)
            
            # Get CLIP image features
            image_inputs = self.clip_processor(images=generated_image, return_tensors="pt", do_rescale=False).to(self.device)
            image_features = self.clip_model.get_image_features(**image_inputs)
            
            # Compute alignment direction (push text toward image semantics)
            alignment_direction = image_features - text_features
            
            # Create pseudo-gradient by projecting onto embedding space
            # Match dimensions: CLIP features (512) -> SD embedding (768)
            if alignment_direction.shape[-1] != current_embedding.shape[-1]:
                # Project CLIP features to SD embedding dimension
                projection = torch.nn.functional.pad(
                    alignment_direction, 
                    (0, current_embedding.shape[-1] - alignment_direction.shape[-1])
                )
                feedback_gradient = projection.unsqueeze(1).expand_as(current_embedding)
            else:
                feedback_gradient = alignment_direction.unsqueeze(1).expand_as(current_embedding)
            
            # Scale by alignment score (stronger feedback when misaligned)
            feedback_scale = (1.0 - torch.clamp(torch.tensor(clipscore / 100.0), 0, 1)).item()
            feedback_gradient = feedback_gradient * feedback_scale
        
        # Strategy 1: Global gradient-based update (for overall alignment)
        # Use provided alpha or default to moderate value
        if alpha is None:
            alpha = 0.12  # Default to stronger feedback for proactive approach
        
        global_updated = self.update_prompt_embedding(
            current_embedding, 
            feedback_gradient,
            alpha=alpha
        )
        
        # Strategy 2: Selective token re-weighting (for missing concepts)
        if use_per_token and weak_tokens:
            # Apply selective boost to underrepresented tokens
            updated_embedding = self.selective_token_reweight(
                global_updated,
                weak_tokens,
                prompt,
                boost_factor=1.8  # Stronger boost for weak tokens
            )
        else:
            updated_embedding = global_updated
        
        # Calculate embedding shift magnitude
        embedding_shift = torch.norm(updated_embedding - current_embedding).item()
        
        return {
            'updated_embedding': updated_embedding,
            'clip_score': clipscore,
            'embedding_shift': embedding_shift,
            'step': step,
            'weak_tokens': list(weak_tokens.keys()) if weak_tokens else [],
            'token_analysis': token_analysis
        }

    def run_generation(self, prompt, initial_embedding, denoising_steps, real_images=None):
        # Example main loop for generation and metric logging
        prompt_embedding = initial_embedding
        generated_images = []
        start_time = time.time()
        for step in range(denoising_steps):
            # ...generate image using current prompt_embedding...
            image = torch.rand(1, 3, 512, 512).to(self.device)  # Placeholder for generated image
            generated_images.append(image)
            # Feedback at intermediate steps
            if step % 10 == 0:
                prompt_embedding = self.feedback_loop(image, prompt, prompt_embedding)
        end_time = time.time()

        # Quantitative metrics
        clipscore = self.compute_clipscore(generated_images[-1], prompt)
        fid_score = None
        if real_images is not None:
            fid_score = self.compute_fid(torch.cat(generated_images), real_images)
        compositional_accuracy = self.compute_compositional_accuracy(generated_images[-1], prompt)
        gen_time = self.measure_generation_time(start_time, end_time)

        return {
            'clipscore': clipscore,
            'fid_score': fid_score,
            'compositional_accuracy': compositional_accuracy,
            'generation_time': gen_time,
            'generated_images': generated_images
        }

    def compute_metrics(self, prompt, images, real_images=None):
        """
        Compute evaluation metrics for generated images
        
        Args:
            prompt: Text prompt (str)
            images: Generated images tensor (batch_size, 3, H, W) in [0, 1]
            real_images: Optional real images for FID computation
            
        Returns:
            dict with 'clip_score', 'fid_score', 'compositional_accuracy'
        """
        # Compute CLIP score
        clip_score = self.compute_clipscore(images, prompt)
        
        # Compute FID if real images provided
        fid_score = None
        if real_images is not None:
            fid_score = self.compute_fid(images, real_images)
        
        # Compute compositional accuracy (placeholder)
        compositional_accuracy = self.compute_compositional_accuracy(images, prompt)
        
        return {
            'clip_score': clip_score,
            'fid_score': fid_score,
            'compositional_accuracy': compositional_accuracy
        }


# =====================
# DynaPrompt Integration Example
# =====================
#
# In your Stable Diffusion denoising loop, use DynaPrompt as follows:
#
# from dynaprompt.core import DynaPrompt
#
# # Initialize DynaPrompt feedback module
# dynaprompt = DynaPrompt(device="cuda")
#
# # After text encoding (using SD's CLIP encoder):
# prompt_embedding = initial_embedding  # output from SD's text encoder
#
# for step in range(num_steps):
#     # Denoising step using current prompt_embedding
#     image = unet_denoise(latent, prompt_embedding)  # your U-Net denoising function
#
#     # At selected intermediate steps, apply DynaPrompt feedback
#     if step % feedback_interval == 0:
#         prompt_embedding = dynaprompt.feedback_loop(image, prompt, prompt_embedding)
#
# # After generation, compute metrics
# clipscore = dynaprompt.compute_clipscore(image, prompt)
# # Optionally: fid_score, compositional_accuracy, generation_time
#
#
# This approach keeps DynaPrompt external to the SD architecture, only affecting prompt conditioning during denoising.
