
import torch
from transformers import CLIPProcessor, CLIPModel
import time

# Lazy import for FID to avoid matplotlib circular import issues
_FID_MODULE = None
_BLIP2_MODULE = None

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

def _get_blip2_module():
    """Lazy load BLIP-2 for compositional accuracy"""
    global _BLIP2_MODULE
    if _BLIP2_MODULE is None:
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            _BLIP2_MODULE = {
                'processor': Blip2Processor,
                'model': Blip2ForConditionalGeneration
            }
        except ImportError as e:
            print(f"Warning: Could not import BLIP-2: {e}")
            _BLIP2_MODULE = False
    return _BLIP2_MODULE

# BLIP-2 placeholder import (replace with actual BLIP-2 integration)
# from blip2 import Blip2Model

class DynaPrompt:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.fid = None  # Lazy-loaded when needed
        self.blip2 = None  # Lazy-loaded when needed
        self.blip2_processor = None

    def compute_clipscore(self, image, prompt):
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model(**inputs)
        clipscore = outputs.logits_per_image[0][0].item()
        return clipscore

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
        Compute compositional accuracy using BLIP-2 to check if generated image
        contains objects/attributes mentioned in prompt
        
        Args:
            image: Generated image tensor (1, 3, H, W) in [0, 1]
            prompt: Text prompt (str)
            
        Returns:
            Compositional accuracy score or None if BLIP-2 not available
        """
        BLIP2_MODULES = _get_blip2_module()
        if BLIP2_MODULES is False:
            return None
        
        # Lazy initialize BLIP-2
        if self.blip2 is None:
            try:
                print("Loading BLIP-2 for compositional accuracy...")
                self.blip2_processor = BLIP2_MODULES['processor'].from_pretrained("Salesforce/blip2-opt-2.7b")
                self.blip2 = BLIP2_MODULES['model'].from_pretrained("Salesforce/blip2-opt-2.7b").to(self.device)
            except Exception as e:
                print(f"Failed to load BLIP-2: {e}")
                return None
        
        try:
            # Convert tensor to PIL Image for BLIP-2
            import torchvision.transforms as T
            to_pil = T.ToPILImage()
            pil_image = to_pil(image.squeeze(0))
            
            # Generate caption from image
            inputs = self.blip2_processor(pil_image, return_tensors="pt").to(self.device)
            generated_ids = self.blip2.generate(**inputs, max_length=50)
            generated_caption = self.blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Simple compositional accuracy: check if key words from prompt appear in caption
            prompt_words = set(prompt.lower().split())
            caption_words = set(generated_caption.lower().split())
            
            # Remove common stop words
            stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'to', 'is', 'are'}
            prompt_words -= stop_words
            caption_words -= stop_words
            
            if len(prompt_words) == 0:
                return 0.0
            
            # Calculate overlap
            overlap = len(prompt_words & caption_words)
            accuracy = overlap / len(prompt_words)
            
            return accuracy
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

    def feedback_loop(self, prompt, current_embedding, generated_image, step):
        """
        Main DynaPrompt feedback loop - computes semantic alignment and updates embeddings
        
        Args:
            prompt: Original text prompt (str)
            current_embedding: Current prompt embedding tensor (1, seq_len, dim)
            generated_image: Current generated image (1, 3, H, W) in [0, 1]
            step: Current denoising step (int)
            
        Returns:
            dict with 'updated_embedding', 'clip_score', and 'embedding_shift'
        """
        # Compute CLIP score for semantic alignment
        clipscore = self.compute_clipscore(generated_image, prompt)
        
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
        
        # Update embedding based on feedback with conservative alpha
        updated_embedding = self.update_prompt_embedding(
            current_embedding, 
            feedback_gradient,
            alpha=0.05  # Very conservative to avoid corruption
        )
        
        # Calculate embedding shift magnitude
        embedding_shift = torch.norm(updated_embedding - current_embedding).item()
        
        return {
            'updated_embedding': updated_embedding,
            'clip_score': clipscore,
            'embedding_shift': embedding_shift,
            'step': step
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
