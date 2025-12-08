"""
DynaPrompt Wrapper: Integrates DynaPrompt with Stable Diffusion
This script demonstrates full integration of DynaPrompt feedback with SD denoising
"""

import torch
import yaml
import numpy as np
from tqdm import tqdm
from .core import DynaPrompt
from .sd_loader import load_sd_model


class DynaPromptPipeline:
    """
    Complete DynaPrompt + Stable Diffusion pipeline
    Integrates real-time semantic feedback into the denoising loop
    """
    
    def __init__(self, config_path='configs/dynaprompt_config.yaml', ckpt_path=None, device=None):
        """
        Initialize the DynaPrompt + SD pipeline
        
        Args:
            config_path: Path to DynaPrompt configuration
            ckpt_path: Path to SD checkpoint (uses default if None)
            device: Torch device (auto-detected if None)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        print(f"Initializing DynaPrompt Pipeline on {self.device}...")
        
        # Load Stable Diffusion model
        print("Loading Stable Diffusion model...")
        self.sd = load_sd_model(ckpt_path=ckpt_path, device=self.device)
        
        # Initialize DynaPrompt feedback system
        print("Loading CLIP for DynaPrompt feedback...")
        clip_model = self.config.get('clip', {}).get('model_name', None)
        if clip_model is None:
            # Fallback to feedback.clip_model for backward compatibility
            clip_model = self.config.get('feedback', {}).get('clip_model', 'openai/clip-vit-base-patch32')
            # Map ViT-B/32 to HuggingFace format
            if clip_model == 'ViT-B/32':
                clip_model = 'openai/clip-vit-base-patch32'
            elif clip_model == 'ViT-L/14':
                clip_model = 'openai/clip-vit-large-patch14'
        
        self.dynaprompt = DynaPrompt(
            clip_model_name=clip_model,
            device=self.device
        )
        
        print("✓ Pipeline initialized successfully!")
    
    @torch.no_grad()
    def generate_with_feedback(self, prompt, steps=None, cfg_scale=None, height=512, width=512,
                               seed=None, feedback_enabled=None, sampler_type="ddim", eta=0.0):
        """
        Generate images with DynaPrompt real-time feedback
        
        Args:
            prompt: Text prompt (str)
            steps: Number of denoising steps (from config if None)
            cfg_scale: Classifier-free guidance scale (from config if None)
            height, width: Output dimensions
            seed: Random seed
            feedback_enabled: Whether to use DynaPrompt feedback (from config if None)
            sampler_type: "ddim" or "plms"
            eta: DDIM eta parameter
            
        Returns:
            dict with 'images', 'metrics', and 'trajectory'
        """
        # Use config defaults
        if steps is None:
            steps = self.config.get('sampling', {}).get('ddim_steps', 50)
        if cfg_scale is None:
            cfg_scale = self.config.get('sampling', {}).get('scale', 7.5)
        if feedback_enabled is None:
            feedback_enabled = self.config.get('feedback', {}).get('enabled', True)
        
        if seed is not None:
            torch.manual_seed(seed)
        
        print(f"\nGenerating: '{prompt}'")
        print(f"Steps: {steps}, CFG: {cfg_scale}, Feedback: {feedback_enabled}")
        
        # Start timing
        import time
        start_time = time.time()
        
        # Encode prompt
        c = self.sd.encode_text([prompt])  # (1, 77, 768)
        uc = self.sd.encode_text([""])      # Unconditional for CFG
        
        # Keep track of original embedding for feedback
        c_original = c.clone()
        
        # Create sampler
        sampler = self.sd.create_sampler(sampler_type)
        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)
        
        # Initialize latent
        shape = [1, 4, height // 8, width // 8]
        latents = torch.randn(shape, device=self.device)
        
        # Feedback configuration
        feedback_freq = self.config.get('feedback', {}).get('feedback_frequency', 10)
        feedback_start = self.config.get('feedback', {}).get('feedback_start_step', 0)
        feedback_end = self.config.get('feedback', {}).get('feedback_end_step', steps)
        update_alpha = self.config.get('prompt_update', {}).get('update_alpha', 0.3)
        
        # Storage for metrics and trajectory
        metrics_history = []
        embedding_trajectory = []
        
        # Denoising loop with DynaPrompt feedback
        timesteps = sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        
        print(f"Running denoising with {total_steps} steps...")
        iterator = tqdm(enumerate(time_range), total=total_steps, desc="DynaPrompt+SD")
        
        for i, step in iterator:
            index = total_steps - i - 1
            ts = torch.full((1,), step, device=self.device, dtype=torch.long)
            
            # === DynaPrompt Feedback Integration ===
            # Apply feedback at specified intervals
            if (feedback_enabled and 
                i % feedback_freq == 0 and 
                feedback_start <= i < feedback_end):
                
                # Decode current latents to get intermediate image
                with torch.no_grad():
                    intermediate_image = self.sd.decode_latents(latents)
                    # Clamp to [-1, 1] and normalize to [0, 1] for CLIP
                    intermediate_image = torch.clamp((intermediate_image + 1.0) / 2.0, min=0.0, max=1.0)
                
                # Compute CLIP feedback
                feedback_result = self.dynaprompt.feedback_loop(
                    prompt=prompt,
                    current_embedding=c,
                    generated_image=intermediate_image,
                    step=i
                )
                
                # Update conditioning with feedback
                # Blend original and adjusted embedding
                c = (1 - update_alpha) * c + update_alpha * feedback_result['updated_embedding']
                
                # Store metrics
                metrics_history.append({
                    'step': i,
                    'clip_score': feedback_result['clip_score'],
                    'embedding_shift': feedback_result.get('embedding_shift', 0.0),
                    'weak_tokens': feedback_result.get('weak_tokens', []),
                    'token_analysis': feedback_result.get('token_analysis', None)
                })
                
                # Update progress bar with weak tokens info
                weak_tokens_str = ', '.join(feedback_result.get('weak_tokens', [])[:3])  # Show first 3
                if weak_tokens_str:
                    iterator.set_postfix({
                        'CLIP': f"{feedback_result['clip_score']:.3f}",
                        'Weak': weak_tokens_str[:30]  # Truncate if too long
                    })
                else:
                    iterator.set_postfix({'CLIP': f"{feedback_result['clip_score']:.3f}"})
            
            # Store embedding trajectory
            if i % 10 == 0:
                embedding_trajectory.append(c.clone().cpu())
            
            # === Standard DDIM Denoising Step ===
            # Classifier-free guidance
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([ts] * 2)
            c_in = torch.cat([uc, c])
            
            # U-Net prediction
            e_t_uncond, e_t = self.sd.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + cfg_scale * (e_t - e_t_uncond)
            
            # DDIM step
            latents, pred_x0 = sampler.p_sample_ddim(
                x=latents,
                c=c,
                t=ts,
                index=index,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
                use_original_steps=False
            )
        
        # Decode final latents to images
        print("Decoding latents...")
        images = self.sd.decode_latents(latents)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        
        # End timing
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Compute final metrics
        final_metrics = self.dynaprompt.compute_metrics(
            prompt=prompt,
            images=images
        )
        
        # Add generation time to metrics
        final_metrics['generation_time'] = generation_time
        
        results = {
            'images': images,
            'metrics': final_metrics,
            'metrics_history': metrics_history,
            'embedding_trajectory': embedding_trajectory,
            'final_clip_score': final_metrics.get('clip_score', 0.0),
            'fid_score': final_metrics.get('fid_score', None),
            'compositional_accuracy': final_metrics.get('compositional_accuracy', None),
            'generation_time': generation_time,
            'prompt': prompt
        }
        
        print(f"✓ Generation complete! Final CLIP Score: {results['final_clip_score']:.3f}")
        print(f"  Generation Time: {generation_time:.2f}s")
        if results['fid_score'] is not None:
            print(f"  FID Score: {results['fid_score']:.3f}")
        if results['compositional_accuracy'] is not None:
            print(f"  Compositional Accuracy: {results['compositional_accuracy']:.3f}")
        
        # Print weak tokens analysis
        if metrics_history:
            all_weak_tokens = set()
            for entry in metrics_history:
                all_weak_tokens.update(entry.get('weak_tokens', []))
            if all_weak_tokens:
                print(f"  Underrepresented concepts detected: {', '.join(sorted(all_weak_tokens)[:5])}")
        
        return results


def run_dynaprompt_generation(prompt, config_path='configs/dynaprompt_config.yaml', **kwargs):
    """
    Convenience function to run DynaPrompt generation
    
    Args:
        prompt: Text prompt
        config_path: Path to DynaPrompt config
        **kwargs: Additional arguments for generate_with_feedback
        
    Returns:
        Generation results dict
    """
    pipeline = DynaPromptPipeline(config_path=config_path)
    return pipeline.generate_with_feedback(prompt, **kwargs)


if __name__ == "__main__":
    # Test DynaPrompt generation
    prompt = "A golden retriever playing with a red ball in a snowy park"
    results = run_dynaprompt_generation(prompt, steps=30, cfg_scale=7.5, seed=42)
    print(f"Generated {results['images'].shape[0]} images")
    print(f"Final CLIP Score: {results['final_clip_score']:.3f}")
