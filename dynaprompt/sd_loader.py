"""
Stable Diffusion Model Loader for DynaPrompt Integration
Loads v1.5 checkpoint and provides interfaces for text encoding, denoising, and decoding
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# Add CompVis SD to path
SD_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'stable_diffusion_compvis')
sys.path.insert(0, SD_PATH)

# Add taming-transformers to path
TAMING_PATH = os.path.join(SD_PATH, 'src', 'taming-transformers')
if os.path.exists(TAMING_PATH) and TAMING_PATH not in sys.path:
    sys.path.insert(0, TAMING_PATH)

# Add CLIP to path
CLIP_PATH = os.path.join(SD_PATH, 'src', 'clip')
if os.path.exists(CLIP_PATH) and CLIP_PATH not in sys.path:
    sys.path.insert(0, CLIP_PATH)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class StableDiffusionLoader:
    """Loads and manages Stable Diffusion v1.5 model components"""
    
    def __init__(self, ckpt_path, config_path, device=None):
        """
        Initialize SD model from checkpoint and config
        
        Args:
            ckpt_path: Path to .ckpt file (e.g., v1-5-pruned-emaonly.ckpt)
            config_path: Path to model config (e.g., v1-inference.yaml)
            device: torch device (auto-detected if None)
        """
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        
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
            
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load SD model from checkpoint"""
        print(f"Loading Stable Diffusion from {self.ckpt_path}")
        
        # Load config
        config = OmegaConf.load(self.config_path)
        
        # Load checkpoint
        pl_sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        
        sd = pl_sd["state_dict"]
        
        # Instantiate model
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        
        if len(m) > 0:
            print(f"Missing keys: {len(m)}")
        if len(u) > 0:
            print(f"Unexpected keys: {len(u)}")
            
        # Move to device and set eval mode
        model = model.to(self.device)
        model.eval()
        
        print("Model loaded successfully!")
        return model
    
    def get_text_encoder(self):
        """Get the CLIP text encoder"""
        return self.model.cond_stage_model
    
    def get_unet(self):
        """Get the U-Net diffusion model"""
        return self.model.model
    
    def get_vae(self):
        """Get the VAE (first stage model)"""
        return self.model.first_stage_model
    
    @torch.no_grad()
    def encode_text(self, prompts):
        """
        Encode text prompts to conditioning embeddings
        
        Args:
            prompts: List of text strings or single string
            
        Returns:
            Text embeddings (batch_size, seq_len, dim)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Use the model's text encoder
        c = self.model.get_learned_conditioning(prompts)
        return c
    
    @torch.no_grad()
    def decode_latents(self, latents):
        """
        Decode latent representations to images
        
        Args:
            latents: Latent codes (batch_size, 4, h//8, w//8)
            
        Returns:
            Images (batch_size, 3, height, width) in [-1, 1]
        """
        # Scale latents
        latents = 1. / self.model.scale_factor * latents
        
        # Decode with VAE
        images = self.model.decode_first_stage(latents)
        return images
    
    def create_sampler(self, sampler_type="ddim"):
        """
        Create a sampler for the model
        
        Args:
            sampler_type: "ddim" or "plms"
            
        Returns:
            Sampler instance
        """
        if sampler_type == "ddim":
            return DDIMSampler(self.model)
        elif sampler_type == "plms":
            return PLMSSampler(self.model)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    @torch.no_grad()
    def generate_baseline(self, prompts, steps=50, cfg_scale=7.5, height=512, width=512, 
                         sampler_type="ddim", eta=0.0, seed=None):
        """
        Generate images using vanilla Stable Diffusion (no DynaPrompt)
        
        Args:
            prompts: Text prompts (str or list)
            steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            height, width: Output dimensions
            sampler_type: "ddim" or "plms"
            eta: DDIM eta parameter
            seed: Random seed
            
        Returns:
            images: Generated images as torch tensors
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        if isinstance(prompts, str):
            prompts = [prompts]
        
        batch_size = len(prompts)
        
        # Encode prompts
        c = self.encode_text(prompts)
        
        # Unconditional conditioning for CFG
        uc = self.model.get_learned_conditioning(batch_size * [""])
        
        # Create sampler
        sampler = self.create_sampler(sampler_type)
        
        # Sample
        shape = [4, height // 8, width // 8]
        samples, _ = sampler.sample(
            S=steps,
            conditioning=c,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=uc,
            eta=eta
        )
        
        # Decode to images
        images = self.decode_latents(samples)
        
        return images


def load_sd_model(ckpt_path=None, config_path=None, device=None):
    """
    Convenience function to load SD model with default paths
    
    Args:
        ckpt_path: Path to checkpoint (defaults to v1-5-pruned-emaonly.ckpt)
        config_path: Path to config (defaults to v1-inference.yaml)
        device: Torch device
        
    Returns:
        StableDiffusionLoader instance
    """
    # Default paths
    if ckpt_path is None:
        ckpt_path = os.path.join(
            os.path.dirname(__file__), 
            '../models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt'
        )
    
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            '../models/stable_diffusion_compvis/configs/stable-diffusion/v1-inference.yaml'
        )
    
    # Normalize paths
    ckpt_path = os.path.normpath(ckpt_path)
    config_path = os.path.normpath(config_path)
    
    # Check files exist
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    return StableDiffusionLoader(ckpt_path, config_path, device)


if __name__ == "__main__":
    # Test loading
    print("Testing SD model loading...")
    sd = load_sd_model()
    
    # Test text encoding
    print("\nTesting text encoding...")
    embeddings = sd.encode_text("A beautiful sunset over mountains")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Test baseline generation
    print("\nTesting baseline generation...")
    images = sd.generate_baseline(
        "A golden retriever playing with a red ball",
        steps=20,
        cfg_scale=7.5
    )
    print(f"Generated image shape: {images.shape}")
    print("✓ All tests passed!")
