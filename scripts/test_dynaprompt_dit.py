"""
Test runner for dynaprompt_dit using a placeholder DiT components interface.
Wire this to your actual DiT implementation.
"""

import argparse
import torch
from dataclasses import dataclass

from dynaprompt_dit.sampler import DiTDynaPromptSampler, DiTComponents


# Placeholder tokenizer/text encoder compatible interface
class DummyTokenizer:
    def __init__(self):
        self.model_max_length = 77
        self.pad_token_id = 0
    
    def __call__(self, text, **kwargs):
        # Handle both list and string inputs
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1
            text = [text]
        
        # Check if this is for word tokenization (no padding kwarg or add_special_tokens only)
        if 'add_special_tokens' in kwargs and len(kwargs) == 1:
            # Simple word tokenization
            return {"input_ids": [ord(c) % 100 for c in text[0]]}
        
        # HF-like tokenizer call with padding
        input_ids = torch.randint(1, 100, (batch_size, self.model_max_length))
        return type("DummyOut", (), {"input_ids": input_ids, "to": lambda self, d: self})()

class DummyTextEncoder:
    def __call__(self, input_ids):
        # Return embeddings tensor
        B, L = input_ids.shape
        return torch.randn(B, L, 768)

class DummyScheduler:
    def set_timesteps(self, steps):
        self.timesteps = list(range(steps))

class DummyDiT:
    latent_dim = 256
    def modules(self):
        return []
    def __call__(self, latents, t, text_embeddings, cfg_scale):
        # Simulate diffusion update
        return latents * 0.95 + torch.randn_like(latents) * 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--cfg', type=float, default=4.0)
    args = parser.parse_args()

    comps = DiTComponents(
        tokenizer=DummyTokenizer(),
        text_encoder=DummyTextEncoder(),
        dit_model=DummyDiT(),
        scheduler=DummyScheduler(),
    )

    sampler = DiTDynaPromptSampler(comps, check_step=3)
    latents, meta = sampler.sample(prompt=args.prompt, steps=args.steps, cfg_scale=args.cfg)
    print("Sampling complete. Latents shape:", latents.shape)
    print("Meta:", meta)


if __name__ == '__main__':
    main()
