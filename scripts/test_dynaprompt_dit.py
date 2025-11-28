"""
Test runner for dynaprompt_dit using a placeholder DiT components interface.
Wire this to your actual DiT implementation.
"""

import argparse
import torch
from dataclasses import dataclass

from dynaprompt_dit.sampler import DiTDynaPromptSampler, DiTComponents


# Placeholder tokenizer/text encoder compatible interface
@dataclass
class DummyTokenizer:
    model_max_length: int = 77
    pad_token_id: int = 0
    def __call__(self, text_list, **kwargs):
        # Simulate HF-like output
        input_ids = torch.randint(1, 100, (len(text_list), self.model_max_length))
        return type("DummyOut", (), {"input_ids": input_ids, "to": lambda self, d: self})()
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 100 for c in text]}

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
