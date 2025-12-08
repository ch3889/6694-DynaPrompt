"""
dynaprompt_dit: Diffusion Transformer-based dynamic prompt guidance

This package provides a sampler that performs early attention detection and
runtime token reweighting for DiT-style architectures.
"""

__all__ = [
    "DiTAttentionBooster",
    "DiTDynaPromptSampler",
]
