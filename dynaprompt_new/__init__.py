"""
dynaprompt_new: Fresh, minimal implementation of dynamic prompt guidance

This package provides a clean sampler that detects under-attended tokens
early and optionally boosts their cross-attention weights to reduce
"missing word" failures, using the CompVis Stable Diffusion repo.
"""

__all__ = [
    "AttentionBooster",
    "DynapromptNewSampler",
]
