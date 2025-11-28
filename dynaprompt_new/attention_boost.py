import torch
from typing import List, Optional


class AttentionBooster:
    """
    Lightweight attention booster that amplifies cross-attention weights
    for under-attended token indices. Keeps behavior contained to a simple
    wrapper around the CrossAttention forward, without touching other code.
    """

    def __init__(self, boost_factor: float = 6.0, max_multiplier: float = 18.0):
        self.boost_factor = boost_factor
        self.max_multiplier = max_multiplier
        self.indices: List[int] = []
        self.enabled: bool = False

    def set_indices(self, indices: List[int]):
        self.indices = list(indices)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def apply(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Apply boosting to attention tensor [B*H, pixels, tokens].
        """
        if not self.enabled or len(self.indices) == 0:
            return attn

        boosted = attn.clone()

        # Adaptive boost based on current mean attention per token
        for idx in self.indices:
            if idx >= boosted.shape[-1]:
                continue
            current = boosted[:, :, idx].mean().item()
            if current < 1e-3:
                mult = self.boost_factor * 3.0
            elif current < 5e-3:
                mult = self.boost_factor * 2.0
            elif current < 1e-2:
                mult = self.boost_factor * 1.5
            else:
                mult = self.boost_factor
            mult = min(mult, self.max_multiplier)
            boosted[:, :, idx] *= mult

        # Renormalize
        boosted = boosted / boosted.sum(dim=-1, keepdim=True)
        return boosted
