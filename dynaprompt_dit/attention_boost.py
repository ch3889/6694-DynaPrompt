import torch
from typing import List


class DiTAttentionBooster:
    """
    Attention booster for Diffusion Transformer attention maps.
    Applies adaptive per-token scaling and renormalization.
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
        attn: [B, heads, Q, K] or [B*H, Q, K] depending on DiT implementation.
        We boost along the token/key dimension K.
        """
        if not self.enabled or len(self.indices) == 0:
            return attn

        boosted = attn.clone()

        # Handle [B, H, Q, K] by merging B and H for simplicity
        merged = False
        if boosted.dim() == 4:
            B, H, Q, K = boosted.shape
            boosted = boosted.reshape(B * H, Q, K)
            merged = True

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

        boosted = boosted / boosted.sum(dim=-1, keepdim=True)

        if merged:
            boosted = boosted.reshape(B, H, Q, K)

        return boosted
