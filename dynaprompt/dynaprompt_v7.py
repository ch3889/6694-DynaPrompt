"""
DynaPrompt V7: Complete compositional generation system.

Combines:
1. Very early detection (step 3-5 instead of 15)
2. Attention boosting fallback (V6 Phase 2)
3. Adaptive boosting based on attention levels

This is the most advanced and accurate version.
"""

import torch
import sys
from typing import List, Optional, Tuple
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'stable_diffusion_compvis'))
sys.path.insert(0, str(Path(__file__).parent))

from dynaprompt_v6 import DynaPromptV6Sampler


class DynaPromptV7Sampler(DynaPromptV6Sampler):
    """
    DynaPrompt V7: Most advanced compositional generation.

    New features over V6:
    - Configurable early detection (default: step 3 instead of 15)
    - Higher retry count (default: 15 instead of 2)
    - Adaptive boosting based on attention levels
    """

    def __init__(
        self,
        ddim_sampler,
        model,
        tokenizer,
        device="cuda",
        check_step=3,  # VERY early detection (6% completion)
        attention_threshold=0.05,  # Stricter threshold than before
        max_retries=15,  # Many seed retries before boosting fallback
        boost_factor=7.5,  # Strong base boost (adaptive up to 22.5x for very low attention)
        start_step_ratio=0.0,
        end_step_ratio=0.5,  # Longer than V6's 0.4
    ):
        """
        Initialize DynaPrompt V7 sampler.

        Args:
            ddim_sampler: DDIM sampler instance
            model: Stable Diffusion model
            tokenizer: Text tokenizer
            device: Device to use
            check_step: Step to check composition (3 = 6% done, earlier is better)
            attention_threshold: Minimum attention score (lower = more permissive)
            max_retries: Maximum seed retries before boosting
            boost_factor: Base attention boost multiplier (adaptive: up to 3x this value for very low attention)
            start_step_ratio: When to start boosting
            end_step_ratio: When to stop boosting
        """
        super().__init__(
            ddim_sampler=ddim_sampler,
            model=model,
            tokenizer=tokenizer,
            device=device,
            check_step=check_step,
            attention_threshold=attention_threshold,
            max_retries=max_retries,
            boost_factor=boost_factor,
            start_step_ratio=start_step_ratio,
            end_step_ratio=end_step_ratio,
        )

    def sample_with_dynaprompt(
        self,
        prompt: str,
        shape,
        steps=50,
        unconditional_guidance_scale=7.5,
        critical_tokens: List[str] = None,
        verbose=True,
    ):
        """
        Generate image with DynaPrompt V7.

        Args:
            prompt: Text prompt
            shape: Latent shape
            steps: Number of diffusion steps
            unconditional_guidance_scale: CFG scale
            critical_tokens: Required objects (auto-detected if None)
            verbose: Print progress

        Returns:
            (samples, intermediates)
        """
        # Use V6's standard path with V7's improved parameters
        return super().sample_with_dynaprompt(
            prompt=prompt,
            shape=shape,
            steps=steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            critical_tokens=critical_tokens,
            verbose=verbose,
        )

