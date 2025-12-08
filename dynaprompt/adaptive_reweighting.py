"""
Adaptive Reweighting Module for Hybrid DynaPrompt

Implements dynamic weight adjustment for both embedding updates and attention boosting
based on feedback effectiveness and current generation state.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class AdaptiveReweighter:
    """
    Dynamically adjusts feedback weights based on effectiveness
    
    Features:
    - Adaptive alpha for embedding updates (learns from CLIP score improvement)
    - Adaptive boost factor for attention (based on attention strength)
    - Momentum-based smoothing to prevent oscillation
    """
    
    def __init__(
        self,
        initial_alpha: float = 0.08,
        initial_boost: float = 1.3,
        min_alpha: float = 0.01,
        max_alpha: float = 0.3,
        min_boost: float = 1.1,
        max_boost: float = 2.0,
        momentum: float = 0.9,
        adaptation_rate: float = 0.1
    ):
        """
        Args:
            initial_alpha: Starting embedding update strength
            initial_boost: Starting attention boost factor
            min_alpha, max_alpha: Bounds for alpha
            min_boost, max_boost: Bounds for boost factor
            momentum: Smoothing factor for weight updates (0-1)
            adaptation_rate: How aggressively to adapt (0-1)
        """
        self.alpha = initial_alpha
        self.boost_factor = initial_boost
        
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_boost = min_boost
        self.max_boost = max_boost
        
        self.momentum = momentum
        self.adaptation_rate = adaptation_rate
        
        # History tracking
        self.clip_score_history = []
        self.alpha_history = []
        self.boost_history = []
        self.weak_token_counts = []
        
    def update_alpha(self, current_clip: float, previous_clip: Optional[float] = None) -> float:
        """
        Adaptively adjust alpha based on CLIP score improvement
        
        Strategy:
        - If CLIP score improved significantly → keep or increase alpha
        - If CLIP score stagnated/decreased → reduce alpha
        - Use momentum to smooth changes
        
        Args:
            current_clip: Current CLIP score
            previous_clip: Previous CLIP score (None for first iteration)
            
        Returns:
            Updated alpha value
        """
        self.clip_score_history.append(current_clip)
        
        if previous_clip is not None:
            improvement = current_clip - previous_clip
            
            # Adaptive adjustment based on improvement - favor growth
            if improvement > 0.01:  # Good improvement
                # Increase alpha to reinforce
                alpha_adjustment = self.adaptation_rate * 0.2
            elif improvement > 0:  # Slight improvement
                # Still increase slightly
                alpha_adjustment = self.adaptation_rate * 0.1
            elif improvement > -0.005:  # Very slight degradation (noise)
                # Keep stable, don't penalize
                alpha_adjustment = 0
            else:  # Significant degradation
                # Reduce moderately
                alpha_adjustment = -self.adaptation_rate * 0.1
            
            # Apply momentum smoothing
            new_alpha = self.alpha + alpha_adjustment
            self.alpha = self.momentum * self.alpha + (1 - self.momentum) * new_alpha
        
        # Clip to bounds
        self.alpha = np.clip(self.alpha, self.min_alpha, self.max_alpha)
        self.alpha_history.append(self.alpha)
        
        return self.alpha
    
    def update_boost_factor(
        self,
        weak_token_count: int,
        avg_attention: Optional[float] = None,
        total_tokens: int = 77
    ) -> float:
        """
        Adaptively adjust attention boost factor based on weak token prevalence
        
        Strategy:
        - More weak tokens → stronger boosting needed
        - Higher average attention → less boosting needed (already strong)
        - Fewer weak tokens → can reduce boosting (problem solving)
        
        Args:
            weak_token_count: Number of underrepresented tokens detected
            avg_attention: Average attention strength (optional)
            total_tokens: Total number of tokens in prompt
            
        Returns:
            Updated boost factor
        """
        self.weak_token_counts.append(weak_token_count)
        
        # Calculate weak token ratio
        weak_ratio = weak_token_count / total_tokens if total_tokens > 0 else 0
        
        # Base adjustment on weak token prevalence - be aggressive
        if weak_ratio > 0.2:  # Many weak tokens (>20%)
            # Maximum boosting
            target_boost = self.max_boost
        elif weak_ratio > 0.1:  # Moderate weak tokens (10-20%)
            # Strong boosting
            target_boost = self.max_boost * 0.8
        elif weak_ratio > 0.05:  # Few weak tokens (5-10%)
            # Moderate boosting
            target_boost = self.max_boost * 0.6
        else:  # Very few weak tokens (<5%)
            # Still maintain decent boost
            target_boost = self.max_boost * 0.5
        
        # Further adjust based on average attention if provided
        if avg_attention is not None:
            if avg_attention < 0.05:  # Very weak attention
                target_boost *= 1.2
            elif avg_attention > 0.15:  # Strong attention
                target_boost *= 0.8
        
        # Apply momentum smoothing
        self.boost_factor = (
            self.momentum * self.boost_factor +
            (1 - self.momentum) * target_boost
        )
        
        # Clip to bounds
        self.boost_factor = np.clip(self.boost_factor, self.min_boost, self.max_boost)
        self.boost_history.append(self.boost_factor)
        
        return self.boost_factor
    
    def get_step_dependent_weights(self, current_step: int, total_steps: int) -> Dict[str, float]:
        """
        Apply step-dependent weight scheduling
        
        Strategy:
        - Early steps: Stronger feedback (structure formation)
        - Middle steps: Moderate feedback (refinement)
        - Late steps: Weaker feedback (avoid over-correction)
        
        Args:
            current_step: Current denoising step
            total_steps: Total number of steps
            
        Returns:
            Dict with 'alpha_scale' and 'boost_scale' multipliers
        """
        progress = current_step / total_steps
        
        if progress < 0.3:  # Early phase (0-30%)
            # Strong feedback for structure formation
            alpha_scale = 1.3
            boost_scale = 1.5
        elif progress < 0.7:  # Middle phase (30-70%)
            # Sustained feedback for refinement
            alpha_scale = 1.2
            boost_scale = 1.3
        else:  # Late phase (70-100%)
            # Maintain feedback strength for quality
            alpha_scale = 1.0
            boost_scale = 1.0
        
        return {
            'alpha_scale': alpha_scale,
            'boost_scale': boost_scale,
            'scaled_alpha': self.alpha * alpha_scale,
            'scaled_boost': self.boost_factor * boost_scale
        }
    
    def get_statistics(self) -> Dict:
        """Get reweighting statistics for logging/analysis"""
        return {
            'current_alpha': self.alpha,
            'current_boost': self.boost_factor,
            'alpha_history': self.alpha_history.copy(),
            'boost_history': self.boost_history.copy(),
            'clip_score_history': self.clip_score_history.copy(),
            'weak_token_counts': self.weak_token_counts.copy(),
            'avg_alpha': np.mean(self.alpha_history) if self.alpha_history else self.alpha,
            'avg_boost': np.mean(self.boost_history) if self.boost_history else self.boost_factor
        }
    
    def reset(self):
        """Reset histories (call between generations)"""
        self.clip_score_history = []
        self.alpha_history = []
        self.boost_history = []
        self.weak_token_counts = []
