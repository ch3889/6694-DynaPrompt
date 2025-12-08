"""
Prompt Token Updater for DynaPrompt

This module updates prompt embeddings by boosting underrepresented tokens
based on CLIP feedback.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np


class PromptUpdater:
    """
    Updates prompt embeddings by re-weighting tokens based on CLIP feedback.
    """

    def __init__(self, boost_factor: float = 1.5, max_boost: float = 3.0):
        """
        Initialize prompt updater.

        Args:
            boost_factor: Multiplicative factor to boost underrepresented tokens
            max_boost: Maximum cumulative boost allowed per token
        """
        self.boost_factor = boost_factor
        self.max_boost = max_boost
        self.token_boosts = {}  # Track cumulative boosts per token

    def reset(self):
        """Reset token boost tracking."""
        self.token_boosts = {}

    def update_embeddings(
        self,
        embeddings: torch.Tensor,
        tokenized_prompt: torch.Tensor,
        underrepresented: List[Tuple[str, float]],
        tokenizer,
        text_encoder
    ) -> torch.Tensor:
        """
        Update prompt embeddings by boosting underrepresented tokens.

        Args:
            embeddings: Current prompt embeddings [1, seq_len, embed_dim]
            tokenized_prompt: Tokenized prompt tensor [1, seq_len]
            underrepresented: List of (token_text, score) for underrepresented tokens
            tokenizer: CLIP tokenizer
            text_encoder: CLIP text encoder

        Returns:
            Updated embeddings tensor
        """
        if len(underrepresented) == 0:
            return embeddings

        # Clone embeddings to avoid in-place modification
        updated_embeddings = embeddings.clone()

        # Get token ids for underrepresented tokens
        underrep_tokens = [tok for tok, _ in underrepresented]

        print(f"\n   Boosting {len(underrep_tokens)} underrepresented tokens:")

        for token_text, score in underrepresented:
            # Tokenize the individual token to find its ID
            token_encoded = tokenizer(
                token_text,
                padding=False,
                truncation=False,
                return_tensors="pt"
            )

            # Get the actual token ID (skip start token)
            if token_encoded['input_ids'].shape[1] > 1:
                token_id = token_encoded['input_ids'][0, 1].item()
            else:
                continue

            # Find positions of this token in the prompt
            positions = (tokenized_prompt[0] == token_id).nonzero(as_tuple=True)[0]

            if len(positions) > 0:
                # Calculate boost for this token
                current_boost = self.token_boosts.get(token_text, 1.0)
                new_boost = min(current_boost * self.boost_factor, self.max_boost)
                self.token_boosts[token_text] = new_boost

                # Apply boost to embeddings at these positions
                for pos in positions:
                    updated_embeddings[0, pos] *= new_boost

                print(f"     • {token_text:15s} (score: {score:.3f}) -> boost: {new_boost:.2f}x at positions {positions.tolist()}")

        return updated_embeddings

    def adaptive_boost(
        self,
        embeddings: torch.Tensor,
        tokenized_prompt: torch.Tensor,
        token_scores: Dict[str, float],
        underrepresented: List[Tuple[str, float]],
        tokenizer,
        text_encoder
    ) -> torch.Tensor:
        """
        Adaptive boosting based on how underrepresented each token is.
        Tokens with lower scores get stronger boosts.

        Args:
            embeddings: Current prompt embeddings [1, seq_len, embed_dim]
            tokenized_prompt: Tokenized prompt tensor [1, seq_len]
            token_scores: All token CLIP scores
            underrepresented: List of (token_text, score) for underrepresented tokens
            tokenizer: CLIP tokenizer
            text_encoder: CLIP text encoder

        Returns:
            Updated embeddings tensor
        """
        if len(underrepresented) == 0:
            return embeddings

        # Clone embeddings
        updated_embeddings = embeddings.clone()

        # Get score range for normalization
        all_scores = list(token_scores.values())
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        print(f"\n   Adaptive boosting for {len(underrepresented)} tokens:")

        for token_text, score in underrepresented:
            # Calculate adaptive boost based on how far below max score
            # Lower scores get higher boosts
            normalized_gap = (max_score - score) / score_range
            adaptive_factor = 1.0 + (normalized_gap * (self.boost_factor - 1.0))

            # Apply max boost limit
            current_boost = self.token_boosts.get(token_text, 1.0)
            new_boost = min(current_boost * adaptive_factor, self.max_boost)
            self.token_boosts[token_text] = new_boost

            # Tokenize to find token ID
            token_encoded = tokenizer(
                token_text,
                padding=False,
                truncation=False,
                return_tensors="pt"
            )

            if token_encoded['input_ids'].shape[1] > 1:
                token_id = token_encoded['input_ids'][0, 1].item()
            else:
                continue

            # Find positions in prompt
            positions = (tokenized_prompt[0] == token_id).nonzero(as_tuple=True)[0]

            if len(positions) > 0:
                for pos in positions:
                    updated_embeddings[0, pos] *= new_boost

                print(f"     • {token_text:15s} (score: {score:.3f}, gap: {normalized_gap:.3f}) -> {new_boost:.2f}x")

        return updated_embeddings


def test_prompt_updater():
    """Test the PromptUpdater module independently."""
    print("="*80)
    print("Testing PromptUpdater Module")
    print("="*80)

    # Initialize
    print("\n1. Initializing PromptUpdater...")
    updater = PromptUpdater(boost_factor=1.5, max_boost=3.0)
    print(f"   ✓ Boost factor: {updater.boost_factor}")
    print(f"   ✓ Max boost: {updater.max_boost}")

    # Create dummy embeddings
    print("\n2. Creating dummy embeddings...")
    seq_len = 8
    embed_dim = 768
    dummy_embeddings = torch.randn(1, seq_len, embed_dim)
    print(f"   Embedding shape: {dummy_embeddings.shape}")
    print(f"   Initial embedding norm: {dummy_embeddings.norm():.4f}")

    # Simulate underrepresented tokens
    print("\n3. Simulating underrepresented tokens...")
    underrepresented = [
        ("cat", 0.22),
        ("chair", 0.21)
    ]
    print(f"   Underrepresented: {underrepresented}")

    # Test boost tracking
    print("\n4. Testing boost accumulation over multiple iterations...")
    for iteration in range(3):
        print(f"\n   Iteration {iteration + 1}:")
        # In real usage, we would update embeddings here
        # For testing, just track the boost values
        for token, score in underrepresented:
            current_boost = updater.token_boosts.get(token, 1.0)
            new_boost = min(current_boost * updater.boost_factor, updater.max_boost)
            updater.token_boosts[token] = new_boost
            print(f"     {token:15s}: {current_boost:.2f}x -> {new_boost:.2f}x")

    print("\n5. Testing reset...")
    updater.reset()
    print(f"   Token boosts after reset: {updater.token_boosts}")

    print("\n" + "="*80)
    print("✓ PromptUpdater module test complete!")
    print("="*80)


if __name__ == "__main__":
    test_prompt_updater()
