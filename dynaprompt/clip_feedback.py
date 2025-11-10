"""
CLIP Feedback Module for DynaPrompt

This module provides CLIP-based feedback during diffusion sampling to identify
underrepresented concepts in the generated image.
"""

import torch
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple


class CLIPFeedback:
    """
    CLIP feedback module that evaluates intermediate images during generation
    and identifies underrepresented tokens.
    """

    def __init__(self, device="cuda", model_name="ViT-B/32"):
        """
        Initialize CLIP feedback module.

        Args:
            device: Device to run CLIP on (cuda/cpu)
            model_name: CLIP model variant to use
        """
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def decode_latents(self, latents, vae):
        """
        Decode latent representation to pixel space using VAE decoder.

        Args:
            latents: Latent tensor [1, 4, H, W]
            vae: VAE model from Stable Diffusion

        Returns:
            PIL Image
        """
        # Scale latents back to proper range
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            # Decode to pixel space
            image = vae.decode(latents)

        # Convert to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        return pil_image

    def compute_token_similarities(
        self,
        image: Image.Image,
        tokens: List[str]
    ) -> Dict[str, float]:
        """
        Compute CLIP similarity between image and each individual token.

        Args:
            image: PIL Image to evaluate
            tokens: List of text tokens from the prompt

        Returns:
            Dictionary mapping token -> similarity score
        """
        # Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Get image features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)

        # Compute similarity for each token
        token_scores = {}
        for token in tokens:
            # Skip special tokens
            if token in ['<|startoftext|>', '<|endoftext|>', '']:
                continue

            # Encode token as text
            text_input = clip.tokenize([token]).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                text_features = F.normalize(text_features, dim=-1)

                # Compute cosine similarity
                similarity = (image_features @ text_features.T).item()
                token_scores[token] = similarity

        return token_scores

    def identify_underrepresented(
        self,
        token_scores: Dict[str, float],
        threshold: float = None,
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Identify underrepresented tokens based on CLIP scores.

        Args:
            token_scores: Dictionary of token -> similarity scores
            threshold: Minimum score threshold (tokens below this are underrepresented)
            top_k: Return the k most underrepresented tokens

        Returns:
            List of (token, score) tuples sorted by increasing score
        """
        # Sort tokens by score (ascending - lowest scores first)
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1])

        if threshold is not None:
            # Filter by threshold
            underrepresented = [(tok, score) for tok, score in sorted_tokens if score < threshold]
        elif top_k is not None:
            # Take top-k worst scoring tokens
            underrepresented = sorted_tokens[:top_k]
        else:
            # Default: bottom 30% of tokens
            k = max(1, len(sorted_tokens) // 3)
            underrepresented = sorted_tokens[:k]

        return underrepresented

    def get_feedback(
        self,
        latents: torch.Tensor,
        vae,
        tokens: List[str],
        threshold: float = None,
        top_k: int = None
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """
        Complete feedback pipeline: decode latents, compute scores, identify underrepresented.

        Args:
            latents: Latent tensor to evaluate
            vae: VAE decoder
            tokens: List of prompt tokens
            threshold: Score threshold for underrepresentation
            top_k: Number of underrepresented tokens to identify

        Returns:
            Tuple of (all_token_scores, underrepresented_tokens)
        """
        # Decode latents to image
        image = self.decode_latents(latents, vae)

        # Compute CLIP scores for all tokens
        token_scores = self.compute_token_similarities(image, tokens)

        # Identify underrepresented tokens
        underrepresented = self.identify_underrepresented(
            token_scores,
            threshold=threshold,
            top_k=top_k
        )

        return token_scores, underrepresented


def test_clip_feedback():
    """Test the CLIPFeedback module independently."""
    print("="*80)
    print("Testing CLIPFeedback Module")
    print("="*80)

    # Initialize
    print("\n1. Initializing CLIPFeedback...")
    feedback = CLIPFeedback(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ CLIP model loaded on {feedback.device}")

    # Test with a sample image
    print("\n2. Testing with sample prompt...")
    test_prompt = "a blue cat sitting on a red chair"
    tokens = test_prompt.split()

    # Create a dummy image (for testing)
    dummy_image = Image.new('RGB', (512, 512), color=(100, 150, 200))
    print(f"   Prompt: '{test_prompt}'")
    print(f"   Tokens: {tokens}")

    # Compute token similarities
    print("\n3. Computing CLIP similarities for each token...")
    token_scores = feedback.compute_token_similarities(dummy_image, tokens)

    print("   Token Scores:")
    for token, score in sorted(token_scores.items(), key=lambda x: -x[1]):
        print(f"     {token:15s}: {score:.4f}")

    # Identify underrepresented
    print("\n4. Identifying underrepresented tokens (bottom 30%)...")
    underrepresented = feedback.identify_underrepresented(token_scores)

    print("   Underrepresented tokens:")
    for token, score in underrepresented:
        print(f"     ⚠  {token:15s}: {score:.4f}")

    print("\n" + "="*80)
    print("✓ CLIPFeedback module test complete!")
    print("="*80)


if __name__ == "__main__":
    test_clip_feedback()
