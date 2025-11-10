"""
Attention Map Modification for DynaPrompt

This module implements attention re-weighting by hooking into the U-Net's
cross-attention layers and amplifying attention to underrepresented tokens.

Based on the Attend-and-Excite approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class AttentionStore:
    """
    Stores attention maps from cross-attention layers during forward pass.
    """

    def __init__(self):
        self.step_store = {}
        self.attention_store = {}
        self.curr_step = 0

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        """Called by attention hooks to store attention maps."""
        if is_cross:
            key = f"{place_in_unet}_cross"
            if key not in self.step_store:
                self.step_store[key] = []
            self.step_store[key].append(attn.cpu().detach())

    def step_callback(self):
        """Called at the end of each diffusion step."""
        self.attention_store[self.curr_step] = self.step_store
        self.step_store = {}
        self.curr_step += 1

    def get_average_attention(self):
        """Get average attention across all cross-attention layers."""
        attention_maps = []
        for step_maps in self.attention_store.values():
            for location_maps in step_maps.values():
                attention_maps.extend(location_maps)

        if len(attention_maps) == 0:
            return None

        # Different layers have different spatial resolutions
        # We need to average over tokens only, accounting for different resolutions
        # Shape: (batch*heads, pixels, tokens) but pixels varies
        # Average across spatial dimension first to get (batch*heads, tokens)
        token_attentions = []
        for attn_map in attention_maps:
            # Average over spatial dim: (batch*heads, pixels, tokens) -> (batch*heads, tokens)
            token_attn = attn_map.mean(dim=1)
            token_attentions.append(token_attn)

        # Now stack and average: (num_maps, batch*heads, tokens) -> (batch*heads, tokens)
        avg_attention = torch.stack(token_attentions).mean(0)
        return avg_attention

    def reset(self):
        """Reset stored attention maps."""
        self.step_store = {}
        self.attention_store = {}
        self.curr_step = 0


class AttentionModifier:
    """
    Modifies attention maps during generation to boost underrepresented tokens.
    """

    def __init__(
        self,
        tokenizer,
        boost_factor: float = 1.3,
        threshold: float = 0.3,
        start_step: int = 15,
        end_step: int = 35
    ):
        """
        Initialize attention modifier.

        Args:
            tokenizer: CLIP tokenizer
            boost_factor: How much to amplify attention (1.3 = 30% increase)
            threshold: Attention threshold below which tokens are considered underrepresented
            start_step: First step to apply modification
            end_step: Last step to apply modification
        """
        self.tokenizer = tokenizer
        self.boost_factor = boost_factor
        self.threshold = threshold
        self.start_step = start_step
        self.end_step = end_step

        self.attention_store = AttentionStore()
        self.underrepresented_indices = []
        self.hooks = []
        self.enabled = True

    def register_hooks(self, unet):
        """
        Register forward hooks on all cross-attention layers in the U-Net.

        Args:
            unet: The U-Net model
        """
        # Find all CrossAttention modules
        def register_recr(net, count, place_in_unet):
            if net.__class__.__name__ == 'CrossAttention':
                hook = net.register_forward_hook(
                    self.make_attention_hook(place_in_unet, count)
                )
                self.hooks.append(hook)
                return count + 1
            elif hasattr(net, 'children'):
                for child in net.children():
                    count = register_recr(child, count, place_in_unet)
            return count

        # Register hooks in different parts of U-Net
        count = 0
        for name in ['down', 'mid', 'up']:
            if hasattr(unet, name + '_blocks') or hasattr(unet, name):
                block = getattr(unet, name + '_blocks', None) or getattr(unet, name, None)
                if block is not None:
                    count = register_recr(block, count, name)

        print(f"Registered {len(self.hooks)} attention hooks")

    def make_attention_hook(self, place_in_unet: str, count: int):
        """
        Create a hook function for a specific attention layer.

        Args:
            place_in_unet: Location in U-Net (down/mid/up)
            count: Layer index

        Returns:
            Hook function
        """
        def hook(module, input, output):
            if not self.enabled:
                return

            # The forward pass computes attention internally
            # We need to intercept it by modifying the module temporarily
            # For now, we'll store it after computation
            # In the next iteration, we'll modify the forward pass
            pass

        return hook

    def modify_attention_forward(self, original_forward):
        """
        Wrap the CrossAttention forward method to modify attention maps.

        Args:
            original_forward: Original forward method of CrossAttention

        Returns:
            Modified forward method
        """
        def modified_forward(x, context=None, mask=None):
            # Get the module (self in the original method)
            # Handle both bound methods (have __self__) and wrapped functions (don't have __self__)
            if hasattr(original_forward, '__self__'):
                module = original_forward.__self__
            else:
                # If it's a wrapped function from Phase 1, we can't use it
                # This shouldn't happen if unpatch worked correctly
                raise RuntimeError("Cannot patch already-wrapped forward method. Unpatch failed?")

            h = module.heads

            q = module.to_q(x)
            from ldm.modules.attention import default
            context = default(context, x)
            k = module.to_k(context)
            v = module.to_v(context)

            from einops import rearrange
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            # Use torch.einsum instead of einops.einsum
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * module.scale

            if mask is not None:
                from ldm.modules.attention import exists, repeat
                if exists(mask):
                    mask = rearrange(mask, 'b ... -> b (...)')
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = repeat(mask, 'b j -> (b h) () j', h=h)
                    sim.masked_fill_(~mask, max_neg_value)

            # Compute attention
            attn = sim.softmax(dim=-1)

            # Store attention for analysis
            is_cross = context is not None and context.shape != x.shape
            if is_cross:
                self.attention_store(attn, is_cross=True, place_in_unet="cross")

            # MODIFY ATTENTION HERE if we have underrepresented tokens
            if self.enabled and len(self.underrepresented_indices) > 0 and is_cross:
                attn = self.boost_attention(attn)

            out = torch.einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return module.to_out(out)

        return modified_forward

    def boost_attention(self, attn):
        """
        Boost attention weights for underrepresented token indices with adaptive boosting.

        Tokens with lower attention get boosted MORE aggressively.

        Args:
            attn: Attention weights [batch*heads, pixels, tokens]

        Returns:
            Modified attention weights
        """
        # Clone to avoid in-place modification
        modified_attn = attn.clone()

        # For each underrepresented token index, boost its attention adaptively
        for token_idx in self.underrepresented_indices:
            if token_idx < modified_attn.shape[-1]:
                # Calculate current attention for this token (average across spatial dims)
                current_attn = modified_attn[:, :, token_idx].mean().item()

                # Adaptive boost: Lower attention → Higher boost factor
                # If attention is 0.001, boost by 10x
                # If attention is 0.01, boost by 5x
                # If attention is 0.02, boost by 3x (base factor)
                if current_attn < 0.001:
                    adaptive_factor = self.boost_factor * 3.0  # Very aggressive
                elif current_attn < 0.005:
                    adaptive_factor = self.boost_factor * 2.0  # Aggressive
                elif current_attn < 0.01:
                    adaptive_factor = self.boost_factor * 1.5  # Moderate
                else:
                    adaptive_factor = self.boost_factor  # Base

                # Cap maximum boost to prevent numerical instability
                adaptive_factor = min(adaptive_factor, 15.0)

                # Increase attention to this token
                modified_attn[:, :, token_idx] *= adaptive_factor

        # Re-normalize so attention sums to 1
        modified_attn = modified_attn / modified_attn.sum(dim=-1, keepdim=True)

        return modified_attn

    def patch_attention_layers(self, unet):
        """
        Patch all CrossAttention forward methods to enable modification.

        Args:
            unet: The U-Net model
        """
        def patch_recr(net):
            if net.__class__.__name__ == 'CrossAttention':
                # Replace forward method with our modified version
                net.forward = self.modify_attention_forward(net.forward)
            elif hasattr(net, 'children'):
                for child in net.children():
                    patch_recr(child)

        patch_recr(unet)
        print("Patched CrossAttention layers for attention modification")

    def identify_underrepresented_tokens(self, prompt: str, attention_threshold: Optional[float] = None):
        """
        Identify underrepresented tokens based on average attention.

        Args:
            prompt: The text prompt
            attention_threshold: Threshold below which tokens are underrepresented

        Returns:
            List of underrepresented token indices
        """
        if attention_threshold is None:
            attention_threshold = self.threshold

        # Get average attention across all layers and steps
        avg_attention = self.attention_store.get_average_attention()

        if avg_attention is None:
            return []

        # avg_attention is now [batch*heads, tokens] after spatial averaging
        # Average across batch*heads: [batch*heads, tokens] -> [tokens]
        token_attention = avg_attention.mean(dim=0)

        # Tokenize prompt to get token count
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Get actual tokens (excluding padding)
        actual_tokens = (text_input['input_ids'][0] != self.tokenizer.pad_token_id).sum().item()

        # Find tokens with low attention (excluding start/end tokens)
        underrepresented = []
        for i in range(1, min(actual_tokens - 1, len(token_attention))):
            if token_attention[i] < attention_threshold:
                underrepresented.append(i)

        return underrepresented

    def set_underrepresented_indices(self, indices: List[int]):
        """Set which token indices to boost."""
        self.underrepresented_indices = indices

    def should_modify(self, step: int) -> bool:
        """Check if we should modify attention at this step."""
        return self.start_step <= step <= self.end_step

    def enable(self):
        """Enable attention modification."""
        self.enabled = True

    def disable(self):
        """Disable attention modification."""
        self.enabled = False

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset(self):
        """Reset attention store and underrepresented indices."""
        self.attention_store.reset()
        self.underrepresented_indices = []


def test_attention_modifier():
    """Test the AttentionModifier independently."""
    print("="*80)
    print("Testing AttentionModifier")
    print("="*80)

    # Mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.model_max_length = 77
            self.pad_token_id = 0

        def __call__(self, text, **kwargs):
            # Return mock tokenized output
            tokens = torch.randint(1, 100, (1, 77))
            tokens[0, 0] = 49406  # Start token
            tokens[0, 10:] = 0  # Padding
            return {'input_ids': tokens}

    tokenizer = MockTokenizer()

    print("\n1. Initializing AttentionModifier...")
    modifier = AttentionModifier(
        tokenizer=tokenizer,
        boost_factor=1.3,
        threshold=0.3,
        start_step=15,
        end_step=35
    )
    print(f"   ✓ Boost factor: {modifier.boost_factor}")
    print(f"   ✓ Threshold: {modifier.threshold}")
    print(f"   ✓ Active steps: {modifier.start_step}-{modifier.end_step}")

    print("\n2. Testing attention boosting...")
    # Create mock attention weights
    batch_heads = 8
    pixels = 64
    tokens = 77
    attn = torch.rand(batch_heads, pixels, tokens)
    attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize

    print(f"   Original attention shape: {attn.shape}")
    print(f"   Original attention sum: {attn.sum(dim=-1)[0, 0]:.4f}")

    # Set some tokens as underrepresented
    modifier.set_underrepresented_indices([5, 10, 15])

    # Boost attention
    boosted_attn = modifier.boost_attention(attn)

    print(f"   Boosted attention shape: {boosted_attn.shape}")
    print(f"   Boosted attention sum: {boosted_attn.sum(dim=-1)[0, 0]:.4f}")
    print(f"   Attention to token 5: {attn[0, 0, 5]:.4f} -> {boosted_attn[0, 0, 5]:.4f}")
    print(f"   Attention to token 10: {attn[0, 0, 10]:.4f} -> {boosted_attn[0, 0, 10]:.4f}")

    print("\n3. Testing attention store...")
    modifier.attention_store.reset()

    # Simulate storing attention across multiple layers
    for i in range(3):
        mock_attn = torch.rand(8, 64, 77)
        modifier.attention_store(mock_attn, is_cross=True, place_in_unet=f"layer{i}")

    modifier.attention_store.step_callback()

    avg_attn = modifier.attention_store.get_average_attention()
    print(f"   Average attention shape: {avg_attn.shape if avg_attn is not None else 'None'}")

    print("\n" + "="*80)
    print("✓ AttentionModifier test complete!")
    print("="*80)


if __name__ == "__main__":
    test_attention_modifier()
