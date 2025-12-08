"""
DynaPrompt V6: Hybrid Early Detection + Attention Boosting

Strategy: Combine the best of V3 and V5:
1. Early detection (V5): Check if objects are present at step 15
2. Adaptive restart: Try different seeds until we find a good one
3. Attention boosting fallback (V3): If no good seed found, use best attempt + boost

This is the most robust approach - we try to find a naturally good seed,
but if that fails, we apply attention boosting to help the generation.
"""

import torch
import sys
import os
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional

# Add the stable diffusion path
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt/models/stable_diffusion_compvis')

from dynaprompt.attention_modifier import AttentionModifier, AttentionStore


class DynaPromptV6Sampler:
    """
    DynaPrompt V6: Hybrid early detection with attention boosting fallback.
    """

    def __init__(
        self,
        ddim_sampler,
        model,
        tokenizer,
        device="cuda",
        check_step=15,
        attention_threshold=0.05,
        max_retries=5,
        boost_factor=2.5,
        start_step_ratio=0.0,
        end_step_ratio=0.4
    ):
        """
        Initialize DynaPrompt V6 sampler.

        Args:
            ddim_sampler: Base DDIM sampler
            model: Stable Diffusion model
            tokenizer: CLIP tokenizer
            device: Device to run on
            check_step: Step at which to check composition (default: 15)
            attention_threshold: Minimum attention for object presence (default: 0.05)
            max_retries: Maximum restart attempts (default: 5)
            boost_factor: Attention boost factor for fallback (default: 2.5)
            start_step_ratio: Start boosting at this ratio (default: 0.0)
            end_step_ratio: End boosting at this ratio (default: 0.4)
        """
        self.ddim_sampler = ddim_sampler
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.check_step = check_step
        self.attention_threshold = attention_threshold
        self.max_retries = max_retries
        self.boost_factor = boost_factor
        self.start_step_ratio = start_step_ratio
        self.end_step_ratio = end_step_ratio

        # Store original forward methods for unpatching
        self.original_forwards = {}

    def sample_with_dynaprompt(
        self,
        prompt: str,
        shape,
        steps=50,
        unconditional_guidance_scale=7.5,
        critical_tokens: List[str] = None,
        verbose=True,
        **kwargs
    ):
        """
        Sample with hybrid early detection + attention boosting.

        Args:
            prompt: Text prompt
            shape: Latent shape
            steps: Number of DDIM steps
            unconditional_guidance_scale: CFG scale
            critical_tokens: List of critical words (auto-detected if None)
            verbose: Print feedback info
            **kwargs: Additional arguments

        Returns:
            Generated latent and intermediates
        """
        print(f"\nDynaPrompt V6 Sampling (Hybrid: Detection + Boosting)")
        print(f"Prompt: {prompt}")
        print(f"Strategy: Try {self.max_retries + 1} seeds, fallback to attention boosting")
        print(f"Check step: {self.check_step}")
        print(f"Attention threshold: {self.attention_threshold}")
        print(f"Boost factor (fallback): {self.boost_factor}x")
        print("="*80)

        # Auto-detect critical tokens if not provided
        if critical_tokens is None:
            critical_tokens = self._extract_critical_tokens(prompt)
            if verbose:
                print(f"Auto-detected critical tokens: {critical_tokens}")

        # Get token indices
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        critical_indices = self._get_token_indices(text_input.input_ids[0], critical_tokens)

        if verbose:
            print(f"Critical token indices: {critical_indices}")
            print("="*80 + "\n")

        # Encode prompt
        with torch.no_grad():
            text_embeddings = self.model.cond_stage_model.transformer(text_input.input_ids)[0]
            unconditional_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.device)
            unconditional_embeddings = self.model.cond_stage_model.transformer(unconditional_input.input_ids)[0]

        # Phase 1: Try finding a good seed
        best_attempt = None
        best_attention_scores = None
        best_seed = None

        # IMPORTANT: Save truly original forward methods BEFORE any patching
        # Clear any previous saves to handle multiple calls on same instance
        self.original_forwards = {}
        self._save_original_forwards(self.model.model.diffusion_model)

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                # Use completely different random seed for retry
                retry_seed = torch.randint(0, 1000000, (1,)).item()
                torch.manual_seed(retry_seed)
                torch.cuda.manual_seed(retry_seed)

                print(f"\n{'='*80}")
                print(f"RETRY #{attempt}: Trying random seed {retry_seed}")
                print(f"{'='*80}\n")

            # Sample with early detection
            samples, intermediates, success, attention_scores, seed_used = self._sample_with_early_detection(
                text_embeddings=text_embeddings,
                unconditional_embeddings=unconditional_embeddings,
                critical_indices=critical_indices,
                critical_tokens=critical_tokens,
                shape=shape,
                steps=steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                attempt=attempt,
                verbose=verbose
            )

            # Track best attempt
            if best_attempt is None or (attention_scores and sum(attention_scores.values()) > sum(best_attention_scores.values())):
                best_attempt = (samples, intermediates)
                best_attention_scores = attention_scores
                best_seed = seed_used

            if success:
                print(f"\n{'='*80}")
                print(f"✓ Found good seed {seed_used}! All objects detected at step {self.check_step}")
                print(f"{'='*80}\n")
                return samples, intermediates

        # Phase 2: No good seed found, use attention boosting on best attempt
        print(f"\n{'='*80}")
        print(f"⚠ No seed with sufficient attention found after {self.max_retries + 1} attempts")
        print(f"Switching to PHASE 2: Attention boosting on best seed {best_seed}")
        print(f"Best attention scores: {best_attention_scores}")
        print(f"{'='*80}\n")

        # Restore original forward methods from Phase 1 patches
        print("Restoring original attention layers...")
        self._unpatch_attention_layers(self.model.model.diffusion_model)

        # Set seed to best one
        torch.manual_seed(best_seed)
        torch.cuda.manual_seed(best_seed)

        # Sample with attention boosting
        samples, intermediates = self._sample_with_attention_boosting(
            text_embeddings=text_embeddings,
            unconditional_embeddings=unconditional_embeddings,
            text_input_ids=text_input.input_ids,
            shape=shape,
            steps=steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            prompt=prompt,
            verbose=verbose
        )

        print(f"\n{'='*80}")
        print(f"✓ DynaPrompt V6 complete (used attention boosting fallback)")
        print(f"{'='*80}\n")

        # Clean up Phase 2 patches for next use
        print("Cleaning up Phase 2 patches...")
        self._unpatch_attention_layers(self.model.model.diffusion_model)
        self.original_forwards = {}  # Clear references

        return samples, intermediates

    def _extract_critical_tokens(self, prompt: str) -> List[str]:
        """Extract critical nouns from prompt."""
        ignore = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'next', 'parked'}
        words = prompt.lower().split()
        critical = []
        for word in words:
            word = word.strip('.,!?;:')
            if word and word not in ignore and len(word) > 2:
                critical.append(word)
        return critical

    def _get_token_indices(self, token_ids: torch.Tensor, critical_words: List[str]) -> List[int]:
        """Get token indices for critical words."""
        indices = []
        for word in critical_words:
            word_tokens = self.tokenizer(word, add_special_tokens=False)['input_ids']
            for i, token_id in enumerate(token_ids):
                if token_id in word_tokens:
                    indices.append(i)
        return list(set(indices))

    def _sample_with_early_detection(
        self,
        text_embeddings,
        unconditional_embeddings,
        critical_indices: List[int],
        critical_tokens: List[str],
        shape,
        steps,
        unconditional_guidance_scale,
        attempt=0,
        verbose=True
    ) -> Tuple[torch.Tensor, dict, bool, dict, int]:
        """
        Sample with early detection.

        Returns:
            (samples, intermediates, success, attention_scores, seed_used)
        """
        # Remember the seed
        seed_used = torch.initial_seed()

        # Prepare sampling
        self.ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)

        device = self.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        # Setup attention tracking with simple store
        attention_store = AttentionStore()
        self._patch_attention_layers(self.model.model.diffusion_model, attention_store)

        timesteps = self.ddim_sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        iterator = tqdm(time_range, desc=f'V6 Phase 1 (Attempt {attempt + 1})', total=total_steps)

        composition_checked = False
        composition_success = False
        attention_scores = {}

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Perform DDIM step
            with torch.no_grad():
                outs = self.ddim_sampler.p_sample_ddim(
                    img, text_embeddings, ts,
                    index=index,
                    use_original_steps=False,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_embeddings
                )
                img, pred_x0 = outs

            # Check composition at specified step
            if i == self.check_step and not composition_checked:
                composition_checked = True

                avg_attention = attention_store.get_average_attention()
                if avg_attention is not None:
                    token_attention = avg_attention.mean(dim=0)

                    missing_tokens = []
                    for idx, token_name in zip(critical_indices, critical_tokens):
                        if idx < len(token_attention):
                            attn_score = token_attention[idx].item()
                            attention_scores[token_name] = attn_score
                            if attn_score < self.attention_threshold:
                                missing_tokens.append((token_name, idx, attn_score))

                    if len(missing_tokens) > 0:
                        if verbose:
                            print(f"\n   ⚠ Step {i}: {len(missing_tokens)} underrepresented (threshold {self.attention_threshold}):")
                            for token_name, idx, score in missing_tokens[:3]:
                                print(f"     • '{token_name}': {score:.4f}")
                        composition_success = False

                        if attempt < self.max_retries:
                            # Early abort to save compute
                            return None, None, False, attention_scores, seed_used
                    else:
                        if verbose:
                            print(f"\n   ✓ Step {i}: All objects detected!")
                            for token_name, score in list(attention_scores.items())[:3]:
                                print(f"     • '{token_name}': {score:.4f}")
                            print()
                        composition_success = True

            # Store intermediates
            if index % 10 == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

            attention_store.step_callback()

        return img, intermediates, composition_success, attention_scores, seed_used

    def _save_original_forwards(self, unet):
        """Save truly original forward methods before any patching."""
        module_counter = [0]

        def save_recr(net):
            if net.__class__.__name__ == 'CrossAttention':
                module_id = module_counter[0]
                # Only save if it's a bound method (not already patched)
                if hasattr(net.forward, '__self__'):
                    self.original_forwards[module_id] = net.forward
                module_counter[0] += 1
            elif hasattr(net, 'children'):
                for child in net.children():
                    save_recr(child)

        save_recr(unet)
        print(f"Saved {len(self.original_forwards)} original CrossAttention forward methods")

    def _sample_with_attention_boosting(
        self,
        text_embeddings,
        unconditional_embeddings,
        text_input_ids,
        shape,
        steps,
        unconditional_guidance_scale,
        prompt,
        verbose=True
    ):
        """
        Sample with V3-style attention boosting.
        """
        print(f"Using attention boosting (V3 strategy)")
        print(f"Boost factor: {self.boost_factor}x")
        print(f"Active steps: {int(steps * self.start_step_ratio)}-{int(steps * self.end_step_ratio)}")
        print("="*80)

        start_step = int(steps * self.start_step_ratio)
        end_step = int(steps * self.end_step_ratio)

        # Initialize attention modifier
        attention_modifier = AttentionModifier(
            tokenizer=self.tokenizer,
            boost_factor=self.boost_factor,
            threshold=self.attention_threshold,
            start_step=start_step,
            end_step=end_step
        )

        # Patch attention layers
        attention_modifier.patch_attention_layers(self.model.model.diffusion_model)

        # Prepare sampling
        self.ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)

        device = self.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        timesteps = self.ddim_sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        iterator = tqdm(time_range, desc='V6 Phase 2 (Boosting)', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Enable/disable attention modification
            if attention_modifier.should_modify(i):
                attention_modifier.enable()

                # Identify underrepresented tokens every 3 steps
                if i % 3 == 0:
                    underrep = attention_modifier.identify_underrepresented_tokens(
                        prompt=prompt,
                        attention_threshold=self.attention_threshold
                    )
                    if len(underrep) > 0:
                        attention_modifier.set_underrepresented_indices(underrep)
                        if verbose and i == start_step:
                            print(f"\n   [Step {i}] Boosting {len(underrep)} tokens")
            else:
                attention_modifier.disable()

            # Perform DDIM step
            with torch.no_grad():
                outs = self.ddim_sampler.p_sample_ddim(
                    img, text_embeddings, ts,
                    index=index,
                    use_original_steps=False,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_embeddings
                )
                img, pred_x0 = outs

            # Store intermediates
            if index % 10 == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

            attention_modifier.attention_store.step_callback()

        return img, intermediates

    def _patch_attention_layers(self, unet, attention_store: AttentionStore):
        """Patch CrossAttention layers to capture attention."""

        def modify_forward(module, module_id):
            # Note: original forwards should already be saved by _save_original_forwards
            # We don't save here because module.forward might already be patched

            def new_forward(x, context=None, mask=None):
                h = module.heads
                q = module.to_q(x)

                from ldm.modules.attention import default
                context = default(context, x)
                k = module.to_k(context)
                v = module.to_v(context)

                from einops import rearrange
                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

                sim = torch.einsum('b i d, b j d -> b i j', q, k) * module.scale

                if mask is not None:
                    from ldm.modules.attention import exists, repeat
                    if exists(mask):
                        mask = rearrange(mask, 'b ... -> b (...)')
                        max_neg_value = -torch.finfo(sim.dtype).max
                        mask = repeat(mask, 'b j -> (b h) () j', h=h)
                        sim.masked_fill_(~mask, max_neg_value)

                attn = sim.softmax(dim=-1)

                # Store attention if cross-attention
                is_cross = context is not None and context.shape != x.shape
                if is_cross:
                    attention_store(attn.cpu().detach(), is_cross=True, place_in_unet="cross")

                out = torch.einsum('b i j, b j d -> b i d', attn, v)
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                return module.to_out(out)

            module.forward = new_forward

        module_counter = [0]  # Use list to allow modification in nested function

        def patch_recr(net):
            if net.__class__.__name__ == 'CrossAttention':
                module_id = module_counter[0]
                modify_forward(net, module_id)
                module_counter[0] += 1
            elif hasattr(net, 'children'):
                for child in net.children():
                    patch_recr(child)

        patch_recr(unet)

    def _unpatch_attention_layers(self, unet):
        """Restore original forward methods to CrossAttention layers."""

        module_counter = [0]
        restored_count = [0]

        def unpatch_recr(net):
            if net.__class__.__name__ == 'CrossAttention':
                module_id = module_counter[0]
                if module_id in self.original_forwards:
                    net.forward = self.original_forwards[module_id]
                    restored_count[0] += 1
                module_counter[0] += 1
            elif hasattr(net, 'children'):
                for child in net.children():
                    unpatch_recr(child)

        unpatch_recr(unet)
        print(f"Restored {restored_count[0]} CrossAttention forward methods")
