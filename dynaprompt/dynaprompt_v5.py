"""
DynaPrompt V5: Early Detection with Adaptive Restart

Strategy: Instead of trying to fix broken trajectories, detect compositional
failures early (step 10-15) and restart with noise perturbation.

Key insight: If attention to critical tokens is too low at step 15, the final
image will be missing those objects. Rather than try to steer the trajectory,
restart with a modified latent that has better initial conditions.
"""

import torch
import sys
import os
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

# Add the stable diffusion path
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt/models/stable_diffusion_compvis')

from dynaprompt.attention_modifier import AttentionStore


class DynaPromptV5Sampler:
    """
    DynaPrompt V5: Early detection and adaptive restart.
    """

    def __init__(
        self,
        ddim_sampler,
        model,
        tokenizer,
        device="cuda",
        check_step=15,
        attention_threshold=0.05,
        max_retries=3,
        noise_perturbation=0.3
    ):
        """
        Initialize DynaPrompt V5 sampler.

        Args:
            ddim_sampler: Base DDIM sampler
            model: Stable Diffusion model
            tokenizer: CLIP tokenizer
            device: Device to run on
            check_step: Step at which to check composition (default: 15)
            attention_threshold: Minimum attention for object presence (default: 0.05)
            max_retries: Maximum restart attempts (default: 3)
            noise_perturbation: How much to perturb initial noise on retry (default: 0.3)
        """
        self.ddim_sampler = ddim_sampler
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.check_step = check_step
        self.attention_threshold = attention_threshold
        self.max_retries = max_retries
        self.noise_perturbation = noise_perturbation

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
        Sample with early detection and adaptive restart.

        Args:
            prompt: Text prompt
            shape: Latent shape
            steps: Number of DDIM steps
            unconditional_guidance_scale: CFG scale
            critical_tokens: List of critical words that must appear (auto-detected if None)
            verbose: Print feedback info
            **kwargs: Additional arguments

        Returns:
            Generated latent and intermediates
        """
        print(f"\nDynaPrompt V5 Sampling (Early Detection + Adaptive Restart)")
        print(f"Prompt: {prompt}")
        print(f"Check step: {self.check_step}")
        print(f"Attention threshold: {self.attention_threshold}")
        print(f"Max retries: {self.max_retries}")
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

        # Try sampling with retries
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                # Use completely different random seed for retry
                retry_seed = torch.randint(0, 1000000, (1,)).item()
                torch.manual_seed(retry_seed)
                torch.cuda.manual_seed(retry_seed)

                print(f"\n{'='*80}")
                print(f"RETRY #{attempt}: Regenerating with new random seed {retry_seed}")
                print(f"{'='*80}\n")

            # Sample with early detection
            samples, intermediates, success = self._sample_with_early_detection(
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

            if success:
                print(f"\n{'='*80}")
                print(f"✓ DynaPrompt V5 sampling complete! (Attempt {attempt + 1}/{self.max_retries + 1})")
                print(f"{'='*80}\n")
                return samples, intermediates

        # Failed all retries
        print(f"\n{'='*80}")
        print(f"⚠ Warning: All {self.max_retries + 1} attempts had missing objects")
        print(f"Returning best attempt, but composition may be incomplete")
        print(f"{'='*80}\n")
        return samples, intermediates

    def _extract_critical_tokens(self, prompt: str) -> List[str]:
        """
        Extract critical nouns from prompt (simple heuristic).
        """
        # Common words to ignore
        ignore = {'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'is', 'are', 'was', 'were'}

        words = prompt.lower().split()
        critical = []

        for word in words:
            # Remove punctuation
            word = word.strip('.,!?;:')
            if word and word not in ignore and len(word) > 2:
                critical.append(word)

        return critical

    def _get_token_indices(self, token_ids: torch.Tensor, critical_words: List[str]) -> List[int]:
        """
        Get token indices for critical words.
        """
        indices = []

        for word in critical_words:
            # Tokenize the word
            word_tokens = self.tokenizer(word, add_special_tokens=False)['input_ids']

            # Find where this token appears in the full sequence
            for i, token_id in enumerate(token_ids):
                if token_id in word_tokens:
                    indices.append(i)

        return list(set(indices))  # Remove duplicates

    def _patch_attention_layers(self, unet, attention_store: AttentionStore):
        """Patch CrossAttention layers to capture attention maps."""

        def modify_forward(module):
            original_forward = module.forward

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

        def patch_recr(net):
            if net.__class__.__name__ == 'CrossAttention':
                modify_forward(net)
            elif hasattr(net, 'children'):
                for child in net.children():
                    patch_recr(child)

        patch_recr(unet)

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
    ) -> Tuple[torch.Tensor, dict, bool]:
        """
        Sample with early detection of missing objects.

        Returns:
            (samples, intermediates, success)
        """
        # Prepare sampling
        self.ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)

        device = self.model.betas.device
        b = shape[0]

        # Generate initial noise (fresh random for all attempts since seed changes)
        img = torch.randn(shape, device=device)

        # Setup attention tracking
        attention_store = AttentionStore()
        self._patch_attention_layers(self.model.model.diffusion_model, attention_store)

        timesteps = self.ddim_sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        iterator = tqdm(time_range, desc=f'DynaPrompt V5 (Attempt {attempt + 1})', total=total_steps)

        composition_checked = False
        composition_success = False

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
                iterator.set_description(f'DynaPrompt V5 (Checking composition...)')

                # Get average attention
                avg_attention = attention_store.get_average_attention()

                if avg_attention is not None:
                    token_attention = avg_attention.mean(dim=0)  # [tokens]

                    # Check if all critical tokens have sufficient attention
                    missing_tokens = []
                    for idx, token_name in zip(critical_indices, critical_tokens):
                        if idx < len(token_attention):
                            attn_score = token_attention[idx].item()
                            if attn_score < self.attention_threshold:
                                missing_tokens.append((token_name, idx, attn_score))

                    if len(missing_tokens) > 0:
                        print(f"\n   ⚠ Step {i}: Detected {len(missing_tokens)} underrepresented objects:")
                        for token_name, idx, score in missing_tokens[:5]:
                            print(f"     • '{token_name}' (token {idx}): attention = {score:.4f} < {self.attention_threshold}")

                        if attempt < self.max_retries:
                            print(f"   → Will retry with perturbed noise\n")
                            return None, None, False  # Trigger restart
                        else:
                            print(f"   → Final attempt, continuing anyway\n")
                            composition_success = False
                    else:
                        print(f"\n   ✓ Step {i}: All critical objects detected!")
                        for token_name, idx in zip(critical_tokens[:3], critical_indices[:3]):
                            if idx < len(token_attention):
                                print(f"     • '{token_name}' (token {idx}): attention = {token_attention[idx]:.4f}")
                        print()
                        composition_success = True

            # Store intermediates
            if index % 10 == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

            # Update attention store
            attention_store.step_callback()

        return img, intermediates, composition_success or (attempt == self.max_retries)
