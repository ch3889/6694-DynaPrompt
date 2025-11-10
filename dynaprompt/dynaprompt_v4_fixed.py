"""
DynaPrompt V4: Gradient-Based Latent Refinement (Fixed Implementation)

This version uses a simpler and more direct approach:
- Aggregate attention maps across all cross-attention layers
- Compute loss directly on attention aggregation
- Backpropagate to latent to maximize attention to underrepresented tokens
"""

import torch
import sys
import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List

# Add the stable diffusion path
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt/models/stable_diffusion_compvis')


class GradientAttentionStore:
    """
    Stores attention maps WITH gradients for backpropagation.
    """

    def __init__(self):
        self.attention_maps = []

    def __call__(self, attn):
        """Store attention map (keep on GPU with gradients)."""
        self.attention_maps.append(attn)

    def get_average_attention(self):
        """Get average attention, keeping gradients."""
        if len(self.attention_maps) == 0:
            return None

        # Average over spatial dimension for each map
        token_attentions = []
        for attn_map in self.attention_maps:
            # attn_map: (batch*heads, pixels, tokens)
            token_attn = attn_map.mean(dim=1)  # (batch*heads, tokens)
            token_attentions.append(token_attn)

        # Stack and average
        avg_attention = torch.stack(token_attentions).mean(0)  # (batch*heads, tokens)
        return avg_attention

    def reset(self):
        """Clear stored attention maps."""
        self.attention_maps = []


class DynaPromptV4Sampler:
    """
    DynaPrompt V4 sampler with gradient-based latent refinement.
    """

    def __init__(
        self,
        ddim_sampler,
        model,
        tokenizer,
        device="cuda",
        feedback_interval=3,
        refinement_steps=20,
        learning_rate=20.0,
        attention_threshold=0.3,
        start_step_ratio=0.0,
        end_step_ratio=0.4
    ):
        """
        Initialize DynaPrompt V4 sampler.
        """
        self.ddim_sampler = ddim_sampler
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.feedback_interval = feedback_interval
        self.refinement_steps = refinement_steps
        self.learning_rate = learning_rate
        self.attention_threshold = attention_threshold
        self.start_step_ratio = start_step_ratio
        self.end_step_ratio = end_step_ratio
        self.attention_store = GradientAttentionStore()

    def sample_with_dynaprompt(
        self,
        prompt: str,
        shape,
        steps=50,
        unconditional_guidance_scale=7.5,
        verbose=True,
        **kwargs
    ):
        """
        Sample using DynaPrompt V4 with gradient-based latent refinement.
        """
        print(f"\nDynaPrompt V4 Sampling (Gradient-Based Refinement - Fixed)")
        print(f"Prompt: {prompt}")
        print(f"Refinement steps per feedback: {self.refinement_steps}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Attention threshold: {self.attention_threshold}")
        print(f"Feedback interval: every {self.feedback_interval} steps")
        print(f"Active steps: {int(steps * self.start_step_ratio)}-{int(steps * self.end_step_ratio)}")
        print("="*80)

        # Calculate step range
        start_step = int(steps * self.start_step_ratio)
        end_step = int(steps * self.end_step_ratio)

        # Patch attention layers
        self._patch_attention_layers(self.model.model.diffusion_model)

        # Encode prompt
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_embeddings = self.model.cond_stage_model.transformer(text_input.input_ids)[0]
            unconditional_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.device)
            unconditional_embeddings = self.model.cond_stage_model.transformer(unconditional_input.input_ids)[0]

        # Sample
        samples, intermediates = self._sample_with_gradient_guidance(
            text_embeddings=text_embeddings,
            unconditional_embeddings=unconditional_embeddings,
            text_input_ids=text_input.input_ids,
            shape=shape,
            steps=steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            start_step=start_step,
            end_step=end_step,
            prompt=prompt,
            verbose=verbose
        )

        print(f"\n{'='*80}")
        print(f"✓ DynaPrompt V4 sampling complete!")
        print(f"{'='*80}\n")

        return samples, intermediates

    def _patch_attention_layers(self, unet):
        """Patch CrossAttention layers to capture attention WITH gradients."""

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

                # Store attention if cross-attention (keep gradients!)
                is_cross = context is not None and context.shape != x.shape
                if is_cross:
                    self.attention_store(attn)

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
        print("Patched CrossAttention layers for gradient-aware attention capture")

    def _sample_with_gradient_guidance(
        self,
        text_embeddings,
        unconditional_embeddings,
        text_input_ids,
        shape,
        steps,
        unconditional_guidance_scale,
        start_step,
        end_step,
        prompt,
        verbose=True
    ):
        """Internal sampling with gradient-based guidance."""
        self.ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)

        device = self.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        timesteps = self.ddim_sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        iterator = tqdm(time_range, desc='DynaPrompt V4', total=total_steps)

        refinement_count = 0

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Check if we should refine
            should_refine = (start_step <= i <= end_step and i > 0 and i % self.feedback_interval == 0)

            if should_refine:
                refinement_count += 1
                iterator.set_description(f'DynaPrompt V4 (Refinement #{refinement_count})')

                # Refine latent
                img = self._refine_latent(
                    latent=img,
                    cond=text_embeddings,
                    uncond=unconditional_embeddings,
                    timestep=ts,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    text_input_ids=text_input_ids,
                    step_num=i,
                    verbose=verbose
                )

            # Perform DDIM step (without gradients)
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

        return img, intermediates

    def _refine_latent(
        self,
        latent,
        cond,
        uncond,
        timestep,
        unconditional_guidance_scale,
        text_input_ids,
        step_num,
        verbose=True
    ):
        """
        Refine latent using gradient descent on attention maps.
        """
        # First pass: identify underrepresented tokens
        self.attention_store.reset()

        with torch.no_grad():
            _ = self.model.apply_model(latent, timestep, cond)

        avg_attention = self.attention_store.get_average_attention()

        if avg_attention is None:
            return latent

        token_attention = avg_attention.mean(dim=0)  # [tokens]

        # Identify underrepresented tokens
        actual_tokens = (text_input_ids[0] != self.tokenizer.pad_token_id).sum().item()
        underrepresented = []

        for i in range(1, min(actual_tokens - 1, len(token_attention))):
            if token_attention[i] < self.attention_threshold:
                underrepresented.append(i)

        if len(underrepresented) == 0:
            return latent

        if verbose:
            print(f"\n   [Step {step_num}] Found {len(underrepresented)} underrepresented tokens:")
            for idx in underrepresented[:3]:
                if idx < len(text_input_ids[0]):
                    token_text = self.tokenizer.decode([text_input_ids[0][idx]])
                    print(f"     • Token {idx}: '{token_text}' (attn: {token_attention[idx]:.4f})")

        # Iterative refinement
        refined_latent = latent.clone().detach()

        for refine_step in range(self.refinement_steps):
            # Enable gradients for latent
            latent_for_grad = refined_latent.clone().requires_grad_(True)

            # Reset attention store
            self.attention_store.reset()

            # Forward pass with gradients - directly through U-Net
            # We need to call the diffusion model directly, not apply_model
            noise_pred = self.model.model.diffusion_model(
                latent_for_grad,
                timestep,
                context=cond
            )

            # Get attention
            avg_attention = self.attention_store.get_average_attention()

            if avg_attention is None:
                break

            # Compute loss: maximize attention to underrepresented tokens
            # Start with first token's attention (which has grad_fn)
            loss = None

            for token_idx in underrepresented:
                if token_idx < avg_attention.shape[-1]:
                    # Maximize mean attention to this token
                    token_attn = avg_attention[:, token_idx].mean()
                    if loss is None:
                        loss = -token_attn
                    else:
                        loss = loss - token_attn

            if loss is None:
                break

            # Check if we got gradients
            try:
                # Backpropagate
                loss.backward(retain_graph=False)

                # Update latent
                if latent_for_grad.grad is not None:
                    with torch.no_grad():
                        grad = latent_for_grad.grad
                        grad_norm = grad.norm().item()

                        if grad_norm > 0:
                            # Gradient descent step
                            refined_latent = refined_latent - self.learning_rate * grad

                            if verbose and refine_step == 0:
                                print(f"     → Gradient norm: {grad_norm:.6f}, Loss: {loss.item():.6f}")
                        else:
                            if verbose and refine_step == 0:
                                print(f"     → Warning: Zero gradient norm")
                            break
                else:
                    if verbose and refine_step == 0:
                        print(f"     → Warning: No gradients computed")
                    break

            except RuntimeError as e:
                if verbose and refine_step == 0:
                    print(f"     → Gradient computation failed: {e}")
                break

        if verbose:
            print(f"     → Applied {self.refinement_steps} refinement steps")

        return refined_latent
