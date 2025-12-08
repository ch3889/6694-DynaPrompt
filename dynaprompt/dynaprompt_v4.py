"""
DynaPrompt V4: Gradient-Based Latent Refinement

This version uses gradient-based guidance to actually STEER the latent trajectory,
not just modify attention weights. Based on the Attend-and-Excite approach.

Key improvements over V3:
- Compute attention loss for underrepresented tokens
- Backpropagate through denoising to get latent gradients
- Iteratively refine latent to maximize attention to missing concepts
- This CREATES features instead of just amplifying non-existent signals
"""

import torch
import sys
import os
import numpy as np
from tqdm import tqdm

# Add the stable diffusion path
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt/models/stable_diffusion_compvis')

from dynaprompt.attention_modifier import AttentionStore


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
        refinement_steps=5,
        learning_rate=20.0,
        attention_threshold=0.3,
        start_step_ratio=0.0,
        end_step_ratio=0.4
    ):
        """
        Initialize DynaPrompt V4 sampler.

        Args:
            ddim_sampler: Base DDIM sampler
            model: Stable Diffusion model
            tokenizer: CLIP tokenizer
            device: Device to run on
            feedback_interval: Analyze attention every N steps (default: 3)
            refinement_steps: Number of gradient steps per feedback (default: 5)
            learning_rate: Gradient descent learning rate (default: 20.0)
            attention_threshold: Threshold for underrepresented tokens
            start_step_ratio: Start refinement at this ratio (default: 0.0 = step 0)
            end_step_ratio: End refinement at this ratio (default: 0.4 = step 20)
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

        Args:
            prompt: Text prompt
            shape: Latent shape
            steps: Number of DDIM steps
            unconditional_guidance_scale: CFG scale
            verbose: Print feedback info
            **kwargs: Additional arguments for DDIM sampler

        Returns:
            Generated latent and intermediates
        """
        print(f"\nDynaPrompt V4 Sampling (Gradient-Based Refinement)")
        print(f"Prompt: {prompt}")
        print(f"Refinement steps per feedback: {self.refinement_steps}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Attention threshold: {self.attention_threshold}")
        print(f"Feedback interval: every {self.feedback_interval} steps")
        print(f"Active steps: {int(steps * self.start_step_ratio)}-{int(steps * self.end_step_ratio)} (EARLY)")
        print("="*80)

        # Calculate step range for modification
        start_step = int(steps * self.start_step_ratio)
        end_step = int(steps * self.end_step_ratio)

        # Initialize attention store
        attention_store = AttentionStore()

        # Patch the U-Net's cross-attention layers to capture attention
        self._patch_attention_layers(self.model.model.diffusion_model, attention_store)

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

        # Unconditional embeddings for CFG
        unconditional_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            unconditional_embeddings = self.model.cond_stage_model.transformer(unconditional_input.input_ids)[0]

        # Sample with gradient-based refinement
        samples, intermediates = self._sample_with_gradient_guidance(
            text_embeddings=text_embeddings,
            unconditional_embeddings=unconditional_embeddings,
            text_input_ids=text_input.input_ids,
            shape=shape,
            steps=steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            attention_store=attention_store,
            start_step=start_step,
            end_step=end_step,
            prompt=prompt,
            verbose=verbose,
            **kwargs
        )

        print(f"\n{'='*80}")
        print(f"✓ DynaPrompt V4 sampling complete!")
        print(f"{'='*80}\n")

        return samples, intermediates

    def _patch_attention_layers(self, unet, attention_store):
        """
        Patch all CrossAttention layers to capture attention maps.

        Args:
            unet: The U-Net model
            attention_store: AttentionStore instance
        """
        def make_forward_hook(place_in_unet):
            def forward_hook(module, input, output):
                # The output is the result after to_out projection
                # We need to capture attention from the forward pass
                # This is stored in the modified forward function
                pass
            return forward_hook

        def modify_forward(module, place_in_unet):
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

                # Store attention if this is cross-attention
                is_cross = context is not None and context.shape != x.shape
                if is_cross:
                    attention_store(attn, is_cross=True, place_in_unet=place_in_unet)

                out = torch.einsum('b i j, b j d -> b i d', attn, v)
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                return module.to_out(out)

            module.forward = new_forward

        def patch_recr(net, place_in_unet):
            if net.__class__.__name__ == 'CrossAttention':
                modify_forward(net, place_in_unet)
            elif hasattr(net, 'children'):
                for child in net.children():
                    patch_recr(child, place_in_unet)

        # Patch all U-Net blocks
        for name in ['down', 'mid', 'up']:
            if hasattr(unet, name + '_blocks') or hasattr(unet, name):
                block = getattr(unet, name + '_blocks', None) or getattr(unet, name, None)
                if block is not None:
                    patch_recr(block, name)

        print("Patched CrossAttention layers for attention capture")

    def _sample_with_gradient_guidance(
        self,
        text_embeddings,
        unconditional_embeddings,
        text_input_ids,
        shape,
        steps,
        unconditional_guidance_scale,
        attention_store,
        start_step,
        end_step,
        prompt,
        verbose=True,
        **kwargs
    ):
        """
        Internal sampling with gradient-based guidance.
        """
        # Prepare sampling
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

            # Prepare conditioning
            cond = text_embeddings
            uncond = unconditional_embeddings

            # Check if we should refine at this step
            should_refine = (
                start_step <= i <= end_step and
                i > 0 and
                i % self.feedback_interval == 0
            )

            if should_refine:
                refinement_count += 1
                iterator.set_description(f'DynaPrompt V4 (Refinement #{refinement_count})')

                # Perform gradient-based latent refinement
                img = self._refine_latent(
                    latent=img,
                    cond=cond,
                    uncond=uncond,
                    timestep=ts,
                    index=index,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    attention_store=attention_store,
                    text_input_ids=text_input_ids,
                    prompt=prompt,
                    step_num=i,
                    verbose=verbose
                )

            # Perform DDIM step (without gradient tracking)
            with torch.no_grad():
                outs = self.ddim_sampler.p_sample_ddim(
                    img, cond, ts,
                    index=index,
                    use_original_steps=False,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=uncond
                )
                img, pred_x0 = outs

            # Store intermediates
            if index % 10 == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

            # Step callback for attention store
            attention_store.step_callback()

        return img, intermediates

    def _refine_latent(
        self,
        latent,
        cond,
        uncond,
        timestep,
        index,
        unconditional_guidance_scale,
        attention_store,
        text_input_ids,
        prompt,
        step_num,
        verbose=True
    ):
        """
        Refine latent using gradient descent to maximize attention to underrepresented tokens.

        Args:
            latent: Current latent
            cond: Conditional embeddings
            uncond: Unconditional embeddings
            timestep: Current timestep
            index: Index in timestep schedule
            unconditional_guidance_scale: CFG scale
            attention_store: AttentionStore instance
            text_input_ids: Tokenized text input IDs
            prompt: Text prompt
            step_num: Current step number
            verbose: Print info

        Returns:
            Refined latent
        """
        # First, perform a forward pass to get attention maps
        attention_store.reset()

        with torch.no_grad():
            _ = self.ddim_sampler.p_sample_ddim(
                latent, cond, timestep,
                index=index,
                use_original_steps=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uncond
            )

        # Get average attention
        avg_attention = attention_store.get_average_attention()

        if avg_attention is None:
            return latent

        # avg_attention: [batch*heads, tokens]
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

        # Perform iterative refinement using a simpler approach
        # Instead of backprop through sampling, compute noise prediction gradient
        refined_latent = latent.clone().detach()

        for refine_step in range(self.refinement_steps):
            # Enable gradient for latent
            refined_latent_grad = refined_latent.clone().detach().requires_grad_(True)

            # Reset attention store
            attention_store.reset()

            # Get noise prediction with gradients
            t = timestep[0].item()

            # Predict noise (with gradients)
            noise_pred = self.model.apply_model(refined_latent_grad, timestep, cond)

            # Get attention maps (stored during apply_model call)
            avg_attention = attention_store.get_average_attention()

            if avg_attention is not None and avg_attention.requires_grad:
                # Compute attention loss
                attention_loss = torch.tensor(0.0, device=latent.device, requires_grad=True)
                for token_idx in underrepresented:
                    if token_idx < avg_attention.shape[-1]:
                        # Maximize attention to this token
                        attention_loss = attention_loss - avg_attention[:, token_idx].sum()

                # Backpropagate to get gradient w.r.t. latent
                if attention_loss.requires_grad:
                    attention_loss.backward()

                    # Update latent with gradient descent
                    if refined_latent_grad.grad is not None:
                        with torch.no_grad():
                            grad_norm = refined_latent_grad.grad.norm().item()
                            if grad_norm > 0:
                                refined_latent = refined_latent - self.learning_rate * refined_latent_grad.grad
                                if verbose and refine_step == 0:
                                    print(f"     → Gradient norm: {grad_norm:.4f}")

        if verbose:
            print(f"     → Applied {self.refinement_steps} refinement steps")

        return refined_latent
