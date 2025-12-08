"""
DynaPrompt V2: Attention-Based Dynamic Prompt Guidance

This version uses attention re-weighting instead of embedding boosting.
Based on the Attend-and-Excite approach.
"""

import torch
import sys
import os
import numpy as np
from tqdm import tqdm

# Add the stable diffusion path
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt/models/stable_diffusion_compvis')

from dynaprompt.attention_modifier import AttentionModifier


class DynaPromptV2Sampler:
    """
    DynaPrompt sampler using attention re-weighting.
    """

    def __init__(
        self,
        ddim_sampler,
        model,
        tokenizer,
        device="cuda",
        feedback_interval=5,
        boost_factor=1.3,
        attention_threshold=0.3,
        start_step_ratio=0.3,
        end_step_ratio=0.7
    ):
        """
        Initialize DynaPrompt V2 sampler.

        Args:
            ddim_sampler: Base DDIM sampler
            model: Stable Diffusion model
            tokenizer: CLIP tokenizer
            device: Device to run on
            feedback_interval: Analyze attention every N steps
            boost_factor: How much to boost attention (1.3 = 30% increase)
            attention_threshold: Threshold for underrepresented tokens
            start_step_ratio: Start attention modification at this ratio of total steps
            end_step_ratio: End attention modification at this ratio of total steps
        """
        self.ddim_sampler = ddim_sampler
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.feedback_interval = feedback_interval
        self.boost_factor = boost_factor
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
        Sample using DynaPrompt V2 with attention re-weighting.

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
        print(f"\nDynaPrompt V2 Sampling (Attention-Based)")
        print(f"Prompt: {prompt}")
        print(f"Boost factor: {self.boost_factor}x")
        print(f"Attention threshold: {self.attention_threshold}")
        print(f"Feedback interval: every {self.feedback_interval} steps")
        print(f"Active steps: {int(steps * self.start_step_ratio)}-{int(steps * self.end_step_ratio)}")
        print("="*80)

        # Calculate step range for modification
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

        # Patch the U-Net's cross-attention layers
        attention_modifier.patch_attention_layers(self.model.model.diffusion_model)

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

        # Sample with attention modification
        samples, intermediates = self._sample_with_attention_feedback(
            text_embeddings=text_embeddings,
            unconditional_embeddings=unconditional_embeddings,
            shape=shape,
            steps=steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            attention_modifier=attention_modifier,
            prompt=prompt,
            verbose=verbose,
            **kwargs
        )

        print(f"\n{'='*80}")
        print(f"✓ DynaPrompt V2 sampling complete!")
        print(f"  Final underrepresented tokens: {attention_modifier.underrepresented_indices}")
        print(f"{'='*80}\n")

        return samples, intermediates

    def _sample_with_attention_feedback(
        self,
        text_embeddings,
        unconditional_embeddings,
        shape,
        steps,
        unconditional_guidance_scale,
        attention_modifier,
        prompt,
        verbose=True,
        **kwargs
    ):
        """
        Internal sampling with attention feedback.
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
        iterator = tqdm(time_range, desc='DynaPrompt V2', total=total_steps)

        feedback_count = 0

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Prepare conditioning
            cond = text_embeddings
            uncond = unconditional_embeddings

            # Enable/disable attention modification based on step
            if attention_modifier.should_modify(i):
                attention_modifier.enable()
            else:
                attention_modifier.disable()

            # Perform DDIM step
            outs = self.ddim_sampler.p_sample_ddim(
                img, cond, ts,
                index=index,
                use_original_steps=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uncond
            )
            img, pred_x0 = outs

            # Analyze attention and update underrepresented tokens
            if i > 0 and i % self.feedback_interval == 0 and attention_modifier.should_modify(i):
                feedback_count += 1
                iterator.set_description(f'DynaPrompt V2 (Feedback #{feedback_count})')

                # Identify underrepresented tokens based on attention maps
                underrep = attention_modifier.identify_underrepresented_tokens(
                    prompt=prompt,
                    attention_threshold=self.attention_threshold
                )

                if len(underrep) > 0 and verbose:
                    # Get token names for display
                    token_ids = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )['input_ids'][0]

                    print(f"\n   [Step {i}] Found {len(underrep)} underrepresented tokens:")
                    for idx in underrep[:3]:  # Show top 3
                        if idx < len(token_ids):
                            token_text = self.tokenizer.decode([token_ids[idx]])
                            print(f"     • Token {idx}: {token_text}")

                # Update modifier with new underrepresented indices
                attention_modifier.set_underrepresented_indices(underrep)

                # Step callback to store this iteration's attention
                attention_modifier.attention_store.step_callback()

            # Store intermediates
            if index % 10 == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates
