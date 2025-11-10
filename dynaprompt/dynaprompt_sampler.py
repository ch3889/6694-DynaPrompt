"""
DynaPrompt-enabled DDIM Sampler

This module extends the DDIM sampler with dynamic prompt guidance using CLIP feedback.
"""

import torch
import sys
import os

# Add the stable diffusion path
sys.path.insert(0, '/home/cursedfox/6694-DynaPrompt/models/stable_diffusion_compvis')

from dynaprompt.clip_feedback import CLIPFeedback
from dynaprompt.prompt_updater import PromptUpdater


class DynaPromptSampler:
    """
    Wrapper around DDIM sampler that adds dynamic prompt guidance.
    """

    def __init__(
        self,
        ddim_sampler,
        model,
        tokenizer,
        text_encoder,
        device="cuda",
        feedback_interval=10,
        boost_factor=1.5,
        max_boost=3.0,
        use_adaptive=True
    ):
        """
        Initialize DynaPrompt sampler.

        Args:
            ddim_sampler: Base DDIM sampler
            model: Stable Diffusion model
            tokenizer: CLIP tokenizer
            text_encoder: CLIP text encoder
            device: Device to run on
            feedback_interval: Apply CLIP feedback every N steps
            boost_factor: How much to boost underrepresented tokens
            max_boost: Maximum cumulative boost
            use_adaptive: Use adaptive boosting based on score gaps
        """
        self.ddim_sampler = ddim_sampler
        self.model = model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.feedback_interval = feedback_interval
        self.use_adaptive = use_adaptive

        # Initialize modules
        self.clip_feedback = CLIPFeedback(device=device)
        self.prompt_updater = PromptUpdater(
            boost_factor=boost_factor,
            max_boost=max_boost
        )

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
        Sample using DynaPrompt with dynamic prompt guidance.

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
        # Reset prompt updater
        self.prompt_updater.reset()

        # Tokenize prompt
        tokens = prompt.split()
        print(f"\nDynaPrompt Sampling")
        print(f"Prompt: {prompt}")
        print(f"Tokens: {tokens}")
        print(f"Feedback interval: every {self.feedback_interval} steps")
        print(f"Adaptive boosting: {self.use_adaptive}")
        print("="*80)

        # Encode prompt
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids)[0]

        # Unconditional embeddings for CFG
        unconditional_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids)[0]

        # Store original embeddings
        original_embeddings = text_embeddings.clone()

        # Custom callback for CLIP feedback
        feedback_count = 0

        def dynaprompt_callback(i, pred_x0=None):
            nonlocal feedback_count, text_embeddings

            # Apply feedback at specified intervals
            if i > 0 and i % self.feedback_interval == 0 and pred_x0 is not None:
                feedback_count += 1
                print(f"\n[Step {i}/{steps}] Applying CLIP feedback #{feedback_count}...")

                # Get CLIP feedback
                token_scores, underrepresented = self.clip_feedback.get_feedback(
                    latents=pred_x0,
                    vae=self.model.first_stage_model,
                    tokens=tokens,
                    top_k=3  # Focus on top 3 worst tokens
                )

                if verbose:
                    print(f"   Token scores:")
                    for tok, score in sorted(token_scores.items(), key=lambda x: -x[1])[:5]:
                        print(f"     {tok:15s}: {score:.3f}")

                # Update embeddings
                if len(underrepresented) > 0:
                    if self.use_adaptive:
                        text_embeddings = self.prompt_updater.adaptive_boost(
                            embeddings=text_embeddings,
                            tokenized_prompt=text_input.input_ids,
                            token_scores=token_scores,
                            underrepresented=underrepresented,
                            tokenizer=self.tokenizer,
                            text_encoder=self.text_encoder
                        )
                    else:
                        text_embeddings = self.prompt_updater.update_embeddings(
                            embeddings=text_embeddings,
                            tokenized_prompt=text_input.input_ids,
                            underrepresented=underrepresented,
                            tokenizer=self.tokenizer,
                            text_encoder=self.text_encoder
                        )

                    # Update conditioning for next steps
                    # Note: This is tricky - we need to pass updated embeddings to the sampler
                    # For now, we'll store them and they'll be used in the next iteration

        # Modified DDIM sampling with dynamic conditioning
        # We'll need to modify the DDIM sampler to accept a callback with pred_x0
        # For now, let's use the standard sampler and add hooks

        # Store callback
        self.current_embeddings = text_embeddings
        self.unconditional_embeddings = unconditional_embeddings
        self.dynaprompt_callback = dynaprompt_callback
        self.step_count = 0
        self.total_steps = steps

        # Sample using modified DDIM
        samples, intermediates = self._sample_with_feedback(
            text_embeddings=text_embeddings,
            unconditional_embeddings=unconditional_embeddings,
            shape=shape,
            steps=steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            tokens=tokens,
            text_input=text_input,
            **kwargs
        )

        print(f"\n{'='*80}")
        print(f"✓ DynaPrompt sampling complete!")
        print(f"  Total CLIP feedback iterations: {feedback_count}")
        print(f"  Final token boosts: {self.prompt_updater.token_boosts}")
        print(f"{'='*80}\n")

        return samples, intermediates

    def _sample_with_feedback(
        self,
        text_embeddings,
        unconditional_embeddings,
        shape,
        steps,
        unconditional_guidance_scale,
        tokens,
        text_input,
        **kwargs
    ):
        """
        Internal sampling with CLIP feedback integration.
        """
        import numpy as np
        from tqdm import tqdm

        # Prepare sampling
        self.ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)

        device = self.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        timesteps = self.ddim_sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        iterator = tqdm(time_range, desc='DynaPrompt Sampler', total=total_steps)

        feedback_count = 0
        current_text_embeddings = text_embeddings.clone()

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # Prepare conditioning with current embeddings
            cond = current_text_embeddings
            uncond = unconditional_embeddings

            # Perform DDIM step
            outs = self.ddim_sampler.p_sample_ddim(
                img, cond, ts,
                index=index,
                use_original_steps=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uncond
            )
            img, pred_x0 = outs

            # Apply CLIP feedback at intervals
            if i > 0 and i % self.feedback_interval == 0:
                feedback_count += 1
                iterator.set_description(f'DynaPrompt (Feedback #{feedback_count})')

                # Get CLIP feedback
                token_scores, underrepresented = self.clip_feedback.get_feedback(
                    latents=pred_x0,
                    vae=self.model.first_stage_model,
                    tokens=tokens,
                    top_k=2
                )

                # Update embeddings if needed
                if len(underrepresented) > 0:
                    if self.use_adaptive:
                        current_text_embeddings = self.prompt_updater.adaptive_boost(
                            embeddings=current_text_embeddings,
                            tokenized_prompt=text_input.input_ids,
                            token_scores=token_scores,
                            underrepresented=underrepresented,
                            tokenizer=self.tokenizer,
                            text_encoder=self.text_encoder
                        )
                    else:
                        current_text_embeddings = self.prompt_updater.update_embeddings(
                            embeddings=current_text_embeddings,
                            tokenized_prompt=text_input.input_ids,
                            underrepresented=underrepresented,
                            tokenizer=self.tokenizer,
                            text_encoder=self.text_encoder
                        )

            # Store intermediates
            if index % 10 == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates
