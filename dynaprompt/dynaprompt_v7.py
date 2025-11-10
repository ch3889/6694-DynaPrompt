"""
DynaPrompt V7: Complete compositional generation system.

Combines:
1. LLM-based prompt rewriting (automatic prompt optimization)
2. Negative prompt guidance (steer away from failures)
3. Very early detection (step 3-5 instead of 15)
4. Attention boosting fallback (V6 Phase 2)

This is the most advanced and accurate version.
"""

import torch
import sys
from typing import List, Optional, Tuple
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'models' / 'stable_diffusion_compvis'))
sys.path.insert(0, str(Path(__file__).parent))

from dynaprompt_v6 import DynaPromptV6Sampler
from prompt_rewriter import PromptRewriter


class DynaPromptV7Sampler(DynaPromptV6Sampler):
    """
    DynaPrompt V7: Most advanced compositional generation.

    New features over V6:
    - Automatic prompt rewriting for better compositions
    - Negative prompt support
    - Configurable early detection (default: step 5 instead of 15)
    - Higher retry count (default: 15 instead of 2)
    """

    def __init__(
        self,
        ddim_sampler,
        model,
        tokenizer,
        device="cuda",
        check_step=3,  # VERY early detection (6% completion)
        attention_threshold=0.05,  # Stricter threshold than before
        max_retries=15,  # Many seed retries before boosting fallback
        boost_factor=7.5,  # Strong base boost (adaptive up to 22.5x for very low attention)
        start_step_ratio=0.0,
        end_step_ratio=0.5,  # Longer than V6's 0.4
        use_prompt_rewriting=False,  # DISABLED: TinyLlama hallucinations too severe
        llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_llm_gpu=True,
    ):
        """
        Initialize DynaPrompt V7 sampler.

        Args:
            ddim_sampler: DDIM sampler instance
            model: Stable Diffusion model
            tokenizer: Text tokenizer
            device: Device to use
            check_step: Step to check composition (5 = 10% done, earlier is better)
            attention_threshold: Minimum attention score (lower = more permissive)
            max_retries: Maximum seed retries before boosting
            boost_factor: Base attention boost multiplier (adaptive: up to 3x this value for very low attention)
            start_step_ratio: When to start boosting
            end_step_ratio: When to stop boosting
            use_prompt_rewriting: Enable automatic prompt optimization
            llm_model_name: Local LLM for prompt rewriting
            use_llm_gpu: Use GPU for LLM
        """
        super().__init__(
            ddim_sampler=ddim_sampler,
            model=model,
            tokenizer=tokenizer,
            device=device,
            check_step=check_step,
            attention_threshold=attention_threshold,
            max_retries=max_retries,
            boost_factor=boost_factor,
            start_step_ratio=start_step_ratio,
            end_step_ratio=end_step_ratio,
        )

        # V7 additions
        self.use_prompt_rewriting = use_prompt_rewriting
        self.prompt_rewriter = None

        if use_prompt_rewriting:
            self.prompt_rewriter = PromptRewriter(
                model_name=llm_model_name,
                use_gpu=use_llm_gpu
            )

    def sample_with_dynaprompt(
        self,
        prompt: str,
        shape,
        steps=50,
        unconditional_guidance_scale=7.5,
        critical_tokens: List[str] = None,
        verbose=True,
        negative_prompt: Optional[str] = None,
    ):
        """
        Generate image with DynaPrompt V7.

        Args:
            prompt: Text prompt
            shape: Latent shape
            steps: Number of diffusion steps
            unconditional_guidance_scale: CFG scale
            critical_tokens: Required objects (auto-detected if None)
            verbose: Print progress
            negative_prompt: Negative prompt (auto-generated if None)

        Returns:
            (samples, intermediates)
        """

        # Step 1: Rewrite prompt if enabled
        original_prompt = prompt
        if self.use_prompt_rewriting and self.prompt_rewriter is not None:
            if verbose:
                print(f"\n{'='*80}")
                print(f"DynaPrompt V7: Optimizing prompt...")
                print(f"{'='*80}\n")

            enhanced_prompt, auto_negative = self.prompt_rewriter.rewrite_for_accuracy(prompt)

            if verbose:
                print(f"Original prompt: {original_prompt}")
                print(f"Enhanced prompt: {enhanced_prompt}")
                if negative_prompt is None:
                    print(f"Negative prompt: {auto_negative}")
                print()

            prompt = enhanced_prompt

            # Use auto-generated negative if not provided
            if negative_prompt is None:
                negative_prompt = auto_negative

        # Step 2: Generate negative embeddings if provided
        negative_embeddings = None
        if negative_prompt is not None:
            negative_input = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            negative_embeddings = self.model.cond_stage_model.transformer(
                negative_input.input_ids
            )[0]

        # Step 3: Call V6's detection + boosting with negative embeddings
        # Note: V6 uses unconditional_embeddings, we override with negative if provided
        if negative_embeddings is not None:
            # Modify V6's behavior by replacing unconditional with negative
            return self._sample_with_negative(
                prompt=prompt,
                negative_embeddings=negative_embeddings,
                shape=shape,
                steps=steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                critical_tokens=critical_tokens,
                verbose=verbose,
            )
        else:
            # Standard V6 path
            return super().sample_with_dynaprompt(
                prompt=prompt,
                shape=shape,
                steps=steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                critical_tokens=critical_tokens,
                verbose=verbose,
            )

    def _sample_with_negative(
        self,
        prompt: str,
        negative_embeddings,
        shape,
        steps,
        unconditional_guidance_scale,
        critical_tokens,
        verbose,
    ):
        """
        Sample with negative prompt guidance.

        This modifies V6's sampling to use negative embeddings instead of
        unconditional (empty string) embeddings for classifier-free guidance.
        """

        # Get text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        text_embeddings = self.model.cond_stage_model.transformer(text_input.input_ids)[0]

        # Extract critical tokens
        if critical_tokens is None:
            critical_tokens = self._extract_critical_tokens(prompt)

        if verbose:
            print(f"\n{'='*80}")
            print(f"DynaPrompt V7 Sampling (with Negative Prompts)")
            print(f"Prompt: {prompt}")
            print(f"Strategy: Try {self.max_retries + 1} seeds, fallback to attention boosting")
            print(f"Check step: {self.check_step} (~{100*self.check_step//steps}% done)")
            print(f"Attention threshold: {self.attention_threshold}")
            print(f"Boost factor (fallback): {self.boost_factor}x")
            print(f"{'='*80}")

        # Get critical token indices
        critical_indices = []
        for token in critical_tokens:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_id) > 0:
                for idx, tid in enumerate(text_input.input_ids[0]):
                    if tid == token_id[0]:
                        critical_indices.append(idx)
                        break

        if verbose:
            print(f"Auto-detected critical tokens: {critical_tokens}")
            print(f"Critical token indices: {critical_indices}")
            print(f"{'='*80}\n")

        # Phase 1: Try finding a good seed with negative prompts
        best_attempt = None
        best_attention_scores = None
        best_seed = None

        # Save original forwards
        self.original_forwards = {}
        self._save_original_forwards(self.model.model.diffusion_model)

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                retry_seed = torch.randint(0, 1000000, (1,)).item()
                torch.manual_seed(retry_seed)
                torch.cuda.manual_seed(retry_seed)

                print(f"\n{'='*80}")
                print(f"RETRY #{attempt}: Trying random seed {retry_seed}")
                print(f"{'='*80}\n")

            # Sample with early detection (using negative instead of unconditional)
            # Use V6's method but pass negative_embeddings as unconditional
            img, intermediates, success, attention_scores, seed_used = (
                self._sample_with_early_detection(
                    text_embeddings=text_embeddings,
                    unconditional_embeddings=negative_embeddings,  # Use negative here!
                    critical_indices=critical_indices,
                    critical_tokens=critical_tokens,
                    shape=shape,
                    steps=steps,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    attempt=attempt,
                    verbose=verbose,
                )
            )

            # Track best attempt
            if best_attempt is None or sum(attention_scores.values()) > sum(
                best_attention_scores.values()
            ):
                best_attempt = (img, intermediates)
                best_attention_scores = attention_scores
                best_seed = seed_used

            if success:
                print(f"\n{'='*80}")
                print(f"✓ DynaPrompt V7 complete (found good seed: {seed_used})")
                print(f"{'='*80}\n")
                return img, intermediates

        # Phase 2: Use attention boosting
        print(f"\n{'='*80}")
        print(f"⚠ No seed with sufficient attention found after {self.max_retries + 1} attempts")
        print(f"Switching to PHASE 2: Attention boosting on best seed {best_seed}")
        print(f"Best attention scores: {best_attention_scores}")
        print(f"{'='*80}\n")

        # Restore originals
        print("Restoring original attention layers...")
        self._unpatch_attention_layers(self.model.model.diffusion_model)

        # Set best seed
        torch.manual_seed(best_seed)
        torch.cuda.manual_seed(best_seed)

        # Sample with boosting (using negative embeddings)
        samples, intermediates = self._sample_with_attention_boosting(
            text_embeddings=text_embeddings,
            unconditional_embeddings=negative_embeddings,  # Use negative for CFG
            text_input_ids=text_input.input_ids,
            shape=shape,
            steps=steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            prompt=prompt,
            verbose=verbose,
        )

        print(f"\n{'='*80}")
        print(f"✓ DynaPrompt V7 complete (used attention boosting fallback)")
        print(f"{'='*80}\n")

        # Clean up
        print("Cleaning up Phase 2 patches...")
        self._unpatch_attention_layers(self.model.model.diffusion_model)
        self.original_forwards = {}

        return samples, intermediates

    # No need for custom early detection method - V6's method works with negative embeddings
    # by just passing them as unconditional_embeddings parameter
