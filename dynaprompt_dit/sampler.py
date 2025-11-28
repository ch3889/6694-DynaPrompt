"""
DiT-based sampler with early attention detection and fallback boosting.
This is framework-agnostic scaffolding; wire it to your DiT implementation.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

from .attention_boost import DiTAttentionBooster


@dataclass
class DiTComponents:
    tokenizer: any  # Should provide HF-like API
    text_encoder: any  # Encodes token ids -> embeddings
    dit_model: any  # Diffusion Transformer UNet-like with attention blocks
    scheduler: any  # DDIM/DPMSolver-like scheduler for steps


class DiTDynaPromptSampler:
    def __init__(
        self,
        comps: DiTComponents,
        device: str = "cuda",
        check_step: int = 5,
        attention_threshold: float = 0.05,
        max_retries: int = 10,
        boost_factor: float = 6.0,
        start_step_ratio: float = 0.0,
        end_step_ratio: float = 0.5,
    ):
        self.c = comps
        self.device = device
        self.check_step = check_step
        self.attention_threshold = attention_threshold
        self.max_retries = max_retries
        self.booster = DiTAttentionBooster(boost_factor=boost_factor)
        self.start_step_ratio = start_step_ratio
        self.end_step_ratio = end_step_ratio

        self._original_attn_forwards = []

    def _get_token_indices(self, token_ids: torch.Tensor, words: List[str]) -> List[int]:
        indices = []
        for word in words:
            toks = self.c.tokenizer(word, add_special_tokens=False)["input_ids"]
            for i, tid in enumerate(token_ids):
                if tid in toks:
                    indices.append(i)
        return sorted(list(set(indices)))

    def _capture_attn(self):
        """
        Patch DiT attention blocks to capture attn matrices.
        This is pseudocode; adapt to your DiT's attention class name.
        """
        store = []
        self._original_attn_forwards = []

        def wrap(module):
            orig = module.forward
            self._original_attn_forwards.append((module, orig))

            def new_forward(*args, **kwargs):
                out = orig(*args, **kwargs)
                # Assume module exposes last attention as module.last_attn
                attn = getattr(module, "last_attn", None)
                if attn is not None:
                    store.append(attn.detach().cpu())
                return out

            module.forward = new_forward

        # Walk modules and wrap attention blocks
        for m in self.c.dit_model.modules():
            if m.__class__.__name__.lower().endswith("attention"):
                wrap(m)

        return store

    def _unpatch_attn(self):
        for m, f in self._original_attn_forwards:
            m.forward = f
        self._original_attn_forwards = []

    def _patch_boost(self):
        self._original_attn_forwards = []

        def wrap(module):
            orig = module.forward
            self._original_attn_forwards.append((module, orig))

            def new_forward(*args, **kwargs):
                out = orig(*args, **kwargs)
                attn = getattr(module, "last_attn", None)
                if attn is not None:
                    boosted = self.booster.apply(attn)
                    setattr(module, "last_attn", boosted)
                return out

            module.forward = new_forward

        for m in self.c.dit_model.modules():
            if m.__class__.__name__.lower().endswith("attention"):
                wrap(m)

    def sample(
        self,
        prompt: str,
        steps: int = 50,
        cfg_scale: float = 7.5,
        critical_words: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        # Tokenize
        text_input = self.c.tokenizer(
            [prompt],
            padding="max_length",
            max_length=getattr(self.c.tokenizer, "model_max_length", 77),
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_embeddings = self.c.text_encoder(text_input.input_ids)

        # Critical words
        if critical_words is None:
            ignore = {"a","an","the","of","in","on","at","to","for","with","by","from","is","are","was","were"}
            tokens = [w.strip('.,!?;:').lower() for w in prompt.split()]
            critical_words = [w for w in tokens if w and w not in ignore and len(w) > 2]
        crit_indices = self._get_token_indices(text_input.input_ids[0], critical_words)

        if verbose:
            print("DiT dynaprompt: early detection + fallback boosting")
            print(f"Prompt: {prompt}")
            print(f"Critical words: {critical_words}")
            print(f"Critical indices: {crit_indices}")

        # Prepare latents/scheduler
        latents = torch.randn((1, self.c.dit_model.latent_dim), device=self.device)
        self.c.scheduler.set_timesteps(steps)

        best_seed = None
        best_score = -1.0

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                seed = torch.randint(0, 10**9, (1,)).item()
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                if verbose:
                    print(f"Retry {attempt}: seed {seed}")

            store = self._capture_attn()

            latents = torch.randn((1, self.c.dit_model.latent_dim), device=self.device)
            for i, t in enumerate(tqdm(self.c.scheduler.timesteps, total=steps, desc=f"Attempt {attempt+1}")):
                with torch.no_grad():
                    latents = self.c.dit_model(latents, t, text_embeddings, cfg_scale)

                if i == self.check_step:
                    # Compute avg per-token attention
                    token_attn_all = []
                    for att in store:
                        # attn expected [B, H, Q, K]; average over Q
                        if att.dim() == 4:
                            token_attn_all.append(att.mean(dim=2))  # [B, H, K]
                        elif att.dim() == 3:
                            token_attn_all.append(att.mean(dim=1))  # [B*H, K]
                    if len(token_attn_all) > 0:
                        avg = torch.stack([a.reshape(-1, a.shape[-1]) for a in token_attn_all]).mean(0).mean(0)
                        missing = []
                        scores = {}
                        for idx, name in zip(crit_indices, critical_words):
                            if idx < avg.shape[0]:
                                s = float(avg[idx].item())
                                scores[name] = s
                                if s < self.attention_threshold:
                                    missing.append(idx)
                        score_sum = sum(scores.values()) if scores else 0.0
                        if score_sum > best_score:
                            best_score = score_sum
                            best_seed = torch.initial_seed()
                        self._unpatch_attn()
                        if len(missing) == 0:
                            if verbose:
                                print("✓ Adequate attention found (Phase 1)")
                            return latents, {"scores": scores}
                        else:
                            break

            self._unpatch_attn()

        # Phase 2: boosting
        if verbose:
            print("No adequate seed; switching to attention boosting (Phase 2)")
        if best_seed is not None:
            torch.manual_seed(best_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(best_seed)

        self.booster.set_indices(crit_indices)
        self._patch_boost()

        latents = torch.randn((1, self.c.dit_model.latent_dim), device=self.device)
        start_i = int(steps * self.start_step_ratio)
        end_i = int(steps * self.end_step_ratio)

        for i, t in enumerate(tqdm(self.c.scheduler.timesteps, total=steps, desc="Boosting")):
            self.booster.enable() if start_i <= i <= end_i else self.booster.disable()
            with torch.no_grad():
                latents = self.c.dit_model(latents, t, text_embeddings, cfg_scale)

        self._unpatch_attn()
        return latents, {}
