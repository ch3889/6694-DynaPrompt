"""
Fresh sampler that integrates early token-attention checking and optional
attention boosting with the CompVis Stable Diffusion repo.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange

from .attention_boost import AttentionBooster


class DynapromptNewSampler:
    def __init__(
        self,
        ddim_sampler,
        model,
        tokenizer,
        device: str = "cuda",
        check_step: int = 5,
        attention_threshold: float = 0.05,
        max_retries: int = 10,
        boost_factor: float = 6.0,
        start_step_ratio: float = 0.0,
        end_step_ratio: float = 0.5,
    ):
        self.ddim_sampler = ddim_sampler
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.check_step = check_step
        self.attention_threshold = attention_threshold
        self.max_retries = max_retries
        self.booster = AttentionBooster(boost_factor=boost_factor)
        self.start_step_ratio = start_step_ratio
        self.end_step_ratio = end_step_ratio

        self._original_forwards = {}

    def _get_token_indices(self, token_ids: torch.Tensor, words: List[str]) -> List[int]:
        indices = []
        for word in words:
            toks = self.tokenizer(word, add_special_tokens=False)["input_ids"]
            for i, tid in enumerate(token_ids):
                if tid in toks:
                    indices.append(i)
        return sorted(list(set(indices)))

    def _save_original_forwards(self, unet):
        counter = [0]

        def walk(net):
            if net.__class__.__name__ == "CrossAttention":
                mid = counter[0]
                if hasattr(net.forward, "__self__"):
                    self._original_forwards[mid] = net.forward
                counter[0] += 1
            elif hasattr(net, "children"):
                for c in net.children():
                    walk(c)
        walk(unet)

    def _unpatch_attention(self, unet):
        counter = [0]
        def walk(net):
            if net.__class__.__name__ == "CrossAttention":
                mid = counter[0]
                if mid in self._original_forwards:
                    net.forward = self._original_forwards[mid]
                counter[0] += 1
            elif hasattr(net, "children"):
                for c in net.children():
                    walk(c)
        walk(unet)

    def _patch_attention_capture(self, unet, store_list: list):
        def modify(module):
            def new_forward(x, context=None, mask=None):
                h = module.heads
                q = module.to_q(x)
                from ldm.modules.attention import default, exists, repeat
                context = default(context, x)
                k = module.to_k(context)
                v = module.to_v(context)

                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * module.scale

                if mask is not None and exists(mask):
                    m = rearrange(mask, 'b ... -> b (...)')
                    max_neg = -torch.finfo(sim.dtype).max
                    m = repeat(m, 'b j -> (b h) () j', h=h)
                    sim.masked_fill_(~m, max_neg)

                attn = sim.softmax(dim=-1)

                # Capture only cross-attention
                is_cross = context is not None and context.shape != x.shape
                if is_cross:
                    store_list.append(attn.detach().cpu())

                out = torch.einsum('b i j, b j d -> b i d', attn, v)
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                return module.to_out(out)

            module.forward = new_forward

        def walk(net):
            if net.__class__.__name__ == "CrossAttention":
                modify(net)
            elif hasattr(net, "children"):
                for c in net.children():
                    walk(c)
        walk(unet)

    def _patch_attention_boost(self, unet):
        def modify(module):
            def new_forward(x, context=None, mask=None):
                h = module.heads
                q = module.to_q(x)
                from ldm.modules.attention import default, exists, repeat
                context = default(context, x)
                k = module.to_k(context)
                v = module.to_v(context)

                q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * module.scale

                if mask is not None and exists(mask):
                    m = rearrange(mask, 'b ... -> b (...)')
                    max_neg = -torch.finfo(sim.dtype).max
                    m = repeat(m, 'b j -> (b h) () j', h=h)
                    sim.masked_fill_(~m, max_neg)

                attn = sim.softmax(dim=-1)

                # Only boost for cross-attention
                is_cross = context is not None and context.shape != x.shape
                if is_cross:
                    attn = self.booster.apply(attn)

                out = torch.einsum('b i j, b j d -> b i d', attn, v)
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                return module.to_out(out)

            module.forward = new_forward

        def walk(net):
            if net.__class__.__name__ == "CrossAttention":
                modify(net)
            elif hasattr(net, "children"):
                for c in net.children():
                    walk(c)
        walk(unet)

    def sample(
        self,
        prompt: str,
        shape: List[int],
        steps: int = 50,
        cfg_scale: float = 7.5,
        critical_words: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        # Tokenize
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_embeddings = self.model.cond_stage_model.transformer(text_input.input_ids)[0]
            uncond_input = self.tokenizer([""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
            uncond_embeddings = self.model.cond_stage_model.transformer(uncond_input.input_ids)[0]

        # Derive critical words if not provided (simple heuristic)
        if critical_words is None:
            ignore = {"a","an","the","of","in","on","at","to","for","with","by","from","is","are","was","were"}
            tokens = [w.strip('.,!?;:').lower() for w in prompt.split()]
            critical_words = [w for w in tokens if w and w not in ignore and len(w) > 2]

        critical_indices = self._get_token_indices(text_input.input_ids[0], critical_words)

        if verbose:
            print("dynaprompt_new: Early detection + fallback boosting")
            print(f"Prompt: {prompt}")
            print(f"Critical words: {critical_words}")
            print(f"Critical indices: {critical_indices}")

        # Prepare schedule
        self.ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=0.0, verbose=False)
        device = self.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        timesteps = self.ddim_sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        intermediates = {"x_inter": [img], "pred_x0": [img]}

        # Phase 1: Early detection, try seeds
        best_seed = None
        best_score_sum = -1.0
        best_attempt = None

        # Save originals
        self._original_forwards = {}
        self._save_original_forwards(self.model.model.diffusion_model)

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                retry_seed = torch.randint(0, 10**9, (1,)).item()
                torch.manual_seed(retry_seed)
                torch.cuda.manual_seed(retry_seed)
                if verbose:
                    print(f"Retry {attempt}: seed {retry_seed}")

            # Patch to capture attention
            store = []
            self._patch_attention_capture(self.model.model.diffusion_model, store)

            img = torch.randn(shape, device=device)
            composition_checked = False
            composition_success = False
            attention_scores = {}

            for i, step in enumerate(tqdm(time_range, total=total_steps, desc=f"Attempt {attempt+1}")):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)

                with torch.no_grad():
                    outs = self.ddim_sampler.p_sample_ddim(
                        img, text_embeddings, ts,
                        index=index,
                        use_original_steps=False,
                        unconditional_guidance_scale=cfg_scale,
                        unconditional_conditioning=uncond_embeddings,
                    )
                    img, pred_x0 = outs

                if i == self.check_step and not composition_checked:
                    composition_checked = True
                    # Compute avg token attention
                    token_attn_all = []
                    for att in store:
                        token_attn_all.append(att.mean(dim=1))  # [B*H, tokens]
                    if len(token_attn_all) > 0:
                        avg_token_attn = torch.stack(token_attn_all).mean(0).mean(0)  # [tokens]
                        missing = []
                        for idx, name in zip(critical_indices, critical_words):
                            if idx < avg_token_attn.shape[0]:
                                score = float(avg_token_attn[idx].item())
                                attention_scores[name] = score
                                if score < self.attention_threshold:
                                    missing.append(idx)
                        score_sum = sum(attention_scores.values()) if attention_scores else 0.0
                        if score_sum > best_score_sum:
                            best_score_sum = score_sum
                            best_seed = torch.initial_seed()
                            best_attempt = (img.clone(), pred_x0.clone())
                        if len(missing) == 0:
                            composition_success = True
                            break
                        else:
                            # Abort early to retry another seed
                            break

                if index % 10 == 0 or index == total_steps - 1:
                    intermediates["x_inter"].append(img)
                    intermediates["pred_x0"].append(pred_x0)

            # Restore original after each attempt
            self._unpatch_attention(self.model.model.diffusion_model)

            if composition_success:
                if verbose:
                    print("✓ Found seed with adequate attention (Phase 1)")
                return img, intermediates

        # Phase 2: Boosting using best seed
        if verbose:
            print("No adequate seed found. Switching to attention boosting (Phase 2)")

        self._unpatch_attention(self.model.model.diffusion_model)
        if best_seed is not None:
            torch.manual_seed(best_seed)
            torch.cuda.manual_seed(best_seed)

        # Identify indices once more from prompt
        self.booster.set_indices(critical_indices)

        start_step = int(steps * self.start_step_ratio)
        end_step = int(steps * self.end_step_ratio)

        # Patch booster
        self._patch_attention_boost(self.model.model.diffusion_model)

        img = torch.randn(shape, device=self.model.betas.device)
        for i, step in enumerate(tqdm(time_range, total=total_steps, desc="Boosting")):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.model.betas.device, dtype=torch.long)

            if start_step <= i <= end_step:
                self.booster.enable()
            else:
                self.booster.disable()

            with torch.no_grad():
                outs = self.ddim_sampler.p_sample_ddim(
                    img, text_embeddings, ts,
                    index=index,
                    use_original_steps=False,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning=uncond_embeddings,
                )
                img, pred_x0 = outs

            if index % 10 == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        # Cleanup
        self._unpatch_attention(self.model.model.diffusion_model)
        self._original_forwards = {}

        return img, intermediates
