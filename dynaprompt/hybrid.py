"""
DynaPrompt Hybrid: Combines zk2295 (Embedding) + ch3889 (Attention) Approaches

This module implements a hybrid feedback system that leverages both:
1. zk2295: External CLIP-based embedding updates (global + selective)
2. ch3889: Internal U-Net attention amplification

Architecture:
    For each feedback step:
        Phase 1 (zk2295): Decode image → CLIP analysis → Update embeddings
        Phase 2 (ch3889): Pass updated embeddings → Amplify attention to weak tokens
        
Result: Double reinforcement of underrepresented concepts
"""

# Fix pytorch_lightning compatibility issue
try:
    import pytorch_lightning
except ImportError:
    pass
else:
    if not hasattr(pytorch_lightning, 'utilities') or not hasattr(pytorch_lightning.utilities, 'distributed'):
        import pytorch_lightning.utilities
        class _DistributedShim:
            @staticmethod
            def rank_zero_only(fn):
                return fn
        pytorch_lightning.utilities.distributed = _DistributedShim()

import torch
import yaml
import numpy as np
from tqdm import tqdm
from .core import DynaPrompt
from .sd_loader import load_sd_model
from .attention_modifier import AttentionModifier
from .adaptive_reweighting import AdaptiveReweighter

class HybridDynaPrompt:
    """
    Hybrid DynaPrompt combining embedding updates (zk2295) with attention boosting (ch3889)
    
    This integrates two complementary techniques:
    - Embedding feedback: Improves WHAT SD receives as input
    - Attention boosting: Improves HOW SD processes that input
    """
    
    def __init__(self, config_path='configs/dynaprompt_config.yaml', ckpt_path=None, device=None):
        """
        Initialize Hybrid DynaPrompt pipeline
        
        Args:
            config_path: Path to configuration file
            ckpt_path: Path to SD checkpoint (uses default if None)
            device: Torch device (auto-detected if None)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        print(f"Initializing Hybrid DynaPrompt Pipeline on {self.device}...")
        
        # Load Stable Diffusion model
        print("Loading Stable Diffusion model...")
        self.sd = load_sd_model(ckpt_path=ckpt_path, device=self.device)
        
        # Initialize Phase 1: Embedding feedback (zk2295)
        print("Initializing embedding feedback system (zk2295)...")
        clip_model = self.config.get('clip', {}).get('model_name', 'openai/clip-vit-base-patch32')
        self.dynaprompt = DynaPrompt(clip_model_name=clip_model, device=self.device)
        
        # Initialize Phase 2: Attention boosting (ch3889)
        print("Initializing attention boosting system (ch3889)...")
        self.attention_modifier = AttentionModifier(
            tokenizer=self.sd.model.cond_stage_model.tokenizer,
            boost_factor=self.config.get('attention', {}).get('boost_factor', 1.3),
            threshold=self.config.get('attention', {}).get('threshold', 0.05),
            start_step=self.config.get('attention', {}).get('start_step', 0),
            end_step=self.config.get('attention', {}).get('end_step', 20)
        )
        
        # Patch U-Net attention layers for Phase 2
        print("Patching U-Net attention layers...")
        self.attention_modifier.patch_attention_layers(self.sd.model.model)
        
        # Initialize adaptive reweighting
        print("Initializing adaptive reweighting system...")
        reweight_config = self.config.get('adaptive_reweighting', {})
        self.reweighter = AdaptiveReweighter(
            initial_alpha=self.config.get('prompt_update', {}).get('update_alpha', 0.08),
            initial_boost=self.config.get('attention', {}).get('boost_factor', 1.3),
            momentum=reweight_config.get('momentum', 0.9),
            adaptation_rate=reweight_config.get('adaptation_rate', 0.1)
        )
        
        print("✓ Hybrid DynaPrompt initialized successfully!")
    
    def map_concepts_to_token_positions(self, weak_tokens, prompt, tokenizer):
        """
        Map weak concepts to token positions for attention boosting
        
        Args:
            weak_tokens: Dict of {concept: score}
            prompt: Original text prompt
            tokenizer: SD tokenizer
            
        Returns:
            List of token indices to boost
        """
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        token_indices = []
        
        # Handle both dict and list formats
        if isinstance(weak_tokens, dict):
            weak_concepts = list(weak_tokens.keys())
        else:
            weak_concepts = weak_tokens
        
        for weak_concept in weak_concepts:
            concept_words = weak_concept.split()
            
            # Search for concept in prompt
            for i in range(len(words)):
                match = True
                for j, cword in enumerate(concept_words):
                    if i + j >= len(words) or words[i + j] != cword:
                        match = False
                        break
                
                if match:
                    # Add token positions (offset by 1 for BOS token)
                    for j in range(len(concept_words)):
                        token_idx = i + j + 1
                        if token_idx not in token_indices:
                            token_indices.append(token_idx)
        
        return token_indices
    
    def compute_token_clip_scores(self, image, prompt, weak_tokens):
        """Compute CLIP score for each weak token to determine boost strength
        
        Args:
            image: Current generated image
            prompt: Text prompt
            weak_tokens: Dict or list of weak token concepts
            
        Returns:
            Dict mapping token concept to CLIP score
        """
        token_scores = {}
        
        # Handle both dict and list formats
        if isinstance(weak_tokens, dict):
            concepts = list(weak_tokens.keys())
        else:
            concepts = weak_tokens
        
        # Compute CLIP score for each weak token
        for concept in concepts:
            try:
                score = self.dynaprompt.compute_clipscore(image, concept)
                token_scores[concept] = score
            except:
                token_scores[concept] = 0.0
        
        return token_scores
    
    def compute_adaptive_boost_factor(self, token_clip_score, base_boost=1.3, scene_difficulty='standard'):
        """Compute adaptive boost using smooth inverse scaling based on CLIP score
        
        Principle: Boost inversely proportional to current alignment strength
        Uses smooth gradient instead of hard thresholds for stable optimization
        Adapts boost intensity based on scene difficulty
        
        Args:
            token_clip_score: CLIP score for this specific token (15-35 typical range)
            base_boost: Base boost factor from config
            scene_difficulty: 'easy' for well-composed scenes, 'standard' for difficult ones
            
        Returns:
            Adaptive boost factor with smooth gradient (1.0x to base_boost×multiplier)
        """
        # Define CLIP score range observed during generation
        min_score = 15.0  # Weakest typical CLIP score
        max_score = 35.0  # Strongest typical CLIP score
        
        # Adaptive boost multiplier based on scene difficulty
        # Easy scenes (baseline already good): gentler boost to avoid over-correction
        # Standard scenes (baseline needs help): standard boost for correction
        if scene_difficulty == 'easy':
            boost_multiplier = 1.2  # Gentler: 1.3 × 1.2 = 1.56x max
        else:
            boost_multiplier = 1.5  # Standard: 1.3 × 1.5 = 1.95x max
        
        # Define boost range
        max_boost = base_boost * boost_multiplier
        min_boost = 1.0  # No boost for strong tokens
        
        # Normalize score to [0, 1] range
        normalized = (token_clip_score - min_score) / (max_score - min_score)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to valid range
        
        # Inverse linear interpolation: low score → high boost, high score → low boost
        boost = max_boost - (max_boost - min_boost) * normalized
        
        return boost
    
    def decompose_prompt_by_stage(self, prompt, current_step, total_steps):
        """Decompose prompt into stages for progressive concept building
        
        Stage 1 (0-33%): Establish main subjects
        Stage 2 (34-66%): Add attributes (colors, sizes, materials)
        Stage 3 (67-100%): Add objects and spatial relationships
        
        Args:
            prompt: Full text prompt
            current_step: Current denoising step
            total_steps: Total number of steps
            
        Returns:
            Dict with token emphasis weights
        """
        # Calculate stage (0.0 to 1.0)
        progress = current_step / total_steps
        
        # Tokenize prompt
        words = prompt.lower().split()
        
        # Token categories
        subjects = ['cat', 'dog', 'table', 'car', 'person', 'bird', 'animal']
        attributes = ['red', 'blue', 'green', 'yellow', 'orange', 'white', 'black',
                     'tiny', 'small', 'large', 'fluffy', 'wooden', 'metal', 'golden']
        objects = ['hat', 'vase', 'flower', 'ball', 'apple', 'banana', 'carrot', 'umbrella']
        spatial = ['wearing', 'next', 'arranged', 'row', 'behind', 'front', 'sitting', 'playing']
        
        # Compute emphasis weights based on stage (VERY GENTLE - max 1.3x)
        emphasis = {}
        
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?')
            
            if progress < 0.33:  # Stage 1: Focus on subjects
                if any(subj in clean_word for subj in subjects):
                    emphasis[i] = 2.0  # Strong emphasis on current stage
                elif any(attr in clean_word for attr in attributes):
                    emphasis[i] = 0.8  # Suppress non-stage concepts
                elif any(obj in clean_word for obj in objects):
                    emphasis[i] = 0.8  # Suppress non-stage concepts
                else:
                    emphasis[i] = 1.0
                    
            elif progress < 0.66:  # Stage 2: Add attributes
                if any(subj in clean_word for subj in subjects):
                    emphasis[i] = 1.0  # Maintain established subjects
                elif any(attr in clean_word for attr in attributes):
                    emphasis[i] = 2.0  # Strong emphasis on current stage
                elif any(obj in clean_word for obj in objects):
                    emphasis[i] = 0.8  # Suppress not-yet-focused
                else:
                    emphasis[i] = 1.0
                    
            else:  # Stage 3: Add objects and spatial
                if any(subj in clean_word for subj in subjects):
                    emphasis[i] = 1.0  # Maintain subjects
                elif any(attr in clean_word for attr in attributes):
                    emphasis[i] = 1.0  # Maintain attributes
                elif any(obj in clean_word for obj in objects):
                    emphasis[i] = 2.0  # Strong emphasis on objects
                elif any(spat in clean_word for spat in spatial):
                    emphasis[i] = 1.8  # Strong emphasis on spatial
                else:
                    emphasis[i] = 1.0
        
        return emphasis
    
    def generate_negative_prompts(self, weak_tokens, token_clip_scores):
        """Generate dynamic negative prompts based on missing concepts
        
        Args:
            weak_tokens: Dict or list of weak tokens (can be phrases like "red hat")
            token_clip_scores: CLIP scores for each token
            
        Returns:
            Negative prompt string
        """
        negatives = []
        
        # Handle both dict and list formats
        if isinstance(weak_tokens, dict):
            concepts = list(weak_tokens.keys())
        else:
            concepts = weak_tokens
        
        # Mapping of KEYWORDS (not phrases) to negatives
        negative_map = {
            'hat': 'no hat, bare head',
            'vase': 'no vase',
            'wearing': 'not wearing, bare',
            'red': 'wrong color, not red',
            'blue': 'wrong color, not blue',
            'green': 'wrong color, not green',
            'yellow': 'wrong color, not yellow',
            'orange': 'wrong color, not orange',
            'banana': 'no banana',
            'apple': 'no apple',
            'carrot': 'no carrot',
            'flower': 'no flower',
            'arranged': 'scattered, disorganized',
            'row': 'piled together, not in a row',
            'tiny': 'large, oversized',
            'fluffy': 'smooth, sleek'
        }
        
        # Extract keywords from phrases and check against negative map
        # E.g., "red hat" → check both "red" and "hat"
        seen_keys = set()
        for concept in concepts:
            score = token_clip_scores.get(concept, 0)
            if score < 15:  # Only very missing tokens (conservative threshold)
                # Split phrase into words
                words = concept.lower().split()
                for word in words:
                    # Check if this word matches any key in negative_map
                    for key, neg in negative_map.items():
                        if key in word and key not in seen_keys:
                            negatives.append(neg)
                            seen_keys.add(key)
                            break
        
        # Combine into single negative prompt
        if negatives:
            return ', '.join(negatives[:5])  # Limit to 5 to avoid overload
        else:
            return ''
    
    def pre_analyze_prompt(self, prompt):
        """Pre-analyze prompt to identify potentially weak tokens before generation
        
        Uses heuristics based on:
        1. Token rarity (rare tokens tend to be underrepresented)
        2. Critical visual attributes (colors, sizes, materials)
        3. Key objects that SD often misses
        
        Args:
            prompt: Text prompt string
            
        Returns:
            List of token indices that are likely to be weak
        """
        # Tokenize prompt
        tokens = self.sd.model.cond_stage_model.tokenizer.encode(prompt)
        text_tokens = self.sd.model.cond_stage_model.tokenizer.convert_ids_to_tokens(tokens)
        
        weak_indices = []
        
        # HIGH PRIORITY: Tokens SD commonly misses (very selective)
        high_priority = [
            'red', 'blue', 'yellow', 'orange', 'purple', 'pink',  # Specific colors
            'tiny', 'small',  # Size modifiers
            'hat', 'vase', 'flower',  # Small objects
            'wearing', 'arranged'  # Actions
        ]
        
        # MEDIUM PRIORITY: Context-dependent tokens (boost only if multiple present)
        medium_priority = [
            'green', 'white', 'golden',  # Common colors
            'fluffy', 'wooden',  # Materials/textures
            'apple', 'banana', 'carrot',  # Food items
            'next', 'row'  # Spatial
        ]
        
        print("\nToken analysis:")
        high_priority_count = 0
        medium_priority_count = 0
        
        for idx, token in enumerate(text_tokens[1:-1], start=1):  # Skip BOS/EOS
            token_str = token.replace('</w>', '').lower()
            
            # Check high priority tokens - always boost
            if any(pattern == token_str or token_str.startswith(pattern) for pattern in high_priority):
                weak_indices.append(idx)
                high_priority_count += 1
                print(f"  Token {idx}: '{token_str}' -> HIGH PRIORITY BOOST")
            # Check medium priority - track but don't boost yet
            elif any(pattern == token_str or token_str.startswith(pattern) for pattern in medium_priority):
                medium_priority_count += 1
        
        # Only boost medium priority if we have 3+ of them (compositionally complex prompt)
        if medium_priority_count >= 3:
            for idx, token in enumerate(text_tokens[1:-1], start=1):
                token_str = token.replace('</w>', '').lower()
                if any(pattern == token_str or token_str.startswith(pattern) for pattern in medium_priority):
                    if idx not in weak_indices:
                        weak_indices.append(idx)
                        print(f"  Token {idx}: '{token_str}' -> MEDIUM PRIORITY BOOST")
        
        print(f"\n→ Selected {len(weak_indices)} critical tokens for boosting")
        print(f"   (High priority: {high_priority_count}, Medium priority: {medium_priority_count})")
        return weak_indices
    
    def generate(
        self,
        prompt,
        height=512,
        width=512,
        steps=50,
        cfg_scale=7.5,
        sampler_type='ddim',
        eta=0.0,
        seed=None,
        embedding_feedback=True,
        attention_feedback=True
    ):
        """
        Generate image with hybrid DynaPrompt feedback
        
        Args:
            prompt: Text prompt
            height: Image height (default 512)
            width: Image width (default 512)
            steps: Number of denoising steps (default 50)
            cfg_scale: Classifier-free guidance scale (default 7.5)
            sampler_type: Sampler type ('ddim' or 'plms')
            eta: DDIM eta parameter
            seed: Random seed (None for random)
            embedding_feedback: Enable Phase 1 (zk2295) embedding updates
            attention_feedback: Enable Phase 2 (ch3889) attention boosting
            
        Returns:
            dict with 'image', 'metrics', 'embedding_trajectory'
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        print(f"\n{'='*60}")
        print(f"HYBRID DYNAPROMPT GENERATION")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Steps: {steps}, CFG: {cfg_scale}")
        print(f"Embedding Feedback (zk2295): {'✓' if embedding_feedback else '✗'}")
        print(f"Attention Boosting (ch3889): {'✓' if attention_feedback else '✗'}")
        print(f"{'='*60}\n")
        
        import time
        start_time = time.time()
        
        # Reset adaptive reweighting stats for this generation
        self.reweighter.reset()
        
        # Encode prompt
        c_original = self.sd.encode_text([prompt])
        uc = self.sd.encode_text([""])
        c = c_original.clone()  # Start with original, update during feedback
        
        # Create sampler
        sampler = self.sd.create_sampler(sampler_type)
        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)
        
        # Initialize latent
        shape = [1, 4, height // 8, width // 8]
        latents = torch.randn(shape, device=self.device)
        
        # Configuration
        feedback_freq = self.config.get('feedback', {}).get('feedback_frequency', 4)
        feedback_start = self.config.get('feedback', {}).get('feedback_start_step', 5)
        feedback_end = self.config.get('feedback', {}).get('feedback_end_step', 42)
        
        # Storage
        metrics_history = []
        embedding_trajectory = []
        weak_tokens_history = []
        
        # PRE-GENERATION ANALYSIS: Identify weak tokens upfront
        pre_weak_indices = []
        if attention_feedback:
            print("\nPre-analyzing prompt for potentially weak tokens...")
            pre_weak_indices = self.pre_analyze_prompt(prompt)
            if pre_weak_indices:
                print(f"Enabling proactive attention boost for {len(pre_weak_indices)} tokens from step 0")
                self.attention_modifier.set_underrepresented_indices(pre_weak_indices)
        
        # Denoising loop
        timesteps = sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        
        print(f"Running hybrid denoising with {total_steps} steps...")
        iterator = tqdm(enumerate(time_range), total=total_steps, desc="Hybrid DynaPrompt")
        
        for i, step in iterator:
            index = total_steps - i - 1
            ts = torch.full((1,), step, device=self.device, dtype=torch.long)
            
            # === HYBRID FEEDBACK (zk2295 + ch3889) ===
            # Trigger at feedback_start, then every feedback_freq steps
            if (i >= feedback_start and i < feedback_end and 
                (i - feedback_start) % feedback_freq == 0):
                
                # Decode current latent to get intermediate image
                with torch.no_grad():
                    latents_scaled = 1 / 0.18215 * latents
                    intermediate_image = self.sd.model.first_stage_model.decode(latents_scaled)
                    intermediate_image = torch.clamp((intermediate_image + 1.0) / 2.0, min=0.0, max=1.0)
                
                # === PHASE 1: Embedding Feedback (zk2295) ===
                if embedding_feedback:
                    # Get stage-based emphasis weights
                    stage_emphasis = self.decompose_prompt_by_stage(prompt, i, total_steps)
                    
                    # Use MAXIMUM emphasis of focused tokens (not average which dilutes)
                    # Only consider tokens with emphasis >= 1.2 (actively boosted)
                    boosted_values = [v for v in stage_emphasis.values() if v >= 1.2]
                    if boosted_values:
                        max_emphasis = max(boosted_values)
                    else:
                        max_emphasis = 1.0  # No boosting this stage
                    
                    # Determine current stage for logging
                    progress = i / total_steps
                    if progress < 0.33:
                        stage_name = "Stage 1 (Subjects)"
                    elif progress < 0.66:
                        stage_name = "Stage 2 (Attributes)"
                    else:
                        stage_name = "Stage 3 (Objects)"
                    
                    print(f"\n  [Step {i}/{total_steps}] {stage_name}, Emphasis: {max_emphasis:.2f}x")
                    
                    # Use zk2295's CLIP gradient feedback with STAGE-ADJUSTED alpha
                    base_alpha = self.config.get('prompt_update', {}).get('update_alpha', 0.08)
                    base_boost = self.config.get('attention', {}).get('boost_factor', 1.3)
                    adjusted_alpha = base_alpha * max_emphasis  # Scale by stage emphasis
                    
                    print(f"    Alpha: {base_alpha:.3f} * {max_emphasis:.2f} = {adjusted_alpha:.3f}")
                    
                    feedback_result = self.dynaprompt.feedback_loop(
                        prompt=prompt,
                        current_embedding=c,
                        generated_image=intermediate_image,
                        step=i,
                        use_per_token=True,
                        alpha=adjusted_alpha,
                        boost_factor=base_boost  # Pass boost_factor from config
                    )
                    
                    # Update embedding with CLIP guidance
                    c = feedback_result['updated_embedding']
                    weak_tokens = feedback_result['weak_tokens']
                    
                    print(f"    CLIP Score: {feedback_result['clip_score']:.2f}")
                    print(f"    Weak tokens: {list(weak_tokens.keys()) if isinstance(weak_tokens, dict) else weak_tokens}")
                    
                    # Detect scene difficulty for adaptive boost intensity
                    # If early CLIP score is high (>25), scene is already well-composed
                    if i <= 10 and feedback_result['clip_score'] >= 25:
                        scene_difficulty = 'easy'  # Use gentler boost (1.2x multiplier)
                    else:
                        scene_difficulty = 'standard'  # Use standard boost (1.5x multiplier)
                        
                    metrics_history.append({
                        'step': i,
                        'clipscore': feedback_result['clip_score'],
                        'weak_tokens': weak_tokens,
                        'stage_emphasis': max_emphasis
                    })
                    
                    # === PHASE 1.5: Dynamic Negative Prompts ===
                    # Generate negative prompts for missing concepts
                    negative_prompt = ""
                    if weak_tokens:
                        # Filter to individual tokens only (1-2 words max), not phrases
                        if isinstance(weak_tokens, dict):
                            individual_tokens = {k: v for k, v in weak_tokens.items() if len(k.split()) <= 2}
                        else:
                            individual_tokens = [t for t in weak_tokens if len(t.split()) <= 2]
                        
                        if individual_tokens:
                            token_clip_scores = self.compute_token_clip_scores(
                                intermediate_image, prompt, individual_tokens
                            )
                            
                            negative_prompt = self.generate_negative_prompts(
                                individual_tokens, token_clip_scores
                            )
                            
                            if negative_prompt:
                                # Encode negative prompt and strengthen unconditional guidance
                                uc_negative = self.sd.encode_text([negative_prompt])
                                # GENTLE blending: 0.8 original + 0.2 negative (subtle guidance)
                                uc = 0.8 * uc + 0.2 * uc_negative
                    
                    # Always log negative prompt status (even if empty)
                    metrics_history[-1]['negative_prompt'] = negative_prompt if negative_prompt else "(none)"
                    print(f"    Negative prompt: '{negative_prompt if negative_prompt else '(none)'}'")                    # === PHASE 2: Attention Boosting (ch3889) ===
                    if attention_feedback and weak_tokens:
                        # Compute per-token CLIP scores for adaptive boosting
                        token_clip_scores = self.compute_token_clip_scores(
                            intermediate_image, prompt, weak_tokens
                        )
                        
                        # Map weak tokens to indices
                        token_indices = self.map_concepts_to_token_positions(
                            weak_tokens, prompt, self.sd.model.cond_stage_model.tokenizer
                        )
                        
                        if token_indices:
                            # ADAPTIVE: Set per-token boost factors with budget balancing
                            # Principle: Redistribute attention budget based on relative weakness
                            # instead of amplifying all weak tokens (which can exceed budget)
                            
                            base_boost = self.config.get('attention', {}).get('boost_factor', 1.3)
                            
                            # Get concepts for each token index
                            if isinstance(weak_tokens, dict):
                                concepts = list(weak_tokens.keys())
                            else:
                                concepts = weak_tokens
                            
                            # Step 1: Calculate raw adaptive boosts for each concept
                            raw_boosts = {}
                            concept_to_indices = {}
                            
                            for concept in concepts:
                                clip_score = token_clip_scores.get(concept, 0.0)
                                raw_boost = self.compute_adaptive_boost_factor(
                                    clip_score, base_boost, scene_difficulty
                                )
                                
                                # Find token indices for this concept
                                concept_indices = self.map_concepts_to_token_positions(
                                    {concept: 0} if isinstance(weak_tokens, dict) else [concept],
                                    prompt, 
                                    self.sd.model.cond_stage_model.tokenizer
                                )
                                
                                if concept_indices:
                                    raw_boosts[concept] = raw_boost
                                    concept_to_indices[concept] = concept_indices
                            
                            # Step 2: Normalize boosts to stay within attention budget
                            # Total attention should not exceed reasonable multiplier (1.5x of base)
                            if raw_boosts:
                                total_raw_boost = sum(raw_boosts.values())
                                num_concepts = len(raw_boosts)
                                max_total_budget = base_boost * num_concepts  # Each concept can get base_boost on average
                                
                                # If total exceeds budget, normalize down
                                if total_raw_boost > max_total_budget:
                                    normalization_factor = max_total_budget / total_raw_boost
                                else:
                                    normalization_factor = 1.0
                                
                                # Apply normalized boosts
                                adaptive_boosts = {}
                                for concept, raw_boost in raw_boosts.items():
                                    normalized_boost = raw_boost * normalization_factor
                                    # Ensure minimum boost of 1.0 (no suppression)
                                    normalized_boost = max(1.0, normalized_boost)
                                    
                                    for idx in concept_to_indices[concept]:
                                        adaptive_boosts[idx] = normalized_boost
                                
                                # Set token-specific boost factors in attention modifier
                                self.attention_modifier.set_underrepresented_indices(token_indices)
                                self.attention_modifier.set_adaptive_boosts(adaptive_boosts)
                                self.attention_modifier.enable()
                                
                                # Store for metrics
                                metrics_history[-1]['adaptive_boosts'] = adaptive_boosts
                                metrics_history[-1]['token_clip_scores'] = token_clip_scores
                                metrics_history[-1]['normalization_factor'] = normalization_factor
                            else:
                                self.attention_modifier.disable()
                        else:
                            self.attention_modifier.disable()
                    else:
                        self.attention_modifier.disable()
            
            # Check if attention should be active at this step
            if attention_feedback and not self.attention_modifier.should_modify(i):
                self.attention_modifier.disable()
            
            # Regular denoising step (attention hooks active if enabled)
            latents = sampler.p_sample_ddim(
                x=latents,
                c=c,
                t=ts,
                index=index,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc
            )[0]
        
        # Disable attention modification after generation
        self.attention_modifier.disable()
        
        # Final decode
        print("\nDecoding final image...")
        with torch.no_grad():
            latents_scaled = 1 / 0.18215 * latents
            image = self.sd.model.first_stage_model.decode(latents_scaled)
            image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        
        # Compute final metrics
        print("Computing final metrics...")
        final_clipscore = self.dynaprompt.compute_clipscore(image, prompt)
        
        final_analysis = self.dynaprompt.compute_per_token_alignment(
            image, prompt, self.sd.model.cond_stage_model.tokenizer
        )
        final_compositional = self.dynaprompt.compute_compositional_accuracy(
            image, prompt
        )
        
        generation_time = time.time() - start_time
        
        # Compile statistics
        feedback_stats = {
            'num_feedback_steps': len(metrics_history),
            'feedback_applied': len(metrics_history) > 0,
            'avg_clipscore': sum(m['clipscore'] for m in metrics_history) / len(metrics_history) if metrics_history else 0,
            'negative_prompts_used': sum(1 for m in metrics_history if m.get('negative_prompt')) if metrics_history else 0,
            'avg_adaptive_boost': sum(sum(m.get('adaptive_boosts', {}).values()) / max(len(m.get('adaptive_boosts', {})), 1) 
                                     for m in metrics_history) / max(len(metrics_history), 1) if metrics_history else 0
        }
        
        print(f"\n{'='*60}")
        print(f"HYBRID GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Time: {generation_time:.2f}s")
        print(f"Final CLIP Score: {final_clipscore:.4f}")
        print(f"Compositional Accuracy: {final_compositional:.4f}")
        print(f"Feedback steps: {feedback_stats['num_feedback_steps']}")
        if feedback_stats.get('negative_prompts_used', 0) > 0:
            print(f"Negative prompts used: {feedback_stats['negative_prompts_used']}")
        if feedback_stats.get('avg_adaptive_boost', 0) > 0:
            print(f"Avg adaptive boost: {feedback_stats['avg_adaptive_boost']:.2f}x")
        print(f"{'='*60}\n")
        
        return {
            'image': image,
            'final_clipscore': final_clipscore,
            'compositional_accuracy': final_compositional,
            'token_analysis': final_analysis,
            'metrics_history': metrics_history,
            'embedding_trajectory': embedding_trajectory,
            'weak_tokens_history': weak_tokens_history,
            'adaptive_stats': feedback_stats,
            'generation_time': generation_time,
            'prompt': prompt,
            'config': {
                'embedding_feedback': embedding_feedback,
                'attention_feedback': attention_feedback,
                'steps': steps,
                'cfg_scale': cfg_scale,
                'seed': seed
            }
        }
    
    def cleanup(self):
        """Remove attention hooks and clean up resources"""
        self.attention_modifier.remove_hooks()
        print("✓ Cleaned up attention hooks")


def test_hybrid_dynaprompt():
    """
    Test the hybrid DynaPrompt system with a challenging compositional prompt
    """
    print("="*80)
    print("TESTING HYBRID DYNAPROMPT")
    print("="*80)
    
    # Initialize hybrid system
    hybrid = HybridDynaPrompt(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test prompts (known challenging cases)
    test_prompts = [
        "a silver car parked next to a golden bicycle",
        "a red cube and a blue sphere on a wooden table",
        "a tiny red bicycle next to a giant blue umbrella"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"Testing: {prompt}")
        print(f"{'='*80}\n")
        
        # Generate with hybrid approach
        result = hybrid.generate(
            prompt=prompt,
            steps=50,
            cfg_scale=7.5,
            seed=42,
            embedding_feedback=True,
            attention_feedback=True
        )
        
        print(f"\nResults for: {prompt}")
        print(f"  CLIP Score: {result['final_clipscore']:.4f}")
        print(f"  Compositional Accuracy: {result['compositional_accuracy']:.4f}")
        print(f"  Generation Time: {result['generation_time']:.2f}s")
        print(f"  Feedback Steps: {len(result['metrics_history'])}")
        
        # Save image (optional)
        from torchvision.utils import save_image
        import os
        os.makedirs('outputs/hybrid', exist_ok=True)
        save_path = f"outputs/hybrid/{prompt.replace(' ', '_')[:50]}.png"
        save_image(result['image'], save_path)
        print(f"  Saved to: {save_path}")
    
    # Cleanup
    hybrid.cleanup()
    
    print("\n" + "="*80)
    print("✓ HYBRID DYNAPROMPT TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_hybrid_dynaprompt()
