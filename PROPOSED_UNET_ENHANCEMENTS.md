# U-Net Enhancement Plan for DynaPrompt

Based on our research and findings, here's the roadmap to achieve better compositional generation with U-Net (SD 1.5).

## Current Status: V7 Baseline

**What works:**
- ✅ Both objects generated (silver car + bicycle)
- ✅ Spatial composition correct (car and bicycle present)
- ✅ Attention boosting successfully prevents object neglect

**What needs improvement:**
- ⚠️ Color accuracy (bicycle is silver/gray, not golden)
- ⚠️ Attribute binding (golden attribute not applied to bicycle)

## Proposed Enhancements (Priority Order)

### **Phase 1: Add CLIP Validation to V7** (Quick Win - 1-2 days)

**Goal**: Detect attribute failures and retry with stronger boosting

**Approach:**
```python
# Pseudo-code
while not all_attributes_validated:
    1. Generate with current boost_factor
    2. Validate with CLIP
    3. If attribute missing/wrong: increase boost_factor for that token
    4. Retry generation
```

**Why this works:**
- V7 already generates both objects
- Just needs stronger boost for color attributes
- CLIP provides objective validation

**Implementation:**
- Add CLIP scoring to V7
- Implement adaptive boost (increase from 7.5x to 15x for failing attributes)
- Self-correction loop (max 3 retries)

**Expected improvement**: 70-80% success rate (vs current ~50%)

---

### **Phase 2: Implement Attend-and-Excite Integration** (Medium - 3-5 days)

**Goal**: Add iterative latent optimization to strengthen weak tokens

**What is Attend-and-Excite:**
- Training-free method (works with existing models)
- Optimizes latents during generation to maximize attention on neglected tokens
- Proven to work well with U-Net cross-attention

**How it works:**
```python
At each denoising step:
  1. Check cross-attention maps for each token
  2. If token attention < threshold:
     - Compute gradient of attention w.r.t. latents
     - Update latents to increase attention
  3. Continue denoising
```

**Why combine with V7:**
- V7's attention boosting: modifies attention weights (global)
- Attend-and-Excite: optimizes latents (local, per-step)
- Complementary approaches!

**Implementation:**
```python
class DynaPromptV10_AttendExcite(DynaPromptV7):
    def __init__(self, ...):
        super().__init__(...)
        self.attend_excite_steps = 5  # Apply A&E at steps 10-15
        self.attention_threshold = 0.05

    def _apply_attend_excite(self, latents, timestep, text_embeddings):
        """Apply Attend-and-Excite optimization."""
        # Get cross-attention maps
        attention_maps = self._extract_attention_maps()

        # Find weak tokens
        weak_tokens = self._find_weak_tokens(attention_maps)

        if weak_tokens:
            # Optimize latents to strengthen weak tokens
            latents = self._optimize_latents_for_tokens(
                latents, weak_tokens, num_steps=5
            )

        return latents
```

**Expected improvement**: 85-90% success rate

---

### **Phase 3: Token-Specific Guidance** (Advanced - 5-7 days)

**Goal**: Apply different guidance scales to different tokens

**Concept:**
- Instead of global CFG scale (7.5 for all tokens)
- Use higher CFG for critical attribute tokens
- Lower CFG for generic tokens

**Example:**
```
"a silver car parked next to a golden bicycle"

Token CFG scales:
- "silver": 12.0 (boost color)
- "car": 7.5 (standard)
- "parked": 5.0 (reduce generic descriptor)
- "next to": 5.0 (reduce)
- "golden": 12.0 (boost color)
- "bicycle": 9.0 (boost object)
```

**Implementation:**
```python
def _compute_token_specific_guidance(
    self,
    noise_pred_uncond,
    noise_pred_text,
    token_cfg_scales  # Dict[token_id: scale]
):
    """Apply per-token CFG scaling."""
    # Get cross-attention maps to determine which regions each token affects
    attention_maps = self._get_cross_attention_maps()

    # For each token, apply its specific CFG scale to its influenced regions
    guided_noise = noise_pred_uncond.clone()

    for token_id, cfg_scale in token_cfg_scales.items():
        # Get attention mask for this token
        mask = attention_maps[token_id]

        # Apply token-specific guidance in masked regions
        guidance = noise_pred_text - noise_pred_uncond
        guided_noise += mask * guidance * cfg_scale

    return guided_noise
```

**Expected improvement**: 90-95% success rate

---

### **Phase 4: LLM-Enhanced Prompt Decomposition** (Research - 1 week)

**Goal**: Use Ollama to intelligently parse and prioritize tokens

**What we learned from V9:**
- ✅ Ollama + qwen2.5 works perfectly for decomposition
- ✅ Free and local
- ✅ Can identify objects, attributes, relationships

**Enhanced V7 with LLM:**
```python
class DynaPromptV11_LLM(DynaPromptV10_AttendExcite):
    def __init__(self, ...):
        super().__init__(...)
        self.ollama_model = "qwen2.5:7b"

    def sample_with_llm_planning(self, prompt, ...):
        # Step 1: LLM decomposes prompt
        decomposition = self._llm_decompose(prompt)
        # Returns: {
        #   "objects": ["silver car", "golden bicycle"],
        #   "critical_attributes": {
        #     "car": ["silver"],
        #     "bicycle": ["golden"]
        #   },
        #   "relationships": ["next to"]
        # }

        # Step 2: Assign boost factors based on decomposition
        boost_factors = {}
        for obj, attrs in decomposition["critical_attributes"].items():
            for attr in attrs:
                boost_factors[attr] = 15.0  # High boost for colors

        # Step 3: Generate with LLM-guided boosting
        return self.sample_with_adaptive_boost(
            prompt, boost_factors=boost_factors
        )
```

**Expected improvement**: 95%+ success rate

---

## Implementation Priority

### **Week 1: Quick Wins**
1. ✅ Validate V7 works (DONE - bicycle is present!)
2. **Add CLIP validation to V7** (Phase 1)
   - Files to modify: `dynaprompt/dynaprompt_v7.py`
   - Add CLIP model loading
   - Add validation loop
   - Test on 5 hard prompts

### **Week 2: Core Enhancement**
3. **Implement Attend-and-Excite** (Phase 2)
   - Create `dynaprompt/dynaprompt_v10_attend_excite.py`
   - Extract attention maps during generation
   - Add latent optimization
   - Test on full 30 prompt eval

### **Week 3-4: Advanced Features**
4. **Token-specific guidance** (Phase 3)
5. **LLM integration** (Phase 4)

---

## Recommended Immediate Next Steps

**Option A: Quick Enhancement (Recommended)**
1. Add CLIP validation to V7
2. Implement retry with increased boost for failing attributes
3. Test on 10 hard prompts
4. **Time**: 1-2 days
5. **Expected success rate**: 70-80%

**Option B: Research Implementation**
1. Implement full Attend-and-Excite
2. More complex but higher success rate
3. **Time**: 3-5 days
4. **Expected success rate**: 85-90%

**Option C: Hybrid Approach**
1. Start with Option A (CLIP validation)
2. If results promising, add Attend-and-Excite (Option B)
3. Iterative improvement
4. **Time**: 1 week total
5. **Expected success rate**: 90%+

---

## Technical Details: Why U-Net is Better

### Cross-Attention in U-Net (SD 1.5)

```python
# In each U-Net block:
def forward(self, x, context):
    # x: image features [B, C, H, W]
    # context: text embeddings [B, 77, 768]  # 77 tokens

    # Query from image
    Q = self.to_q(x)  # [B, H*W, dim]

    # Key/Value from text
    K = self.to_k(context)  # [B, 77, dim]
    V = self.to_v(context)  # [B, 77, dim]

    # Attention: image queries × text keys
    attention = softmax(Q @ K.T / sqrt(dim))  # [B, H*W, 77]

    # Each spatial location attends to each text token
    # This is what we can boost!

    out = attention @ V
    return out
```

**Key advantage**: We can access and modify `attention[spatial_loc, token_id]`

### DiT (SDXL) - Why it's harder

```python
# In DiT:
def forward(self, image_tokens, text_tokens):
    # Concatenate all tokens
    all_tokens = concat([text_tokens, image_tokens])  # [B, 77+H*W, dim]

    # Self-attention on everything
    attention = softmax(all_tokens @ all_tokens.T / sqrt(dim))

    # Text-image relationship is IMPLICIT in this attention matrix
    # No clear separation of "which text token → which image location"
```

**Problem**: Can't easily isolate text→image attention

---

## Conclusion

**U-Net (SD 1.5) with enhanced DynaPrompt is the best path forward because:**

1. ✅ **Proven to work**: V7 already generates both objects
2. ✅ **Explicit control**: Cross-attention can be directly modified
3. ✅ **Research-backed**: All compositional methods target U-Net
4. ✅ **Incremental improvement**: Can add CLIP, A&E, LLM step-by-step

**Recommended first step:**
- **Implement Phase 1 (CLIP validation + adaptive boost)**
- Low effort, high impact
- Builds on working V7 baseline

Would you like me to start implementing Phase 1?
