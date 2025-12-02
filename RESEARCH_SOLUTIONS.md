# Research Solutions for Compositional Text-to-Image Generation

Based on recent research (2024-2025), here are promising approaches to solve our compositional generation problem.

## Problem Recap

Our challenge: Models fail to generate images with correct attribute binding (e.g., "silver car + golden bicycle" → missing bicycle or wrong colors).

## Current State-of-the-Art Solutions

### 1. **CompAgent** (Training-Free, 2024) ⭐ **MOST PROMISING**

**Paper**: "Divide and Conquer: Language Models can Plan and Self-Correct for Compositional Text-to-Image Generation"

**How it works**:
- Uses an LLM agent to decompose complex prompts into simpler sub-tasks
- Generates individual objects separately
- Composes them together with layout control
- Includes verification + self-correction loop

**Performance**:
- **>10% improvement** on T2I-CompBench (comprehensive benchmark)
- Training-free approach
- Works with existing diffusion models

**Why it's perfect for us**:
- ✅ Training-free (can apply immediately)
- ✅ Addresses exact problem (attribute binding, multiple objects)
- ✅ Self-correction mechanism (similar to our goal)
- ✅ State-of-the-art results

**Implementation approach**:
```
1. LLM parses prompt: "silver car + golden bicycle"
2. Generate separately:
   - Image 1: "a silver car"
   - Image 2: "a golden bicycle"
3. LLM plans layout: car on right, bicycle on left
4. Compose using inpainting/layout control
5. Verify with CLIP → self-correct if needed
```

### 2. **Attend-and-Excite** (Training-Free, 2023-2024)

**Paper**: "Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models"

**How it works**:
- Monitors cross-attention maps during generation
- Identifies "neglected" tokens (low attention)
- Optimizes latents to increase attention on those tokens
- Iteratively refines until all subjects appear

**Why relevant**:
- ✅ Training-free
- ✅ Works at inference time
- ✅ Directly addresses "catastrophic neglect" (missing objects)

**Limitations**:
- ⚠️ Still faces "attribute leakage" (wrong color binding)
- Focuses on presence, not accuracy

**Potential use**: Combine with our CLIP validation for better results

### 3. **Layout-to-Image with ControlNet** (2024)

**Paper**: "Layout-to-Image Generation with Localized Descriptions using ControlNet with Cross-Attention Control"

**How it works**:
- Uses layout (bounding boxes) to specify object locations
- Cross-attention manipulation for localized text control
- Training-free modification of ControlNet

**Why relevant**:
- ✅ Precise spatial control
- ✅ Localized descriptions prevent attribute leakage
- ✅ Training-free adaptation

**Challenge**:
- Requires layout specification (could use LLM to generate)

### 4. **BlobGEN** (2024)

**Paper**: "Compositional Text-to-Image Generation with Dense Blob Representations"

**How it works**:
- Decomposes scene into "blob" representations
- Each blob = object with fine-grained details
- More precise than bounding boxes
- Better reconstruction of details

**Why relevant**:
- ✅ Fine-grained compositional control
- ✅ Better attribute binding than boxes

**Challenge**:
- Requires training blob-grounded model

### 5. **Improved Text Embeddings** (2024)

**Paper**: "Improving Compositional Attribute Binding in Text-to-Image Generative Models via Enhanced Text Embeddings"

**Approach**:
- Fine-tune only CLIP projection layer (parameter-efficient)
- Uses small dataset of compositional image-text pairs
- Maintains FID score while improving composition

**Why relevant**:
- ✅ Minimal training (just linear projection)
- ✅ Small dataset needed
- ✅ No harm to general quality

### 6. **RealCompo** (NeurIPS 2024)

**Paper**: "Balancing Realism and Compositionality Improves Text-to-Image"

**Approach**:
- Dual-branch architecture
- Fidelity branch: focus on contours/colors
- Compositional branch: manipulate object positions
- Dynamic weighting between branches

**Why relevant**:
- ✅ Explicitly balances quality vs composition
- ✅ Outperforms GLIGEN on challenging prompts

## Recommended Solutions for DynaPrompt

### **Option A: Implement CompAgent Approach** (Recommended)

**Rationale**: Training-free, proven >10% improvement, addresses our exact problem

**Implementation**:
1. Use GPT/Claude API to parse compositional prompts
2. Generate individual objects with SDXL
3. Use ControlNet + inpainting for composition
4. CLIP validation for self-correction loop

**Pros**:
- No training required
- Can work with existing SDXL model
- Self-correction aligns with DynaPrompt philosophy
- State-of-the-art performance

**Cons**:
- Requires LLM API calls
- More complex pipeline
- Multiple generation steps

**Estimated effort**: 1-2 weeks

### **Option B: Combine Attend-and-Excite + CLIP Guidance**

**Rationale**: Both training-free, complementary strengths

**Implementation**:
1. Use Attend-and-Excite to ensure all objects present
2. Add our CLIP guidance for attribute verification
3. Iterative refinement loop

**Pros**:
- Addresses both neglect (A&E) and attribute binding (CLIP)
- Training-free
- Works during single generation pass

**Cons**:
- A&E still has attribute leakage issues
- May be slow (multiple optimization passes)

**Estimated effort**: 1 week

### **Option C: Fine-tune CLIP Projection** (Longer-term)

**Rationale**: Minimal training, permanent improvement

**Implementation**:
1. Collect ~1000 compositional image-text pairs
2. Fine-tune only CLIP's linear projection in SDXL
3. Deploy improved model

**Pros**:
- Permanent fix
- Minimal parameters (<1M)
- Maintains general quality

**Cons**:
- Requires dataset collection
- Some training needed
- Not purely training-free

**Estimated effort**: 2-3 weeks

## Benchmarks to Track

Use **T2I-CompBench++** (2024 standard) which tests:
1. Attribute binding
2. Object relationships
3. Generative numeracy
4. Complex compositions

Our current evaluation (30 prompts) aligns with these categories.

## Recommended Next Steps

1. **Immediate** (This week):
   - Implement simplified CompAgent:
     - Use LLM to decompose prompts
     - Generate objects separately
     - Simple composition (no fancy layout yet)
     - CLIP-based verification

2. **Short-term** (Next 2 weeks):
   - Add Attend-and-Excite to pipeline
   - Implement proper layout control (ControlNet)
   - Full self-correction loop

3. **Medium-term** (1-2 months):
   - Collect compositional dataset
   - Fine-tune CLIP projection
   - Comprehensive evaluation on T2I-CompBench++

## Key Insights from Research

1. **Divide-and-conquer works best**: Generating complex scenes as separate objects then composing outperforms end-to-end generation

2. **LLMs are crucial**: Modern solutions use LLMs for:
   - Prompt decomposition
   - Layout planning
   - Verification and feedback

3. **Training-free is viable**: Multiple 2024 papers show significant improvements without model retraining

4. **Attribute binding is hardest**: All papers identify this as the primary challenge (not just missing objects)

5. **SDXL struggles**: Research confirms SDXL/DiT models have compositional weaknesses despite better single-object quality

## Resources

- CompAgent: https://zhenyuw16.github.io/CompAgent/
- Attend-and-Excite: https://yuval-alaluf.github.io/Attend-and-Excite/
- T2I-CompBench++: https://arxiv.org/abs/2307.06350
- BlobGEN: https://blobgen-2d.github.io/
