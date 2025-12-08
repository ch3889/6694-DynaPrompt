# CH3889 Branch: Attention Boosting Method

## Intuition
When diffusion models "forget" objects, those tokens receive **low cross-attention scores**. By amplifying attention to underrepresented tokens during early denoising steps, we can encourage the model to include neglected concepts.

---

## Core Formulation

**Standard Cross-Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

**Our Modification:** Multiply attention scores for target tokens by boost factor β
```
Attention'(Q, K, V) = softmax(QK^T / √d · M_boost) · V

where M_boost[i,j] = β  if token j is underrepresented
                   = 1  otherwise
```

**Parameters:**
- **Boost factor β:** 2.5x (applied to neglected tokens)
- **Active window:** Steps 0-20 (early denoising, where composition is decided)
- **Detection:** CLIP similarity per token < threshold → boost that token

---

## Engineering for Fair Comparison

| Design Choice | Rationale |
|--------------|-----------|
| **Single seed per prompt** | Deterministic comparison; avoids seed lottery confounding results |
| **Fixed 50 DDIM steps** | Consistent across all methods |
| **Same CLIP model (ViT-B/32)** | Identical evaluation across branches |
| **No retry/restart** | Isolates attention boosting effect from seed search |

**Why no seed variation?** DrawBench evaluation with multiple seeds per prompt is computationally expensive. Single-seed evaluation reveals the **true capability** of attention boosting alone.

---

## Takeaway

**What attention boosting CAN do:**
- Strengthen existing object representations
- Improve attribute binding when objects are already initialized

**What attention boosting CANNOT do:**
- Create objects that weren't seeded in early noise
- Override strong training biases (e.g., "golden bicycle" remains rare)

> Attention boosting is a **refinement tool**, not a **generation tool** — it amplifies signals but cannot create them from nothing.
