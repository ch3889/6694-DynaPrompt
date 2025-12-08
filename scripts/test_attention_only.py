"""
Test attention-only (ch3889) vs hybrid (zk2295+ch3889)
To diagnose why hybrid underperforms ch3889 alone
"""

import sys
sys.path.append('.')

import torch
from dynaprompt.hybrid import HybridDynaPrompt

def test_configurations(prompt, seed=42):
    """Test three configurations:
    1. Baseline (no feedback)
    2. Attention ONLY (ch3889)
    3. Hybrid (embedding + attention)
    """
    
    print("="*80)
    print("TESTING: Attention-Only vs Hybrid")
    print("="*80)
    print(f"Prompt: {prompt}")
    print(f"Seed: {seed}")
    print("="*80)
    
    pipeline = HybridDynaPrompt()
    
    # 1. Baseline
    print("\n[1/3] BASELINE (no feedback)")
    result_baseline = pipeline.generate(
        prompt=prompt,
        steps=30,
        seed=seed,
        embedding_feedback=False,  # DISABLE embedding
        attention_feedback=False   # DISABLE attention
    )
    print(f"✓ CLIP: {result_baseline['metrics']['final_clip_score']:.4f}")
    print(f"✓ Comp: {result_baseline['metrics']['compositional_accuracy']:.4f}")
    
    # 2. Attention ONLY (ch3889's approach)
    print("\n[2/3] ATTENTION ONLY (ch3889)")
    result_attention = pipeline.generate(
        prompt=prompt,
        steps=30,
        seed=seed,
        embedding_feedback=False,  # DISABLE embedding feedback
        attention_feedback=True    # ENABLE attention only
    )
    print(f"✓ CLIP: {result_attention['metrics']['final_clip_score']:.4f}")
    print(f"✓ Comp: {result_attention['metrics']['compositional_accuracy']:.4f}")
    delta_attn = result_attention['metrics']['final_clip_score'] - result_baseline['metrics']['final_clip_score']
    print(f"→ Δ vs Baseline: {delta_attn:+.4f} ({delta_attn/result_baseline['metrics']['final_clip_score']*100:+.2f}%)")
    
    # 3. Hybrid (embedding + attention)
    print("\n[3/3] HYBRID (embedding + attention)")
    result_hybrid = pipeline.generate(
        prompt=prompt,
        steps=30,
        seed=seed,
        embedding_feedback=True,   # ENABLE embedding feedback
        attention_feedback=True    # ENABLE attention
    )
    print(f"✓ CLIP: {result_hybrid['metrics']['final_clip_score']:.4f}")
    print(f"✓ Comp: {result_hybrid['metrics']['compositional_accuracy']:.4f}")
    delta_hybrid = result_hybrid['metrics']['final_clip_score'] - result_baseline['metrics']['final_clip_score']
    print(f"→ Δ vs Baseline: {delta_hybrid:+.4f} ({delta_hybrid/result_baseline['metrics']['final_clip_score']*100:+.2f}%)")
    
    # Comparison
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Baseline:        {result_baseline['metrics']['final_clip_score']:.4f}")
    print(f"Attention-Only:  {result_attention['metrics']['final_clip_score']:.4f} ({delta_attn:+.4f}, {delta_attn/result_baseline['metrics']['final_clip_score']*100:+.2f}%)")
    print(f"Hybrid:          {result_hybrid['metrics']['final_clip_score']:.4f} ({delta_hybrid:+.4f}, {delta_hybrid/result_baseline['metrics']['final_clip_score']*100:+.2f}%)")
    print("="*80)
    
    if delta_attn > 0 and delta_hybrid < 0:
        print("\n⚠️  ATTENTION-ONLY POSITIVE, HYBRID NEGATIVE")
        print("→ Embedding feedback is CORRUPTING attention boosting")
    elif delta_attn > delta_hybrid:
        print("\n⚠️  ATTENTION-ONLY BETTER THAN HYBRID")
        print("→ Embedding feedback is DEGRADING results")
    else:
        print("\n✓ Hybrid improves over attention-only")

if __name__ == "__main__":
    test_configurations(
        prompt="a fluffy white cat wearing a tiny red hat sitting next to a blue flower vase",
        seed=42
    )
