"""
Quick test script for Hybrid DynaPrompt - generates images only
No baseline comparison, just tests that hybrid works
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dynaprompt.hybrid import HybridDynaPrompt
from torchvision.utils import save_image


def quick_test():
    """Quick test of hybrid generation"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Find checkpoint
    import os
    possible_paths = [
        'models/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/v1-5-pruned-emaonly.ckpt',
        'models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt'
    ]
    ckpt_path = None
    for path in possible_paths:
        if os.path.exists(path):
            ckpt_path = path
            break
    
    # Initialize hybrid pipeline
    print("\n" + "="*60)
    print("INITIALIZING HYBRID DYNAPROMPT")
    print("="*60)
    hybrid = HybridDynaPrompt(ckpt_path=ckpt_path, device=device)
    
    # Test prompts - just 2 quick ones
    test_prompts = [
        "a red cube and a blue sphere",
        "a golden retriever playing with a red ball"
    ]
    
    output_dir = 'outputs/hybrid_quick_test'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n\n{'='*60}")
        print(f"TEST {i}/{len(test_prompts)}: {prompt}")
        print(f"{'='*60}")
        
        try:
            result = hybrid.generate(
                prompt=prompt,
                steps=30,  # Reduced steps for speed
                cfg_scale=7.5,
                seed=42,
                embedding_feedback=True,
                attention_feedback=True
            )
            
            # Save image
            safe_prompt = prompt.replace(' ', '_')[:40]
            img_path = os.path.join(output_dir, f'{i}_{safe_prompt}.png')
            save_image(result['image'], img_path)
            
            print(f"\n✓ SUCCESS")
            print(f"  Saved: {img_path}")
            print(f"  CLIP Score: {result['final_clipscore']:.4f}")
            print(f"  Compositional Acc: {result['compositional_accuracy']:.4f}")
            print(f"  Time: {result['generation_time']:.1f}s")
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n\n{'='*60}")
    print("TESTS COMPLETE")
    print(f"Images saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    quick_test()
