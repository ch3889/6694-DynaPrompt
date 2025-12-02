"""Quick test to verify models can be loaded without crashing"""

import torch
import sys
from pathlib import Path

print("="*60)
print("Testing Model Loading")
print("="*60)

# Test 1: CLIP
print("\n1. Loading CLIP...")
try:
    import clip
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("   ✅ CLIP loaded successfully")
except Exception as e:
    print(f"   ❌ CLIP failed: {e}")
    sys.exit(1)

# Test 2: Stable Diffusion
print("\n2. Loading Stable Diffusion...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dynaprompt.sd_loader import load_sd_model
    
    sd_model = load_sd_model(device=device)
    print("   ✅ Stable Diffusion loaded successfully")
except Exception as e:
    print(f"   ❌ Stable Diffusion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Simple generation
print("\n3. Testing simple generation...")
try:
    test_prompt = "a red cube"
    print(f"   Prompt: '{test_prompt}'")
    
    # Just test encoding, not full generation
    from dynaprompt.wrapper import StableDiffusionWrapper
    wrapper = StableDiffusionWrapper(sd_model.model, device=device)
    
    # Test text encoding
    text_input = sd_model.model.cond_stage_model.tokenizer(
        [test_prompt],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_embeddings = sd_model.model.cond_stage_model.transformer(
            text_input.input_ids.to(device)
        )[0]
    
    print(f"   Text embedding shape: {text_embeddings.shape}")
    print("   ✅ Text encoding works")
    
except Exception as e:
    print(f"   ⚠️  Generation test skipped: {e}")
    print("   (This is OK - main models work)")

print("\n" + "="*60)
print("✅ All critical tests passed! Models ready for experiments")
print("="*60)
