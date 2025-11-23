"""
Test script to verify DynaPrompt + Stable Diffusion integration
"""

import torch
import os
import sys

# Ensure we're in the right directory
sys.path.insert(0, os.path.dirname(__file__))

def test_sd_loader():
    """Test SD model loading"""
    print("=" * 60)
    print("TEST 1: Stable Diffusion Model Loading")
    print("=" * 60)
    
    from dynaprompt.sd_loader import load_sd_model
    
    try:
        sd = load_sd_model()
        print("✓ SD model loaded successfully!")
        
        # Test text encoding
        print("\nTesting text encoder...")
        text_emb = sd.encode_text("A golden retriever")
        print(f"✓ Text embedding shape: {text_emb.shape}")
        assert text_emb.shape[1] == 77, "Expected 77 tokens"
        assert text_emb.shape[2] == 768, "Expected 768 dimensions"
        
        # Test VAE access
        print("\nTesting VAE access...")
        vae = sd.get_vae()
        print(f"✓ VAE loaded: {type(vae).__name__}")
        
        # Test U-Net access
        print("\nTesting U-Net access...")
        unet = sd.get_unet()
        print(f"✓ U-Net loaded: {type(unet).__name__}")
        
        print("\n✓ All SD loader tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ SD loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynaprompt_core():
    """Test DynaPrompt feedback module"""
    print("\n" + "=" * 60)
    print("TEST 2: DynaPrompt Core Module")
    print("=" * 60)
    
    from dynaprompt.core import DynaPrompt
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        dynaprompt = DynaPrompt(device=device)
        print("✓ DynaPrompt initialized!")
        
        # Test CLIP score computation
        print("\nTesting CLIP score computation...")
        dummy_image = torch.rand(1, 3, 512, 512).to(device)
        prompt = "A golden retriever"
        clip_score = dynaprompt.compute_clipscore(dummy_image, prompt)
        print(f"✓ CLIP score computed: {clip_score:.3f}")
        
        # Test feedback loop
        print("\nTesting feedback loop...")
        dummy_embedding = torch.randn(1, 77, 768).to(device)
        feedback_result = dynaprompt.feedback_loop(
            prompt=prompt,
            current_embedding=dummy_embedding,
            generated_image=dummy_image,
            step=10
        )
        print(f"✓ Feedback result keys: {list(feedback_result.keys())}")
        print(f"  - CLIP score: {feedback_result['clip_score']:.3f}")
        print(f"  - Embedding shift: {feedback_result['embedding_shift']:.4f}")
        
        # Test metrics computation
        print("\nTesting metrics computation...")
        metrics = dynaprompt.compute_metrics(prompt, dummy_image)
        print(f"✓ Metrics computed: {list(metrics.keys())}")
        
        print("\n✓ All DynaPrompt core tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ DynaPrompt core test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test full DynaPrompt + SD pipeline"""
    print("\n" + "=" * 60)
    print("TEST 3: Full Pipeline Integration")
    print("=" * 60)
    
    from dynaprompt.wrapper import DynaPromptPipeline
    
    try:
        print("Initializing DynaPromptPipeline...")
        print("(This will load SD model + CLIP, may take ~30 seconds)")
        
        pipeline = DynaPromptPipeline(
            config_path='configs/dynaprompt_config.yaml'
        )
        print("✓ Pipeline initialized!")
        
        print("\n✓ All pipeline integration tests passed!")
        print("\nNOTE: To run full generation test, use:")
        print("  python test_integration.py --full-generation")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_generation():
    """Test full generation (slower, optional)"""
    print("\n" + "=" * 60)
    print("TEST 4: Full Image Generation (THIS WILL TAKE ~1 MINUTE)")
    print("=" * 60)
    
    from dynaprompt.wrapper import run_dynaprompt_generation
    
    try:
        prompt = "A golden retriever playing with a red ball"
        print(f"\nGenerating with prompt: '{prompt}'")
        print("Using: 20 steps, CFG=7.5, feedback enabled")
        
        results = run_dynaprompt_generation(
            prompt=prompt,
            steps=20,
            cfg_scale=7.5,
            seed=42
        )
        
        print(f"\n✓ Generation complete!")
        print(f"  - Image shape: {results['images'].shape}")
        print(f"  - Final CLIP score: {results['final_clip_score']:.3f}")
        print(f"  - Feedback applications: {len(results['metrics_history'])}")
        
        print("\n✓ Full generation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Full generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DynaPrompt integration")
    parser.add_argument("--full-generation", action="store_true",
                       help="Run full generation test (slow)")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("DynaPrompt + Stable Diffusion Integration Tests")
    print("=" * 60)
    
    # Run tests
    results = []
    
    results.append(("SD Loader", test_sd_loader()))
    results.append(("DynaPrompt Core", test_dynaprompt_core()))
    results.append(("Pipeline Integration", test_pipeline_integration()))
    
    if args.full_generation:
        results.append(("Full Generation", test_full_generation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour DynaPrompt integration is working correctly!")
        print("\nNext steps:")
        print("1. Run with --full-generation to test actual image generation")
        print("2. Use dynaprompt.wrapper.run_dynaprompt_generation() in your code")
        print("3. Adjust configs/dynaprompt_config.yaml for your experiments")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Please check the error messages above.")
        sys.exit(1)
