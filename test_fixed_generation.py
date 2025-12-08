"""
Test fixed DynaPrompt generation with proper gradient feedback
"""

from dynaprompt.wrapper import run_dynaprompt_generation
from torchvision.utils import save_image
import os

print("=" * 70)
print("Testing FIXED DynaPrompt Generation")
print("=" * 70)
print("\nFixes applied:")
print("  ✓ Proper CLIP feature alignment (no scalar addition)")
print("  ✓ Normalized gradient updates")
print("  ✓ Conservative alpha = 0.05")
print("  ✓ More frequent feedback (every 5 steps)")
print("  ✓ Earlier start (step 5)")

# Test with simple prompt first
prompt = "A golden retriever playing with a red ball"

print(f"\nPrompt: '{prompt}'")
print("Generating with fixed feedback mechanism...")

# Generate with fixed DynaPrompt
results = run_dynaprompt_generation(
    prompt=prompt,
    steps=30,
    cfg_scale=7.5,
    seed=42,
    feedback_enabled=True
)

# Get results
images = results['images']
clip_score = results['final_clip_score']
feedback_history = results['metrics_history']

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"✓ Generated {images.shape[0]} image(s)")
print(f"✓ Image shape: {images.shape}")
print(f"✓ Final CLIP Score: {clip_score:.3f}")
print(f"✓ Feedback applications: {len(feedback_history)}")
print(f"✓ Generation time: {results['generation_time']:.2f}s")

# Show feedback trajectory
if feedback_history:
    print("\n📊 Feedback Trajectory:")
    for entry in feedback_history:
        print(f"  Step {entry['step']:3d}: CLIP={entry['clip_score']:.3f}, Shift={entry.get('embedding_shift', 0):.4f}")
    
    # Check for improvement
    if len(feedback_history) > 1:
        first_score = feedback_history[0]['clip_score']
        last_score = feedback_history[-1]['clip_score']
        improvement = last_score - first_score
        print(f"\n{'✓' if improvement > 0 else '✗'} CLIP Score Change: {improvement:+.3f}")

# Save image
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "fixed_dynaprompt_output.png")
save_image(images, output_path)
print(f"\n💾 Image saved to: {output_path}")

# Compare with baseline
print("\n" + "=" * 70)
print("Generating Baseline (no feedback) for comparison...")
print("=" * 70)

baseline_results = run_dynaprompt_generation(
    prompt=prompt,
    steps=30,
    cfg_scale=7.5,
    seed=42,
    feedback_enabled=False  # Disable feedback
)

baseline_images = baseline_results['images']
baseline_clip_score = baseline_results['final_clip_score']

baseline_path = os.path.join(output_dir, "baseline_output.png")
save_image(baseline_images, baseline_path)

print(f"\n✓ Baseline CLIP Score: {baseline_clip_score:.3f}")
print(f"✓ DynaPrompt CLIP Score: {clip_score:.3f}")
print(f"{'✓' if clip_score > baseline_clip_score else '✗'} Improvement: {clip_score - baseline_clip_score:+.3f}")
print(f"💾 Baseline saved to: {baseline_path}")

print("\n" + "=" * 70)
print("Generation complete! Check outputs/ for results.")
print("=" * 70)
