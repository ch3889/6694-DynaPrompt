"""
Quick test of DynaPrompt generation
"""

from dynaprompt.wrapper import run_dynaprompt_generation
from torchvision.utils import save_image
import os

print("=" * 60)
print("Running DynaPrompt Generation")
print("=" * 60)

# Generate with DynaPrompt feedback
results = run_dynaprompt_generation(
    prompt="A golden retriever playing with a red ball",
    steps=30,  # Using 30 steps for faster generation
    cfg_scale=7.5,
    seed=42
)

# Get results
images = results['images']  # Generated images
clip_score = results['final_clip_score']  # Quality metric
feedback_history = results['metrics_history']  # Feedback trajectory

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"✓ Generated {images.shape[0]} image(s)")
print(f"✓ Image shape: {images.shape}")
print(f"✓ Final CLIP Score: {clip_score:.3f}")
print(f"✓ Feedback applications: {len(feedback_history)}")

# Show feedback trajectory
if feedback_history:
    print("\nFeedback Trajectory:")
    for entry in feedback_history:
        print(f"  Step {entry['step']:3d}: CLIP={entry['clip_score']:.3f}, Shift={entry['embedding_shift']:.4f}")

# Save image
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "dynaprompt_output.png")
save_image(images, output_path)
print(f"\n✓ Image saved to: {output_path}")

print("\n" + "=" * 60)
print("Generation complete!")
print("=" * 60)
