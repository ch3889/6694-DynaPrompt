"""
Compare Baseline SD vs DynaPrompt Generation
Generates images with and without feedback to compare results
"""

from dynaprompt.wrapper import DynaPromptPipeline
from torchvision.utils import save_image
import torch
import os

print("=" * 60)
print("Comparing Baseline SD vs DynaPrompt")
print("=" * 60)

# Initialize pipeline
pipeline = DynaPromptPipeline(config_path='configs/dynaprompt_config.yaml')

prompt = "A golden retriever playing with a red ball"
seed = 42
steps = 30
cfg_scale = 7.5

print(f"\nPrompt: '{prompt}'")
print(f"Settings: {steps} steps, CFG={cfg_scale}, seed={seed}\n")

# Generate with Baseline SD (no feedback)
print("=" * 60)
print("1. BASELINE: Vanilla Stable Diffusion (no feedback)")
print("=" * 60)
baseline = pipeline.generate_with_feedback(
    prompt=prompt,
    steps=steps,
    cfg_scale=cfg_scale,
    seed=seed,
    feedback_enabled=False  # ← Disable DynaPrompt
)

print(f"\n✓ Baseline complete!")
print(f"  Final CLIP Score: {baseline['final_clip_score']:.3f}")

# Generate with DynaPrompt (with feedback)
print("\n" + "=" * 60)
print("2. DYNAPROMPT: With Real-time Feedback")
print("=" * 60)
torch.manual_seed(seed)  # Reset seed for fair comparison
dynaprompt = pipeline.generate_with_feedback(
    prompt=prompt,
    steps=steps,
    cfg_scale=cfg_scale,
    seed=seed,
    feedback_enabled=True  # ← Enable DynaPrompt
)

print(f"\n✓ DynaPrompt complete!")
print(f"  Final CLIP Score: {dynaprompt['final_clip_score']:.3f}")
print(f"  Feedback applications: {len(dynaprompt['metrics_history'])}")

# Show feedback trajectory
if dynaprompt['metrics_history']:
    print("\n  Feedback Trajectory:")
    for entry in dynaprompt['metrics_history']:
        print(f"    Step {entry['step']:3d}: CLIP={entry['clip_score']:.3f}, Shift={entry['embedding_shift']:.4f}")

# Save images
output_dir = "outputs/comparison"
os.makedirs(output_dir, exist_ok=True)

baseline_path = os.path.join(output_dir, "baseline_sd.png")
dynaprompt_path = os.path.join(output_dir, "dynaprompt_sd.png")

save_image(baseline['images'], baseline_path)
save_image(dynaprompt['images'], dynaprompt_path)

# Create side-by-side comparison
comparison = torch.cat([baseline['images'], dynaprompt['images']], dim=3)  # Concatenate horizontally
comparison_path = os.path.join(output_dir, "comparison_side_by_side.png")
save_image(comparison, comparison_path)

# Results summary
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)
print(f"\n{'Method':<20} {'CLIP Score':<15} {'Improvement':<15}")
print("-" * 50)
baseline_score = baseline['final_clip_score']
dynaprompt_score = dynaprompt['final_clip_score']
improvement = dynaprompt_score - baseline_score
print(f"{'Baseline SD':<20} {baseline_score:<15.3f} {'-':<15}")
print(f"{'DynaPrompt SD':<20} {dynaprompt_score:<15.3f} {f'+{improvement:.3f}':<15}")
print("-" * 50)

improvement_pct = (improvement / baseline_score) * 100
print(f"\nImprovement: {improvement_pct:+.2f}%")

print("\n" + "=" * 60)
print("SAVED FILES")
print("=" * 60)
print(f"✓ Baseline image:     {baseline_path}")
print(f"✓ DynaPrompt image:   {dynaprompt_path}")
print(f"✓ Side-by-side:       {comparison_path}")

print("\n" + "=" * 60)
print("Comparison complete!")
print("=" * 60)
print("\nOpen the comparison_side_by_side.png to see both images together.")
print("Baseline (left) vs DynaPrompt (right)")
