"""
Quick Comparison: Generate baseline only and compare with existing DynaPrompt output
This saves time by reusing the DynaPrompt image you already generated
"""

from dynaprompt.wrapper import DynaPromptPipeline
from torchvision.utils import save_image
from PIL import Image
import torch
import os

print("=" * 60)
print("Quick Baseline Comparison")
print("=" * 60)

# Check if DynaPrompt output exists
dynaprompt_img_path = "outputs/dynaprompt_output.png"
if not os.path.exists(dynaprompt_img_path):
    print(f"\n✗ Error: DynaPrompt output not found at {dynaprompt_img_path}")
    print("Run 'python run_dynaprompt.py' first to generate the DynaPrompt image.")
    exit(1)

print(f"\n✓ Found existing DynaPrompt image: {dynaprompt_img_path}")

# Initialize pipeline
print("\nInitializing pipeline...")
pipeline = DynaPromptPipeline(config_path='configs/dynaprompt_config.yaml')

prompt = "A golden retriever playing with a red ball"
seed = 42
steps = 30
cfg_scale = 7.5

print(f"\nPrompt: '{prompt}'")
print(f"Settings: {steps} steps, CFG={cfg_scale}, seed={seed}")

# Generate ONLY baseline (no feedback)
print("\n" + "=" * 60)
print("Generating BASELINE (no feedback)...")
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

# Load existing DynaPrompt image
print(f"\nLoading existing DynaPrompt image from {dynaprompt_img_path}...")
dynaprompt_img = Image.open(dynaprompt_img_path)
dynaprompt_tensor = torch.from_numpy(np.array(dynaprompt_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# Compute CLIP score for existing DynaPrompt image
print("Computing CLIP score for DynaPrompt image...")
dynaprompt_metrics = pipeline.dynaprompt.compute_metrics(prompt, dynaprompt_tensor)
dynaprompt_score = dynaprompt_metrics['clip_score']

print(f"✓ DynaPrompt CLIP Score: {dynaprompt_score:.3f}")

# Save baseline image
output_dir = "outputs/comparison"
os.makedirs(output_dir, exist_ok=True)

baseline_path = os.path.join(output_dir, "baseline_sd.png")
save_image(baseline['images'], baseline_path)

# Copy DynaPrompt image to comparison folder
import shutil
dynaprompt_comparison_path = os.path.join(output_dir, "dynaprompt_sd.png")
shutil.copy(dynaprompt_img_path, dynaprompt_comparison_path)

# Create side-by-side comparison
comparison = torch.cat([baseline['images'], dynaprompt_tensor], dim=3)  # Concatenate horizontally
comparison_path = os.path.join(output_dir, "comparison_side_by_side.png")
save_image(comparison, comparison_path)

# Results summary
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)
baseline_score = baseline['final_clip_score']
improvement = dynaprompt_score - baseline_score
improvement_pct = (improvement / baseline_score) * 100

print(f"\n{'Method':<20} {'CLIP Score':<15} {'Improvement':<15}")
print("-" * 50)
print(f"{'Baseline SD':<20} {baseline_score:<15.3f} {'-':<15}")
print(f"{'DynaPrompt SD':<20} {dynaprompt_score:<15.3f} {f'+{improvement:.3f}':<15}")
print("-" * 50)
print(f"\nImprovement: {improvement_pct:+.2f}%")

print("\n" + "=" * 60)
print("SAVED FILES")
print("=" * 60)
print(f"✓ Baseline image:     {baseline_path}")
print(f"✓ DynaPrompt image:   {dynaprompt_comparison_path}")
print(f"✓ Side-by-side:       {comparison_path}")

print("\n" + "=" * 60)
print("Comparison complete!")
print("=" * 60)
print("\nOpen comparison_side_by_side.png to see both images together.")
print("Baseline (left) vs DynaPrompt (right)")
