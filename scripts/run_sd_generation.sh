#!/bin/bash
# Quick image generation with Stable Diffusion via HuggingFace diffusers

set -e

PROMPT="${1:-A blue cat sitting on a red chair with a yellow ball}"
OUTPUT="${2:-outputs/sd_generated.png}"

echo "=== Stable Diffusion Image Generation ==="
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"
echo ""

# Check if dependencies are installed
if ! python -c "import diffusers" 2>/dev/null; then
    echo "Installing diffusers library..."
    pip install -q diffusers transformers accelerate safetensors
fi

# Create output directory
mkdir -p outputs

# Run generation
python scripts/test_sd_dit.py \
    --prompt "$PROMPT" \
    --output "$OUTPUT" \
    --steps 25 \
    --cfg 7.5 \
    --seed 42

echo ""
echo "✓ Image generation complete!"
echo "View: open $OUTPUT"
