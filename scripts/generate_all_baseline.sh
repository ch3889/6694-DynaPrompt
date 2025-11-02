#!/bin/bash
# Generate baseline images for all test prompts
# Usage: bash scripts/generate_all_baseline.sh

set -e  # Exit on error

PROJECT_ROOT="/home/cursedfox/6694-DynaPrompt"
SD_PATH="$PROJECT_ROOT/models/stable_diffusion_compvis"
PROMPTS_FILE="$PROJECT_ROOT/data/test_prompts.txt"
OUTPUT_DIR="$PROJECT_ROOT/data/images/baseline"
CKPT="$SD_PATH/v1-5-pruned-emaonly.ckpt"

# Parameters
N_SAMPLES=1
HEIGHT=512
WIDTH=512
STEPS=50
SEED=42

echo "==================================================================="
echo "Baseline Image Generation for DynaPrompt"
echo "==================================================================="
echo "Checkpoint: $CKPT"
echo "Output: $OUTPUT_DIR"
echo "Steps: $STEPS, Seed: $SEED"
echo "==================================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save metadata
METADATA="$OUTPUT_DIR/metadata.txt"
echo "Generation timestamp: $(date)" > "$METADATA"
echo "Checkpoint: $CKPT" >> "$METADATA"
echo "Steps: $STEPS" >> "$METADATA"
echo "Seed: $SEED" >> "$METADATA"
echo "Samples per prompt: $N_SAMPLES" >> "$METADATA"
echo "" >> "$METADATA"
echo "==================================================================" >> "$METADATA"
echo "" >> "$METADATA"

# Read prompts and generate images
PROMPT_IDX=1
while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue

    PROMPT="$line"

    echo "[$PROMPT_IDX] Generating: $PROMPT"

    # Create safe directory name
    SAFE_PROMPT=$(echo "$PROMPT" | sed 's/[^a-zA-Z0-9 _-]/_/g' | cut -c1-80)
    PROMPT_DIR="$OUTPUT_DIR/$(printf "%03d" $PROMPT_IDX)_${SAFE_PROMPT}"
    mkdir -p "$PROMPT_DIR"

    # Save prompt text
    echo "$PROMPT" > "$PROMPT_DIR/prompt.txt"

    # Generate image
    cd "$SD_PATH"
    python scripts/txt2img.py \
        --prompt "$PROMPT" \
        --ckpt "$CKPT" \
        --outdir "$PROMPT_DIR" \
        --n_samples "$N_SAMPLES" \
        --H "$HEIGHT" \
        --W "$WIDTH" \
        --ddim_steps "$STEPS" \
        --seed "$SEED" \
        2>&1 | grep -E "(Loading model|Data shape|Sampling|Your samples)" || true

    if [ $? -eq 0 ]; then
        echo "  ✓ Generated successfully"
        echo "Prompt $PROMPT_IDX: $PROMPT" >> "$METADATA"
        echo "Output: $PROMPT_DIR" >> "$METADATA"
        echo "Status: Success" >> "$METADATA"
        echo "" >> "$METADATA"
    else
        echo "  ✗ Generation failed"
        echo "Prompt $PROMPT_IDX: $PROMPT" >> "$METADATA"
        echo "Output: $PROMPT_DIR" >> "$METADATA"
        echo "Status: Failed" >> "$METADATA"
        echo "" >> "$METADATA"
    fi

    ((PROMPT_IDX++))
    echo ""

done < "$PROMPTS_FILE"

echo "==================================================================="
echo "✓ Baseline generation complete!"
echo "Output directory: $OUTPUT_DIR"
echo "==================================================================="
