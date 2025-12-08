#!/bin/bash
# Download Stable Diffusion v1.5 weights and CLIP models

echo "=== DynaPrompt Model Download Script ==="
echo ""

# Create models directory if it doesn't exist
mkdir -p models/stable_diffusion_compvis

# Download Stable Diffusion v1.5 checkpoint
echo "Downloading Stable Diffusion v1.5 checkpoint..."
echo "Note: This is a large file (~4GB). Make sure you have enough space."
echo ""

cd models/stable_diffusion_compvis

# Option 1: Using wget
if command -v wget &> /dev/null; then
    echo "Using wget to download..."
    wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
elif command -v curl &> /dev/null; then
    echo "Using curl to download..."
    curl -L -o v1-5-pruned-emaonly.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
else
    echo "Error: Neither wget nor curl is installed."
    echo "Please install one of them or manually download from:"
    echo "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    exit 1
fi

cd ../..

echo ""
echo ""
echo "Cloning CompVis Stable Diffusion repo at specific commit..."

# Clone the CompVis repository if not already present
if [ ! -d "stable-diffusion" ]; then
    git clone https://github.com/CompVis/stable-diffusion.git
fi

cd stable-diffusion

# Checkout the requested commit
git fetch --all
git checkout 21f890f9da3cfbeaba8e2ac3c425ee9e998d5229

echo "Applying macOS compatibility patch (if available)..."
PATCH_FILE="../../patches/compvis_mac_compatibility.patch"
if [ -f "$PATCH_FILE" ]; then
    git apply "$PATCH_FILE" || {
        echo "Patch apply failed or already applied. Continuing..."
    }
else
    echo "No macOS patch found at $PATCH_FILE. Skipping."
fi

cd ../..

echo ""
echo "=== Download Complete ==="
echo ""
echo "Next steps:"
echo "1. Install dependencies for CompVis repo (in a Python 3.10 env):"
echo "   pip install -r models/stable_diffusion_compvis/stable-diffusion/requirements.txt"
echo "   pip install git+https://github.com/openai/CLIP.git"
echo "2. Place the SD v1.5 checkpoint at: models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt"
echo "3. Test inference:"
echo "   python models/stable_diffusion_compvis/stable-diffusion/scripts/txt2img.py --prompt 'A blue cat on a red chair' --ckpt models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt --H 512 --W 512 --n_samples 1 --n_iter 1 --seed 42"
