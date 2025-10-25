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
echo "=== Download Complete ==="
echo ""
echo "Next steps:"
echo "1. Clone CompVis Stable Diffusion repo into models/stable_diffusion_compvis/"
echo "2. Install CLIP: pip install git+https://github.com/openai/CLIP.git"
echo "3. Test inference with: python models/stable_diffusion_compvis/scripts/txt2img.py"
