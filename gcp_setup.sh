#!/bin/bash
# GCP VM Setup Script for DynaPrompt with GPU
# Run this on your GCP VM after SSH connection

set -e

echo "========================================"
echo "DynaPrompt GCP Setup with GPU"
echo "========================================"

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install NVIDIA drivers and CUDA (if not pre-installed)
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA drivers already installed"
    nvidia-smi
else
    echo "Installing NVIDIA drivers..."
    # For Ubuntu 20.04/22.04 with CUDA 11.8
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-11-8
    sudo apt-get -y install nvidia-driver-525
    echo "Please reboot the VM: sudo reboot"
    exit 0
fi

# Install Miniconda
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    source ~/.bashrc
else
    echo "✓ Conda already installed"
fi

# Create conda environment
echo "Creating dynaprompt conda environment..."
conda create -n dynaprompt python=3.10 -y

# Activate environment
source $HOME/miniconda3/bin/activate dynaprompt

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA 11.8..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
echo "Installing dependencies..."
pip install transformers==4.35.0
pip install diffusers==0.24.0
pip install accelerate==0.25.0
pip install omegaconf==2.3.0
pip install einops==0.7.0
pip install kornia==0.7.0
pip install pytorch-lightning==2.1.0
pip install torchmetrics==1.2.0
pip install pillow==10.1.0
pip install numpy==1.24.3
pip install tqdm
pip install matplotlib

# Clone repository
echo "Cloning DynaPrompt repository..."
cd ~
if [ ! -d "6694-DynaPrompt" ]; then
    git clone https://github.com/ch3889/6694-DynaPrompt.git
    cd 6694-DynaPrompt
    git checkout zk2295
else
    cd 6694-DynaPrompt
    git pull origin zk2295
fi

# Download Stable Diffusion checkpoint
echo "Checking for SD v1.5 checkpoint..."
CKPT_PATH="models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "Downloading Stable Diffusion v1.5 checkpoint..."
    mkdir -p models/stable_diffusion_compvis
    cd models/stable_diffusion_compvis
    
    # Option 1: wget (you need to provide the download URL)
    # wget -O v1-5-pruned-emaonly.ckpt "YOUR_DOWNLOAD_URL_HERE"
    
    # Option 2: Using HuggingFace CLI
    pip install huggingface-hub
    huggingface-cli download runwayml/stable-diffusion-v1-5 v1-5-pruned-emaonly.ckpt --local-dir . --local-dir-use-symlinks False
    
    cd ../..
else
    echo "✓ SD checkpoint already exists"
fi

# Test GPU
echo ""
echo "========================================"
echo "Testing GPU setup..."
echo "========================================"
python check_gpu.py

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate environment: conda activate dynaprompt"
echo "To run DynaPrompt: python run_dynaprompt.py"
echo "To run comparison: python compare_baseline.py"
echo ""
echo "For faster generation with GPU, use fewer steps:"
echo "  python run_dynaprompt.py  # Should be ~100x faster on GPU"
