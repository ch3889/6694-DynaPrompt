#!/bin/bash
# Setup script for GCP Deep Learning VM (PyTorch pre-installed)
# Run this after connecting to your DL VM

set -e

echo "========================================"
echo "DynaPrompt Setup on DL VM"
echo "========================================"

# Check PyTorch and CUDA
echo "Checking PyTorch installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# Check GPU
nvidia-smi

# Install additional dependencies
echo "Installing dependencies..."
pip install transformers==4.35.0
pip install diffusers==0.24.0
pip install accelerate==0.25.0
pip install omegaconf==2.3.0
pip install einops==0.7.0
pip install kornia==0.7.0
pip install pytorch-lightning==2.1.0
pip install torchmetrics==1.2.0
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

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Upload SD checkpoint:"
echo "   gcloud compute scp v1-5-pruned-emaonly.ckpt dynaprompt-dl:~/6694-DynaPrompt/models/stable_diffusion_compvis/ --zone=us-central1-a"
echo ""
echo "2. Test GPU:"
echo "   python check_gpu.py"
echo ""
echo "3. Run DynaPrompt:"
echo "   python run_dynaprompt.py"
