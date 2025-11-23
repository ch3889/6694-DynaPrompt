"""
GPU Configuration Guide for DynaPrompt

To use GPU instead of CPU, you need:
1. NVIDIA GPU with CUDA support
2. CUDA-enabled PyTorch installed

OPTION 1: If you have access to a GPU machine
==========================================
1. Move this project to the GPU machine
2. Install CUDA toolkit
3. Install PyTorch with CUDA:
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
4. Run the scripts - device will auto-detect GPU

OPTION 2: Use Google Colab (Free GPU)
======================================
Upload your code to Google Drive and run in Colab:

```python
# In Colab notebook:
!git clone https://github.com/ch3889/6694-DynaPrompt.git
%cd 6694-DynaPrompt

# Install dependencies
!pip install -r requirements.txt
!pip install kornia

# Upload your checkpoint to Colab or download it
# Then run:
!python run_dynaprompt.py
```

OPTION 3: For this machine - Force GPU if available
===================================================
If your machine has a GPU but it's not being detected:

1. Check GPU availability:
   python -c "import torch; print(torch.cuda.is_available())"

2. If False, reinstall PyTorch with CUDA:
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

PERFORMANCE COMPARISON
======================
CPU (your current setup):  ~90 minutes for 30 steps
GPU (RTX 3090):           ~30 seconds for 30 steps  
GPU (T4 - Colab free):    ~60 seconds for 30 steps

CURRENT AUTO-DETECTION
======================
The scripts automatically detect the best device:
- CUDA GPU (if available)
- MPS (Mac M1/M2)
- CPU (fallback)

No code changes needed - just install CUDA PyTorch!
"""

# Quick device check
import torch

print("=" * 60)
print("Device Detection")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No CUDA GPU detected - using CPU")
    print("\nTo use GPU, see instructions in GPU_SETUP.md")
print("=" * 60)
