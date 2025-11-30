# Running DynaPrompt on GCP with GPU

## Step 1: Create GCP VM Instance

### Using GCP Console:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Compute Engine > VM Instances**
3. Click **CREATE INSTANCE**

### Recommended Configuration:

```
Name: dynaprompt-gpu
Region: us-central1 (or closest to you)
Zone: us-central1-a

Machine Configuration:
  Series: N1
  Machine type: n1-standard-4 (4 vCPU, 15 GB memory)

GPU:
  GPU type: NVIDIA Tesla T4
  Number of GPUs: 1

Boot Disk:
  Operating System: Ubuntu
  Version: Ubuntu 22.04 LTS
  Boot disk type: SSD persistent disk
  Size: 100 GB

Firewall:
  ✓ Allow HTTP traffic
  ✓ Allow HTTPS traffic
```

### Using gcloud CLI:

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create VM with T4 GPU
gcloud compute instances create dynaprompt-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata=install-nvidia-driver=True
```

**Cost Estimate**: ~$0.50-0.70/hour with T4 GPU

## Step 2: SSH into VM

```bash
# From your local machine
gcloud compute ssh dynaprompt-gpu --zone=us-central1-a

# Or use the SSH button in GCP Console
```

## Step 3: Run Setup Script

```bash
# Upload setup script to VM
gcloud compute scp gcp_setup.sh dynaprompt-gpu:~ --zone=us-central1-a

# SSH into VM and run setup
ssh into VM
chmod +x gcp_setup.sh
./gcp_setup.sh

# If NVIDIA drivers were installed, reboot
sudo reboot

# Reconnect after reboot
gcloud compute ssh dynaprompt-gpu --zone=us-central1-a
```

## Step 4: Upload Your Code (if modified locally)

```bash
# From your local machine
gcloud compute scp --recurse C:\Users\zisho\6694-DynaPrompt dynaprompt-gpu:~ --zone=us-central1-a

# Or just pull latest from GitHub (already in setup script)
```

## Step 5: Upload SD Checkpoint

**Option A: Transfer from local machine** (if you have it):
```bash
gcloud compute scp C:\Users\zisho\6694-DynaPrompt\models\stable_diffusion_compvis\v1-5-pruned-emaonly.ckpt \
    dynaprompt-gpu:~/6694-DynaPrompt/models/stable_diffusion_compvis/ \
    --zone=us-central1-a
```

**Option B: Download directly on VM**:
```bash
# SSH into VM
cd ~/6694-DynaPrompt/models/stable_diffusion_compvis

# Download from HuggingFace
pip install huggingface-hub
huggingface-cli login  # Enter your HF token
huggingface-cli download runwayml/stable-diffusion-v1-5 \
    v1-5-pruned-emaonly.ckpt \
    --local-dir . \
    --local-dir-use-symlinks False

# Or use wget with direct URL (if you have access)
# wget -O v1-5-pruned-emaonly.ckpt "YOUR_URL_HERE"
```

## Step 6: Run DynaPrompt on GPU

```bash
# Activate environment
conda activate dynaprompt

# Verify GPU
python check_gpu.py

# Expected output:
# ✓ CUDA Available: True
# ✓ CUDA Device Count: 1
# ✓ Current CUDA Device: NVIDIA Tesla T4
# ✓ CUDA Version: 11.8

# Run single generation (should take ~2-3 minutes on T4)
python run_dynaprompt.py

# Run comparison (baseline vs DynaPrompt)
python compare_baseline.py

# Or use test script
python test_fixed_generation.py
```

## Step 7: Download Results

```bash
# From your local machine
gcloud compute scp --recurse \
    dynaprompt-gpu:~/6694-DynaPrompt/outputs \
    C:\Users\zisho\6694-DynaPrompt\ \
    --zone=us-central1-a
```

## Step 8: Stop VM (Important!)

```bash
# Stop VM to avoid charges when not in use
gcloud compute instances stop dynaprompt-gpu --zone=us-central1-a

# Start again when needed
gcloud compute instances start dynaprompt-gpu --zone=us-central1-a

# Delete VM when completely done
gcloud compute instances delete dynaprompt-gpu --zone=us-central1-a
```

## Performance Comparison

| Hardware | Steps | Time | Cost |
|----------|-------|------|------|
| **CPU (local)** | 30 | ~90 min | $0 |
| **GCP T4 GPU** | 30 | ~2 min | ~$0.02 |
| **GCP T4 GPU** | 50 | ~3 min | ~$0.03 |

## GPU Options on GCP

| GPU | Memory | Speed | Cost/hour | Best for |
|-----|--------|-------|-----------|----------|
| **T4** | 16 GB | 1x | $0.35 | Development/Testing |
| **V100** | 16 GB | 2x | $2.48 | Faster generation |
| **A100** | 40 GB | 3x | $3.67 | Batch generation |

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size or image resolution
python run_dynaprompt.py  # Uses 512x512 by default
```

### NVIDIA Driver Issues
```bash
# Check driver installation
nvidia-smi

# Reinstall if needed
sudo apt-get install --reinstall nvidia-driver-525
sudo reboot
```

### Environment Issues
```bash
# Recreate environment
conda env remove -n dynaprompt
./gcp_setup.sh
```

### Checkpoint Issues
```bash
# Verify checkpoint size (should be ~4GB)
ls -lh models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt

# Re-download if corrupted
rm models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt
# Then use Option B above to re-download
```

## Cost Optimization Tips

1. **Use Preemptible VMs** (70% cheaper but can be terminated):
   ```bash
   gcloud compute instances create dynaprompt-gpu --preemptible ...
   ```

2. **Stop VM when not in use** (storage costs ~$5/month, running costs ~$15/day)

3. **Use Cloud Storage** for checkpoints instead of VM disk

4. **Use smaller GPU** (T4 is sufficient for this project)

5. **Set automatic shutdown**:
   ```bash
   # Shutdown after 2 hours of inactivity
   sudo shutdown -h +120
   ```

## Quick Commands Reference

```bash
# Create VM
gcloud compute instances create dynaprompt-gpu --zone=us-central1-a --machine-type=n1-standard-4 --accelerator=type=nvidia-tesla-t4,count=1

# SSH
gcloud compute ssh dynaprompt-gpu --zone=us-central1-a

# Transfer files
gcloud compute scp LOCAL_FILE dynaprompt-gpu:REMOTE_PATH --zone=us-central1-a

# Stop VM
gcloud compute instances stop dynaprompt-gpu --zone=us-central1-a

# Start VM
gcloud compute instances start dynaprompt-gpu --zone=us-central1-a

# Check status
gcloud compute instances list

# View logs
gcloud compute instances get-serial-port-output dynaprompt-gpu --zone=us-central1-a
```

## Alternative: Jupyter Notebook on GCP

If you prefer Jupyter:

```bash
# On VM
conda activate dynaprompt
jupyter notebook --no-browser --port=8888

# On local machine (in new terminal)
gcloud compute ssh dynaprompt-gpu --zone=us-central1-a -- -L 8888:localhost:8888

# Open in browser: http://localhost:8888
```
