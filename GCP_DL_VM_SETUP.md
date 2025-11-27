# DynaPrompt on GCP Deep Learning VM

## Step 1: Create Deep Learning VM

1. Go to [GCP Console > Marketplace](https://console.cloud.google.com/marketplace)
2. Search for **"Deep Learning VM"**
3. Click **"LAUNCH"**

### Configuration:

```
Deployment name: dynaprompt-dl
Zone: us-central1-a

Framework:
  - PyTorch 2.0 (CUDA 11.8) ← Recommended
  - OR PyTorch 2.1 (CUDA 12.1)

Machine configuration:
  - Machine type: n1-standard-4
    (4 vCPUs, 15 GB memory)

GPUs:
  - GPU type: NVIDIA Tesla T4
  - Number of GPUs: 1

Boot disk:
  - Disk type: Standard Persistent Disk
  - Size (GB): 100

Networking:
  ✓ Install NVIDIA GPU driver automatically
  ✓ Enable access to all Cloud APIs

Firewall:
  ✓ Allow HTTP traffic
  ✓ Allow HTTPS traffic
```

Click **"DEPLOY"** (takes 3-5 minutes)

## Step 2: Connect to VM

```powershell
# From your local PowerShell
gcloud compute ssh dynaprompt-dl --zone=us-central1-a
```

## Step 3: Run Setup Script

```bash
# Upload setup script
gcloud compute scp gcp_dl_vm_setup.sh dynaprompt-dl:~ --zone=us-central1-a

# SSH and run
gcloud compute ssh dynaprompt-dl --zone=us-central1-a
chmod +x gcp_dl_vm_setup.sh
./gcp_dl_vm_setup.sh
```

## Step 4: Upload SD Checkpoint

```powershell
# From local PowerShell (4GB upload, ~5-10 minutes)
gcloud compute scp C:\Users\zisho\6694-DynaPrompt\models\stable_diffusion_compvis\v1-5-pruned-emaonly.ckpt dynaprompt-dl:~/6694-DynaPrompt/models/stable_diffusion_compvis/ --zone=us-central1-a
```

**OR download on VM:**

```bash
# Inside VM
cd ~/6694-DynaPrompt/models/stable_diffusion_compvis
pip install huggingface-hub
huggingface-cli login  # Enter your HF token
huggingface-cli download runwayml/stable-diffusion-v1-5 v1-5-pruned-emaonly.ckpt --local-dir . --local-dir-use-symlinks False
```

## Step 5: Test GPU

```bash
cd ~/6694-DynaPrompt
python check_gpu.py
```

Expected output:
```
✓ CUDA Available: True
✓ CUDA Device: NVIDIA Tesla T4
✓ Device Count: 1
```

## Step 6: Run DynaPrompt

```bash
# Single generation (~2-3 minutes on T4)
python run_dynaprompt.py

# Full comparison
python compare_baseline.py

# Test fixed version
python test_fixed_generation.py
```

## Step 7: Download Results

```powershell
# From local PowerShell
gcloud compute scp --recurse dynaprompt-dl:~/6694-DynaPrompt/outputs C:\Users\zisho\6694-DynaPrompt\ --zone=us-central1-a
```

## Cost Management

### Stop VM when not in use:
```powershell
gcloud compute instances stop dynaprompt-dl --zone=us-central1-a
```

### Start VM again:
```powershell
gcloud compute instances start dynaprompt-dl --zone=us-central1-a
```

### Check your spending:
- Go to [Billing](https://console.cloud.google.com/billing)
- View your $300 credit usage

### Set budget alerts:
1. Go to **Billing > Budgets & alerts**
2. Create budget: $50, $100, $200 thresholds
3. Get email notifications

## Performance Expectations

| Task | CPU (local) | T4 GPU | Speedup |
|------|-------------|--------|---------|
| 30 steps | ~90 min | ~2-3 min | 30-45x |
| 50 steps | ~150 min | ~4-5 min | 30-40x |
| Comparison | ~180 min | ~5-6 min | 30-35x |

## Troubleshooting

### GPU not detected:
```bash
# Check driver
nvidia-smi

# Reinstall if needed
sudo /opt/deeplearning/install-driver.sh
sudo reboot
```

### CUDA out of memory:
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Reduce batch size if needed (already 1 in config)
```

### Slow download speeds:
```bash
# Use gsutil for large files
gsutil cp gs://your-bucket/checkpoint.ckpt ~/
```

## Cost Optimization Tips

1. **Use preemptible instances** (70% cheaper):
   - Check "Enable Preemptible" in deployment
   - VM can be terminated but costs ~$0.16/hr instead of $0.54/hr

2. **Stop VM between sessions**:
   - Stopped VM: ~$0.40/day (storage only)
   - Running VM: ~$13/day

3. **Use smaller disk**:
   - 50GB disk if you don't need many generations
   - Saves ~$0.34/month

4. **Delete VM when completely done**:
   ```bash
   gcloud compute instances delete dynaprompt-dl --zone=us-central1-a
   ```

5. **Set automatic shutdown**:
   ```bash
   # Shutdown after 2 hours
   sudo shutdown -h +120
   ```

## Quick Commands

```bash
# Connect
gcloud compute ssh dynaprompt-dl --zone=us-central1-a

# Upload file
gcloud compute scp LOCAL_FILE dynaprompt-dl:REMOTE_PATH --zone=us-central1-a

# Download results
gcloud compute scp --recurse dynaprompt-dl:~/6694-DynaPrompt/outputs . --zone=us-central1-a

# Stop VM
gcloud compute instances stop dynaprompt-dl --zone=us-central1-a

# Start VM
gcloud compute instances start dynaprompt-dl --zone=us-central1-a

# Check status
gcloud compute instances list

# SSH with port forwarding (for Jupyter)
gcloud compute ssh dynaprompt-dl --zone=us-central1-a -- -L 8888:localhost:8888
```

## $300 Credit Usage Estimate

| Usage Pattern | Duration | Generations | Credit Left |
|--------------|----------|-------------|-------------|
| **Conservative** | 100 hours | ~2000 | ~$246 |
| **Moderate** | 200 hours | ~4000 | ~$192 |
| **Heavy** | 400 hours | ~8000 | ~$84 |

**Recommendation**: Use ~100 hours for development/testing, keep $200+ for final experiments.
