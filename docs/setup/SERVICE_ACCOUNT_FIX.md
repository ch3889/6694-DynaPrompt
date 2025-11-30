# Quick Fix Options for Service Account Error

## Option 1: Use Simplified Terraform (Recommended)

```powershell
cd C:\Users\zisho\6694-DynaPrompt\terraform

# Backup current config
mv main.tf main_old.tf

# Use simplified version (no service account issues)
mv main_simple.tf main.tf

# Re-initialize and deploy
terraform init
terraform apply
```

## Option 2: Skip Terraform - Use gcloud CLI Directly

```powershell
gcloud compute instances create dynaprompt-dl `
    --project=gen-ai-479417 `
    --zone=us-central1-a `
    --machine-type=n1-standard-4 `
    --accelerator=type=nvidia-tesla-t4,count=1 `
    --image-family=pytorch-latest-gpu `
    --image-project=deeplearning-platform-release `
    --boot-disk-size=100GB `
    --boot-disk-type=pd-standard `
    --maintenance-policy=TERMINATE `
    --metadata=install-nvidia-driver=True `
    --scopes=cloud-platform `
    --tags=http-server,https-server
```

This is the simplest and most reliable method.

## Option 3: Use GCP Console Directly

1. Go to: https://console.cloud.google.com/marketplace/product/click-to-deploy-images/deeplearning
2. Click **"LAUNCH"**
3. Fill in:
   ```
   Deployment name: dynaprompt-dl
   Zone: us-central1-a
   Machine type: n1-standard-4
   Framework: PyTorch 2.0 (CUDA 11.8)
   GPUs: 1 x NVIDIA Tesla T4
   Disk: 100GB Standard
   ```
4. Click **"DEPLOY"**

This avoids all Terraform/CLI issues.

## Option 4: Fix Service Account in Terraform

If you want to keep using Terraform with the fixed main.tf:

```powershell
cd C:\Users\zisho\6694-DynaPrompt\terraform

# Destroy any partial deployment
terraform destroy

# Re-initialize
terraform init -upgrade

# Try again
terraform apply
```

The updated main.tf now uses the default Compute Engine service account instead of trying to create a custom one.

## Recommended: Use gcloud CLI (Fastest)

This is the most reliable and avoids all Terraform complexity:

```powershell
# Single command to create VM
gcloud compute instances create dynaprompt-dl `
    --project=gen-ai-479417 `
    --zone=us-central1-a `
    --machine-type=n1-standard-4 `
    --accelerator=type=nvidia-tesla-t4,count=1 `
    --image-family=pytorch-latest-gpu `
    --image-project=deeplearning-platform-release `
    --boot-disk-size=100GB `
    --metadata=install-nvidia-driver=True `
    --scopes=cloud-platform

# Wait 3-5 minutes, then connect
gcloud compute ssh dynaprompt-dl --zone=us-central1-a

# Inside VM, clone and setup
git clone https://github.com/ch3889/6694-DynaPrompt.git
cd 6694-DynaPrompt
git checkout zk2295
pip install -r requirements_gpu.txt
```

## If Still Getting Errors

Check if you have GPU quota:
```powershell
gcloud compute project-info describe --project=gen-ai-479417
```

Request quota if needed:
https://console.cloud.google.com/iam-admin/quotas?project=gen-ai-479417
