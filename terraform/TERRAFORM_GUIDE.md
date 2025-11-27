# DynaPrompt Terraform Deployment Guide

## Prerequisites

1. **Install Terraform**
   - Download from: https://www.terraform.io/downloads
   - Or use Chocolatey: `choco install terraform`
   - Verify: `terraform version`

2. **Install Google Cloud SDK**
   - Already installed if `gcloud` works
   - If not: https://cloud.google.com/sdk/docs/install

3. **Authenticate**
   ```powershell
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

## Quick Start

### Step 1: Navigate to Terraform directory
```powershell
cd C:\Users\zisho\6694-DynaPrompt\terraform
```

### Step 2: Create terraform.tfvars
```powershell
# Copy example file
cp terraform.tfvars.example terraform.tfvars

# Edit with your project ID
notepad terraform.tfvars
```

Replace `your-project-id-here` with your actual GCP project ID.

### Step 3: Initialize Terraform
```powershell
terraform init
```

### Step 4: Preview the deployment
```powershell
terraform plan
```

This shows what will be created without actually creating it.

### Step 5: Deploy
```powershell
terraform apply
```

Type `yes` when prompted. Deployment takes ~3-5 minutes.

### Step 6: Get connection info
```powershell
terraform output
```

Shows:
- SSH command
- External IP
- Estimated cost
- Setup status check

### Step 7: Connect to VM
```powershell
# Use the SSH command from terraform output
gcloud compute ssh dynaprompt-dl --zone=us-central1-a

# Or get it directly
terraform output -raw ssh_command | Invoke-Expression
```

### Step 8: Check setup completion
```powershell
# Wait 2-3 minutes for startup script, then check
gcloud compute ssh dynaprompt-dl --zone=us-central1-a --command='cat /tmp/setup_complete.txt'
```

### Step 9: Upload SD checkpoint
```powershell
# From local PowerShell
gcloud compute scp C:\Users\zisho\6694-DynaPrompt\models\stable_diffusion_compvis\v1-5-pruned-emaonly.ckpt dynaprompt-dl:~/6694-DynaPrompt/models/stable_diffusion_compvis/ --zone=us-central1-a
```

### Step 10: Run DynaPrompt
```bash
# Inside VM
cd ~/6694-DynaPrompt
python check_gpu.py
python run_dynaprompt.py
```

## Customization Options

### Use preemptible instance (70% cheaper)
Edit `main.tf`, line ~78:
```hcl
scheduling {
  preemptible = true  # Change from false to true
}
```

### Change GPU type
```powershell
terraform apply -var="gpu_type=nvidia-tesla-v100"
```

### Change machine type
```powershell
terraform apply -var="machine_type=n1-standard-8"
```

### Change disk size
```powershell
terraform apply -var="disk_size_gb=200"
```

## Management Commands

### Stop VM (save money)
```powershell
gcloud compute instances stop dynaprompt-dl --zone=us-central1-a
```

### Start VM
```powershell
gcloud compute instances start dynaprompt-dl --zone=us-central1-a
```

### Check VM status
```powershell
terraform show
```

### Destroy everything (cleanup)
```powershell
terraform destroy
```

Type `yes` to confirm. This deletes all resources.

## Cost Management

### View current costs
```powershell
# Check your outputs
terraform output estimated_cost_per_hour

# Monitor actual costs in GCP Console
# https://console.cloud.google.com/billing
```

### Set budget alerts
```powershell
# Not in Terraform - do this in GCP Console
# Billing > Budgets & alerts
# Set alerts at $50, $100, $200
```

### Cost optimization
```hcl
# In terraform.tfvars or -var flags:

# Cheaper disk
disk_size_gb = 50  # Instead of 100

# Preemptible (70% cheaper)
# Edit main.tf: preemptible = true

# Smaller GPU (if available)
gpu_type = "nvidia-tesla-t4"  # Already cheapest option
```

## Troubleshooting

### "Quota exceeded" error
```
Error: googleapi: Error 403: Quota 'GPUS_ALL_REGIONS' exceeded
```

**Solution**: Request GPU quota increase
1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter: "GPUs (all regions)"
3. Select quota → Edit quotas
4. Request limit: 1 (for T4)
5. Wait for approval (~5 minutes to 1 day)

### "Terraform not found"
```powershell
# Install Terraform
choco install terraform

# Or download manually
# https://www.terraform.io/downloads
```

### GPU not detected after deployment
```bash
# SSH into VM and check
nvidia-smi

# If not working, reinstall driver
sudo /opt/deeplearning/install-driver.sh
sudo reboot
```

### Startup script didn't run
```bash
# Check logs
sudo journalctl -u google-startup-scripts.service

# Run setup manually
cd ~
git clone https://github.com/ch3889/6694-DynaPrompt.git
cd 6694-DynaPrompt
git checkout zk2295
pip install -r requirements_gpu.txt
```

### Can't SSH into VM
```powershell
# Check firewall rules
gcloud compute firewall-rules list

# Add SSH rule if missing
gcloud compute firewall-rules create allow-ssh --allow tcp:22
```

## State Management

Terraform stores state in `terraform.tfstate`. **Do not delete this file** or you'll lose track of your resources.

### Backup state
```powershell
cp terraform.tfstate terraform.tfstate.backup
```

### View state
```powershell
terraform state list
terraform state show google_compute_instance.dynaprompt_gpu
```

## Advanced Usage

### Multiple environments
```powershell
# Create dev and prod workspaces
terraform workspace new dev
terraform workspace new prod

# Switch between them
terraform workspace select dev
terraform apply

terraform workspace select prod
terraform apply
```

### Remote state (team collaboration)
Edit `main.tf`, add:
```hcl
terraform {
  backend "gcs" {
    bucket = "your-terraform-state-bucket"
    prefix = "dynaprompt/state"
  }
}
```

## Complete Workflow Example

```powershell
# 1. Setup
cd C:\Users\zisho\6694-DynaPrompt\terraform
cp terraform.tfvars.example terraform.tfvars
notepad terraform.tfvars  # Add your project_id

# 2. Deploy
terraform init
terraform plan
terraform apply  # Type 'yes'

# 3. Wait for deployment (3-5 min)
timeout /t 180

# 4. Check setup
terraform output -raw setup_status | Invoke-Expression

# 5. Upload checkpoint
gcloud compute scp C:\Users\zisho\6694-DynaPrompt\models\stable_diffusion_compvis\v1-5-pruned-emaonly.ckpt dynaprompt-dl:~/6694-DynaPrompt/models/stable_diffusion_compvis/ --zone=us-central1-a

# 6. Connect and run
gcloud compute ssh dynaprompt-dl --zone=us-central1-a
# Now inside VM:
cd ~/6694-DynaPrompt
python check_gpu.py
python run_dynaprompt.py

# 7. Download results (from local PowerShell)
gcloud compute scp --recurse dynaprompt-dl:~/6694-DynaPrompt/outputs . --zone=us-central1-a

# 8. Stop VM when done
gcloud compute instances stop dynaprompt-dl --zone=us-central1-a

# 9. Cleanup when completely finished
terraform destroy  # Type 'yes'
```

## Estimated Timeline

- Terraform init: 30 seconds
- Terraform apply: 3-5 minutes
- Startup script: 2-3 minutes
- Checkpoint upload: 5-10 minutes (4GB file)
- First generation: 2-3 minutes on GPU

**Total**: ~15-20 minutes to first generation

## Cost Tracking

| Action | Cost Impact |
|--------|-------------|
| Deploy VM | $0.54/hour starts |
| Stop VM | ~$0.02/day (storage only) |
| Start VM | $0.54/hour resumes |
| Destroy VM | $0 (everything deleted) |

With $300 credit:
- 555 hours of runtime
- ~1,100+ generations
- Or 50+ days of 4-hour sessions
