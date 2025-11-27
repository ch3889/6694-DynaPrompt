# How to Request GPU Quota in GCP

## Quick Fix

1. **Go to Quotas page**
   - Visit: https://console.cloud.google.com/iam-admin/quotas
   - Or: GCP Console → IAM & Admin → Quotas

2. **Filter for GPU quotas**
   - In the filter box, type: `GPUs (all regions)`
   - OR search: `NVIDIA T4 GPUs`

3. **Select and edit quota**
   - Check the box next to "GPUs (all regions)"
   - Click "EDIT QUOTAS" at the top
   - Set new limit: `1` (or higher if you want)
   - Add justification: "Running deep learning research for university project"
   - Click SUBMIT REQUEST

4. **Wait for approval**
   - Usually approved in 5-30 minutes
   - Check email for confirmation
   - Sometimes instant for small requests

## Alternative: Use Different Region

Some regions have quota available by default. Try these zones:

```powershell
# Edit terraform/terraform.tfvars, add:
zone = "us-west1-b"    # Or try us-east1-b, europe-west4-a
```

Then:
```powershell
terraform apply
```

## Alternative: Use Preemptible GPU (Sometimes has separate quota)

Edit `terraform/main.tf` line 78:
```hcl
scheduling {
  preemptible = true  # 70% cheaper + may bypass quota
}
```

## Check Your Current Quotas

```powershell
gcloud compute project-info describe --project=YOUR_PROJECT_ID
```

Or visit: https://console.cloud.google.com/iam-admin/quotas

## What to Request

| Quota Name | Current | Request | Reason |
|------------|---------|---------|--------|
| GPUs (all regions) | 0 | 1 | For T4 GPU |
| NVIDIA T4 GPUs | 0 | 1 | Specific T4 |
| Compute Engine API CPUs | - | 4+ | For n1-standard-4 |

## If Quota Request is Denied

Try these alternatives:

### Option 1: Use CPU-only (slower but free quota)
```powershell
# In terraform/terraform.tfvars, comment out GPU:
# gpu_count = 0

# Or deploy without Terraform:
gcloud compute instances create dynaprompt-cpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --image-family=pytorch-latest-cpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB
```

### Option 2: Use Colab Pro ($10/month, instant T4 access)
- Go to: https://colab.research.google.com/
- Upload your code
- Enable GPU: Runtime → Change runtime type → T4 GPU
- No quota needed!

### Option 3: Use different cloud provider
- AWS SageMaker (free tier)
- Azure ML (free credits for students)
- Lambda Labs ($0.50/hr for A10)

## While Waiting for Quota

Test your code on CPU (slower but works):

```powershell
# Connect to local environment
cd C:\Users\zisho\6694-DynaPrompt

# Run with CPU (will take 90 min instead of 2 min)
C:\Users\zisho\anaconda3\envs\dynaprompt\python.exe run_dynaprompt.py
```

Or just run a few steps for testing:
```python
# Edit run_dynaprompt.py, change:
steps=5  # Instead of 30, just for testing (~15 min on CPU)
```

## Check Quota Status

```powershell
# List all quotas
gcloud compute project-info describe --project=YOUR_PROJECT_ID | Select-String -Pattern "GPUS"

# Check specific region
gcloud compute regions describe us-central1 | Select-String -Pattern "GPUS"
```

## Contact Support (for faster approval)

If urgent:
1. Go to: https://console.cloud.google.com/support
2. Create case: "Need GPU quota for academic research"
3. Mention: "$300 free trial, university project, need 1 T4 GPU"
4. Usually get response within hours

## Expected Timeline

| Method | Approval Time |
|--------|---------------|
| Self-service quota request | 5-30 minutes |
| Different region | Instant (if available) |
| Support ticket | 1-4 hours |
| Preemptible GPU | Instant (if quota available) |
