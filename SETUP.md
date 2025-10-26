# DynaPrompt Setup Guide

Step-by-step instructions to get the project running.

---

## Step 1: Clone Repository

```bash
git clone git@github.com:ch3889/6694-DynaPrompt.git
cd 6694-DynaPrompt
git checkout ch3889
```

---

## Step 2: Create Environment

```bash
conda create -n dynaprompt python=3.10
conda activate dynaprompt
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install invisible-watermark
pip install pytorch-lightning==1.9.0
pip install taming-transformers-rom1504
pip install kornia
```

---

## Step 4: Get CompVis Stable Diffusion

```bash
git submodule update --init --recursive
cd models/stable_diffusion_compvis
pip install -e .
cd ../..
```

---

## Step 5: Apply Compatibility Patches

Open these files in `models/stable_diffusion_compvis/` and make the following changes:

### scripts/txt2img.py

**Line 51:** Add `weights_only=False`
```python
pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
```

**Lines 64-73:** Replace device selection
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
    model = model.to(device)
elif torch.cuda.is_available():
    model.cuda()
else:
    pass
model.eval()
return model
```

**Lines 256-265:** Replace device selection
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
model = model.to(device)
```

### ldm/models/diffusion/ddim.py

**Lines 19-28:** Replace register_buffer method
```python
def register_buffer(self, name, attr):
    if type(attr) == torch.Tensor:
        model_device = next(self.model.parameters()).device
        if attr.device != model_device:
            if model_device.type == 'mps' and attr.dtype == torch.float64:
                attr = attr.float()
            attr = attr.to(model_device)
    setattr(self, name, attr)
```

### ldm/modules/encoders/modules.py

**Lines 155-157:** Replace device assignment
```python
device = next(self.transformer.parameters()).device
tokens = batch_encoding["input_ids"].to(device)
```

See `patches/compvis_mac_compatibility.patch` for detailed patch locations.

---

## Step 6: Download Model Weights

```bash
cd models/stable_diffusion_compvis
curl -L -o v1-5-pruned-emaonly.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
cd ../..
```

This downloads ~4GB. Manual download: https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main

---

## Step 7: Test

```bash
cd models/stable_diffusion_compvis
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python scripts/txt2img.py \
  --prompt "A golden retriever playing with a red ball" \
  --ckpt v1-5-pruned-emaonly.ckpt \
  --n_samples 1 \
  --H 512 \
  --W 512 \
  --ddim_steps 50
```

Output image location: `outputs/txt2img-samples/samples/`

---

## Troubleshooting

**No module named 'ldm'**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Out of memory**
```bash
python scripts/txt2img.py --prompt "test" --ckpt v1-5-pruned-emaonly.ckpt --H 256 --W 256
```

---

## Contact

- Charles: ch3889@columbia.edu
- Max: zk2295@columbia.edu
- Swapnil: sb5041@columbia.edu
