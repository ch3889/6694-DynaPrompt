# DynaPrompt

EECS 6694 Deep Learning Project - Columbia University

**Team:** Charles Hou (ch3889), Max Kim (zk2295), Swapnil Banerjee (sb5041)

## What is this?

Stable Diffusion sometimes ignores parts of your prompt. Ask for "a red cube on a blue cube" and you might get two gray cubes.

DynaPrompt fixes this by boosting attention to underrepresented tokens during generation. It modifies the U-Net's cross-attention layers to make the model pay more attention to the things it's ignoring.

## How it works

1. During diffusion, we monitor which tokens get low attention
2. We boost attention weights for those tokens (colors, objects, etc.)
3. The model generates images that better match the prompt

No retraining needed - just hooks into existing Stable Diffusion.

## Quick Start

```bash
# Setup
conda create -n dynaprompt python=3.10
conda activate dynaprompt
pip install -r requirements.txt

# Download SD 1.5 weights to models/stable_diffusion_compvis/

# Run
python scripts/test_v3.py --prompt "a red cube on top of a blue cube"
```

## Project Structure

```
dynaprompt/           # Core attention modification code
  dynaprompt_v3.py    # Main sampler with attention boosting
  attention_modifier.py # Hooks into U-Net cross-attention

scripts/              # Evaluation scripts
  eval_drawbench_v3.py # DrawBench benchmark (50 prompts, multiple boost levels)

data/                 # Generated images and results
```

## Results

Testing on DrawBench (color binding, multi-object, spatial prompts) with different boost factors:
- 1.0x (baseline) - no boosting
- 2.5x, 5.0x, 7.5x, 10.0x - increasing attention boost

Higher boost = model pays more attention to underrepresented tokens.

## Dependencies

- PyTorch
- Stable Diffusion 1.5 (CompVis)
- CLIP (for evaluation)
- transformers
