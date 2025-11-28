"""
Simple test script for dynaprompt_new.
Uses CompVis Stable Diffusion repo at the specified commit.
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

# Workspace-relative paths
WORKSPACE = Path(__file__).resolve().parents[1]
SD_ROOT = WORKSPACE / 'models' / 'stable_diffusion_compvis'
SD_REPO = SD_ROOT / 'stable-diffusion'

sys.path.insert(0, str(SD_REPO))
sys.path.insert(0, str(WORKSPACE))

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
except ModuleNotFoundError as e:
    print("[ERROR] Could not import 'ldm'. Ensure the CompVis repo is cloned under 'models/stable_diffusion_compvis/stable-diffusion' and PYTHONPATH is set.")
    print("Attempted path:", SD_REPO)
    print("Add to PYTHONPATH: export PYTHONPATH=$PWD/models/stable_diffusion_compvis/stable-diffusion:$PYTHONPATH")
    raise
from dynaprompt_new.sampler import DynapromptNewSampler


def load_model():
    config_path = SD_REPO / 'configs' / 'stable-diffusion' / 'v1-inference.yaml'
    ckpt_path = SD_ROOT / 'v1-5-pruned-emaonly.ckpt'
    print(f"Loading model from {ckpt_path}")

    config = OmegaConf.load(str(config_path))
    pl_sd = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    sd = pl_sd['state_dict']

    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Test dynaprompt_new")
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', type=str, default='data/images/dynaprompt_new')
    parser.add_argument('--check_step', type=int, default=5)
    parser.add_argument('--max_retries', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--boost_factor', type=float, default=6.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = load_model()
    sampler = DDIMSampler(model)

    dp = DynapromptNewSampler(
        ddim_sampler=sampler,
        model=model,
        tokenizer=model.cond_stage_model.tokenizer,
        check_step=args.check_step,
        attention_threshold=args.threshold,
        max_retries=args.max_retries,
        boost_factor=args.boost_factor,
        start_step_ratio=0.0,
        end_step_ratio=0.5,
    )

    print("Generating...")
    with torch.no_grad():
        shape = [1, 4, 64, 64]
        latents, _ = dp.sample(
            prompt=args.prompt,
            shape=shape,
            steps=args.steps,
            cfg_scale=args.cfg,
            verbose=True,
        )

    print("Decoding...")
    with torch.no_grad():
        imgs = model.decode_first_stage(latents)
        imgs = torch.clamp((imgs + 1.0) / 2.0, 0.0, 1.0)

    for i, x in enumerate(imgs):
        arr = (255.0 * x.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
        Image.fromarray(arr).save(outdir / f"dynaprompt_new_{i:04d}.png")
        print(f"Saved: {outdir / f'dynaprompt_new_{i:04d}.png'}")


if __name__ == '__main__':
    main()
