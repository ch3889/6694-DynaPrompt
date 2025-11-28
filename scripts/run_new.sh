#!/usr/bin/env bash
# Convenience wrapper to set up venv (if missing) and run the fresh sampler.
# ENV VARS (override by exporting before running or inline):
#   PROMPT        - Text prompt (default below)
#   STEPS         - Diffusion steps (default 50)
#   CFG           - Guidance scale (default 7.5)
#   SEED          - Random seed (default 123)
#   CHECK_STEP    - Early detection step (default 5)
#   MAX_RETRIES   - Seed retries before boosting (default 10)
#   THRESHOLD     - Attention threshold (default 0.05)
#   BOOST         - Base boost factor (default 6.0)
#   OUTDIR        - Output directory (default data/images/dynaprompt_new)

set -euo pipefail

# Defaults (can be overridden via exported env or inline before the command)
PROMPT=${PROMPT:-${1:-"A blue cat sitting on a red chair with a yellow ball"}}
STEPS=${STEPS:-50}
CFG=${CFG:-7.5}
SEED=${SEED:-123}
CHECK_STEP=${CHECK_STEP:-5}
MAX_RETRIES=${MAX_RETRIES:-10}
THRESHOLD=${THRESHOLD:-0.05}
BOOST=${BOOST:-6.0}
OUTDIR=${OUTDIR:-data/images/dynaprompt_new}

echo "=== DynaPrompt New Runner ==="
echo "Prompt: $PROMPT"
echo "Steps: $STEPS | CFG: $CFG | Seed: $SEED"
echo "Check step: $CHECK_STEP | Max retries: $MAX_RETRIES"
echo "Threshold: $THRESHOLD | Boost factor: $BOOST"
echo "Outdir: $OUTDIR"
echo "================================"

# Ensure venv exists
if [ ! -d ".venv" ]; then
  echo "Creating Python 3.10 virtualenv..."
  python3.10 -m venv .venv
fi

source .venv/bin/activate

# Install requirements if needed
echo "Installing requirements (if not installed)..."
pip install -r requirements.txt || true

# Ensure CompVis repo and weights exist (download script handles both)
if [ ! -d "models/stable_diffusion_compvis/stable-diffusion" ] || [ ! -f "models/stable_diffusion_compvis/v1-5-pruned-emaonly.ckpt" ]; then
  echo "CompVis repo or weights missing. Running download script..."
  bash scripts/download_models.sh
fi

# Run the fresh sampler
python scripts/test_dynaprompt_new.py \
  --prompt "$PROMPT" \
  --steps $STEPS \
  --cfg $CFG \
  --seed $SEED \
  --check_step $CHECK_STEP \
  --max_retries $MAX_RETRIES \
  --threshold $THRESHOLD \
  --boost_factor $BOOST \
  --outdir $OUTDIR

echo "\n✓ Completed. Images saved under: $OUTDIR"
