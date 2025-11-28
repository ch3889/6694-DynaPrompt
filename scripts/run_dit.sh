#!/usr/bin/env zsh
# Convenience runner for the Diffusion Transformer (DiT) dynaprompt sampler.
# ENV VARS (override by exporting before running or inline):
#   PROMPT  - Text prompt (default below)
#   STEPS   - Diffusion steps (default 20)
#   CFG     - Guidance scale (default 4.0)

set -e

# Defaults (can be overridden via exported env or inline before the command)
PROMPT=${PROMPT:-${1:-"A striped cat and a spotted dog playing together in a park"}}
STEPS=${STEPS:-20}
CFG=${CFG:-4.0}

echo "=== DynaPrompt DiT Runner ==="
echo "Prompt: $PROMPT"
echo "Steps: $STEPS | CFG: $CFG"
echo "================================"

# Ensure venv exists
if [ ! -d ".venv" ]; then
  echo "Creating Python 3.10 virtualenv..."
  python3.10 -m venv .venv
fi

source .venv/bin/activate

# Install requirements (torch, numpy, tqdm already in requirements.txt)
echo "Installing requirements (if not installed)..."
pip install -r requirements.txt || true

# Run the DiT sampler test (uses dummy components unless wired to a real DiT)
python scripts/test_dynaprompt_dit.py \
  --prompt "$PROMPT" \
  --steps $STEPS \
  --cfg $CFG

echo "\n✓ Completed DiT run (dummy). Wire real DiT components for images."
