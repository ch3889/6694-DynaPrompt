#!/bin/bash
# Patch CompVis ddpm.py to fix pytorch_lightning import

DDPM_FILE="models/stable_diffusion_compvis/ldm/models/diffusion/ddpm.py"

if [ -f "$DDPM_FILE" ]; then
    echo "Patching $DDPM_FILE for pytorch_lightning compatibility..."
    
    # Create backup
    cp "$DDPM_FILE" "${DDPM_FILE}.backup"
    
    # Replace the problematic import lines
    sed -i '19,21d' "$DDPM_FILE"
    sed -i '18a\
# Fixed for pytorch_lightning compatibility\
try:\
    from pytorch_lightning.utilities.rank_zero import rank_zero_only\
except (ImportError, ModuleNotFoundError):\
    try:\
        from pytorch_lightning.utilities.distributed import rank_zero_only\
    except (ImportError, ModuleNotFoundError):\
        # Create shim if neither import works\
        def rank_zero_only(fn):\
            return fn' "$DDPM_FILE"
    
    echo "✓ Patched successfully"
else
    echo "Error: $DDPM_FILE not found"
    exit 1
fi
