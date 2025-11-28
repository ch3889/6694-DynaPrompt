"""
Patch CompVis ddpm.py to fix pytorch_lightning import issue
Run this before running hybrid tests
"""

import os
import sys

DDPM_PATH = "models/stable_diffusion_compvis/ldm/models/diffusion/ddpm.py"

def patch_ddpm():
    if not os.path.exists(DDPM_PATH):
        print(f"Error: {DDPM_PATH} not found")
        return False
    
    print(f"Patching {DDPM_PATH}...")
    
    with open(DDPM_PATH, 'r') as f:
        lines = f.readlines()
    
    # Find and replace the problematic import section (around lines 18-20)
    fixed_lines = []
    skip_next = 0
    
    for i, line in enumerate(lines):
        if skip_next > 0:
            skip_next -= 1
            continue
            
        # Look for the try/except block for rank_zero_only
        if 'from pytorch_lightning.utilities.rank_zero import rank_zero_only' in line:
            # Replace this entire section with a more robust version
            fixed_lines.append('# Fixed for pytorch_lightning compatibility\n')
            fixed_lines.append('try:\n')
            fixed_lines.append('    from pytorch_lightning.utilities.rank_zero import rank_zero_only\n')
            fixed_lines.append('except (ImportError, ModuleNotFoundError):\n')
            fixed_lines.append('    try:\n')
            fixed_lines.append('        from pytorch_lightning.utilities.distributed import rank_zero_only\n')
            fixed_lines.append('    except (ImportError, ModuleNotFoundError):\n')
            fixed_lines.append('        # Create shim if neither import works\n')
            fixed_lines.append('        def rank_zero_only(fn):\n')
            fixed_lines.append('            return fn\n')
            fixed_lines.append('\n')
            
            # Skip the next 2 lines (the except and the other import)
            skip_next = 2
        else:
            fixed_lines.append(line)
    
    # Write back
    with open(DDPM_PATH, 'w') as f:
        f.writelines(fixed_lines)
    
    print("✓ Patched successfully!")
    print("  Added fallback for missing pytorch_lightning.utilities modules")
    return True

if __name__ == "__main__":
    success = patch_ddpm()
    sys.exit(0 if success else 1)
