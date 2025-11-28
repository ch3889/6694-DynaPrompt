#!/usr/bin/env python3
"""
Simple script to fix pytorch_lightning import in ddpm.py
Run this once on GCP before testing.
"""
import os
import sys

# Get the path to the ddpm.py file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ddpm_path = os.path.join(project_root, "models/stable_diffusion_compvis/ldm/models/diffusion/ddpm.py")

print(f"Fixing {ddpm_path}...")

# Read the file
with open(ddpm_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Check if already patched
if any("Create shim if neither import works" in line for line in lines):
    print("✓ Already patched!")
    sys.exit(0)

# Find and replace the import line
new_lines = []
for i, line in enumerate(lines):
    if "from pytorch_lightning.utilities.distributed import rank_zero_only" in line and "try:" not in lines[i-1]:
        # Replace with the fixed version
        new_lines.append("# Fixed for pytorch_lightning compatibility\n")
        new_lines.append("try:\n")
        new_lines.append("    from pytorch_lightning.utilities.rank_zero import rank_zero_only\n")
        new_lines.append("except (ImportError, ModuleNotFoundError):\n")
        new_lines.append("    try:\n")
        new_lines.append("        from pytorch_lightning.utilities.distributed import rank_zero_only\n")
        new_lines.append("    except (ImportError, ModuleNotFoundError):\n")
        new_lines.append("        # Create shim if neither import works\n")
        new_lines.append("        def rank_zero_only(fn):\n")
        new_lines.append("            return fn\n")
    else:
        new_lines.append(line)

# Write back
with open(ddpm_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✓ Patched successfully!")
print("  Fixed pytorch_lightning.utilities import")
