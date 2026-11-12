"""Inspect transformer checkpoint."""
import sys
import os
import pathlib

# Windows PosixPath compatibility (checkpoint trained on Linux)
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

import torch

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
print('=== model_config ===')
for k, v in ckpt.get('model_config', {}).items():
    print(f'  {k}: {v}')
print(f'\n=== state_dict keys ({len(ckpt["model_state_dict"])}) ===')
for k in list(ckpt['model_state_dict'].keys()):
    print(f'  {k}: {ckpt["model_state_dict"][k].shape}')
