"""Find every erased checkpoint under the given paths and check that the scripts can read it.

    python setup/check_checkpoints.py data/checkpoints

For each checkpoint it prints which component it edits (unet or text_encoder), how many tensors
changed, the edit norm, and the exact --erased / --component values to use. Checkpoints in the
original CompVis/LDM format are reported as such; they need converting to diffusers format.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from edit import compute_edit, load_erased_state_dict  # noqa: E402

WEIGHT_EXT = (".pt", ".pth", ".bin", ".safetensors", ".ckpt")


def find_checkpoints(root):
    """Component folders (config.json + weights), pipeline folders, and loose weight files."""
    if os.path.isfile(root):
        return [root]
    found = []
    for d, subdirs, files in os.walk(root):
        if "model_index.json" in files or "config.json" in files:
            found.append(d)
            subdirs[:] = []  # don't descend into a model folder
            continue
        found += [os.path.join(d, f) for f in sorted(files) if f.endswith(WEIGHT_EXT)]
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--base", default="CompVis/stable-diffusion-v1-4")
    args = p.parse_args()

    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel

    bases = {
        "unet": UNet2DConditionModel.from_pretrained(args.base, subfolder="unet", torch_dtype=torch.float32),
        "text_encoder": CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder", torch_dtype=torch.float32),
    }

    for root in args.paths:
        for ck in find_checkpoints(root):
            print(f"\n== {ck}")
            hit = False
            for comp, module in bases.items():
                try:
                    sd = load_erased_state_dict(ck, comp, type(module))
                except Exception as e:  # wrong class for this folder, unreadable file, ...
                    print(f"   {comp:<12s} cannot load: {str(e).splitlines()[0][:120]}")
                    continue
                keys = [k for k in sd if torch.is_tensor(sd[k])]
                if any(k.startswith("model.diffusion_model.") for k in keys):
                    print("   CompVis/LDM format: convert to diffusers before use "
                          "(diffusers' scripts/convert_original_stable_diffusion_to_diffusers.py)")
                    hit = True
                    break
                try:
                    edit = compute_edit(module, sd)
                except Exception as e:
                    print(f"   {comp:<12s} no match: {str(e).splitlines()[0][:120]}")
                    continue
                print(f"   OK  --component {comp} --erased {ck}")
                print("   " + edit.summary().replace("\n", "\n   "))
                hit = True
                break
            if not hit:
                print("   could not use this checkpoint; print a few of its keys to see the format")


if __name__ == "__main__":
    main()