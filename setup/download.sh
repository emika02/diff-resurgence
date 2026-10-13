#!/usr/bin/env bash
# Downloads everything the erasure-stability scripts need.
#
#   bash setup/download.sh [DATA_DIR]          (default: ./data)
#
# Set HF_HOME / TORCH_HOME beforehand to keep the model caches somewhere else (on a cluster:
# scratch, not home). Use the same values later when running the experiments.
# Skip parts with e.g.  SKIP_BASELINES=1 SKIP_COCO=1 bash setup/download.sh
set -euo pipefail

DATA=$(realpath -m "${1:-./data}")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DATA/checkpoints"
export HF_HOME=${HF_HOME:-$DATA/hf_cache}
export TORCH_HOME=${TORCH_HOME:-$DATA/torch_cache}
echo "data dir: $DATA   HF_HOME: $HF_HOME   TORCH_HOME: $TORCH_HOME"

# ------------------------------------------------------------------ 0. python packages
pip install -q torch torchvision diffusers transformers accelerate safetensors \
               huggingface_hub pandas matplotlib nudenet gdown

# ------------------------------------------------------------------ 1. models (~7 GB)
# Stable Diffusion v1.4 (without the safety checker), CLIP ViT-L/14, ResNet-50, NudeNet,
# and AdvUnlearn's erased text encoders (nudity, Van Gogh, parachute; ~0.5 GB each).
python - <<EOF
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor
from torchvision.models import resnet50, ResNet50_Weights
from huggingface_hub import snapshot_download

StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None,
                                        requires_safety_checker=False)
CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
from nudenet import NudeDetector; NudeDetector()
snapshot_download("OPTML-Group/AdvUnlearn", local_dir="$DATA/checkpoints/advunlearn",
                  allow_patterns=["nudity_unlearned/*", "vangogh_unlearned/*", "parachute_unlearned/*"])
print("models OK")
EOF

# ------------------------------------------------------------------ 2. prompts (small)
# nudity.csv (I2P), vangogh.csv, church/parachute/tench/garbage_truck.csv, coco_10k.csv
if [ ! -d "$DATA/Diffusion-MU-Attack" ]; then
  git clone --depth 1 https://github.com/OPTML-Group/Diffusion-MU-Attack.git "$DATA/Diffusion-MU-Attack"
fi

# ------------------------------------------------------------------ 3. erased U-Nets (large)
# ESD, FMN, UCE, SalUn, ScissorHands, EraseDiff, SPM, as released with AdvUnlearn.
# Each U-Net is ~3.4 GB, so these archives can be tens of GB. Google Drive sometimes refuses
# big downloads ("quota exceeded"); if that happens, open the link in a browser instead:
#   https://drive.google.com/drive/folders/1QZ-zUgFrNNXs1EycpujKVJIRYl-4MkMF
if [ -z "${SKIP_BASELINES:-}" ]; then
  mkdir -p "$DATA/checkpoints/baselines" && cd "$DATA/checkpoints/baselines"
  gdown -c https://drive.google.com/uc?id=1wOqACzdpWpjRjl-a3hWgtjGAhemnelQr   # nudity
  gdown -c https://drive.google.com/uc?id=1-Gyl7Ls-Pa4vJzReG58bXjfI_YWu8teb   # Van Gogh
  gdown -c https://drive.google.com/uc?id=1OjYiFxYwd1B9R7vfKG6mooY27O2txYNg   # objects
  for f in *.zip; do [ -e "$f" ] && unzip -q -n "$f"; done
  for f in *.tar *.tar.gz *.tgz; do [ -e "$f" ] && tar -xf "$f"; done
  cd - > /dev/null
fi

# ------------------------------------------------------------------ 4. COCO val2017 (~1.3 GB)
# Real neutral images + captions for loss_stability.py
if [ -z "${SKIP_COCO:-}" ]; then
  mkdir -p "$DATA/coco" && cd "$DATA/coco"
  [ -d val2017 ] || { curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip && rm val2017.zip; }
  [ -d annotations ] || { curl -L -o ann.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q ann.zip && rm ann.zip; }
  cd - > /dev/null
  python "$HERE/make_coco_captions.py" "$DATA/coco" "$DATA/coco_neutral.csv" --n 512
fi

echo
echo "done. Next: python setup/check_checkpoints.py $DATA/checkpoints"