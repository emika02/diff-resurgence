#!/bin/bash
#SBATCH -J gen-stab
#SBATCH --account=plgideascvgroup1-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1 --cpus-per-task=16 --mem=32G -t 04:00:00
#SBATCH -o logs/%x-%j.out

D=/net/scratch/hscra/plgrid/plgekaczmarczyk/diff-resurgence
module load ML-bundle/24.06a
source /net/scratch/hscra/plgrid/plgekaczmarczyk/LID-project/venv/bin/activate
export HF_HOME=$D/data/hf_cache TORCH_HOME=$D/data/torch_cache HF_HUB_OFFLINE=1
cd $D
set -eo pipefail
echo "python: $(which python)"
python -c "import torch, diffusers, nudenet; print(torch.__version__, torch.cuda.is_available())"

P=data/Diffusion-MU-Attack/prompts
COMMON="--component text_encoder --neutral-prompts $P/coco_10k.csv --max-neutral 50 \
  --eps 2 3 4 6 8 10 12 --conditions erased_noise base_noise --out results/gen_AdvUnlearn_$1"

if [ "$1" = vangogh ]; then
  python run_stability.py --name AdvUnlearn_vangogh --erased data/checkpoints/advunlearn/vangogh_unlearned \
    --concept-prompts $P/vangogh.csv --samples-per-prompt 2 --detector clip \
    --clip-concept "a painting by Vincent van Gogh" \
    --clip-negatives "a painting" "a photograph" "an impressionist painting" "an illustration" \
    --save-images 8 $COMMON
else
  python run_stability.py --name AdvUnlearn_nudity --erased data/checkpoints/advunlearn/nudity_unlearned \
    --concept-prompts $P/nudity.csv --max-prompts 142 --detector nudenet --save-images 8 $COMMON
fi