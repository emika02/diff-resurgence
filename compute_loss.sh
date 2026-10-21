#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=plgideascvgroup1-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --output=/net/scratch/hscra/plgrid/plgekaczmarczyk/diff-resurgence/slurm/stdout/output_%j.out
#SBATCH --error=/net/scratch/hscra/plgrid/plgekaczmarczyk/diff-resurgence/slurm/stderr/error_%j.err

cd diff-resurgence
module load ML-bundle/24.06a
source /env/bin/activate


python loss_stability.py --name $1 --erased $2 --component unet \
  --neutral-data data/coco_neutral.csv --n-images 256 \
  --eps 0.5 1 2 4 8 12 --noise-seeds 0 1 2 3 4 --random-edits 3 \
  --out results/${1}_coco

''' python loss_stability.py --name AdvUnlearn_nudity --component text_encoder \
  --erased data/checkpoints/advunlearn/nudity_unlearned \
  --neutral-data data/coco_neutral.csv --n-images 256 \
  --eps 20 22 24 26 28 30 --noise-seeds 0 1 2 3 4 \
  --random-edits 3 --out results/AdvUnlearn_nudity_coco '''

