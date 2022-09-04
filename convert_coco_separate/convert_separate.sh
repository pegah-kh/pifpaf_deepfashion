#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 90G
#SBATCH --time=5:00:00  
#SBATCH --gres gpu:1
#SBATCH --output=convert_separate_%j.log
#SBATCH --mail-user=pegah.khayatan@epfl.ch
sleep 10
python3 to_coco_and_separate.py --dataset-root /work/vita/pegah/datasets/deepfashion_right_vis \
 --root-save /work/vita/pegah/venv/pifpaf_deepfahsion/full_right_vis