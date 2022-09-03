#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --time 72:00:00
#SBATCH --output=shufflenetv2k16_scratch_0001_%j.log
#SBATCH --gres gpu:2

sleep 5
python3 -u -m openpifpaf.train --dataset=full_right_vis_deepfashion --lr=0.0001 --momentum=0.95 --clip-grad-value=10 \
  --epochs=200 \
  --lr-decay 130 140 --lr-decay-epochs=10 \
  --batch-size=32 \
  --weight-decay=1e-5 \
  --basenet=shufflenetv2k16 \
  --fix-batch-norm=0