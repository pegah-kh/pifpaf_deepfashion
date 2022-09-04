#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 32G
#SBATCH --time=10:00:00  
#SBATCH --gres gpu:1
#SBATCH --output ../../../../../scratch/izar/khayatan/log_files/evaluation/deepfashion_c_crop/shufflenetv2k16_scratch_full_right_%j.log
#SBATCH --mail-user=pegah.khayatan@epfl.ch
sleep 10
python3 -m run_evaluation predict --eval-image-dir /scratch/izar/khayatan/deepfashion/img_test \
 --eval-annotations /scratch/izar/khayatan/deepfashion/test_annotations_MSCOCO_style.json \
 --batch_size 8   --single-epoch True  --confidence_thresh 0.05  --tensorboard_dir crop_testing_smallrot \
 --checkpoint_name_1 shufflenetv2k16_scratch_continue_.epoch200 \
 --ckpt_directory /scratch/izar/khayatan/checkpoints/full_right_vis/ \
 --n_epochs_1 20 --jump 2 --start_epoch_1 10 --long-edge 520