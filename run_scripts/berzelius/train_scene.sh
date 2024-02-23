#!/bin/bash
#SBATCH -A Berzelius-2023-364
#SBATCH --gpus 1
#SBATCH -t 0-03:30:00

export PROJ_DIR=/proj/berzelius-2023-364/users/x_mabus/cloth-splatting

apptainer exec --nv -B "$PROJ_DIR":/workspace  "$PROJ_DIR"/md_splatting_latest.sif \
 python3 "$PROJ_DIR"/train.py -s "$PROJ_DIR"/data/final_scenes/scene_1/ --port 6021 --expname "final_scenes/scene_1" --configs "$PROJ_DIR"/arguments/mdnerf-dataset/cube.py --lambda_w 100000 \
  --lambda_rigidity 0.1 --lambda_spring 0.0 --lambda_isometric 0.3 --lambda_momentum 0.1 \
  --use_wandb --wandb_project final_scene_1 --wandb_name init --k_nearest 5
