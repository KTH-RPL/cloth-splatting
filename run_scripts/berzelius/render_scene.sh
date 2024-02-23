#!/bin/bash
#SBATCH -A Berzelius-2023-364
#SBATCH --gpus 1
#SBATCH -t 0-03:30:00

export PROJ_DIR=/proj/berzelius-2023-364/users/x_mabus/cloth-splatting

apptainer exec --nv -B "$PROJ_DIR":/workspace  "$PROJ_DIR"/md_splatting_latest.sif \
 python3 "$PROJ_DIR"/render_experimental.py --model_path "$PROJ_DIR"/output/final_scenes/scene_1/ \
     --skip_train --skip_video --configs "$PROJ_DIR"/arguments/mdnerf-dataset/cube.py \
     --view_skip 2 --time_skip 2 --log_deform