#!/bin/bash
export SCENE_TYPE="folding_scenes"
export SCENE="SHORTS_01_01"

port=6027
SCENE = "SHORTS_01_01"
EXP_DIR = "output/param_optim/${SCENE}"

for LAMBDA_SSIM in 0.01, 0.1. 0.3;
do
  for LAMBDA_RIGID in 0.01, 0.1. 0.3;
  do
    for LAMBDA_DEFORM_MAGN in 0.01, 0.1. 0.3;
    do
      python3 -s data/folding_scenes/SHORTS_01_01 --expname "residual_simulation/SHORTS_01_01"\
       --configs arguments/cloth_splatting/default.py --view_skip 3\
        --lambda_ssim $LAMBDA_SSIM --lambda_rigid $LAMBDA_RIGID --lambda_deform_magn $LAMBDA_DEFORM_MAGN --iteration 6000
      python3 render_experimental.py -s "data/${SCENE_TYPE}/${SCENE}" \
        --model_path "output/${EXP_DIR}" \
        --configs "arguments/cloth_splatting/default.py" \
        --meshnet_path "output/${EXP_DIR}/meshnet"\
         --view_skip 10 --show_flow --log_deform --track_vertices --skip_train --skip_video
      python3 scripts/align_eval_trajs.py \
      --gt_file "data/${SCENE_TYPE}/${SCENE}/trajectory.npz"\
      --traj_file "output/${EXP_DIR}/test/ours_${ITER}/all_trajs.npz"
      python3 scripts/extract_aligned_trajs.py \
      --src_dir "output/${EXP_DIR}" --target_dir "output/fullev" --iteration 4000 --target_name test.npz

    done
  done
done