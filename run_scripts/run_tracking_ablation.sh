#!/bin/bash
export SCENE_TYPE="folding_scenes"
export SCENE="SHORTS_01_01"

port=6027

python3 train.py -s "data/${SCENE_TYPE}/${SCENE}" --port ${port} --expname "residual_simulation/${SCENE}" --configs "arguments/cloth_splatting/default.py" --view_skip 3  --iterations 6000 \
                  --save_iterations 1500 1625 1750 1875 2000 2500 3000 4000 5000 6000

for ITER in 1500 1575 1625 1750 1875 2000 2500 3000 4000 5000 6000;
do
    echo ${ITER} >> "output/residual_simulation/${SCENE}/align_eval.txt"
    python3 render_experimental.py -s "data/${SCENE_TYPE}/${SCENE}" \
      --model_path "output/residual_simulation/${SCENE}" \
      --configs "arguments/cloth_splatting/default.py" \
      --meshnet_path "output/residual_simulation/${SCENE}/meshnet"\
       --iteration ${ITER} \
       --view_skip 10 --show_flow --log_deform --track_vertices --skip_train --skip_video
    python3 scripts/align_eval_trajs.py \
    --gt_file "data/${SCENE_TYPE}/${SCENE}/trajectory.npz"\
    --traj_file "output/residual_simulation/${SCENE}/test/ours_${ITER}/all_trajs.npz"
done
python3 scripts/extract_aligned_trajs.py \
--src_dir "output/residual_simulation/${SCENE}" \
--target_dir "output/residual_simulation/${SCENE}/tracking_time" \
--take_all