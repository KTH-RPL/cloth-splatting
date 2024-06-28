export TEST_ITER=6000
#!/bin/bash

#rm -r output/full_run
mkdir -p output/full_run
for SCENE in "TOWEL_00_03" "TSHIRT_01_00" "TSHIRT_01_01" "SHORTS_01_00" "SHORTS_01_01"
do
    EXP="full_run/${SCENE}"
    DATA=data/folding_scenes/${SCENE}
#    python3 train.py -s ${DATA} --expname "${EXP}" --configs arguments/cloth_splatting/default.py --view_skip 3 --iterations 6000
    python3 render_experimental.py -s ${DATA} --model_path "output/${EXP}" \
      --configs "arguments/cloth_splatting/default.py"  --meshnet_path "output/${EXP}/meshnet"\
      --show_flow --log_deform --track_vertices --skip_train --skip_video --iteration ${TEST_ITER} --flow_skip 2
    python3 scripts/align_eval_trajs.py  --gt_file "${DATA}/trajectory.npz" --traj_file "output/${EXP}/test/ours_${TEST_ITER}/all_trajs.npz"
    mkdir -p output/full_run/tracking/${SCENE}
    python3 scripts/extract_aligned_trajs.py \
    --src_dir "output/${EXP}" --target_dir output/full_run/tracking/${SCENE} --iteration ${TEST_ITER} --target_name cloth-splatting.npz
done