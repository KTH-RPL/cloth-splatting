export TEST_ITER=6000
#!/bin/bash

#rm -r output/full_run
mkdir -p output/full_run
SCENE="SHORTS_01_01"

ABLATION="no_rot"
EXP="ablation/${ABLATION}"
DATA=data/folding_scenes/${SCENE}
python3 train.py -s ${DATA} --expname "${EXP}" --configs arguments/cloth_splatting/default.py --view_skip 3 --iterations 6000
python3 render.py -s ${DATA} --model_path "output/${EXP}" \
  --configs "arguments/cloth_splatting/default.py"  --meshnet_path "output/${EXP}/meshnet"\
   --view_skip 10 --show_flow --log_deform --track_vertices --skip_train --skip_video --iteration ${TEST_ITER}
python3 scripts/align_eval_trajs.py  --gt_file "${DATA}/trajectory.npz" --traj_file "output/${EXP}/test/ours_${TEST_ITER}/all_trajs.npz"
mkdir -p output/full_run/tracking/${SCENE}
python3 scripts/extract_aligned_trajs.py \
--src_dir "output/${EXP}" --target_dir output/ablation/tr`acking/${SCENE} --iteration ${TEST_ITER} --target_name ${ABLATION}.npz

ABLATION="no_reg"
EXP="ablation/${ABLATION}"
DATA=data/folding_scenes/${SCENE}
python3 train.py -s ${DATA} --expname "${EXP}" --configs arguments/cloth_splatting/default.py --view_skip 3 --iterations 6000 --lambda_deform_mag 0.0 --lambda_rigid 0.0 --lambda_dssim 0.0
python3 render.py -s ${DATA} --model_path "output/${EXP}" \
  --configs "arguments/cloth_splatting/default.py"  --meshnet_path "output/${EXP}/meshnet"\
   --view_skip 10 --show_flow --log_deform --track_vertices --skip_train --skip_video --iteration ${TEST_ITER}
python3 scripts/align_eval_trajs.py  --gt_file "${DATA}/trajectory.npz" --traj_file "output/${EXP}/test/ours_${TEST_ITER}/all_trajs.npz"
mkdir -p output/full_run/tracking/${SCENE}
python3 scripts/extract_aligned_trajs.py \
--src_dir "output/${EXP}" --target_dir output/ablation/tracking/${SCENE} --iteration ${TEST_ITER} --target_name ${ABLATION}.npz

for VIEW_SKIP in 12 ;
do
  ABLATION="view_skip_${VIEW_SKIP}"
  EXP="ablation/${ABLATION}"
  DATA=data/folding_scenes/${SCENE}
 python3 train.py -s ${DATA} --expname "${EXP}" --configs arguments/cloth_splatting/default.py --iterations 6000 --view_skip ${VIEW_SKIP}
  python3 render.py -s ${DATA} --model_path "output/${EXP}" \
    --configs "arguments/cloth_splatting/default.py"  --meshnet_path "output/${EXP}/meshnet"\
     --view_skip 10 --show_flow --log_deform --track_vertices --skip_train --skip_video --iteration ${TEST_ITER}
  python3 scripts/align_eval_trajs.py  --gt_file "${DATA}/trajectory.npz" --traj_file "output/${EXP}/test/ours_${TEST_ITER}/all_trajs.npz"
  mkdir -p output/full_run/tracking/${SCENE}
  python3 scripts/extract_aligned_trajs.py \
  --src_dir "output/${EXP}" --target_dir output/ablation/tracking/${SCENE} --iteration ${TEST_ITER} --target_name ${ABLATION}.npz
done