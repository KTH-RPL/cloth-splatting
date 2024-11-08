export RIGIDITY_LAMBDA=0.1
export LAMBDA_SPRING=0.0
export ISOMETRY=0.01

export SCENE_1="scene_1"
export SCENE_2="scene_2"
export SCENE_3="scene_3"
export SCENE_4="scene_7"
export SCENE_5="scene_5"
export SCENE_6="scene_6"
export SCENE_7="scene_7"

port=6027 
for SCENE in $SCENE_1 $SCENE_2 $SCENE_3 $SCENE_4 $SCENE_5 $SCENE_6;
do
    for isometry in $ISOMETRY;
    do 
        python3 train.py -s "data/final_scenes/${SCENE}" --port $port --expname "shadow_ablation/${SCENE}" --configs arguments/mdnerf-dataset/cube.py --lambda_w 100000 \
        --lambda_rigidity $RIGIDITY_LAMBDA --lambda_spring $LAMBDA_SPRING  --lambda_momentum 0.1 --use_wandb --wandb_project "shadow_ablation_${SCENE}" --no_shadow \
        --wandb_name "no_shadow" --k_nearest 5 --lambda_isometric $isometry
        # add one to port
        port=$((port+1))
    done
done
