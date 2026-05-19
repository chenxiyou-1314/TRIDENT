#!/bin/bash

# custom config
DATA=/data/ccy/ccy-factory/datasets/id_data
# DATA=/path/to/datasets
TRAINER=DVR

DATASET=$1
SHOTS=$2    # number of shots (1, 2, 4, 8, 16)
for SEED in 1
do
    DIR=output/FINAL/debug/ViT14/${DATASET}/${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=1 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.DVR.RESIDUAL_SCALE 0.5
    fi
done