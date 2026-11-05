#!/bin/bash
# ============================================================
# Train DVR on one dataset.
# Usage: bash scripts/train.sh <DATASET> <SHOTS> [SEED]
#   DATASET: ship14_ID | cub100_ID | car98_ID | food50_ID | pet18_ID
#   SHOTS:   1 | 2 | 4 | 8 | 16
#   SEED:    default 1
# Dataset root is read from $DATA_ROOT (default: ./datasets).
# GPU is selected via $CUDA_VISIBLE_DEVICES (default: 0).
# ============================================================
set -e

DATA=${DATA_ROOT:-./datasets}
TRAINER=DVR

DATASET=${1:?Usage: train.sh <DATASET> <SHOTS> [SEED]}
SHOTS=${2:?}
SEED=${3:-1}

DIR=output/FINAL/debug/ViT14/${DATASET}/${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.DVR.RESIDUAL_SCALE 0.5
fi
