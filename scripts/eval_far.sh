#!/bin/bash
# ============================================================
# Far (classic) OOD evaluation with Ship30 as the ID dataset.
# Usage: bash scripts/eval_far.sh [SCORE] [L]
#   SCORE: EOE (default) | MCM | energy | max-logit
#   L:     number of envisioned OOD labels (default 500)
# OOD datasets: iNaturalist / SUN / places365 / dtd
# ============================================================
set -e

SCORE=${1:-EOE}
L=${2:-500}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for i in 0 1 2; do
  echo "Running far-OOD on Ship30, trial=${i}, score=${SCORE}, L=${L}"
  python eval_ood_detection.py \
    --llm_model gpt-4o \
    --ood_task far \
    --score "${SCORE}" \
    --L "${L}" \
    --in_dataset Ship30 \
    --json_number ${i} \
    --model CLIP \
    --CLIP_ckpt ViT-L/14 \
    --update_json 0
done
