#!/bin/bash
# ============================================================
# Fine-grained OOD evaluation (TRIDENT / EOE score)
# Usage: bash scripts/eval_fine_grained.sh <DATASET> <L> <IMAGE_LABEL> [LLM_MODEL]
#   DATASET:     cub100_ID | car98_ID | food50_ID | pet18_ID | ship14_ID (boat14_ID)
#   L:           number of pseudo-OOD categories (100 / 300 / 500)
#   IMAGE_LABEL: number of guidance images for LLM generation (0 / 10 / 30 / 50)
#   LLM_MODEL:   gpt-4o (default) | gemini-1.5-flash | claude-3-5-sonnet-...
# Runs 3 trials (json_number 0..2) for robust evaluation.
# Pre-generated JSONs are loaded from envisioned_classes/<task>_<L>/ by default;
# uncomment --generate_class to call the LLM API instead.
# ============================================================
set -e

DATASET=${1:?Usage: eval_fine_grained.sh <DATASET> <L> <IMAGE_LABEL> [LLM_MODEL]}
L=${2:?}
IMAGE_LABEL=${3:?}
LLM_MODEL=${4:-gpt-4o}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for i in 0 1 2; do
  echo "Running dataset=${DATASET}, trial=${i}, L=${L}, image_label=${IMAGE_LABEL}, llm=${LLM_MODEL}"
  python eval_ood_detection.py \
    --llm_model "${LLM_MODEL}" \
    --ood_task fine_grained \
    --score_ablation EOE \
    --score EOE \
    --L "${L}" \
    --in_dataset "${DATASET}" \
    --json_number ${i} \
    --model CLIP \
    --CLIP_ckpt ViT-L/14 \
    --shot 16 \
    --image_label "${IMAGE_LABEL}" \
    --update_json 0
    # --generate_class   # uncomment to call LLM API instead of loading JSONs
done
