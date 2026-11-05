#!/bin/bash
# ============================================================
# Grid evaluation over (L, image_label) for one dataset.
# Usage: bash scripts/eval_grid.sh <DATASET> [LLM_MODEL]
# Runs L in {100,300,500} x image_label in {0,10,30,50}.
# ============================================================
set -e

DATASET=${1:?Usage: eval_grid.sh <DATASET> [LLM_MODEL]}
LLM_MODEL=${2:-gpt-4o}

for L in 100 300 500; do
  for IMAGE_LABEL in 0 10 30 50; do
    echo "==== L=${L}, image_label=${IMAGE_LABEL} ===="
    bash scripts/eval_fine_grained.sh "${DATASET}" "${L}" "${IMAGE_LABEL}" "${LLM_MODEL}"
  done
done
