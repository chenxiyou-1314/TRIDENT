#!/bin/bash
# ============================================================
# Train DVR on all 5 fine-grained ID datasets (16-shot).
# Usage: bash scripts/train_all.sh [SHOTS]
# ============================================================
set -e

SHOTS=${1:-16}

bash scripts/train.sh ship14_ID ${SHOTS}
bash scripts/train.sh cub100_ID ${SHOTS}
bash scripts/train.sh car98_ID ${SHOTS}
bash scripts/train.sh food50_ID ${SHOTS}
bash scripts/train.sh pet18_ID ${SHOTS}
