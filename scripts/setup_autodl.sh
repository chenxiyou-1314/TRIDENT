#!/bin/bash
# ============================================================
# One-shot environment setup for TRIDENT on AutoDL (or any
# Linux GPU box). Run from the repo root:
#   bash scripts/setup_autodl.sh
#
# Assumes an AutoDL base image with PyTorch 2.0.x + cu118
# (e.g. "PyTorch 2.0.1, Python 3.10, CUDA 11.8").
# ============================================================
set -e

echo "== [1/4] Python dependencies =="
pip install -r requirements.txt

echo "== [2/4] Dassl framework =="
# NOTE: the Dassl.pytorch bundled with this repo is a TRIMMED copy
# (the dassl/data module was removed). Install the full version:
if [ ! -f Dassl.pytorch/dassl/data/datasets/__init__.py ]; then
    rm -rf Dassl.pytorch
    git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
fi
pip install -e Dassl.pytorch/

echo "== [3/4] CLIP weights (ViT-L/14, ~890MB) =="
# Downloaded automatically on first use into ~/.cache/clip.
# Pre-fetch here so the first training run does not stall:
python - <<'PY'
try:
    from clip import clip
    clip.load("ViT-L/14", device="cpu")
    print("CLIP ViT-L/14 cached OK")
except Exception as e:
    print(f"CLIP pre-fetch skipped ({e}); it will download on first run")
PY

echo "== [4/4] Sanity checks =="
python - <<'PY'
import os
os.environ.setdefault("DATA_ROOT", "./datasets")
# 1) fixed Ship30 ID split is present
assert os.path.exists("data/Boat-14/id_classes.json"), "missing data/Boat-14/id_classes.json"
# 2) EOE fixed splits are present
for p in ["data/CUB-100/selected_100_classes.pkl",
          "data/Stanford-Cars-98/selected_98_classes.pkl",
          "data/Food-50/selected_50_classes.pkl",
          "data/Oxford-Pet-18/selected_18_classes.pkl"]:
    assert os.path.exists(p), f"missing {p}"
# 3) Dassl imports and our dataset modules register
from dassl.config import get_cfg_default
import datasets.boat29, datasets.boat29_14
import datasets.cub200, datasets.cub200_100
import datasets.stanford_cars, datasets.stanford_cars_98
import datasets.food101, datasets.food101_50
import datasets.oxford_pets, datasets.oxford_pets_18
print("All sanity checks passed.")
PY

echo ""
echo "Setup done. Next steps:"
echo "  1. bash scripts/download_datasets.sh          # public datasets"
echo "  2. python tools/prepare_ship30.py --raw /path/to/raw_ship30"
echo "  3. export OPENAI_API_KEY=sk-...               # only if regenerating TOS labels"
echo "  4. bash scripts/train_all.sh 16               # DVR training"
