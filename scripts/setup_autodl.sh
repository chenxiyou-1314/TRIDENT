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

echo "== [0/5] Environment preflight =="
python - <<'PY'
import sys
print("python :", sys.version.split()[0])
try:
    import torch
    print("torch  :", torch.__version__, "| cuda:", torch.version.cuda)
    print("gpu    :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE")
    if torch.cuda.is_available():
        torch.zeros(1).cuda()  # fails on arch/cuda mismatch (e.g. sm_89 on cu117)
        print("kernel : GPU matmul smoke test OK")
except ImportError:
    print("torch  : NOT INSTALLED - install torch matching your CUDA before continuing")
try:
    import numpy
    if int(numpy.__version__.split(".")[0]) >= 2:
        print(f"numpy  : {numpy.__version__} (>=2, will be downgraded by requirements)")
    else:
        print("numpy  :", numpy.__version__)
except ImportError:
    pass
PY

echo "== [1/5] Python dependencies =="
pip install -r requirements.txt

echo "== [2/5] Dassl framework =="
# NOTE: the Dassl.pytorch bundled with this repo is a TRIMMED copy
# (the dassl/data module was removed). Install the full version:
if [ ! -f Dassl.pytorch/dassl/data/datasets/__init__.py ]; then
    rm -rf Dassl.pytorch
    git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
fi
pip install -e Dassl.pytorch/

echo "== [2b/5] Patch Dassl defaults (SUBSAMPLE_CLASSES for base2new) =="
# base2new evaluation needs cfg.DATASET.SUBSAMPLE_CLASSES; add it idempotently.
if [ -f Dassl.pytorch/dassl/config/defaults.py ]; then
  grep -q "SUBSAMPLE_CLASSES" Dassl.pytorch/dassl/config/defaults.py || \
    sed -i '/_C.DATASET.ALL_AS_UNLABELED = False/a _C.DATASET.SUBSAMPLE_CLASSES = "all"' \
      Dassl.pytorch/dassl/config/defaults.py
fi

echo "== [3/5] CLIP weights (ViT-L/14, ~890MB) =="
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

echo "== [4/5] Sanity checks =="
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
