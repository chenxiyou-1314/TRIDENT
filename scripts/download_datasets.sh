#!/bin/bash
# ============================================================
# Download the public benchmark datasets for TRIDENT.
# Usage (from repo root):  bash scripts/download_datasets.sh [ID] [OOD]
#   ID  = fine-grained ID datasets (CUB-200, Cars, Food-101, Pets)
#   OOD = classic OOD test sets (iNaturalist/SUN/Places/Texture)
# Downloads into $DATA_ROOT (default ./datasets).
# ~50GB disk total. Run inside AutoDL / gpushare (fast academic network).
# ============================================================
set -e

DATA=${DATA_ROOT:-./datasets}
WHAT=${1:-ALL}
mkdir -p "$DATA"
cd "$DATA"

# ------------------------------------------------------------
# Fine-grained ID datasets (MCM layout, see EOE README)
# ------------------------------------------------------------
if [ "$WHAT" = "ALL" ] || [ "$WHAT" = "ID" ]; then

  # --- CUB-200-2011 (~1.1GB) ---
  # NOTE: Caltech data repository record id changed from w9d68-gec53 to
  # 65de6-vp158 (old link now 404s). The record 302-redirects to a
  # short-lived (~60s) S3 presigned URL, so -L is REQUIRED to follow it.
  if [ ! -d CUB_200_2011 ]; then
    wget -c -q --show-progress -L "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    tar -xzf CUB_200_2011.tgz && rm CUB_200_2011.tgz
  fi

  # --- Food-101 (~5GB) ---
  if [ ! -d food-101 ]; then
    wget -c -q --show-progress -L "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    tar -xzf food-101.tar.gz && rm food-101.tar.gz
  fi

  # --- Oxford-IIIT Pet (~800MB) ---
  if [ ! -d oxford_pets ]; then
    mkdir -p oxford_pets && cd oxford_pets
    wget -c -q --show-progress -L "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    wget -c -q --show-progress -L "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    tar -xzf images.tar.gz && tar -xzf annotations.tar.gz
    rm images.tar.gz annotations.tar.gz && cd ..
  fi

  # --- Stanford Cars (~2GB) ---
  # The original images are hosted on Google Drive (blocked in mainland
  # China without a proxy). Fastest route on AutoDL: transfer via their
  # netdisk or use a mirror. Links:
  #   devkit:  https://github.com/pytorch/vision/files/4566777/devkit.zip
  #   train:   https://drive.google.com/uc?id=1OFdWcbqNTeh4K-engineSeeBelow
  #   Practical mirrors (pick one):
  if [ ! -d stanford_cars ]; then
    echo ">>> Stanford Cars must be downloaded manually (Google Drive):"
    echo ">>>   devkit:  https://github.com/pytorch/vision/files/4566777/devkit.zip"
    echo ">>>   images:  see https://github.com/pytorch/vision/issues/7545 for mirrors"
    echo ">>> After downloading, organize as:"
    echo ">>>   stanford_cars/devkit/cars_meta.mat, cars_train_annos.mat"
    echo ">>>   stanford_cars/cars_train/*.jpg, stanford_cars/cars_test/*.jpg"
  fi

fi

# ------------------------------------------------------------
# Classic OOD test sets curated by MOS (Huang et al. 2021)
# Layout: $DATA/ImageNet_OOD_dataset/{iNaturalist,SUN,Places,Texture}
# Google Drive links from the MOS repo:
#   https://github.com/deeplearning-wisc/large_scale_ood
# ------------------------------------------------------------
if [ "$WHAT" = "ALL" ] || [ "$WHAT" = "OOD" ]; then
  mkdir -p ImageNet_OOD_dataset
  echo ">>> OOD test sets (iNaturalist/SUN/Places/Texture) are shared via"
  echo ">>> Google Drive by the MOS authors:"
  echo ">>>   https://github.com/deeplearning-wisc/large_scale_ood"
  echo ">>> On AutoDL use '学术资源加速' for GitHub, and AutoDL netdisk /"
  echo ">>> gdown (pip install gdown) with a proxy for Drive files."
fi

echo "Done. Verify layout with: ls $DATA"
