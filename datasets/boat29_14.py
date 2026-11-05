import json
import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase

from .oxford_pets import OxfordPets
from .boat29 import boat29
from ._common import filter_items, fewshot_or_cache

# Fixed, documented 14/16 ID/OOD split (data/Boat-14/id_classes.json).
# The legacy selected_14_classes.pkl is generated from this JSON on first run
# so that dataloaders/boat29.py (evaluation side) stays consistent.
SPLIT_JSON = os.path.join("data", "Boat-14", "id_classes.json")
LEGACY_PKL = os.path.join("data", "Boat-14", "selected_14_classes.pkl")


def load_id_classes():
    """Return the fixed list of 14 ID class names (source of truth: JSON)."""
    with open(SPLIT_JSON, "r") as f:
        id_classes = json.load(f)["id_classes"]
    # keep the legacy pkl in sync for dataloaders/boat29.py
    if not os.path.exists(LEGACY_PKL):
        os.makedirs(os.path.dirname(LEGACY_PKL), exist_ok=True)
        with open(LEGACY_PKL, "wb") as f:
            pickle.dump(id_classes, f)
    return id_classes


@DATASET_REGISTRY.register()
class boat14_ID(DatasetBase):
    """Ship30 ID subset: the fixed 14 ID categories used for DVR training."""

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        base = boat29.__new__(boat29)
        base.dataset_dir = os.path.join(root, boat29.dataset_dir)
        base.image_dir = os.path.join(base.dataset_dir, "images")
        base.meta_dir = os.path.join(base.dataset_dir, "meta")

        id_classes = load_id_classes()

        trainval = boat29.read_meta(base, "train.json")
        test = boat29.read_meta(base, "test.json")

        trainval = filter_items(trainval, id_classes)
        test = filter_items(test, id_classes)
        train, val = OxfordPets.split_trainval(trainval)

        train, val = fewshot_or_cache(
            self, train, val, cfg.DATASET.NUM_SHOTS, cfg.SEED,
            os.path.join(base.dataset_dir, "split_fewshot"), "boat14_ID",
        )

        super().__init__(train_x=train, val=val, test=test)
