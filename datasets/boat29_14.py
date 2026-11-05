import json
import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum

from .oxford_pets import OxfordPets
from .boat29 import boat29
from ._common import filter_items, fewshot_or_cache


def to_display(name):
    """'LNG_tanker' -> 'LNG tanker'.  Ship30 meta uses snake_case class keys,
    which hurt CLIP text prompts ('a photo of a LNG_tanker'); keep the
    canonical display names from id_classes.json instead."""
    return name.replace("_", " ").strip()


def relabel_display(items):
    return [
        Datum(impath=it.impath, label=it.label,
              classname=to_display(it.classname))
        for it in items
    ]

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

        # Normalise snake_case class keys to display names BEFORE filtering,
        # so CLIP text prompts use e.g. "LNG tanker" instead of "LNG_tanker".
        trainval = relabel_display(trainval)
        test = relabel_display(test)

        trainval = filter_items(trainval, id_classes)
        test = filter_items(test, id_classes)
        train, val = OxfordPets.split_trainval(trainval)

        # Full 14-class label->classname map (BEFORE base/new subsampling),
        # so the head always covers all 14 classes under base2new.
        _seen = {}
        for it in test:
            _seen[it.label] = it.classname
        full_classnames = [_seen[i] for i in range(len(_seen))]

        subsample = getattr(cfg.DATASET, "SUBSAMPLE_CLASSES", "all")
        if subsample == "base":
            train, val = OxfordPets.subsample_classes(train, val, subsample="base")
            tag = "boat14_ID_b2n"
        else:
            tag = "boat14_ID"

        train, val = fewshot_or_cache(
            self, train, val, cfg.DATASET.NUM_SHOTS, cfg.SEED,
            os.path.join(base.dataset_dir, "split_fewshot"), tag,
        )

        super().__init__(train_x=train, val=val, test=test)

        if subsample == "base":
            self._classnames = full_classnames
            self._num_classes = len(full_classnames)
