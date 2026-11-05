import os

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase

from .oxford_pets import OxfordPets
from .cub200 import bird200
from ._common import filter_items, fewshot_or_cache, load_selected_classes

SELECTED_PKL = os.path.join("data", "CUB-100", "selected_100_classes.pkl")


@DATASET_REGISTRY.register()
class cub100_ID(DatasetBase):
    """CUB-100: the 100 ID categories of CUB-200-2011 (EOE protocol)."""

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        dataset_dir = os.path.join(root, bird200.dataset_dir)
        split_path = os.path.join(dataset_dir, "split_zhou_CUB200.json")
        image_dir = os.path.join(dataset_dir, "images")

        if os.path.exists(split_path):
            train, val, test = OxfordPets.read_split(split_path, image_dir)
        else:
            base = bird200.__new__(bird200)
            base.dataset_dir = dataset_dir
            base.image_dir = image_dir
            train, val, test = bird200.read_cub_metadata(base)

        selected = load_selected_classes(SELECTED_PKL, transform=bird200.raw_to_display)
        train = filter_items(train, selected)
        val = filter_items(val, selected)
        test = filter_items(test, selected)

        train, val = fewshot_or_cache(
            self, train, val, cfg.DATASET.NUM_SHOTS, cfg.SEED,
            os.path.join(dataset_dir, "split_fewshot"), "cub100_ID",
        )

        super().__init__(train_x=train, val=val, test=test)
