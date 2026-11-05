import os

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
from ._common import filter_items, fewshot_or_cache, load_selected_classes

SELECTED_PKL = os.path.join("data", "Food-50", "selected_50_classes.pkl")


@DATASET_REGISTRY.register()
class food50_ID(DatasetBase):
    """Food-50: the 50 ID categories of Food-101 (EOE protocol)."""

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        dataset_dir = os.path.join(root, "food-101")
        split_path = os.path.join(dataset_dir, "split_zhou_Food101.json")
        image_dir = os.path.join(dataset_dir, "images")

        if os.path.exists(split_path):
            train, val, test = OxfordPets.read_split(split_path, image_dir)
        else:
            train, val, test = DTD.read_and_split_data(image_dir)

        selected = load_selected_classes(SELECTED_PKL)
        train = filter_items(train, selected)
        val = filter_items(val, selected)
        test = filter_items(test, selected)

        train, val = fewshot_or_cache(
            self, train, val, cfg.DATASET.NUM_SHOTS, cfg.SEED,
            os.path.join(dataset_dir, "split_fewshot"), "food50_ID",
        )

        super().__init__(train_x=train, val=val, test=test)
