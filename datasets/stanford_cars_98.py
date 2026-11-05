import os

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase

from .oxford_pets import OxfordPets
from .stanford_cars import StanfordCars
from ._common import filter_items, fewshot_or_cache, load_selected_classes, year_first

SELECTED_PKL = os.path.join("data", "Stanford-Cars-98", "selected_98_classes.pkl")


@DATASET_REGISTRY.register()
class car98_ID(DatasetBase):
    """Stanford-Cars-98: the 98 ID categories of Stanford Cars (EOE protocol)."""

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        dataset_dir = os.path.join(root, StanfordCars.dataset_dir)
        split_path = os.path.join(dataset_dir, "split_zhou_StanfordCars.json")

        if os.path.exists(split_path):
            train, val, test = OxfordPets.read_split(split_path, dataset_dir)
        else:
            base = StanfordCars.__new__(StanfordCars)
            base.dataset_dir = dataset_dir
            trainval = base.read_data(
                "cars_train",
                os.path.join(dataset_dir, "devkit", "cars_train_annos.mat"),
                os.path.join(dataset_dir, "devkit", "cars_meta.mat"),
            )
            test = base.read_data(
                "cars_test",
                os.path.join(dataset_dir, "cars_test_annos_withlabels.mat"),
                os.path.join(dataset_dir, "devkit", "cars_meta.mat"),
            )
            train, val = OxfordPets.split_trainval(trainval)

        selected = load_selected_classes(SELECTED_PKL, transform=year_first)
        train = filter_items(train, selected)
        val = filter_items(val, selected)
        test = filter_items(test, selected)

        # Full 98-class label->classname map (BEFORE base/new subsampling),
        # so the head always covers all 98 classes under base2new.
        _seen = {}
        for it in test:
            _seen[it.label] = it.classname
        full_classnames = [_seen[i] for i in range(len(_seen))]

        subsample = getattr(cfg.DATASET, "SUBSAMPLE_CLASSES", "all")
        if subsample == "base":
            train, val = OxfordPets.subsample_classes(train, val, subsample="base")
            tag = "car98_ID_b2n"
        else:
            tag = "car98_ID"

        train, val = fewshot_or_cache(
            self, train, val, cfg.DATASET.NUM_SHOTS, cfg.SEED,
            os.path.join(dataset_dir, "split_fewshot"), tag,
        )

        super().__init__(train_x=train, val=val, test=test)

        if subsample == "base":
            self._classnames = full_classnames
            self._num_classes = len(full_classnames)
