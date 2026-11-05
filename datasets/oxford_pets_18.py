import os

from dassl.data.datasets import DATASET_REGISTRY, DatasetBase

from .oxford_pets import OxfordPets
from ._common import filter_items, fewshot_or_cache, load_selected_classes

SELECTED_PKL = os.path.join("data", "Oxford-Pet-18", "selected_18_classes.pkl")


@DATASET_REGISTRY.register()
class pet18_ID(DatasetBase):
    """Oxford-Pet-18: the 18 ID breeds of Oxford-IIIT Pet (EOE protocol)."""

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        dataset_dir = os.path.join(root, OxfordPets.dataset_dir)
        split_path = os.path.join(dataset_dir, "split_zhou_OxfordPets.json")
        image_dir = os.path.join(dataset_dir, "images")

        if os.path.exists(split_path):
            train, val, test = OxfordPets.read_split(split_path, image_dir)
        else:
            base = OxfordPets.__new__(OxfordPets)
            base.dataset_dir = dataset_dir
            base.image_dir = image_dir
            base.anno_dir = os.path.join(dataset_dir, "annotations")
            trainval = base.read_data(split_file="trainval.txt")
            test = base.read_data(split_file="test.txt")
            train, val = OxfordPets.split_trainval(trainval)

        selected = load_selected_classes(SELECTED_PKL)
        train = filter_items(train, selected)
        val = filter_items(val, selected)
        test = filter_items(test, selected)

        # Full 18-class label->classname map (BEFORE any base/new subsampling).
        # Needed so the classification head always covers all 18 classes even
        # under base2new (where train_x is reduced to base classes only).
        _seen = {}
        for it in train:
            _seen[it.label] = it.classname
        full_classnames = [_seen[i] for i in range(len(_seen))]

        # --- base2new protocol (CIL-style novel-class generalization) ---
        # SUBSAMPLE_CLASSES == "base": train/val keep only the first half of
        # classes (relabeled 0..m-1); the test set stays full (0..K-1) and the
        # head below is forced back to all K classes.
        subsample = getattr(cfg.DATASET, "SUBSAMPLE_CLASSES", "all")
        if subsample == "base":
            train, val = OxfordPets.subsample_classes(train, val, subsample="base")
            tag = "pet18_ID_b2n"
        else:
            tag = "pet18_ID"

        train, val = fewshot_or_cache(
            self, train, val, cfg.DATASET.NUM_SHOTS, cfg.SEED,
            os.path.join(dataset_dir, "split_fewshot"), tag,
        )

        super().__init__(train_x=train, val=val, test=test)

        if subsample == "base":
            # Force the head / classnames back to the FULL 18 classes so that
            # the learned residual text features generalize to unseen (new) classes.
            self._classnames = full_classnames
            self._num_classes = len(full_classnames)
