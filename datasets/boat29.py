import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class boat29(DatasetBase):
    """Ship30 (all 30 categories), Dassl wrapper.

    Expected layout under cfg.DATASET.ROOT:
        boat-29/
            images/           # all images, organized by category subfolder
            meta/
                train.json    # {class_name: [relative_image_path_without_ext, ...]}
                test.json
    """

    dataset_dir = "boat-29"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.meta_dir = os.path.join(self.dataset_dir, "meta")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Ship30.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_meta("train.json")
            test = self.read_meta("test.json")
            train, val = OxfordPets.split_trainval(trainval)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_meta(self, split_file):
        import json
        filepath = os.path.join(self.meta_dir, split_file)
        with open(filepath, "r") as f:
            metadata = json.load(f)

        classes = sorted(metadata.keys(), key=str.lower)
        class_to_idx = dict(zip(classes, range(len(classes))))

        items = []
        for class_label, im_rel_paths in metadata.items():
            label = class_to_idx[class_label]
            for im_rel_path in im_rel_paths:
                impath = os.path.join(self.image_dir, *f"{im_rel_path}.jpg".split("/"))
                item = Datum(impath=impath, label=label, classname=class_label)
                items.append(item)
        return items
