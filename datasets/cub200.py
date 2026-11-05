import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class bird200(DatasetBase):
    """CUB-200-2011 (all 200 categories), Dassl wrapper.

    Expected layout under cfg.DATASET.ROOT:
        CUB_200_2011/
            images.txt
            image_class_labels.txt
            train_test_split.txt
            classes.txt
            images/
    """

    dataset_dir = "CUB_200_2011"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_CUB200.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_cub_metadata()
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

    @staticmethod
    def raw_to_display(raw_name):
        """'002.Laysan_Albatross' -> 'Laysan Albatross'"""
        return raw_name.split(".")[1].replace("_", " ")

    def read_cub_metadata(self):
        def read_txt(name):
            with open(os.path.join(self.dataset_dir, name)) as f:
                return [line.strip().split(" ") for line in f if line.strip()]

        images = {int(i): p for i, p in read_txt("images.txt")}
        labels = {int(i): int(t) for i, t in read_txt("image_class_labels.txt")}
        splits = {int(i): int(s) for i, s in read_txt("train_test_split.txt")}
        class_names = {int(i): n for i, n in read_txt("classes.txt")}

        trainval, test = [], []
        for img_id, rel_path in images.items():
            target = labels[img_id] - 1  # 0-based
            classname = self.raw_to_display(class_names[labels[img_id]])
            impath = os.path.join(self.image_dir, rel_path)
            item = Datum(impath=impath, label=target, classname=classname)
            if splits[img_id] == 1:
                trainval.append(item)
            else:
                test.append(item)

        train, val = OxfordPets.split_trainval(trainval)
        return train, val, test
