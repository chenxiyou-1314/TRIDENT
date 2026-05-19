from pathlib import Path
import json
from typing import Any, Tuple, Callable, Optional
import torch
import PIL.Image
import os
import pickle
import random

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Boat29(VisionDataset):
    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(self.root) / "boat-29"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"
        self.class_names_str = ['Aircraft carrier', 'Bitumen', 'Bulk carrier', 'Catamaran yacht', 'Chemical tanker',
                                'Container ship', 'Crude oil tanker', 'Cruise', 'Destroyer', 'Firefighting',
                                'Fishing ships', 'Frigate', 'Fso', 'General cargo ship', 'Heavy load carrier',
                                'Kayak', 'LNG tanker', 'LPG tanker', 'Monohull sailboat', 'Monohull yacht',
                                'Oil products tanker', 'Passenger cargo ship', 'Passenger ro-ro ship', 'Passenger ship',
                                'Reefer', 'Sailing catamaran', 'Sailing trimaran', 'Submarine', 'Tugboat', 'Vehicles carrier'
                                ]

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())


        self.classes = sorted(metadata.keys(), key=str.lower)
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        for class_label, im_rel_paths in metadata.items():
            label_list = [self.class_to_idx[class_label]] * len(im_rel_paths)

            # 🚀 Debug: 打印 class_label, 对应的 index, 和扩展后的 label_list
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any, str]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, str(image_file)  # 👈 返回路径

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)


class Boat29_14(Boat29):
    def __init__(
            self,
            root: str,
            split: str = "train",
            id: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)

        selected_classes_file = os.path.join('data', 'Boat-14', 'selected_14_classes.pkl')
        if os.path.exists(selected_classes_file):
            with open(selected_classes_file, 'rb') as f:
                selected_classes = pickle.load(f)
        else:
            selected_classes = random.sample(self.classes, 14)
            with open(selected_classes_file, 'wb') as f:
                pickle.dump(selected_classes, f)

        selected_classes = selected_classes if id else [cls for cls in self.classes if cls not in selected_classes]
        selected_ood_classes = [cls for cls in self.classes if cls not in selected_classes] if id else selected_classes

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(selected_classes)}

        selected_image_files = []
        selected_labels = []
        for idx, (image_file, label) in enumerate(zip(self._image_files, self._labels)):
            if self.classes[label] in selected_classes:
                selected_image_files.append(image_file)
                selected_labels.append(label)

        self._image_files = selected_image_files
        self._labels = selected_labels
        self.classes = selected_classes
        self.class_names_str = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls in selected_classes
        ]
        self.ood_class_name_str = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls in selected_ood_classes
        ]


def examine_count(counter, name="train"):
    print(f"in the {name} set")
    for label in counter:
        print(label, counter[label])


if __name__ == "__main__":

    label_names = []
    with open('/data/ccy/ccy-factory/datasets/id_data/boat-29/meta/labels.txt') as f:
        for name in f:
            label_names.append(name.strip())
    print(label_names)

    train_set = Boat29(root="/data/ccy/ccy-factory/datasets/id_data", split="train", download=True)
    test_set = Boat29(root="/data/ccy/ccy-factory/datasets/id_data", split="test")
    print(f"train set len {len(train_set)}")
    print(f"test set len {len(test_set)}")
    from collections import Counter

    train_label_count = Counter(train_set._labels)
    test_label_count = Counter(test_set._labels)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=16, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(test_set,
                                             batch_size=16, shuffle=False, **kwargs)