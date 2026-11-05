"""Shared helpers for fine-grained ID-subset datasets (EOE protocol).

The fixed ID class lists are stored in the pkl files under data/ (copied from
the official EOE repository, https://github.com/tmlr-group/EOE):
    data/CUB-100/selected_100_classes.pkl
    data/Stanford-Cars-98/selected_98_classes.pkl
    data/Food-50/selected_50_classes.pkl
    data/Oxford-Pet-18/selected_18_classes.pkl
For Ship30 the split is documented in data/Boat-14/id_classes.json.
"""
import os
import pickle

from dassl.data.datasets import Datum
from dassl.utils import mkdir_if_missing


def year_first(name):
    """Match TaskRes StanfordCars classname convention: 'McLaren MP4-12C Coupe 2012' -> '2012 McLaren MP4-12C Coupe'."""
    parts = name.split(" ")
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 4:
        return parts[-1] + " " + " ".join(parts[:-1])
    return name


def norm(name):
    """Normalize a class name for matching: '002.Laysan_Albatross' / 'laysan_albatross' -> 'laysan albatross'."""
    name = name.split(".")[-1] if "." in name.split(" ")[0] else name
    return name.replace("_", " ").strip().lower()


def load_selected_classes(pkl_path, transform=None):
    with open(pkl_path, "rb") as f:
        classes = pickle.load(f)
    if transform is not None:
        classes = [transform(c) for c in classes]
    return classes


def filter_items(items, selected_display_names):
    """Keep only ID classes and re-index labels to a contiguous 0..K-1 range."""
    selected = {norm(c) for c in selected_display_names}
    kept = [it for it in items if norm(it.classname) in selected]
    ordered = sorted({it.classname for it in kept}, key=str.lower)
    remap = {c: i for i, c in enumerate(ordered)}
    return [
        Datum(impath=it.impath, label=remap[it.classname], classname=it.classname)
        for it in kept
    ]


def fewshot_or_cache(dataset, train, val, num_shots, seed, cache_dir, tag):
    """Few-shot sampling with on-disk cache; tag avoids collisions with full-set caches."""
    if num_shots < 1:
        return train, val
    mkdir_if_missing(cache_dir)
    preprocessed = os.path.join(cache_dir, f"shot_{num_shots}-seed_{seed}-{tag}.pkl")
    if os.path.exists(preprocessed):
        print(f"Loading preprocessed few-shot data from {preprocessed}")
        with open(preprocessed, "rb") as f:
            data = pickle.load(f)
        return data["train"], data["val"]
    train = dataset.generate_fewshot_dataset(train, num_shots=num_shots)
    val = dataset.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
    print(f"Saving preprocessed few-shot data to {preprocessed}")
    with open(preprocessed, "wb") as f:
        pickle.dump({"train": train, "val": val}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return train, val
