#!/usr/bin/env python3
"""Convert HF Stanford-Cars parquet into TRIDENT's expected layout:

    datasets/stanford_cars/images/*.jpg
    datasets/stanford_cars/split_zhou_StanfordCars.json   (CoOp format)

Class names are mapped to the canonical display names in
data/Stanford-Cars-98/selected_98_classes.pkl (with year_first applied,
matching the year_first transform used by car98_ID). Non-ID classes are kept
in the split file but get a name that filter_items() will discard.
"""
import io
import json
import os
import pickle
import re
import sys

import pyarrow.parquet as pq

ROOT = "/hy-tmp/TRIDENT"
DS = os.path.join(ROOT, "datasets", "stanford_cars")
IMGDIR = os.path.join(DS, "images")
PQDIR = "/hy-tmp/cars_parquet"
README = "/hy-tmp/cars_readme.md"
PKL = os.path.join(ROOT, "data", "Stanford-Cars-98", "selected_98_classes.pkl")
SPLIT = os.path.join(DS, "split_zhou_StanfordCars.json")


def norm(name):
    name = name.split(".")[-1] if "." in name.split(" ")[0] else name
    return name.replace("_", " ").strip().lower()


def year_first(name):
    parts = name.split(" ")
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 4:
        return parts[-1] + " " + " ".join(parts[:-1])
    return name


# --- 1. HF 196 class names from README front matter ---
hf_names = {}
with open(README) as f:
    for line in f:
        m = re.match(r"^\s+'(\d+)':\s*(.+?)\s*$", line)
        if m:
            hf_names[int(m.group(1))] = m.group(2)
print("HF class names:", len(hf_names))
assert len(hf_names) == 196, "expected 196 HF classes"

# --- 2. canonical ID names (year-first display form) ---
with open(PKL, "rb") as f:
    id_raw = pickle.load(f)
canon = {}
for c in id_raw:
    disp = year_first(c)
    canon[norm(disp)] = disp
print("ID classes in pkl:", len(id_raw), "unique canonical:", len(canon))
assert len(canon) == 98, "expected 98 unique ID classes"

# --- 3. map HF label -> display classname ---
label2name = {}
matched = 0
for lab, hf in hf_names.items():
    disp = year_first(hf)
    key = norm(disp)
    if key in canon:
        label2name[lab] = canon[key]      # canonical, will be kept
        matched += 1
    else:
        label2name[lab] = disp            # OOD name, filter_items drops it
print("HF classes matched to ID pkl:", matched, "/ 196")
assert matched == 98, "ID class mapping incomplete -- ABORT"

os.makedirs(IMGDIR, exist_ok=True)

trainval, test = [], []
counts = {"train": 0, "test": 0}

files = sorted(os.listdir(PQDIR))
for fn in files:
    if not fn.endswith(".parquet"):
        continue
    is_test = fn.startswith("test-")
    tag = "test" if is_test else "train"
    table = pq.read_table(os.path.join(PQDIR, fn), columns=["image", "label"])
    print("reading", fn, "rows=", table.num_rows)
    imgs = table.column("image").to_pylist()
    labs = table.column("label").to_pylist()
    for i, (im, lab) in enumerate(zip(imgs, labs)):
        lab = int(lab)
        cname = label2name.get(lab, str(lab))
        # HF image feature stores raw bytes under 'bytes'
        if isinstance(im, dict):
            data = im.get("bytes")
        else:
            data = im
        if data is None:
            continue
        rel = "images/%s_%06d_%06d.jpg" % (tag, lab, counts[tag])
        # HF stores JPEG bytes already -> write straight through (fast, lossless).
        # Only fall back to PIL conversion for non-JPEG payloads.
        if data[:2] == b"\xff\xd8":
            with open(os.path.join(DS, rel), "wb") as f:
                f.write(data)
        else:
            try:
                from PIL import Image
                pil = Image.open(io.BytesIO(data)).convert("RGB")
                pil.save(os.path.join(DS, rel), quality=95)
            except Exception:
                with open(os.path.join(DS, rel), "wb") as f:
                    f.write(data)
        counts[tag] += 1
        rec = [rel, lab, cname]
        (test if is_test else trainval).append(rec)
    del table, imgs, labs

print("trainval:", len(trainval), "test:", len(test))

# --- 4. split trainval into train/val (20% val, per class, deterministic) ---
import random
random.seed(1)
by_class = {}
for rec in trainval:
    by_class.setdefault(rec[1], []).append(rec)
train, val = [], []
for lab, recs in by_class.items():
    random.shuffle(recs)
    n_val = max(1, round(len(recs) * 0.2))
    val.extend(recs[:n_val])
    train.extend(recs[n_val:])
print("train:", len(train), "val:", len(val), "test:", len(test))

split = {"train": train, "val": val, "test": test}
with open(SPLIT, "w") as f:
    json.dump(split, f)
print("wrote", SPLIT, os.path.getsize(SPLIT), "bytes")
print("CARS_SPLIT_DONE")
