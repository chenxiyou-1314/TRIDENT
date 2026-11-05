#!/usr/bin/env python3
"""Prepare the raw Ship30 dataset into the layout expected by TRIDENT.

Raw input (choose one layout, auto-detected):
  A) --raw points to a directory that contains one subfolder per category
     (folder names may use spaces / underscores / different casing):
         raw_ship30/Aircraft carrier/xxx.jpg
         raw_ship30/aircraft_carrier/yyy.jpg
  B) --raw points to a directory of flat images plus a CSV annotation file
     (--csv with columns image,class) -- pass --csv explicitly.

Output (under $DATA_ROOT by default, or --out):
  boat-29/
      images/<class name>/<hash>.jpg
      meta/
          train.json      # {class_name: [relpath_without_ext, ...]}
          test.json
          labels.txt      # one class name per line

The train/test split is stratified per class with a fixed seed (default 0.8/0.2).

Usage:
  python tools/prepare_ship30.py --raw /path/to/raw_ship30 [--out ./datasets] \
      [--train-ratio 0.8] [--seed 1] [--copy]
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil

ALL_CLASSES = [
    'Aircraft carrier', 'Bitumen', 'Bulk carrier', 'Catamaran yacht', 'Chemical tanker',
    'Container ship', 'Crude oil tanker', 'Cruise', 'Destroyer', 'Firefighting',
    'Fishing ships', 'Frigate', 'Fso', 'General cargo ship', 'Heavy load carrier',
    'Kayak', 'LNG tanker', 'LPG tanker', 'Monohull sailboat', 'Monohull yacht',
    'Oil products tanker', 'Passenger cargo ship', 'Passenger ro-ro ship', 'Passenger ship',
    'Reefer', 'Sailing catamaran', 'Sailing trimaran', 'Submarine', 'Tugboat', 'Vehicles carrier',
]

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def norm(name):
    return name.replace('_', ' ').replace('-', ' ').strip().lower()


def build_class_map():
    return {norm(c): c for c in ALL_CLASSES}


def collect_from_folders(raw, class_map):
    """Layout A: one subfolder per category."""
    samples = {c: [] for c in ALL_CLASSES}
    matched = set()
    for entry in sorted(os.listdir(raw)):
        path = os.path.join(raw, entry)
        if not os.path.isdir(path):
            continue
        key = norm(entry)
        if key not in class_map:
            print(f'[WARN] skipping unrecognized folder: {entry}')
            continue
        cname = class_map[key]
        matched.add(cname)
        for root, _, files in os.walk(path):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                    samples[cname].append(os.path.join(root, fn))
    missing = [c for c in ALL_CLASSES if c not in matched]
    if missing:
        print(f'[WARN] categories with NO folder found: {missing}')
    return samples


def collect_from_csv(raw, csv_path, class_map):
    """Layout B: flat images + CSV annotation."""
    samples = {c: [] for c in ALL_CLASSES}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames}
        img_col = cols.get('image') or cols.get('filename') or cols.get('file') or cols.get('path')
        cls_col = cols.get('class') or cols.get('label') or cols.get('category')
        if not img_col or not cls_col:
            raise SystemExit('CSV must have image-ish and class-ish columns')
        for row in reader:
            key = norm(row[cls_col])
            if key not in class_map:
                continue
            img = row[img_col]
            if not os.path.isabs(img):
                img = os.path.join(raw, img)
            if os.path.exists(img):
                samples[class_map[key]].append(img)
    return samples


def rel_out_name(src):
    """Deterministic collision-free output name: <md5-of-path><ext>."""
    ext = os.path.splitext(src)[1].lower()
    if ext == '.jpeg':
        ext = '.jpg'
    h = hashlib.md5(src.encode('utf-8')).hexdigest()[:16]
    return f'{h}{ext}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='raw Ship30 root (see layouts above)')
    ap.add_argument('--csv', default=None, help='annotation CSV if images are flat (layout B)')
    ap.add_argument('--out', default=os.environ.get('DATA_ROOT', './datasets'))
    ap.add_argument('--train-ratio', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--copy', action='store_true',
                    help='copy files instead of symlinks (needed on Windows)')
    args = ap.parse_args()

    class_map = build_class_map()
    if args.csv:
        samples = collect_from_csv(args.raw, args.csv, class_map)
    else:
        samples = collect_from_folders(args.raw, class_map)

    base = os.path.join(args.out, 'boat-29')
    img_dir = os.path.join(base, 'images')
    meta_dir = os.path.join(base, 'meta')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    rng = random.Random(args.seed)
    train_meta, test_meta = {}, {}
    total = 0
    for cname in ALL_CLASSES:
        files = sorted(set(samples[cname]))
        if not files:
            print(f'[WARN] empty category: {cname}')
        rng.shuffle(files)
        n_train = round(len(files) * args.train_ratio)
        cdir = os.path.join(img_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        for i, src in enumerate(files):
            out_name = rel_out_name(src)
            dst = os.path.join(cdir, out_name)
            if not os.path.exists(dst):
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    os.symlink(os.path.abspath(src), dst)
            rel = f'{cname}/{os.path.splitext(out_name)[0]}'
            (train_meta if i < n_train else test_meta).setdefault(cname, []).append(rel)
            total += 1
        print(f'{cname}: {len(files)} images ({n_train} train / {len(files)-n_train} test)')

    with open(os.path.join(meta_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_meta, f, ensure_ascii=False, indent=1)
    with open(os.path.join(meta_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_meta, f, ensure_ascii=False, indent=1)
    with open(os.path.join(meta_dir, 'labels.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(ALL_CLASSES) + '\n')

    print(f'\nDone: {total} images -> {base}')
    print('Next: bash scripts/train.sh ship14_ID 16')


if __name__ == '__main__':
    main()
