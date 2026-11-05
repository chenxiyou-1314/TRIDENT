# datasets/ (MISSING from git history — must be restored)

This package contains the **Dassl-format dataset registrations** imported by
`train.py` (e.g. `datasets.boat29`, `datasets.boat29_14`, `datasets.cub200_100`,
`datasets.stanford_cars_98`, `datasets.food101_50`, `datasets.oxford_pets_18`).

It was previously excluded by an over-broad `datasets/` rule in `.gitignore`
(fixed in this reorganization — only raw image data under `data/` is ignored now).

## How to restore

Copy this whole folder from the training server and commit it:

```bash
git add datasets/
git commit -m "Add Dassl dataset registrations"
```

## Expected files

```
datasets/
├── __init__.py
├── boat29.py          # Ship30 (30 fine-grained classes)
├── boat29_14.py       # Ship-14 (ID split)
├── cub200.py
├── cub200_100.py      # CUB-100 (ID split)
├── stanford_cars.py
├── stanford_cars_98.py
├── food101.py
├── food101_50.py
├── oxford_pets.py
├── oxford_pets_18.py
└── imagenet.py
```

Note: `dataloaders/` (torchvision-style classes used by the eval path) is a
different package and is already committed.
