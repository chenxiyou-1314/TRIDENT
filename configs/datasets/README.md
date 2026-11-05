# configs/datasets/ (MISSING from git history — must be restored)

Per-dataset YAML configs referenced by `scripts/train.sh`:

```bash
--dataset-config-file configs/datasets/${DATASET}.yaml
```

They were never committed. Restore from the training server and commit:

```bash
git add configs/datasets/
git commit -m "Add dataset configs"
```

## Expected files (one per fine-grained ID dataset)

```
configs/datasets/
├── ship14_ID.yaml     # or boat14_ID.yaml (check scripts/train.sh usage)
├── cub100_ID.yaml
├── car98_ID.yaml
├── food50_ID.yaml
└── pet18_ID.yaml
```
