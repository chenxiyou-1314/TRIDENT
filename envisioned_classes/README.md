# envisioned_classes/

Pre-generated pseudo-OOD class labels produced by the TOS module
(`utils/generate_llm_class.py`, folder layout: `<ood_task>_<L>/`).

These JSONs let you reproduce all paper results **without any LLM API key**:
`eval_ood_detection.py` loads them via `--json_number {0,1,2}`
(unless `--generate_class` is passed).

## Expected layout

```
envisioned_classes/
├── far_500/                  # far-OOD labels (Ship30 as ID)
│   ├── gpt-4o_0.json ...
└── fine_grained_500/         # fine-grained labels per ID dataset
    ├── cub100_ID_gpt-4o_0.json ...
```

If this folder is empty, generate labels with `--generate_class`
(requires `OPENAI_API_KEY`).
