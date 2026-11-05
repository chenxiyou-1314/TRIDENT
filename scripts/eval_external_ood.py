"""外部 OOD 评估：Ship30 的 14 个 ID 类 vs 外部 OOD 数据集（如 DTD 子集）。

对应剩余工作清单 2-2 / 2-3。复用 analyze_ship_b2n.py 的特征抽取与指标函数，
不修改 dvr.py。评分用 MSP 与 Energy 代理（**不是论文中的 UOS**），
结果只作第四章辅助实验，不能与 ICME 已发表数字直接比较。

用法：
  python scripts/eval_external_ood.py \
      --root /hy-tmp/TRIDENT/datasets \
      --ckpt output/FINAL/debug/ViT14/ship14_ID/16shots/seed1_b2n/prompt_learner/model.pth.tar-200 \
      --ood-dir /hy-tmp/TRIDENT/datasets/ood_samples/dtd \
      --out /hy-tmp/external_ood.json
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from clip import clip  # noqa: E402
from analyze_ship_b2n import (  # noqa: E402
    ALPHA, auroc, build_transform, display, fpr95, infer, norm,
)


def scores(logits):
    mx = logits.max(1, keepdims=True)
    e = np.exp(logits - mx)
    return {
        "MSP": (e / e.sum(1, keepdims=True)).max(1),
        # logsumexp，数值稳定；越大越像 ID
        "Energy": (np.log(e.sum(1)) + mx.ravel()),
    }


def decode(paths, tfm):
    tensors = []
    for p in paths:
        try:
            with Image.open(p) as im:
                tensors.append(tfm(im.convert("RGB")))
        except Exception:
            continue
    return tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hy-tmp/TRIDENT/datasets")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-json", default="data/Boat-14/id_classes.json")
    ap.add_argument("--ood-dir", required=True)
    ap.add_argument("--out", default="/hy-tmp/external_ood.json")
    ap.add_argument("--max-per-class", type=int, default=1000)
    ap.add_argument("--max-ood", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = os.path.join(args.root, "boat-29")
    test_meta = json.load(open(os.path.join(ds, "meta", "test.json")))
    id_set = {norm(c) for c in json.load(open(args.split_json))["id_classes"]}
    all_display = sorted({display(k) for k in test_meta.keys()}, key=str.lower)
    id_names = [d for d in all_display if norm(d) in id_set]
    assert len(id_names) == 14
    snake = {display(k): k for k in test_meta.keys()}

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck)
    txt = sd["base_text_features"].float() + ALPHA * sd["text_feature_residuals"].float()
    txt = txt / txt.norm(dim=-1, keepdim=True)

    clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
    clip_model = clip_model.float().eval()
    scale = clip_model.logit_scale.exp().item()
    tfm = build_transform()

    id_paths = []
    for n in id_names:
        for r in test_meta.get(snake.get(n, ""), [])[: args.max_per_class]:
            id_paths.append(os.path.join(ds, "images", r + ".jpg"))
    ood_paths = sorted(glob.glob(os.path.join(args.ood_dir, "**", "*.jpg"),
                                 recursive=True))[: args.max_ood]
    ood_paths += sorted(glob.glob(os.path.join(args.ood_dir, "**", "*.png"),
                                  recursive=True))[: args.max_ood]
    print(f"ID images {len(id_paths)} | external OOD images {len(ood_paths)}")

    id_feat = infer(clip_model, decode(id_paths, tfm), device, args.batch_size)
    ood_feat = infer(clip_model, decode(ood_paths, tfm), device, args.batch_size)
    if id_feat.shape[0] == 0 or ood_feat.shape[0] == 0:
        raise RuntimeError("no images decoded - check paths")

    lid = (scale * id_feat @ txt.t()).numpy()
    lood = (scale * ood_feat @ txt.t()).numpy()
    s_id, s_ood = scores(lid), scores(lood)

    out = {
        "note": "MSP/Energy proxy scores on the 14-ID DVR classifier, NOT UOS. "
                "External OOD is a sampled subset; compare only within this table.",
        "ood_dir": args.ood_dir,
        "n_id": int(id_feat.shape[0]),
        "n_ood": int(ood_feat.shape[0]),
        "metrics": {},
    }
    for tag in s_id:
        out["metrics"][tag] = {
            "AUROC": round(auroc(s_id[tag], s_ood[tag]), 4),
            "FPR95": round(fpr95(s_id[tag], s_ood[tag]), 4),
        }
        print(f"[{tag}] AUROC {out['metrics'][tag]['AUROC']} | "
              f"FPR95 {out['metrics'][tag]['FPR95']}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
