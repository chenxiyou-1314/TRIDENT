"""通用 base2new 归因：DVR 训练后 vs CLIP 零样本文本先验。

对应剩余工作清单 2-8。适用于任意已跑过 base2new 的数据集
（CUB-100 / Ship-14 / Cars-98 / Food-50 / Pet-18）。

做法：在**运行时**替换 DVR.test（monkey patch），同时收集训练后 logits 与
「不加载残差、仅用 base_text_features」的零样本 logits，输出
base/new/HM、per-class 准确率、以及 new 类样本被预测进 base 类的比例。
不修改 dvr.py 或任何其他既有文件，与 Claude Code 的文件边界不冲突。

用法（工作目录必须是仓库根）：
  python scripts/analyze_b2n_generic.py \
      --dataset-config-file configs/datasets/cub100_ID.yaml \
      --model-dir output/FINAL/debug/ViT14/cub100_ID/16shots/seed1_b2n \
      --out /hy-tmp/cub_analysis.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train as T  # noqa: E402
from argparse import Namespace  # noqa: E402
from dassl.engine import build_trainer  # noqa: E402
import trainers.dvr as D  # noqa: E402

# train.extend_cfg 引用模块级全局 args.enhanced_base；import 时不会执行 main，
# 这里补一个占位，setup_cfg 才能正常调用。
T.args = Namespace(enhanced_base="none")


def patched_test(self, split=None):
    """替换 DVR.test，额外算零样本对照与偏置统计。"""
    self.set_model_mode("eval")
    self.evaluator.reset()
    loader = self.test_loader
    print("[attrib] Evaluate on the *test* set")

    model = self.model
    core = model.module if hasattr(model, "module") else model
    pl = core.prompt_learner
    base_txt = pl.base_text_features.float()
    base_txt = base_txt / base_txt.norm(dim=-1, keepdim=True)
    scale = core.logit_scale.exp()
    dtype = next(core.parameters()).dtype

    outs, zss, labs = [], [], []
    for batch in tqdm(loader):
        inp, lab = self.parse_batch_test(batch)
        out = self.model_inference(inp)
        with torch.no_grad():
            try:
                imgf = core.image_encoder(inp.type(dtype))
            except Exception:
                imgf = core.image_encoder(inp.float())
            imgf = imgf / imgf.norm(dim=-1, keepdim=True)
            zs = scale * imgf.float() @ base_txt.t().to(imgf.device)
        outs.append(out.detach().float().cpu())
        zss.append(zs.detach().float().cpu())
        labs.append(lab.detach().cpu())

    out = torch.cat(outs, dim=0).numpy()
    zs = torch.cat(zss, dim=0).numpy()
    lab = torch.cat(labs, dim=0).numpy()
    n_cls = base_txt.shape[0]
    m = math.ceil(n_cls / 2)
    names = list(self.dm.dataset.classnames)
    print(f"[attrib] n_class={n_cls} base={m} new={n_cls - m} n_test={len(lab)}")

    def report(logits, tag):
        pred = logits.argmax(1)
        bm, nm = lab < m, lab >= m
        b = float((pred[bm] == lab[bm]).mean() * 100)
        n = float((pred[nm] == lab[nm]).mean() * 100)
        hm = 2 * b * n / (b + n) if b + n else 0.0
        overall = float((pred == lab).mean() * 100)
        to_base = float((pred[nm] < m).mean() * 100)
        per_class = {}
        for i in range(n_cls):
            sel = lab == i
            per_class[names[i] if i < len(names) else str(i)] = {
                "split": "base" if i < m else "new",
                "acc": round(float((pred[sel] == lab[sel]).mean() * 100), 2) if sel.any() else None,
                "n": int(sel.sum()),
            }
        print(f"[{tag}] overall {overall:.2f} | base {b:.2f} | new {n:.2f} | "
              f"HM {hm:.2f} | new->base {to_base:.1f}%")
        return {"overall": round(overall, 2), "base": round(b, 2), "new": round(n, 2),
                "HM": round(hm, 2), "new_pred_into_base_pct": round(to_base, 2),
                "per_class": per_class}

    trained = report(out, "DVR trained")
    zero = report(zs, "CLIP zero-shot")
    result = {
        "note": "zero-shot = base_text_features only (no trained residual); "
                "same test loader as training, so numbers are directly comparable.",
        "dataset": self.cfg.DATASET.NAME,
        "n_class": n_cls,
        "n_base": m,
        "n_test": int(len(lab)),
        "classnames": names,
        "DVR_trained": trained,
        "CLIP_zero_shot": zero,
        "delta_new": round(trained["new"] - zero["new"], 2),
        "delta_base": round(trained["base"] - zero["base"], 2),
    }
    with open(ATTR_OUT, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("[attrib] saved:", ATTR_OUT)
    return trained["HM"]


def main():
    global ATTR_OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hy-tmp/TRIDENT/datasets")
    ap.add_argument("--dataset-config-file", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--load-epoch", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="/hy-tmp/b2n_attrib.json")
    args = ap.parse_args()
    ATTR_OUT = args.out

    cfg_args = Namespace(
        root=args.root, output_dir="", resume="", seed=args.seed,
        source_domains=None, target_domains=None, transforms=None,
        config_file="./configs/trainers/DVR/adam_lr2e-4_B256_ep200_ViT14.yaml",
        dataset_config_file=args.dataset_config_file, trainer="DVR",
        backbone="", head="", eval_only=True, model_dir=args.model_dir,
        load_epoch=args.load_epoch, no_train=True, enhanced_base="none",
        opts=["DATASET.NUM_SHOTS", "16", "TRAINER.DVR.RESIDUAL_SCALE", "0.5",
              "DATASET.SUBSAMPLE_CLASSES", "base"],
    )
    cfg = T.setup_cfg(cfg_args)
    D.DVR.test = patched_test  # 运行时 patch，磁盘上的 dvr.py 不变
    trainer = build_trainer(cfg)
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer.test()


if __name__ == "__main__":
    main()
