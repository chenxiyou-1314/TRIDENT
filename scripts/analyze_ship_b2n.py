"""Ship30 base2new 归因分析 + 全 30 类 OOD 指标（独立脚本，不修改 dvr.py）。

用途（对应剩余工作清单 2-3 / 2-4 / 2-8）：
  A. 用已保存的 DVR checkpoint 复算 base/new/HM，与训练日志对齐做校验；
  B. 输出 per-class 准确率、new 类样本的预测落点分布、以及「CLIP 零样本
     文本先验」对照，用于判断 new acc 掉到 ~47% 的原因；
  C. 在 14 个 ID 类 + 16 个 OOD 类上算 OOD 检测指标 AUROC / FPR95 / OSCR。

设计约束：
  - 不 import 项目 trainer/dataset 模块，不改动任何已有文件；
  - 类别顺序严格复现 datasets/_common.py 的 filter_items（display 名按小写排序）；
  - 文本特征直接复用 checkpoint 里的 base_text_features / text_feature_residuals，
    避免重新编码带来的口径偏差；OOD 新类用 CLIP 文本编码器零样本生成。
  - OOD 评分用 MSP 与 energy（标准代理指标），不是论文中的 UOS，
    结果只作为辅助实验，报告中须注明口径。

用法：
  python scripts/analyze_ship_b2n.py \
      --root /hy-tmp/TRIDENT/datasets \
      --ckpt output/FINAL/debug/ViT14/ship14_ID/16shots/seed1_b2n/prompt_learner/model.pth.tar-200 \
      --split-json data/Boat-14/id_classes.json \
      --out /hy-tmp/ship_analysis.json --max-per-class 100
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 仓库根目录下是本地 clip 包；本脚本位于 scripts/ 下，需手动加入搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clip import clip

TEMPLATE = "a photo of a {}, a type of boat"
ALPHA = 0.5  # TRAINER.DVR.RESIDUAL_SCALE


def norm(name):
    """与 datasets/_common.py 的 norm 保持一致。"""
    return name.replace("_", " ").strip().lower()


def display(name):
    """与 datasets/boat29_14.py 的 to_display 保持一致。"""
    return name.replace("_", " ").strip()


def build_transform():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711],
        ),
    ])


@torch.no_grad()
def infer(clip_model, tensors, device, batch_size=64):
    """批量抽取归一化图像特征；输入张量已与标签对齐。"""
    feats = []
    for i in range(0, len(tensors), batch_size):
        x = torch.stack(tensors[i:i + batch_size]).to(device)
        f = clip_model.visual(x)
        feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty(0, 768)


@torch.no_grad()
def text_features(clip_model, names, device):
    """用与训练相同的模板生成零样本文本特征（用于 OOD 新类）。"""
    tokens = clip.tokenize([TEMPLATE.format(n) for n in names]).to(device)
    f = clip_model.encode_text(tokens)
    return (f / f.norm(dim=-1, keepdim=True)).float()


def auroc(pos, neg):
    """pos=ID 分数，neg=OOD 分数，分数越大越像 ID。"""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列：同分取平均秩
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = sums / counts
    ranks = avg[inv]
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fpr95(pos, neg):
    """TPR=95% 时的 FPR。"""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 0.05)
    return float((neg >= thr).mean())


def oscr(id_scores, id_correct, ood_scores):
    """OSCR 曲线下面积：CCR(正确分类且被接受) vs FPR(误接受 OOD)。"""
    id_scores = np.asarray(id_scores, dtype=np.float64)
    id_correct = np.asarray(id_correct, dtype=bool)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return float("nan")
    cand = np.unique(np.concatenate([id_scores, ood_scores]))
    if len(cand) > 2000:
        cand = np.quantile(cand, np.linspace(0, 1, 2000))
    points = []
    for t in cand:
        ccr = ((id_scores >= t) & id_correct).sum() / n_id
        fpr = (ood_scores >= t).sum() / n_ood
        points.append((fpr, ccr))
    points.sort()
    # 按 FPR 去重取最大 CCR，梯形积分并归一化
    best, area, prev_f, prev_c = {}, 0.0, None, None
    for f, c in points:
        best[f] = max(best.get(f, -1.0), c)
    fs = sorted(best)
    for f in fs:
        c = best[f]
        if prev_f is not None:
            area += (f - prev_f) * (c + prev_c) / 2
        prev_f, prev_c = f, c
    return float(area / max(prev_f, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hy-tmp/TRIDENT/datasets")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-json", default="data/Boat-14/id_classes.json")
    ap.add_argument("--out", default="/hy-tmp/ship_analysis.json")
    ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = os.path.join(args.root, "boat-29")
    img_dir = os.path.join(ds, "images")
    test_meta = json.load(open(os.path.join(ds, "meta", "test.json")))
    split = json.load(open(args.split_json))
    id_json, ood_json = split["id_classes"], split["ood_classes"]

    # 复现训练侧的类别顺序：display 名按小写排序，再过滤出 ID 类
    all_display = sorted({display(k) for k in test_meta.keys()}, key=str.lower)
    id_set = {norm(c) for c in id_json}
    ood_set = {norm(c) for c in ood_json}
    id_names = [d for d in all_display if norm(d) in id_set]
    ood_names = [d for d in all_display if norm(d) in ood_set]
    assert len(id_names) == 14, f"ID classes mismatch: {len(id_names)}"
    assert len(ood_names) == 16, f"OOD classes mismatch: {len(ood_names)}"
    m = math.ceil(len(id_names) / 2)
    print("ID order (first %d = base):" % m, id_names)

    # display -> snake_case（meta 的键）
    snake = {display(k): k for k in test_meta.keys()}

    def paths_of(names):
        out = []
        for n in names:
            rels = test_meta.get(snake.get(n, ""), [])
            for r in rels[: args.max_per_class]:
                out.append((n, os.path.join(img_dir, r + ".jpg")))
        return out

    # ---- 加载 checkpoint ----
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck)
    base_txt = sd["base_text_features"].float()          # [14, 768] 零样本文本特征
    residual = sd["text_feature_residuals"].float()      # [14, 768] 训练残差
    trained_txt = base_txt + ALPHA * residual
    base_txt = base_txt / base_txt.norm(dim=-1, keepdim=True)
    trained_txt = trained_txt / trained_txt.norm(dim=-1, keepdim=True)
    print("checkpoint:", tuple(sd["text_feature_residuals"].shape))

    clip_model, _ = clip.load("ViT-L/14", device=device, jit=False)
    clip_model = clip_model.float().eval()
    scale = clip_model.logit_scale.exp().item()
    tfm = build_transform()

    # ---- 抽取 ID 测试图像特征 ----
    id_items = paths_of(id_names)
    name2idx = {n: i for i, n in enumerate(id_names)}
    tensors, labels = [], []
    for n, p in id_items:
        try:
            with Image.open(p) as im:
                tensors.append(tfm(im.convert("RGB")))
            labels.append(name2idx[n])
        except Exception:
            continue
    id_feat = infer(clip_model, tensors, device, args.batch_size)
    id_lab = torch.tensor(labels)
    print("ID test images:", len(labels))

    def logits_from(feat, txt):
        return (scale * feat @ txt.t()).numpy()

    def acc_report(logits, tag):
        pred = logits.argmax(1)
        lab = id_lab.numpy()
        base_mask, new_mask = lab < m, lab >= m
        b = float((pred[base_mask] == lab[base_mask]).mean() * 100)
        n = float((pred[new_mask] == lab[new_mask]).mean() * 100)
        hm = 2 * b * n / (b + n) if b + n else 0.0
        overall = float((pred == lab).mean() * 100)
        per_class = {}
        for i, name in enumerate(id_names):
            sel = lab == i
            per_class[name] = {
                "split": "base" if i < m else "new",
                "acc": round(float((pred[sel] == lab[sel]).mean() * 100), 2) if sel.any() else None,
                "n": int(sel.sum()),
            }
        # new 类样本预测落在 base 类的比例（head 偏置证据）
        pred_new = pred[new_mask]
        to_base = float((pred_new < m).mean() * 100)
        print(f"[{tag}] overall {overall:.2f} | base {b:.2f} | new {n:.2f} | HM {hm:.2f} "
              f"| new->base {to_base:.1f}%")
        return {"overall": round(overall, 2), "base": round(b, 2), "new": round(n, 2),
                "HM": round(hm, 2), "new_pred_into_base_pct": round(to_base, 2),
                "per_class": per_class}

    # ---- B. 归因：trained vs zero-shot ----
    logits_tr = logits_from(id_feat, trained_txt)
    logits_zs = logits_from(id_feat, base_txt)
    res_trained = acc_report(logits_tr, "DVR trained")
    res_zero = acc_report(logits_zs, "CLIP zero-shot")

    # ---- C. 全 30 类 / OOD 指标 ----
    ood_items = paths_of(ood_names)
    tensors = []
    for n, p in ood_items:
        try:
            with Image.open(p) as im:
                tensors.append(tfm(im.convert("RGB")))
        except Exception:
            continue
    ood_feat = infer(clip_model, tensors, device, args.batch_size)
    print("OOD test images:", ood_feat.shape[0])

    # ID-only 分类器上的 OOD 评分（标准 open-set 设定）
    lid = logits_from(id_feat, trained_txt)
    lood = logits_from(ood_feat, trained_txt) if ood_feat.shape[0] else np.empty((0, 14))

    def msp(l):
        e = np.exp(l - l.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).max(1)

    def energy(l):
        return np.log(np.exp(l).sum(1))

    id_correct = (lid.argmax(1) == id_lab.numpy())
    ood_metrics = {}
    for tag, fn in [("MSP", msp), ("Energy", energy)]:
        s_id, s_ood = fn(lid), fn(lood) if lood.shape[0] else np.array([])
        # 两种分数都已是「越大越像 ID」：MSP 为最大 softmax 概率，
        # Energy 为 logsumexp(logits)（ID 样本能量更高），无需再取负。
        ood_metrics[tag] = {
            "AUROC": round(auroc(s_id, s_ood), 4),
            "FPR95": round(fpr95(s_id, s_ood), 4),
            "OSCR": round(oscr(s_id, id_correct, s_ood), 4),
        }
        print(f"[OOD {tag}] AUROC {ood_metrics[tag]['AUROC']} | "
              f"FPR95 {ood_metrics[tag]['FPR95']} | OSCR {ood_metrics[tag]['OSCR']}")

    # 30 类 closed-set（ID 用 trained、OOD 新类用零样本文本特征，模拟类别空间扩展）
    ood_txt = text_features(clip_model, ood_names, device).cpu()
    txt30 = torch.cat([trained_txt, ood_txt], dim=0)
    names30 = id_names + ood_names
    idx30 = {n: i for i, n in enumerate(names30)}
    l30_id = logits_from(id_feat, txt30)
    l30_ood = logits_from(ood_feat, txt30) if ood_feat.shape[0] else np.empty((0, 30))
    lab30 = np.array([idx30[id_names[i]] for i in id_lab.numpy()])
    closed_id = float((l30_id.argmax(1) == lab30).mean() * 100)
    print(f"[30-class closed-set] ID acc {closed_id:.2f}%")

    out = {
        "note": "OOD scores are MSP/Energy proxies on the 14-ID classifier, NOT UOS. "
                "Preprocessing is an equivalent re-implementation of the Dassl test transform; "
                "small numeric differences vs training logs are expected.",
        "checkpoint": args.ckpt,
        "max_per_class": args.max_per_class,
        "id_order": id_names,
        "ood_order": ood_names,
        "n_base_classes": m,
        "DVR_trained": res_trained,
        "CLIP_zero_shot": res_zero,
        "OOD_metrics_ID_classifier": ood_metrics,
        "closed_set_30": {"id_acc": round(closed_id, 2), "n_id": int(id_feat.shape[0]),
                          "n_ood": int(ood_feat.shape[0])},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
