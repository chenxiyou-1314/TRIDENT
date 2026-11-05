"""从 HF 镜像下载外部 OOD 数据集的抽样子集（绕开被墙的 Google Drive）。

第四章 OOD 检测实验需要 iNaturalist / SUN / Places / Texture 四件套。
镜像可用性（2026-11-05 核实）：
  - DTD (Texture)      cansa/Describable-Textures-Dataset-DTD  逐张 jpg，可直接抽样 ✅
  - Places365          Andron00e/Places365-custom              322 个 parquet，体量大
  - iNaturalist        haideraltahan/wds_inaturalist           tar 分片，体量大
  - SUN397             HuggingFaceM4/sun397                    仅 sun397.py，无数据文件 ❌
本脚本先落地 DTD（最小、最可靠），作为外部 OOD 评估链路的验证集；
其余三个可在磁盘充足时按同样方式分批扩展。

用法：
  python scripts/data_prep/download_ood_subset.py \
      --out /hy-tmp/TRIDENT/datasets/ood_samples/dtd --n 500
"""
import argparse
import concurrent.futures
import json
import os
import random
import urllib.request

REPO = "cansa/Describable-Textures-Dataset-DTD"
API = f"https://hf-mirror.com/api/datasets/{REPO}"
BASE = f"https://hf-mirror.com/datasets/{REPO}/resolve/main/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    req = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        meta = json.load(r)
    files = [s["rfilename"] for s in meta.get("siblings", [])
             if s["rfilename"].endswith((".jpg", ".jpeg", ".png"))]
    print("total image files in repo:", len(files))
    random.seed(args.seed)
    random.shuffle(files)
    files = files[:args.n]
    os.makedirs(args.out, exist_ok=True)

    def dl(rel):
        dst = os.path.join(args.out, rel.replace("/", "_"))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        try:
            q = urllib.request.Request(BASE + rel, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(q, timeout=60) as r, open(dst, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            print("FAIL", rel, str(e)[:80])
            return False

    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        ok = sum(ex.map(dl, files))
    print(f"downloaded {ok}/{len(files)} -> {args.out}")
    print("DTD_SUBSET_DONE" if ok > 0 else "DTD_SUBSET_FAILED")


if __name__ == "__main__":
    main()
