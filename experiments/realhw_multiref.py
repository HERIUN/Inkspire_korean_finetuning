"""ref 를 여러 줄로 — realhw_dump --all-pairs 덤프의 plan 을 그대로 받아, 같은 필기자의 다른 줄 K-1 개를 ref 아래에 더 깔고 생성한다.

추가 줄 = 그 plan 의 다른 ref(목표 문구·첫 ref 제외, 표본 번호로 고정한 순서). 시드·ref 전처리는 realhw_dump 와 같다(K=1 이면 같은 조건).
  python experiments/realhw_multiref.py <out> --src <dump> --ckpt inkspire:…,word_gap=1 --k 3
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--src", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=2, help="보이는 ref 줄 수(1 = 기존)")
    a = ap.parse_args()
    import torch
    from infer.show import load_model, style_tensor
    src, out = Path(a.src), Path(a.out)
    P = json.loads((src / "plan.json").read_text(encoding="utf-8"))
    g = load_model(a.ckpt, torch.device("cuda"))[0]
    load = lambda p: ((style_tensor(np.asarray(Image.open(p).convert("L")))[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    pool = {}   # 필기자 → {ref 전사: ref 이미지}
    for r in P["rows"]:
        pool.setdefault(r["font"], {}).setdefault(r["ref_text"], r["ref_image"])
    (out / "gen").mkdir(parents=True, exist_ok=True)
    for n, r in enumerate(P["rows"]):
        seed = P["seed"] * 1_000_003 + r["i"]
        oth = sorted(t for t in pool[r["font"]] if t not in (r["ref_text"], r["text"]))
        random.Random(r["i"]).shuffle(oth)
        ctx = [(load(pool[r["font"]][t]), t) for t in oth[:a.k - 1]]
        assert len(ctx) == a.k - 1, (r["i"], len(oth))
        Image.fromarray(g.gen_lines(load(r["ref_image"]), r["ref_text"], [r["text"]], seed, ctx)[0]).save(out / "gen" / f"{r['i']:05d}.png")
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(P['rows'])}", flush=True)
    shutil.copy(src / "plan.json", out / "plan.json")
    shutil.copy(src / "gen/transcriptions.json", out / "gen/transcriptions.json")
    (out / "ckpt.txt").write_text(f"{a.ckpt} k={a.k}\n")
    print(f"gen → {out}")


if __name__ == "__main__":
    main()
