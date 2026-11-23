"""realhw_dump 덤프들의 PP-OCR CER 통계 — corpus CER + 표본 부트스트랩 95% CI, 실제 이미지 대비 ΔCER, 모델 쌍 표본별 승패·부호검정.

덤프마다 gen/ppocr_preds.json(eval/ppocr_rec.py --save-preds) 이 있어야 한다. 실제 이미지 바닥은 --gt 덤프의 gt/ppocr_preds.json.
같은 plan(같은 표본 순서)으로 만든 덤프끼리만 비교한다.
  python experiments/realhw_stats.py --gt <dump> 라벨=<dump> 라벨=<dump> …
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

norm = lambda s: re.sub(r"\s+", " ", s).strip()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def sign_p(w, l):
    """양측 부호검정(동률 제외) — 이항 정확 검정."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="실제 이미지 바닥을 읽을 덤프(gt/ppocr_preds.json)")
    ap.add_argument("runs", nargs="+", help="라벨=덤프디렉토리")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rows = json.loads((Path(a.gt) / "plan.json").read_text(encoding="utf-8"))["rows"]
    keys = [f"{r['i']:05d}.png" for r in rows]
    N = [len(norm(r["text"])) for r in rows]
    runs = [("실제 이미지", Path(a.gt) / "gt")] + [(s.split("=", 1)[0], Path(s.split("=", 1)[1]) / "gen") for s in a.runs]
    E = {}
    for lab, d in runs:
        P = json.loads((d / "ppocr_preds.json").read_text(encoding="utf-8"))
        E[lab] = [ed(norm(r["text"]), norm(P[k])) for r, k in zip(rows, keys)]
    rng = random.Random(0)
    idx = [[rng.randrange(len(rows)) for _ in rows] for _ in range(a.boot)]
    base = sum(E["실제 이미지"]) / sum(N)
    print(f"n={len(rows)}  글자 {sum(N)}")
    print(f"| | CER [95% CI] | ΔCER (vs 실제 {base:.3f}) |\n|---|---|---|")
    for lab, _ in runs:
        c = sum(E[lab]) / sum(N)
        bs = sorted(sum(E[lab][i] for i in b) / sum(N[i] for i in b) for b in idx)
        ci = f"[{bs[int(.025 * a.boot)]:.3f}, {bs[int(.975 * a.boot)]:.3f}]"
        print(f"| {lab} | {c:.3f} {ci} | {'—' if lab == '실제 이미지' else f'{c - base:+.3f}'} |")
    labs = [l for l, _ in runs[1:]]
    print("\n| A vs B | A 나음 / B 나음 / 같음 | 부호검정 p |\n|---|---|---|")
    for i, x in enumerate(labs):
        for y in labs[i + 1:]:
            w = sum(p < q for p, q in zip(E[x], E[y]))   # 같은 표본(같은 정답)이라 편집거리 비교 = CER 비교
            l = sum(p > q for p, q in zip(E[x], E[y]))
            print(f"| {x} vs {y} | {w} / {l} / {len(rows) - w - l} | {sign_p(w, l):.2g} |")


if __name__ == "__main__":
    main()
