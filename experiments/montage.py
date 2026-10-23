"""저장된 생성 덤프(cer_dump.py gen / htr_cer.py --dump-dir)를 나란히 몽타주 — 추론 없이 PNG 만 합친다.

행 = 표본, 열 = [ref | GT | 모델1 | 모델2 …]. 생성 셀 라벨 = 모델 · CER · PP-OCR 판독(CER>0 이면 진한 회색 굵게).
같은 plan.json 표본이라 파일명(00000.png …)이 같은 텍스트·폰트·시드다.

  python experiments/montage.py                                   # 기본 3모델, 무작위 24행
  python experiments/montage.py --pick worst --n 16               # 모델 중 최악 CER 이 큰 순
  python experiments/montage.py --pick diff --font NanumBrushScript
  python experiments/montage.py --where "A<B" --one-per-font    # A 가 B 보다 CER 낮은 표본, 격차 큰 순, 폰트당 1개
  python experiments/montage.py --runs "A=finetune_runs/_cer/x" "B=finetune_runs/_cer/y" --out _debug/ab.png
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from infer.show import fit, label_img, render_in_font, style_tensor   # noqa: E402

CER = HERE / "finetune_runs/_cer"
RUNS = [f"FLUX.3-Klein s13000={CER}/klein_s013000_test_sref",
        f"FLUX.1-Fill s13000={CER}/s13000_test_sref",
        f"Eruku s45000={CER}/eruku_test_sref"]


def cell(content, label, cw, ch, highlight=False):
    """infer.show.cell 과 같되 라벨 좌측 정렬(긴 판독문 앞부분이 안 잘리게)."""
    lab = label_img(label, cw, 24, 14, bold=highlight, center=False, bg=200 if highlight else 245)
    return np.pad(np.vstack([lab, np.zeros((1, cw), np.uint8), fit(content, cw, ch)]), 2, constant_values=0)


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS, help="라벨=덤프디렉토리 (gen/ 과 gen/ppocr_preds.json 필요)")
    ap.add_argument("--plan", default=str(CER / "s13000_test_sref/plan.json"), help="폰트·ref 텍스트 (ref 열)")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pick", choices=["random", "worst", "diff"], default="random",
                    help="worst = 모델 중 최대 CER 큰 순 / diff = 모델 간 CER 차이 큰 순")
    ap.add_argument("--font", default=None, help="폰트 이름 부분일치 필터")
    ap.add_argument("--where", default=None, help='"라벨A<라벨B" — A 의 CER 이 B 보다 낮은 표본만, 격차 큰 순')
    ap.add_argument("--one-per-font", action="store_true", help="정렬 순서대로 폰트당 1개")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell-w", type=int, default=560)
    ap.add_argument("--cell-h", type=int, default=60)
    ap.add_argument("--show-ctx", action="store_true", help="덤프에 ctx/ 가 있으면 보강 줄을 생성 위에 같이 보인다(셀 높이 2배 권장)")
    ap.add_argument("--out", default=str(HERE / "_debug/montage.png"))
    a = ap.parse_args()

    runs = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in a.runs]
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    rows = {f"{r['i']:05d}.png": r for r in plan["rows"]}
    preds = [json.loads((d / "gen/ppocr_preds.json").read_text(encoding="utf-8")) for _, d in runs]
    names = [n for n in rows if all((d / "gen" / n).exists() for _, d in runs)]
    if a.font:
        names = [n for n in names if a.font.lower() in Path(rows[n]["font"]).stem.lower()]
    cer = {n: [lev(rows[n]["text"], p.get(n, "")) / max(1, len(rows[n]["text"])) for p in preds] for n in names}
    if a.pick == "random":
        random.Random(a.seed).shuffle(names)
    else:
        key = (lambda n: max(cer[n])) if a.pick == "worst" else (lambda n: max(cer[n]) - min(cer[n]))
        names.sort(key=key, reverse=True)
    if a.where:
        la, lb = (x.strip() for x in a.where.split("<"))
        ia, ib = [lab for lab, _ in runs].index(la), [lab for lab, _ in runs].index(lb)
        names = sorted((n for n in names if cer[n][ia] < cer[n][ib]), key=lambda n: cer[n][ib] - cer[n][ia], reverse=True)
    if a.one_per_font:
        seen = set()
        names = [n for n in names if not (Path(rows[n]["font"]).stem in seen or seen.add(Path(rows[n]["font"]).stem))]
    names = names[:a.n]
    if not names:
        sys.exit("[error] 조건에 맞는 표본 없음")

    W, H = a.cell_w, a.cell_h
    ref_cache, grid = {}, []
    for n in names:
        r = rows[n]
        fp = Path(r["font"])
        if "ref_image" in r:      # realhw_dump.py — ref 가 실제 이미지
            ref_cache[fp] = np.array(Image.open(r["ref_image"]).convert("L"))
        elif fp not in ref_cache:   # 모델이 받은 것과 같은 64px ref
            t = style_tensor(render_in_font(fp, plan["style_ref_text"]))
            ref_cache[fp] = ((t[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        gt = np.array(Image.open(runs[0][1] / "gt" / n).convert("L")) if (runs[0][1] / "gt" / n).exists() \
            else render_in_font(fp, r["text"])
        cells = [cell(ref_cache[fp], f"{n[:5]} {fp.stem[:18]}", W // 2, H),
                 cell(gt, "GT  " + r["text"][:40], W, H)]
        for (lab, d), p, c in zip(runs, preds, cer[n]):
            g = np.array(Image.open(d / "gen" / n).convert("L"))
            if a.show_ctx and (d / "ctx" / n).exists():   # cer_dump --ctx 의 보강 줄을 위에(회색 띠로 구분)
                cx = np.array(Image.open(d / "ctx" / n).convert("L"))
                cx = np.array(Image.fromarray(cx).resize((max(1, cx.shape[1] * g.shape[0] // cx.shape[0]), g.shape[0])))
                Wc = max(cx.shape[1], g.shape[1])
                g = np.vstack([np.pad(cx, ((0, 0), (0, Wc - cx.shape[1])), constant_values=255), np.full((6, Wc), 160, np.uint8),
                               np.pad(g, ((0, 0), (0, Wc - g.shape[1])), constant_values=255)])
            cells.append(cell(g, f"{lab} {c:.2f}  {p.get(n, '')[:40]}", W, H, highlight=c > 0))
        grid.append(np.hstack(cells))
    Wmax = max(g.shape[1] for g in grid)
    img = np.vstack([np.pad(g, ((0, 0), (0, Wmax - g.shape[1])), constant_values=255) for g in grid])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(a.out)
    mean = np.mean([cer[n] for n in names], 0)
    print(f"{len(names)}행 → {a.out}  (이 행들의 평균 CER: " + ", ".join(f"{l} {m:.3f}" for (l, _), m in zip(runs, mean)) + ")")


if __name__ == "__main__":
    main()
