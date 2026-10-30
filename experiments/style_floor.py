"""스타일 지표(HWD)의 하한 — "같은 필기자(폰트)의 다른 표본" 끼리의 거리. 모델 HWD 를 읽는 눈금(EXPERIMENTS §9·§10).

hwd 는 원본 repo .venv 에만 있다(style_scores.py 와 같다). 정규화도 style_scores.norm 그대로.
  font  — §5 표본(held-out 16폰트, n=300): 같은 폰트로 **다른 문구**(150행 떨어진 행의 문구)를 렌더 → GT 와 폰트별 HWD.
          같은 표본 수라 표본 잡음 없이 잰다. sanity(같은 문구 재렌더)=0 이면 렌더 경로가 GT 와 같다.
          ★ 폰트 안에서 GT 를 한 칸 미는 식은 같은 이미지 집합이라 평균 특징이 같아 0 이 된다 — 쓰지 말 것.
  split — 실손글씨처럼 다시 렌더할 수 없을 때: 필기자별 GT 를 겹치지 않는 두 절반(짝/홀)으로 나눠 HWD.
          절반 n 이라 표본 잡음만큼 높게 나온다(같은 폰트 조건에서 font 0.338 vs split 0.541).
          ★ realhw --cross 의 ref/ 를 fake 로 쓰면 안 된다 — ref 는 다른 행 GT 와 같은 원본(크롭만 다름)이다.

  E=/data/work/dgkang/Eruku_korean_finetuning
  CUDA_VISIBLE_DEVICES=2 $E/.venv/bin/python experiments/style_floor.py font
  CUDA_VISIBLE_DEVICES=2 $E/.venv/bin/python experiments/style_floor.py split \\
      --plan finetune_runs/_realhw/inkspire_cross/plan.json --gt finetune_runs/_realhw/inkspire_cross/gt
"""
import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "experiments"))
from style_scores import STD_FONT, norm, render  # noqa: E402

CER = HERE / "finetune_runs/_cer"


def hwd_per_writer(sc, real, fake):
    """hwd EuclideanDistance 와 같은 식(필기자별 평균 특징 거리)을 평균하지 않고 필기자별로."""
    return {w: torch.cdist(real[w].features.mean(0, keepdim=True), fake[w].features.mean(0, keepdim=True)).item()
            for w in sorted(set(real.authors) & set(fake.authors))}


def main():
    from hwd.datasets import FolderDataset
    from hwd.scores import HWDScore
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["font", "split"])
    ap.add_argument("--plan", default=str(CER / "s13000_test_sref/plan.json"))
    ap.add_argument("--gt", default=str(CER / "s13000_test_sref/gt"))
    ap.add_argument("--runs", nargs="*", default=[f"InkSpire={CER}/klein_s013000_refwidth", f"Eruku={CER}/eruku_ctx_cfg1.5"],
                    help="font 모드에서 같이 볼 덤프(라벨=디렉토리, gen/ 필요)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = json.loads(Path(a.plan).read_text(encoding="utf-8"))["rows"]
    gt = Path(a.gt)
    byw = defaultdict(list)
    for r in rows:
        byw[Path(r["font"]).stem].append(r)
    tmp, sc = Path(tempfile.mkdtemp(prefix="floor_")), HWDScore(height=32)

    def digest(tag, rs, src):   # tag/<필기자>/<i>.png
        for r in rs:
            d = tmp / tag / Path(r["font"]).stem
            d.mkdir(parents=True, exist_ok=True)
            s = src(r)
            norm(Image.open(s) if isinstance(s, Path) else s).save(d / f"{r['i']:05d}.png")
        return sc.digest(FolderDataset(str(tmp / tag)))

    try:
        if a.mode == "font":
            oth = {r["i"]: rows[(k + 150) % len(rows)]["text"] for k, r in enumerate(rows)}
            runs = {"sanity": lambda r: render(r["font"], r["text"]),
                    "floor": lambda r: render(r["font"], oth[r["i"]]),
                    **{lab: (lambda r, d=Path(d): d / "gen" / f"{r['i']:05d}.png")
                       for lab, d in (s.split("=", 1) for s in a.runs)},
                    "std-font": lambda r: render(STD_FONT, r["text"])}
            real = digest("real", rows, lambda r: gt / f"{r['i']:05d}.png")
            res = {lab: hwd_per_writer(sc, real, digest(lab, rows, fn)) for lab, fn in runs.items()}
            print(f"{'font':18s}{'n':>4s}" + "".join(f"{k:>10s}" for k in res))
            for w in sorted(res["floor"], key=res["floor"].get):
                print(f"{w:18s}{len(byw[w]):4d}" + "".join(f"{res[k][w]:10.3f}" for k in res))
            print(f"{'mean':18s}{len(rows):4d}" + "".join(f"{sum(v.values()) / len(v):10.3f}" for v in res.values()))
        else:
            half = [[r for rs in byw.values() for r in rs[p::2]] for p in (0, 1)]
            res = hwd_per_writer(sc, digest("A", half[0], lambda r: gt / f"{r['i']:05d}.png"),
                                 digest("B", half[1], lambda r: gt / f"{r['i']:05d}.png"))
            sub = defaultdict(list)
            for w, v in res.items():
                print(f"{w:24s}{len(byw[w]):4d}{v:8.3f}")
                sub[byw[w][0].get("subset", "font")].append(v)
            for s, v in sub.items():
                print(f"mean[{s}] 필기자 {len(v)}  {sum(v) / len(v):.3f}")
        if a.out:
            Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
