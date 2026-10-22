"""저장된 생성 덤프의 스타일 지표 — HWD · FID · BFID · KID (hwd 패키지, Eruku 논문 표 3 과 같은 구현).

hwd 는 원본 repo .venv(py3.11)에만 있으므로 그 파이썬으로 돌린다(ppocr_rec.py 와 같은 방식). 추론 없음.
real = 같은 표본의 GT 렌더(그 폰트로 목표 텍스트), fake = 각 덤프 gen/. 폴더명 = 필기자(폰트) → HWD 는 폰트별 거리의 평균.
기준선 'std-font' = 목표 텍스트를 Xc 표준폰트(NanumGothic)로 렌더한 것 — 스타일을 전혀 안 옮겼을 때의 값.
★ 정규화(기본): 모든 이미지(real 포함)를 배경 중앙값 → 255 로 늘리고, 잉크 bbox 로 자른 뒤 잉크 높이의 15% 여백.
  덤프마다 여백(GT 30% / InkSpire 17% / Eruku 20%)·배경(InkSpire 249~251)이 달라 높이 32 로 줄이면 글자 크기와
  밝기가 달라지고, FID 가 그 차이를 잡는다(정규화 전 InkSpire FID ≈ 표준폰트). --raw 로 끈다.

  E=/data/work/dgkang/Eruku_korean_finetuning
  $E/.venv/bin/python experiments/style_scores.py                 # 기본: FLUX.3-Klein / FLUX.1-Fill / Eruku / std-font
  $E/.venv/bin/python experiments/style_scores.py --runs "A=<dump>" "B=<dump>"
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
CER = HERE / "finetune_runs/_cer"
RUNS = [f"FLUX.3-Klein s13000={CER}/klein_s013000_test_sref",
        f"FLUX.1-Fill s13000={CER}/s13000_test_sref",
        f"Eruku s45000={CER}/eruku_test_sref",
        "std-font=std"]
STD_FONT = HERE / "assets/fonts/label/NanumGothic-Regular.ttf"


def render(fp, text, size=72, pad=14):
    f = ImageFont.truetype(str(fp), size)
    b = f.getbbox(text)
    im = Image.new("L", (b[2] - b[0] + 2 * pad, b[3] - b[1] + 2 * pad), 255)
    ImageDraw.Draw(im).text((pad - b[0], pad - b[1]), text, font=f, fill=0)
    return im


def norm(im):
    """배경 → 255, 잉크 bbox 크롭, 잉크 높이 15% 여백."""
    import numpy as np
    a = np.asarray(im.convert("L"), np.float32)
    a = np.clip(a * 255.0 / max(1.0, float(np.median(a))), 0, 255)
    ys, xs = np.where(a < 200)
    if len(ys):
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    p = max(2, int(round(0.15 * a.shape[0])))
    return Image.fromarray(np.pad(a, p, constant_values=255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=RUNS, help="라벨=덤프디렉토리 (std = 표준폰트 렌더 기준선)")
    ap.add_argument("--plan", default=str(CER / "s13000_test_sref/plan.json"))
    ap.add_argument("--real", default=str(CER / "s13000_test_sref/gt"))
    ap.add_argument("--height", type=int, default=32, help="hwd 입력 높이(Eruku 논문·원본 repo 와 같게 32)")
    ap.add_argument("--raw", action="store_true", help="정규화 없이 덤프 그대로")
    ap.add_argument("--out", default=str(CER / "style_scores.json"))
    a = ap.parse_args()

    from hwd.datasets import FolderDataset
    from hwd.scores import BFIDScore, FIDScore, HWDScore, KIDScore

    rows = json.loads(Path(a.plan).read_text(encoding="utf-8"))["rows"]
    tmp = Path(tempfile.mkdtemp(prefix="style_"))
    try:
        def folder(tag, src):   # tag/<폰트>/<name>.png — src(name, row) → 이미지 경로 또는 PIL
            for r in rows:
                name = f"{r['i']:05d}.png"
                d = tmp / tag / Path(r["font"]).stem
                d.mkdir(parents=True, exist_ok=True)
                s = src(name, r)
                if isinstance(s, Path):
                    if not s.exists():
                        continue
                    if a.raw:
                        (d / name).symlink_to(s); continue
                    s = Image.open(s)
                (s if a.raw else norm(s)).save(d / name)
            return FolderDataset(str(tmp / tag))

        real = folder("real", lambda n, r: Path(a.real) / n)
        scores = {"HWD": HWDScore(height=a.height), "FID": FIDScore(height=a.height),
                  "BFID": BFIDScore(height=a.height), "KID": KIDScore(height=a.height)}
        res = {}
        for i, spec in enumerate(a.runs):
            lab, d = spec.split("=", 1)
            fake = folder(f"fake{i}", (lambda n, r: render(STD_FONT, r["text"])) if d == "std"
                          else (lambda n, r, d=Path(d): d / "gen" / n))
            res[lab] = {k: float(s(fake, real)) for k, s in scores.items()}
            print(f"{lab:22s} " + "  ".join(f"{k} {v:.3f}" for k, v in res[lab].items()), flush=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {a.out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
