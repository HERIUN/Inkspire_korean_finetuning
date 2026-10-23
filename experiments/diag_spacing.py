"""진단: ref 글자 간격이 불규칙하면 InkSpire 가 무너지는가 — 폰트 렌더 ref 의 자간만 흔든다(필체·획은 그대로).

실손글씨에서 InkSpire CER 이 폰트 렌더의 10배(0.023 → 0.246)인데 획 굵기·ref 판독성과는 상관이 없었다.
남은 후보는 X↔Xc 대응표 어긋남: InkSpire 는 ref 의 표준폰트 상자를 줄 폭에 균일하게 늘려 짝을 짓는데(§8),
손글씨는 자간이 불규칙해 상자가 글자에서 밀린다. 같은 폰트로 글자마다 간격을 U(−j, +j)·글자폭 만큼 흔든 ref 를 주고
목표(§5 와 같은 표본, n 개)를 생성해 cer_dump 형식으로 덤프한다 → ppocr_rec.py 로 채점.

  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python experiments/diag_spacing.py finetune_runs/_cer/diag_spacing_j0.4 --jitter 0.4
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from infer.show import load_model, render_in_font   # noqa: E402

CKPT = "inkspire:finetune_runs/inkspire_klein_p512/lora_step_013000"


def jitter_render(fp, text, j, rng, size=72, pad=14):
    """글자마다 따로 그리고 다음 글자 시작을 advance·(1 + U(−j, j)) 만큼 민다(공백 포함). 잉크 bbox 로 크롭."""
    f = ImageFont.truetype(str(fp), size)
    W = int(sum(f.getlength(c) for c in text) * (1 + j) + 4 * pad)
    im = Image.new("L", (W, size * 2), 255)
    d, x = ImageDraw.Draw(im), pad
    for c in text:
        d.text((x, pad), c, font=f, fill=0)
        x += f.getlength(c) * (1 + rng.uniform(-j, j))
    a = np.array(im)
    ys, xs = np.where(a < 250)
    return a[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jitter", type=float, default=0.4)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--plan", default=str(HERE / "finetune_runs/_cer/s13000_test_sref/plan.json"))
    a = ap.parse_args()
    p = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    rows, sref, out = p["rows"][:a.n], p["style_ref_text"], Path(a.out)
    model = load_model(a.ckpt, torch.device("cuda"))[0]
    for sub in ("gen", "gt", "ref"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tr = {}
    for r in rows:
        name, fp = f"{r['i']:05d}.png", Path(r["font"])
        rng = random.Random(r["i"])
        ref = jitter_render(fp, sref, a.jitter, rng) if a.jitter > 0 else render_in_font(fp, sref)
        Image.fromarray(ref).save(out / "ref" / name)
        Image.fromarray(model.gen(ref, sref, r["text"], p["seed"] * 1_000_003 + r["i"])).save(out / "gen" / name)
        Image.fromarray(render_in_font(fp, r["text"])).save(out / "gt" / name)
        tr[name] = r["text"]
    for sub in ("gen", "gt"):
        (out / sub / "transcriptions.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    (out / "plan.json").write_text(json.dumps({**p, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"gen {len(tr)} jitter={a.jitter} → {out}")


if __name__ == "__main__":
    main()
