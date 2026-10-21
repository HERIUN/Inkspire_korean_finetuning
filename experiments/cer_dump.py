"""InkSpire 생성을 원본 repo `eval/htr_cer.py --dump-dir` 와 같은 표본·형식으로 덤프 → 원본 repo 리더로 채점.

이 repo 엔 CER 리더가 없고 원본 repo 엔 InkSpire 코드가 없다. 그래서 표본 계획(텍스트·폰트)은 원본 repo 환경에서
htr_cer.py 와 같은 rng 순서로 뽑아 plan.json 에 쓰고(plan), 생성은 여기서 한다(gen).
출력 <out>/{gen,gt}/{i:05d}.png + transcriptions.json 은 htr_cer.py 덤프와 같은 형식이라
`eval/ppocr_rec.py <out>/gt <out>/gen --save-preds` 로 그대로 읽힌다.

  E=/data/work/dgkang/Eruku_korean_finetuning; I=/data/work/dgkang/inkspire_korean; O=$I/finetune_runs/_cer/s13000_test_sref
  cd $E && PYTHONPATH=$E .venv/bin/python $I/experiments/cer_dump.py plan $O --fonts-dir $E/assets/fonts_korean_v2/test \
      --n 300 --style-ref-text "다람쥐 헌 쳇바퀴에 타고파"
  cd $I && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$I .venv/bin/python experiments/cer_dump.py gen $O \
      --ckpt inkspire:finetune_runs/inkspire_p512/lora_step_013000
  cd $E && .venv-paddle/bin/python eval/ppocr_rec.py $O/gt $O/gen --save-preds
--style-text ocr 이면 gen 앞에: $I/.venv/bin/python cer_dump.py ref $O → .venv-paddle/bin/python eval/ppocr_rec.py $O/ref --save-preds
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def plan(a):
    """htr_cer.py main() 의 표본 루프를 그대로 따른다(rng 소비 순서 포함)."""
    from eval.echo_metrics import build_texts, font_can_render
    rng = random.Random(a.seed)
    fonts = sorted(p for p in Path(a.fonts_dir).glob("*.[ot]tf"))
    texts = build_texts(a.n, rng, english=False, coherent=True, max_words=a.max_words)
    rows = []
    for i, text in enumerate(texts):
        rng.shuffle(fonts)
        fp = next((f for f in fonts if font_can_render(f, text) and font_can_render(f, a.style_ref_text)), None)
        if fp is not None:
            rows.append({"i": i, "text": text, "font": str(fp)})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"seed": a.seed, "style_ref_text": a.style_ref_text, "rows": rows},
                                              ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"plan {len(rows)} rows → {out / 'plan.json'}")


def ref(a):
    """모델이 받는 ref 이미지(style_tensor 거친 64px 줄) 덤프 → 원본 repo ppocr_rec.py --save-preds 로 읽어 --style-text ocr 에 쓴다."""
    from infer.show import render_in_font, style_tensor
    out = Path(a.out)
    p = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    (out / "ref").mkdir(exist_ok=True)
    tr = {}
    for r in p["rows"]:
        name = f"{r['i']:05d}.png"
        t = style_tensor(render_in_font(Path(r["font"]), p["style_ref_text"]))
        Image.fromarray(((t[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(out / "ref" / name)
        tr[name] = p["style_ref_text"]
    (out / "ref" / "transcriptions.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    print(f"ref {len(tr)} → {out / 'ref'}")


def gen(a):
    import torch
    from infer.show import gen_from_style, load_model, render_in_font, style_tensor
    out = Path(a.out)
    p = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    model = load_model(a.ckpt, torch.device("cuda"))[0]
    sref, tr = p["style_ref_text"], {}
    ocr = json.loads((out / "ref/ppocr_preds.json").read_text()) if a.style_text == "ocr" else {}
    for sub in ("gen", "gt"):
        (out / sub).mkdir(exist_ok=True)
    for k, r in enumerate(p["rows"]):
        name = f"{r['i']:05d}.png"
        fp = Path(r["font"])
        g = gen_from_style(model, style_tensor(render_in_font(fp, sref)), {"ref": sref, "none": "", "target": r["text"], "ocr": ocr.get(name)}[a.style_text], r["text"], 1.0, 480,
                           torch.device("cuda"), seed=p["seed"] * 1_000_003 + r["i"])   # htr_cer.py 와 같은 시드
        Image.fromarray(g).save(out / "gen" / name)
        Image.fromarray(render_in_font(fp, r["text"])).save(out / "gt" / name)
        tr[name] = r["text"]
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(p['rows'])}", flush=True)
    for sub in ("gen", "gt"):
        (out / sub / "transcriptions.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    (out / "ckpt.txt").write_text(f"{a.ckpt} --style-text {a.style_text}\n")
    print(f"gen {len(tr)} → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "ref", "gen"])
    ap.add_argument("out")
    ap.add_argument("--fonts-dir")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-words", type=int, default=15)
    ap.add_argument("--style-ref-text", default="다람쥐 헌 쳇바퀴에 타고파")
    ap.add_argument("--ckpt")
    ap.add_argument("--style-text", choices=["ref", "none", "target", "ocr"], default="ref",
                    help="Xc 첫 줄에 줄 전사: ref=실제 ref 전사 / none=빈칸 / target=목표 텍스트(전사 모를 때 대용) / ocr=ref 를 PP-OCR 로 읽은 것(ref 모드+ppocr_rec 먼저)")
    a = ap.parse_args()
    {"plan": plan, "ref": ref, "gen": gen}[a.mode](a)
