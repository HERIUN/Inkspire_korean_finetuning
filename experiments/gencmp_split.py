"""Eruku vs InkSpire 몽타주 — 두 모델을 각자의 repo 환경에서 따로 생성하고 여기서 합친다.

gen_compare.py 는 두 ckpt 를 한 프로세스에 올리는데, 이 repo 의 infer/show.py 는 Eruku 를 못 올리고
원본 repo 엔 InkSpire 코드가 없다. 그래서 셀 단위 PNG 로 나눴다. 계획(폰트×목표)은 결정적이라
두 프로세스가 같은 행을 만든다.

style ref 는 held-out 폰트로 한글 pangram 을 즉석 렌더한다(영어 카테고리는 영어 문장).
★ ref_set 의 혼합 스크립트 줄(영문·숫자·구두점)을 ref 로 쓰면 InkSpire 의 레퍼런스 bbox 근사가
  크게 어긋나 글자가 깨진다(docs/inkspire.md §9.4) — 비교 조건을 CER style-ref 프로토콜에 맞춘다.

  E=/data/work/dgkang/Eruku_korean_finetuning; I=/data/work/dgkang/inkspire_korean; O=$I/finetune_runs/inkspire_p512/_eval_pangram
  cd $E && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$E .venv/bin/python $I/experiments/gencmp_split.py gen eruku $E/finetune_runs/eruku_corpusmax/checkpoint_last.pth $O
  cd $I && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$I .venv/bin/python experiments/gencmp_split.py gen inkspire inkspire:finetune_runs/inkspire_p512/lora_step_013000 $O
  cd $I && PYTHONPATH=$I .venv/bin/python experiments/gencmp_split.py assemble $O
"""
import ast
import sys
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image

HERE = Path(__file__).resolve().parent
FONTS = Path("/data/work/dgkang/Eruku_korean_finetuning/assets/fonts_korean_v2/test")   # = assets/fonts/test
STYLE = {"ko": "다람쥐 헌 쳇바퀴에 타고파", "en": "The quick brown fox jumps"}
N_ROWS = 6
# CATS 는 gen_compare.py 에서 읽는다(import 하면 그쪽 의존을 끌고 온다)
CATS = next(ast.literal_eval(n.value) for n in ast.parse((HERE / "gen_compare.py").read_text()).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CATS")


def can_render(fp, text):
    cm = TTFont(str(fp)).getBestCmap()
    return all(ord(c) in cm for c in text if not c.isspace())


def plan():
    """[(cat, i, font_path, style_text, target)] — 폰트를 행마다 돌려 쓰고, 못 그리는 폰트는 건너뛴다."""
    fonts = sorted(FONTS.glob("*.ttf"))
    out, k = [], 0
    for cat, texts in CATS.items():
        style = STYLE["en" if cat.startswith("en") else "ko"]
        for i in range(N_ROWS):
            tgt = texts[i % len(texts)]
            for _ in range(len(fonts)):
                fp = fonts[k % len(fonts)]; k += 1
                if can_render(fp, tgt) and can_render(fp, style):
                    out.append((cat, i, fp, style, tgt)); break
    return out


def gen(kind, ckpt, out):
    import torch
    from infer.show import gen_from_style, load_model, render_in_font, style_tensor
    model = load_model(ckpt, torch.device("cuda"))[0]
    d = Path(out) / kind; d.mkdir(parents=True, exist_ok=True)
    for cat, i, fp, style, tgt in plan():
        img = gen_from_style(model, style_tensor(render_in_font(fp, style)), style, tgt, 1.25, 320,
                             torch.device("cuda"), seed=0)      # cfg/max_new 는 Eruku 기본값, InkSpire 는 무시
        Image.fromarray(img).save(d / f"{cat}_{i}.png")
        print(kind, cat, fp.stem, tgt, flush=True)


def assemble(out):
    sys.path.insert(0, str(HERE))
    from common import fit_row, header_row
    from infer.show import cell, label_img, render_in_font
    out, CW, CH = Path(out), 380, 72
    labels = ["style ref (held-out, pangram)", "정답 (이 폰트)", "Eruku corpusmax s45000", "InkSpire s13000 std layout"]
    full_w = len(labels) * (CW + 4)
    rows = {}
    for cat, i, fp, style, tgt in plan():
        load = lambda k: np.array(Image.open(out / k / f"{cat}_{i}.png").convert("L"))
        cells = [cell(render_in_font(fp, style), f"ref: {fp.stem}", CW, CH),
                 cell(render_in_font(fp, tgt), f"목표: {tgt[:24]}", CW, CH, highlight=True),
                 cell(load("eruku"), "Eruku", CW, CH), cell(load("inkspire"), "InkSpire", CW, CH)]
        rows.setdefault(cat, []).extend([fit_row(cells, full_w), np.full((6, full_w), 80, np.uint8)])
    for cat, body in rows.items():
        hdr = header_row(labels, CW)
        body = np.vstack([hdr, np.full((3, hdr.shape[1]), 80, np.uint8), *body])
        title = label_img(f"[{cat}] Eruku vs InkSpire — held-out 폰트, pangram style ref, seed 0",
                          body.shape[1], 44, 18, bold=True, center=False, bg=255)
        p = out / f"gencmp_{cat}.png"
        Image.fromarray(np.vstack([title, np.full((2, body.shape[1]), 0, np.uint8), body])).save(p)
        print("saved", p)


if __name__ == "__main__":
    {"gen": lambda: gen(*sys.argv[2:5]), "assemble": lambda: assemble(sys.argv[2])}[sys.argv[1]]()
