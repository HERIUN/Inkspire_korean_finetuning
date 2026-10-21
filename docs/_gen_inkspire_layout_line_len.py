"""docs/inkspire.md §9.4 그림 10 — 같은 레이아웃 모델(s8000)·같은 짧은 목표를 단독 한 줄 vs 긴 두 줄 아래 3번째 줄로."""
import gc, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from infer.inkspire import InkSpireGen
from infer.show import render_in_font, cell, FONTS_DIR
from models.inkspire_layout import load_layout

R = ROOT / "finetune_runs"
ROWS = [("ChosunSm", "roggle", "한국어 평가"), ("IncheonEducationSotong", "(국가공단의)", "인공지능 모델"),
        ("Yeongdo", "lavalier", "새벽 공기"), ("OKGUNG", "2000년;", "형형색색 꽃"),
        ("KCCKimhoon", "153:", "가을 하늘"), ("HakgyoansimDoldam", "심을", "겨울 바다")]
PANGRAM = "다람쥐 헌 쳇바퀴에 타고파"
lines = json.load(open(ROOT / "data/ref_set_clean/train_lines.json"))
refs = []
for w, p, tgt in ROWS:
    r = next(r for r in lines if r["person_key"].split("#")[0] == w and r["text"].startswith(p))
    refs.append((w, np.array(Image.open(r["image_path"]).convert("L")), r["text"], tgt))

cols = {}  # 열 이름 → [이미지 6개]
def run(g, name, pangram=False):
    cols[name] = [g.gen(render_in_font(FONTS_DIR / f"{w}.ttf", PANGRAM) if pangram else arr,
                        PANGRAM if pangram else txt, tgt, 0) for w, arr, txt, tgt in refs]
    print(name, flush=True)

g = InkSpireGen(str(R / "inkspire_p512/lora_step_013000"), None, "cuda")
g.layout = load_layout(str(R / "inkspire_layout/checkpoint_step_008000.pth"), "cuda")[0]
L1, L2 = "다람쥐 헌 쳇바퀴에 타고파", "한국어 손글씨 생성 2024"
cols["layout: 짧은 줄 단독"] = [g.gen(arr, txt, tgt, 0) for w, arr, txt, tgt in refs]
cols["layout: 긴 줄 1번째"] = []; cols["layout: 짧은 줄 3번째"] = []
for w, arr, txt, tgt in refs:
    a1, a2, a3 = g.gen_lines(arr, txt, [L1, L2, tgt], 0)
    cols["layout: 긴 줄 1번째"].append(a1); cols["layout: 짧은 줄 3번째"].append(a3)
print("done", flush=True)
out = []
for i, (w, arr, txt, tgt) in enumerate(refs):
    row = [cell(arr, f"ref {w}", 300, 72), cell(render_in_font(FONTS_DIR / f"{w}.ttf", tgt), f"정답: {tgt}", 300, 72, highlight=True)]
    row += [cell(v[i], k, 300, 72) for k, v in cols.items()]
    out += [np.hstack(row), np.full((6, sum(c.shape[1] for c in row)), 80, np.uint8)]
p = ROOT / "docs/img_inkspire_io/10_layout_line_len.png"
Image.fromarray(np.vstack(out)).save(p); print("saved", p)
