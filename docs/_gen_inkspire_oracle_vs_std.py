"""docs/inkspire.md §9.4 그림 09 — held-out 16폰트, A(std layout 현재 추론) vs B(스타일 폰트 실제 bbox + 학습 page 빌더)."""
import math, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from custom_datasets.korean import page as P
from infer.inkspire import InkSpireGen, _ink_box, _vstack
from infer.show import render_in_font, cell
from models.inkspire import generate

LORA = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "finetune_runs/inkspire_p512/lora_step_013000")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs/img_inkspire_io/09_oracle_vs_std.png"
STYLE = "다람쥐 헌 쳇바퀴에 타고파"
TGTS = ["인공지능 모델", "가을 하늘", "형형색색 꽃들이 활짝 피어난"]
fonts = sorted((ROOT / "assets/fonts/test").glob("*.ttf"))

g = InkSpireGen(LORA, None, "cuda")
t = lambda a: torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0)[None, None].cuda()


def oracle(fp, tgt, seed=0):
    font = P._font(str(fp), P.font_px(fp, 64))
    W = 16 * math.ceil((max(P._adv(font, s) for s in (STYLE, tgt)) + 64) / 16)
    pg = P.render_page(fp, [STYLE, tgt], page_w=W, rng=random.Random(seed), aug=False, pitch=77)
    H = 16 * math.ceil(pg["x"].shape[0] / 16)
    pad = lambda a: np.pad(a, ((0, H - a.shape[0]), (0, 0)), constant_values=255)
    x, xc = pad(pg["x"]), pad(pg["xc"])
    sel = pg["line_id"] == 1
    y_cut = int(pg["bboxes"][sel, 1].min()) - 2
    m = np.zeros_like(x); m[y_cut:] = 1
    with torch.no_grad():
        o = generate(g.tr, g.vae, t(x), t(xc), torch.from_numpy(m.astype(np.float32))[None, None].cuda(),
                     g.cache, steps=g.steps, guidance=g.guidance, seed=seed)
    out = ((o[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    strip = out[max(0, y_cut - 2):int(pg["bboxes"][sel, 3].max()) + 6]
    ib = _ink_box(strip)
    return strip[:, max(0, ib[0] - 4):ib[2] + 4] if ib else strip


rows = []
for fp in fonts:
    ref = render_in_font(fp, STYLE)
    gt = _vstack([render_in_font(fp, s) for s in TGTS])
    a = _vstack(g.gen_lines(ref, STYLE, TGTS, 0))
    b = _vstack([oracle(fp, s) for s in TGTS])
    CH = 72 * len(TGTS)
    rows.append(np.hstack([cell(gt, f"정답: {fp.stem}", 460, CH, highlight=True),
                           cell(a, "A std layout (현재 추론)", 460, CH),
                           cell(b, "B 오라클 bbox + 학습 page", 460, CH)]))
    rows.append(np.full((6, rows[-1].shape[1]), 80, np.uint8))
    print(fp.stem, flush=True)
Image.fromarray(np.vstack(rows)).save(OUT)
print("saved", OUT)
