"""docs/EXPERIMENTS.md §8 그림 14~16 — ref 보강(ctx) 실험. 추론 없음: cer_dump.py --ctx 덤프(finetune_runs/_cer/klein_s013000_ctx_*)를 읽는다.

  14_ctx_canvas.png    모델 입력 캔버스 [X | Xc | 마스크] — 기준선 / adapt-boot / fixed-oracle
  15_ctx_misalign.png  보강 줄 X(파랑)·Xc(주황) 겹쳐보기 — 표준폰트 근사 상자 vs 진짜 글자 상자
  16_ctx_diff.png      조건 간 차이 큰 8표본 몽타주(experiments/montage.py --show-ctx)
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from custom_datasets.korean.page import _font, font_px, render_line
from infer.inkspire import InkSpireGen
from infer.show import label_img, render_in_font, style_tensor
from cer_dump import CTX_FIXED, ctx_text

C = ROOT / "finetune_runs/_cer"
OUT = ROOT / "docs/img_inkspire_io"
plan = json.loads((C / "s13000_test_sref/plan.json").read_text(encoding="utf-8"))
S = plan["style_ref_text"]
u8 = lambda t: ((t[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
g = InkSpireGen(None, device="cpu")
row = {r["i"]: r for r in plan["rows"]}


def canvas(c):
    sep = np.zeros((c["x"].shape[0], 4), np.uint8)
    return np.hstack([c["x"], sep, c["xc"], sep, (1 - c["mask"]) * 255])


def labeled(img, text, w=None):
    w = w or img.shape[1]
    img = np.pad(img, ((0, 0), (0, w - img.shape[1])), constant_values=255)
    return np.vstack([label_img(text, w, 28, 16, center=False, bg=230), img, np.full((10, w), 255, np.uint8)])


def stack(parts):
    W = max(p.shape[1] for p in parts)
    return np.vstack([np.pad(p, ((0, 0), (0, W - p.shape[1])), constant_values=255) for p in parts])


def oracle_ctx(fp, ct):
    ci, _, bx = render_line(_font(str(fp), font_px(fp, 64)), ct, 64 * len(ct) + 64, 0, 8)
    return ci, bx[:, [0, 2]]


# 14 — 같은 표본의 세 입력 캔버스
r = row[5]; fp = Path(r["font"]); name = "00005.png"
ref = u8(style_tensor(render_in_font(fp, S)))
boot = np.array(Image.open(C / "klein_s013000_ctx_adapt_boot/ctx" / name).convert("L"))
ci, xb = oracle_ctx(fp, CTX_FIXED)
cs = [("기준선 — ref 한 줄 + 목표(마스크=검정)", g.prepare(ref, S, [r["text"]])),
      (f"adapt-boot — ref 아래에 모델이 먼저 생성한 '{ctx_text('adapt', S, r['text'])}'", g.prepare(ref, S, [r["text"]], ctx=[(boot, ctx_text("adapt", S, r["text"]))])),
      ("fixed-oracle — ref 아래에 그 폰트로 그린 고정 문장(목표 폭에서 잘림) + 진짜 글자 상자", g.prepare(ref, S, [r["text"]], ctx=[(ci, CTX_FIXED, xb)]))]
ims = [canvas(c) for _, c in cs]; W = max(i.shape[1] for i in ims)
Image.fromarray(stack([labeled(i, t, W) for (t, _), i in zip(cs, ims)])).save(OUT / "14_ctx_canvas.png")


# 15 — 상자 어긋남: 같은 진짜 필체 줄에 표준폰트 근사 상자 vs 진짜 상자
def overlay(c):
    x, xc = c["x"].astype(np.float32) / 255, c["xc"].astype(np.float32) / 255
    rgb = np.stack([np.minimum(x, 1), np.minimum(x, 1), np.ones_like(x)], -1) * 255   # X 잉크 = 파랑
    m = xc < 0.5                                                                   # Xc 잉크 = 주황
    rgb[m] = rgb[m] * 0.35 + np.array([255, 140, 0]) * 0.65
    return rgb.astype(np.uint8)


r = row[4]; fp = Path(r["font"])
ref = u8(style_tensor(render_in_font(fp, S)))
ci, xb = oracle_ctx(fp, CTX_FIXED)
parts = []
for t, c in [("표준폰트 근사 상자 — 숫자·기호에서 X(파랑)와 Xc(주황)가 어긋난다 (반 글자 이상 어긋남 10.4%)", g.prepare(ref, S, [r["text"]], ctx=[(ci, CTX_FIXED)])),
             ("진짜 글자 상자 — 글자마다 겹친다", g.prepare(ref, S, [r["text"]], ctx=[(ci, CTX_FIXED, xb)]))]:
    o = overlay(c)[: 2 * 77 + 8]   # ref + 보강 줄만
    lab = label_img(t, o.shape[1], 28, 16, center=False, bg=230)
    parts.append(np.vstack([np.repeat(lab[..., None], 3, -1), o, np.full((10, o.shape[1], 3), 255, np.uint8)]))
Image.fromarray(np.vstack(parts)).save(OUT / "15_ctx_misalign.png")

# 16 — 몽타주
runs = [f"{lab}={C}/{d}" for lab, d in [("base", "klein_s013000_test_sref"), ("adapt-oracle", "klein_s013000_ctx_adapt_oracle"),
        ("adapt-boot", "klein_s013000_ctx_adapt_boot"), ("fixed-oracle", "klein_s013000_ctx_fixed_oracle"),
        ("fixed-boot", "klein_s013000_ctx_fixed_boot")]]
subprocess.run([sys.executable, str(ROOT / "experiments/montage.py"), "--runs", *runs, "--pick", "diff", "--n", "8",
                "--cell-w", "400", "--cell-h", "120", "--show-ctx", "--out", str(OUT / "16_ctx_diff.png")], check=True)
print("saved 14_ctx_canvas.png 15_ctx_misalign.png 16_ctx_diff.png →", OUT)
