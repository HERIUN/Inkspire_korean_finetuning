"""docs/inkspire.md 용 추론 입출력 예시 — 실제 폰트 하나로 [레퍼런스 | 캔버스 | 생성 | 잘라낸 줄]."""
import sys, numpy as np, cv2
from pathlib import Path
ROOT = Path("/data/work/dgkang/Eruku_korean_finetuning"); sys.path.insert(0, str(ROOT))
from infer.inkspire import InkSpireGen
from infer.show import label_img, render_in_font

FONT = ROOT / "assets/fonts/test/Gaegu.ttf"
STYLE, TARGET = "다람쥐 헌 쳇바퀴에 타고파", "한국어 손글씨 생성 2024"
OUT = ROOT / "docs/img_inkspire_io/05_infer_example.png"

def bgr(g): return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
def lab(t, w, h=26): return bgr(label_img(t, w, h, 15, center=False, bg=240))
def block(t, g, pad=8):
    b = g if g.ndim == 3 else bgr(g)
    return np.vstack([lab(t, b.shape[1]), b, np.full((pad, b.shape[1], 3), 255, np.uint8)])
def stack(bs):
    w = max(b.shape[1] for b in bs)
    return np.vstack([np.pad(b, ((0, 0), (0, w - b.shape[1]), (0, 0)), constant_values=255) for b in bs])

g = InkSpireGen(str(ROOT / "finetune_runs/inkspire_p512/lora_step_013000"), None, "cuda")
style = render_in_font(str(FONT), STYLE)
c = g.prepare(style, STYLE, [TARGET], seed=0)
out = g.gen_lines(style, STYLE, [TARGET], seed=0)[0]

sep = np.full((c["x"].shape[0], 6), 128, np.uint8)
row_in = np.hstack([c["x"], sep, c["xc"], sep, (1 - c["mask"]) * 255])
gt = render_in_font(str(FONT), TARGET)
h = max(out.shape[0], gt.shape[0])
pad = lambda a: np.pad(a, ((0, h - a.shape[0]), (0, 0)), constant_values=255)

cv2.imwrite(str(OUT), stack([
    block(f"① 스타일 레퍼런스 (폰트 {FONT.stem}, 잉크 48px 로 정규화 후 캔버스 첫 줄에 배치)  '{STYLE}'", style),
    block("② 모델 입력 3장:  x (첫 줄만 보임)  |  xc (표준폰트 글리프를 목표 글자 상자에 채움)  |  mask (검정=생성 영역)", row_in),
    block(f"③ 생성 결과에서 잘라낸 목표 줄  '{TARGET}'", pad(out)),
    block(f"④ 같은 폰트로 직접 렌더한 정답 (비교용)", pad(gt)),
]))
print("saved", OUT, "| 캔버스", c["x"].shape, "| 출력", out.shape)
