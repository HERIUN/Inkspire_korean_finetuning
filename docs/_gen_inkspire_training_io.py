"""학습에 실제로 들어가는 x·xc·bbox 를 한 장으로 — docs/img_inkspire_io/07_training_io.png

재생성: PYTHONPATH=. .venv/bin/python docs/_gen_inkspire_training_io.py
"""
from pathlib import Path
import sys, cv2, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from custom_datasets.korean.page import KoreanLayoutDataset, KoreanPageDataset, layout_seq   # noqa: E402
from infer.show import label_img                                                             # noqa: E402

OUT = HERE / "img_inkspire_io/07_training_io.png"
SEED, P = 7, 512


def bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def lab(t, w, h=26):
    return bgr(label_img(t, w, h, 15, center=False, bg=240))


def block(t, img, pad=10):
    b = img if img.ndim == 3 else bgr(img)
    return np.vstack([lab(t, b.shape[1]), b, np.full((pad, b.shape[1], 3), 255, np.uint8)])


def stack(bs):
    w = max(b.shape[1] for b in bs)
    return np.vstack([np.pad(b, ((0, 0), (0, w - b.shape[1]), (0, 0)), constant_values=255) for b in bs])


def draw(img, chars, boxes, thin=1):
    o = bgr(img)
    for ch, (x0, y0, x1, y1) in zip(chars, boxes.astype(int)):
        cv2.rectangle(o, (x0, y0), (x1, y1), (0, 0, 255) if ch != " " else (0, 160, 0), thin)
    return o


lay = KoreanLayoutDataset(P=P, length=100, seed=SEED)          # 레이아웃 학습 데이터 (aug 그대로)
pg, _, _ = lay._page(0)
chars, boxes, lid = pg["chars"], pg["bboxes"], pg["line_id"]
seq = layout_seq(boxes, lid, pg["page_w"])

# 확대: 첫 줄 왼쪽 320px — x 와 xc 에 같은 상자를 그려 대응을 본다
y0, y1 = max(0, pg["line_top"][0] - 8), pg["line_top"][0] + pg["pitch"]
zx, zc = pg["x"][y0:y1, :320], pg["xc"][y0:y1, :320]
sel = [i for i, b in enumerate(boxes) if b[0] < 320 and lid[i] == 0]
zb = boxes[sel].astype(float).copy(); zb[:, [1, 3]] -= y0
zoom = np.hstack([draw(zx, [chars[i] for i in sel], zb), np.full((zx.shape[0], 6, 3), 128, np.uint8),
                  draw(zc, [chars[i] for i in sel], zb)])
zoom = cv2.resize(zoom, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

# 실제 이미지 모델 입력 패치
s = KoreanPageDataset(P=P, mode="rand", length=100, seed=SEED)[0]
u8 = lambda t: ((t[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
px, pxc, m = u8(s["x"]), u8(s["xc"]), s["mask"][0].numpy()
ov = bgr(px); ov[m > 0, 0] = 255; ov[m > 0, 2] = (ov[m > 0, 2] * 0.5).astype(np.uint8)
sep = np.full((P, 6, 3), 128, np.uint8)

tok = " / ".join(f"{chars[i]!r}=[{', '.join(f'{v:.3f}' for v in seq[i])}]" for i in range(4))
cv2.imwrite(str(OUT), stack([
    block(f"① 페이지 x (스타일 폰트 + 증강) + 레이아웃 학습 라벨 bbox  — 빨강=글자, 초록=공백(h 0)   "
          f"{len(chars)}토큰 / {len(pg['line_top'])}줄 / pitch {pg['pitch']}px", draw(pg["x"], chars, boxes)),
    block("② 페이지 xc (같은 bbox 에 표준폰트 글리프를 resize 합성, 증강 없음)", draw(pg["xc"], chars, boxes)),
    block("③ 확대 2× — 왼쪽 x, 오른쪽 xc, 같은 상자. elastic·morph 는 x 에만 걸려 잉크가 상자 안에서 살짝 움직인다", zoom),
    block(f"④ 이미지 모델이 실제로 받는 P×P 패치:  x | xc | mask(파랑=생성, 커버리지 {m.mean():.2f})",
          np.hstack([px if False else bgr(px), sep, bgr(pxc), sep, ov])),
    block(f"⑤ 레이아웃 학습 타깃 [w, h, Δx, Δy] (page_w={pg['page_w']} 로 나눈 값) 앞 4토큰:  {tok}",
          np.full((4, 1200), 255, np.uint8)),
]))
print("saved", OUT)
