"""몽타주 그리기 헬퍼 (Eruku_korean_finetuning 에서 잘라옴).

원본은 Eruku/InkSpire 양쪽 체크포인트를 로드해 비교 몽타주를 만드는 뷰어였다. 여기서는
`infer/inkspire.py` 의 뷰어가 쓰는 그리기 헬퍼 4개만 남긴다 — `load_model` 디스패치와
`gen_from_style` 은 Eruku 모델이 있어야 의미가 있어 원본 repo 에 둔다.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트


GOTHIC = str(HERE / "assets" / "fonts_label" / "NanumGothic-Regular.ttf")


GOTHIC_B = str(HERE / "assets" / "fonts_label" / "NanumGothic-Bold.ttf")


FONTS_DIR = HERE / "assets" / "fonts_korean_v2" / "train"


def label_img(text, w, h, size, bold=False, center=True, bg=255):
    f = ImageFont.truetype(GOTHIC_B if bold else GOTHIC, size)
    im = Image.new("L", (w, h), bg); d = ImageDraw.Draw(im)
    b = d.textbbox((0, 0), text, font=f); tw, th = b[2] - b[0], b[3] - b[1]
    x = (w - tw) // 2 if center else 5
    d.text((x - b[0], (h - th) // 2 - b[1]), text, font=f, fill=0)
    return np.array(im)


def fit(arr, cw, ch, pad=4):
    """grayscale line 을 cell(cw×ch) 안에 높이맞춰 좌측정렬, 흰 패딩."""
    h, w = arr.shape
    th = ch - 2 * pad
    nw = max(1, int(w * th / h))
    if nw > cw - 2 * pad:
        nw = cw - 2 * pad; th = max(1, int(h * nw / w))
    r = cv2.resize(arr, (nw, th), interpolation=cv2.INTER_AREA)
    canvas = np.full((ch, cw), 255, np.uint8)
    y0 = (ch - th) // 2
    canvas[y0:y0 + th, pad:pad + nw] = r
    return canvas


def cell(content, label, cw, ch, lh=26, highlight=False):
    lab = label_img(label, cw, lh, 15, bold=highlight, bg=210 if highlight else 245)
    block = np.vstack([lab, np.full((1, cw), 0, np.uint8), fit(content, cw, ch)])
    return np.pad(block, ((2, 2), (2, 2)), constant_values=0)


def render_in_font(font_path, text, size=72, pad=14):
    f = ImageFont.truetype(str(font_path), size)
    b = f.getbbox(text); w = max(1, b[2] - b[0]); h = max(1, b[3] - b[1])
    # getbbox 가 일부 폰트에서 상/하 여백을 과장 → montage 셀에서 글자가 쭈그러듦.
    # 실제 '잉크 bbox'(그려진 픽셀 min/max)로 타이트하게 crop. 잉크를 전부 포함하므로
    # 글자 부위(받침·디센더)는 절대 안 자름 — 통통 튀는 손글씨 폰트(WagleWagle 등)도 안전.
    # (UlsanJunggu 처럼 '델' 글리프가 본체서 수백px 분리된 malformed 폰트는 이 crop 으로도
    #  거대해지지만, 그 폰트는 eval montage 에서 기본 제외(--exclude-writers)로 처리.)
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    ImageDraw.Draw(im).text((pad - b[0], pad - b[1]), text, font=f, fill=0)
    arr = np.array(im)
    ys, xs = np.where(arr < 250)                       # 잉크(anti-alias 포함) 픽셀
    if len(ys):
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        arr = arr[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    return arr
