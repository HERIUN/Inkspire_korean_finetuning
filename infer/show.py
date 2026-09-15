"""몽타주 그리기 헬퍼 (Eruku_korean_finetuning 에서 잘라옴).

원본은 Eruku/InkSpire 양쪽 체크포인트를 로드해 비교 몽타주를 만드는 뷰어였다. 여기서는
그리기 헬퍼와 **InkSpire 백엔드만 남긴 `load_model`/`gen_from_style`** 을 둔다. Eruku 쪽
(Emuru 로드, decoder embeds 재사용 배치 생성)은 이 repo 에 모델이 없어 안내 후 종료한다 —
Eruku 와의 비교는 각 repo 에서 따로 돌려 숫자를 맞춘다(원본 docs/EXPERIMENTS.md §12 방식).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image, ImageDraw, ImageFont

DEFAULT_MAX_IMG_LEN = 8192   # Eruku 호환 인자(InkSpire 는 무시)

ERUKU_HOWTO = ("Eruku 체크포인트는 이 repo 에 없다 — models/eruku.py 는 "
               "Eruku_korean_finetuning 에 있다.\n"
               "여기서는 `--ckpt inkspire:<lora_dir>[,<layout_ckpt>]` 만 쓴다. "
               "Eruku 수치는 그 repo 에서 같은 프로토콜로 따로 재고 비교한다.")

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


def style_tensor(arr, h=64):
    """라인이미지 배열 → style 텐서 [3,h,W] [-1,1] (BILINEAR h-리사이즈)."""
    img = Image.fromarray(arr).convert("RGB")
    w, hh = img.size
    img = img.resize((max(1, int(w * (h / hh))), h), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
    return t * 2 - 1        # ToTensor + Normalize(0.5, 0.5) — torchvision 한 줄 쓰자고 2GB 안 받는다


def load_model(ckpt, device, vae_checkpoint=None):
    """`inkspire:<lora_dir>[,<layout_ckpt>][,key=value…]` → (InkSpireGen, "inkspire").

    key=value 로 steps · guidance · std_font · trim_ref · degrade 를 스윕할 수 있다.
    """
    if not str(ckpt).startswith("inkspire:"):
        raise SystemExit(f"[error] {ckpt}\n{ERUKU_HOWTO}")
    from infer.inkspire import STD_FONT, InkSpireGen
    parts = str(ckpt)[len("inkspire:"):].split(",")
    kw = {k: (float(v) if v.replace(".", "", 1).isdigit() else v)
          for k, v in (x.split("=") for x in parts if "=" in x)}
    pos = [x for x in parts if "=" not in x]
    return InkSpireGen(pos[0], pos[1] if len(pos) > 1 and pos[1] else None, device,
                       steps=int(kw.get("steps", 20)),
                       guidance=kw.get("guidance", 30.0),
                       std_font=kw.get("std_font", STD_FONT),
                       trim_ref=bool(kw.get("trim_ref", 1)),
                       degrade=kw.get("degrade", 0.0)), "inkspire"


def gen_from_style(model, style_img, style_text, gen_text, cfg, max_new, device,
                   max_img_len=DEFAULT_MAX_IMG_LEN, seed=None):
    """style 텐서([3,h,W] 또는 [1,3,h,W]) → 생성 라인이미지 [H,W] uint8.

    cfg/max_new/max_img_len 은 Eruku 호환용이고 InkSpire 는 무시한다."""
    if not hasattr(model, "gen"):
        raise SystemExit(f"[error] gen_from_style: InkSpire 생성기가 아니다\n{ERUKU_HOWTO}")
    t = style_img[0] if style_img.dim() == 4 else style_img
    arr = ((t[0].float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return model.gen(arr, style_text, gen_text, seed or 0)
