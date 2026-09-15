"""InkSpire 한글 재현용 페이지 렌더러 (모듈 1 — 데이터).

폰트(=writer) 1종으로 여러 줄짜리 **페이지**를 합성하고, 글자마다 bbox 를 렌더 시점에 공짜로
얻는다. 산출물은 두 모델이 나눠 쓴다:
  - 이미지 모델(FLUX-Fill LoRA, models/inkspire.py): P×P 랜덤 패치 {x, xc, mask}
  - 레이아웃 모델(masked CFM, models/inkspire_layout.py): 페이지 전체 토큰 {char_ids, layout, line_id, ref_mask}

좌표 규약
  - bbox = [x0, y0, x1, y1] 페이지 픽셀 int32. 공백도 토큰(w=advance, h=0, y0=y1=baseline).
  - 레이아웃(Fig 2) = [w, h, Δx, Δy] / page_w. Δx = x0 − 이전 토큰 x1(첫 토큰은 0 기준),
    Δy = y0 − 이전 줄 토큰들의 평균 y1(첫 줄은 0 기준). bbox 만으로 역변환 가능(layout_to_bboxes).
  - 텐서: x/xc ∈ [-1,1] float32 [1,P,P], mask float32 [1,P,P] (1 = 생성 영역).

증강은 fontset.py 의 grayscale 함수들을 재사용한다(줄 단위 회전 → 페이지 단위 elastic/morph/
blur/배경/jitter). `fontset.augment()` 는 64px resize 가 박혀 있어 못 쓴다.

self-check:
  .venv/bin/python custom_datasets/korean/page.py --out /tmp/kpage --n 4
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[2]   # 저장소 루트
ASSETS = HERE / "assets"
sys.path.insert(0, str(HERE))
from custom_datasets.korean import fontset as G          # noqa: E402
from custom_datasets.korean import split                 # noqa: E402

STD_FONT = ASSETS / "fonts/label/NanumGothic-Regular.ttf"      # Xc(콘텐츠 이미지) 표준폰트
DEFAULT_FONTS_DIR = ASSETS / "fonts/train"                    # 스타일 폰트 풀 12,951종
#: 표준폰트와 같은 family 는 style 풀에서 뺀다 — Xc 와 X 가 같은 글꼴이면 스타일 학습이 안 됨.
EXCLUDE_FONTS = list(split.DEFAULT_EXCLUDE_FONTS) + ["NanumGothic"]
STRIP_PAD = 32          # 줄 strip 상하 여유 px. ±3° 회전 시 1024px 폭 끝이 ~27px 움직인다.
GRID = 16               # 마스크 격자(FLUX latent 8 × patchify 2)


# ───────────────────────────── 폰트 / 텍스트 ─────────────────────────────
@lru_cache(maxsize=256)
def _font(path: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, px)


def font_px(font_path, line_h: float) -> int:
    """'한글' 잉크높이가 line_h px 가 되는 폰트 크기.

    fonts_sizes.json 은 라틴 기준이라 한글 잉크높이가 폰트마다 25~33px 로 흔들린다 → 직접 프로브.
    0.75: 한글 두 글자 bbox 는 ascender~descender 를 거의 다 쓰므로 line_h 보다 살짝 작게 잡아
    줄 간격(pitch ≥ 1.05·line_h)에서 위아래 줄이 겹치지 않게 한다."""
    b = _font(str(font_path), 100).getbbox("한글")
    h100 = max(1, b[3] - b[1])
    return max(8, int(round(100 * 0.75 * line_h / h100)))


def _adv(font, s: str) -> float:
    """글자별 advance 합(커닝 없음) — render_line 의 전진 규칙과 동일해야 wrap 이 안 넘친다."""
    return sum(font.getlength(c) for c in s)


def wrap_lines(words: list[str], font, max_w: float, n_lines: int) -> list[str]:
    """어절 greedy wrap. max_w 를 혼자 넘는 어절은 버린다. n_lines 를 채우면 멈춘다."""
    lines, cur = [], []
    sp = font.getlength(" ")
    for w in words:
        if len(lines) >= n_lines:
            break
        ww = _adv(font, w)
        if ww > max_w:
            continue
        if cur and _adv(font, " ".join(cur)) + sp + ww > max_w:
            lines.append(" ".join(cur)); cur = []
        cur.append(w)
    if cur and len(lines) < n_lines:
        lines.append(" ".join(cur))
    return lines


# ───────────────────────────── 렌더 ─────────────────────────────
def render_line(font, text: str, page_w: int, line_top: int, left: int = 0):
    """한 줄 렌더. 반환 (ink u8[asc+desc+2·STRIP_PAD, page_w], chars, boxes int32[N,4]).

    strip 의 페이지 y 원점은 `line_top − STRIP_PAD` (호출자가 그 위치에 합성). boxes 는 이미
    페이지 좌표. baseline = line_top + ascent. 공백 = [x, baseline, x+adv, baseline] (h=0).
    w/h ≤ 0 글리프(온글잎 폰트의 0-size 구두점 등)는 토큰째 버린다."""
    asc, desc = font.getmetrics()
    H = asc + desc + 2 * STRIP_PAD
    img = Image.new("L", (page_w, H), 255)
    draw = ImageDraw.Draw(img)
    chars, boxes = [], []
    x = float(left)
    y_off = line_top - STRIP_PAD
    for ch in text:
        adv = font.getlength(ch)
        xi = int(round(x))
        if ch == " ":
            base = STRIP_PAD + asc
            box = [xi, base, xi + int(round(adv)), base]
        else:
            b = font.getbbox(ch)
            box = [xi + b[0], STRIP_PAD + b[1], xi + b[2], STRIP_PAD + b[3]]
            if box[2] <= box[0] or box[3] <= box[1]:
                x += adv
                continue
            draw.text((xi, STRIP_PAD), ch, font=font, fill=0)
        chars.append(ch)
        boxes.append([box[0], box[1] + y_off, box[2], box[3] + y_off])
        x += adv
    return np.array(img, dtype=np.uint8), chars, np.array(boxes, dtype=np.int32).reshape(-1, 4)


@lru_cache(maxsize=8192)
def _std_glyph(std_font_path: str, ch: str):
    """표준폰트 글리프 잉크(u8, 잉크 bbox 로 크롭) — resize 해서 bbox 에 채운다."""
    f = _font(std_font_path, 64)
    b = f.getbbox(ch)
    w, h = b[2] - b[0], b[3] - b[1]
    if w <= 0 or h <= 0:
        return None
    im = Image.new("L", (w, h), 255)
    ImageDraw.Draw(im).text((-b[0], -b[1]), ch, font=f, fill=0)
    return np.array(im, dtype=np.uint8)


def render_content(chars, bboxes, std_font_path=STD_FONT, H: int = 0, W: int = 0) -> np.ndarray:
    """Xc: 표준폰트 글리프를 각 bbox 크기로 resize 해 np.minimum 합성. 공백 skip. 추론에서도 사용."""
    page = np.full((H, W), 255, np.uint8)
    for ch, (x0, y0, x1, y1) in zip(chars, np.asarray(bboxes, dtype=np.int64)):
        w, h = x1 - x0, y1 - y0
        if ch == " " or w <= 0 or h <= 0:
            continue
        g = _std_glyph(str(std_font_path), ch)
        if g is None:
            continue
        g = cv2.resize(g, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        # 페이지 밖 클리핑
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        g = g[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        page[cy0:cy1, cx0:cx1] = np.minimum(page[cy0:cy1, cx0:cx1], g)
    return page


def _rotate_strip(strip, boxes, y_off, deg):
    """strip 회전 + bbox 모서리에 같은 변환 적용. ponytail: 회전된 bbox 는 외접사각형(약간 느슨)."""
    h, w = strip.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    out = cv2.warpAffine(strip, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    if len(boxes) == 0:
        return out, boxes
    b = boxes.astype(np.float32).copy()
    b[:, [1, 3]] -= y_off                                   # strip 좌표로
    pts = np.stack([b[:, [0, 1]], b[:, [2, 1]], b[:, [0, 3]], b[:, [2, 3]]], axis=1)   # [N,4,2]
    pts = cv2.transform(pts, M)                             # [N,4,2]
    nb = np.concatenate([np.floor(pts.min(1)), np.ceil(pts.max(1))], axis=1)
    nb[:, [1, 3]] += y_off
    return out, nb.astype(np.int32)


def render_page(font_path, lines: list[str], std_font_path=STD_FONT, page_w: int = 1024,
                line_h_px: float = 64, rng: random.Random = None, aug: bool = True,
                pitch: int = None) -> dict:
    """페이지 합성. 반환 {x u8[H,W], xc u8[H,W], chars, bboxes int32[N,4], line_id int32[N],
    page_w, pitch, line_top list[int]}.

    pitch 미지정 시 line_h·U(1.05,1.35). x 는 aug 면 줄 회전(p0.5, ±3°) → 페이지 elastic(p0.7)/
    morph(p0.15)/blur(p0.4)/배경합성/jitter(p0.5). xc 는 **회전 후** bbox 로 렌더한 깨끗한 콘텐츠.
    ponytail: elastic 의 ≤3px 이동은 bbox 에 반영하지 않는다(외접사각형이 이미 그만큼 느슨)."""
    rng = rng or random.Random(0)
    px = font_px(font_path, line_h_px)
    font = _font(str(font_path), px)
    pitch = pitch or int(round(line_h_px * rng.uniform(1.05, 1.35)))
    top = rng.randint(0, int(line_h_px // 2))
    left = rng.randint(8, 32)        # ≥8: 첫 글자의 왼쪽 overhang(b[0]<0) 이 x=0 에서 잘리지 않게
    H = top + len(lines) * pitch + STRIP_PAD
    page = np.full((H, page_w), 255, np.uint8)
    chars, boxes, line_id, line_top = [], [], [], []
    for i, text in enumerate(lines):
        lt = top + i * pitch
        line_top.append(lt)
        strip, ch, bx = render_line(font, text, page_w, lt, left)
        y_off = lt - STRIP_PAD
        if aug and rng.random() < 0.5:
            strip, bx = _rotate_strip(strip, bx, y_off, rng.uniform(-3, 3))
        # 페이지에 np.minimum 합성 (세로 클리핑)
        sy0, sy1 = max(0, y_off), min(H, y_off + strip.shape[0])
        if sy1 > sy0:
            page[sy0:sy1] = np.minimum(page[sy0:sy1], strip[sy0 - y_off:sy1 - y_off])
        chars += ch
        boxes.append(bx)
        line_id += [i] * len(ch)
    boxes = np.concatenate(boxes, 0) if boxes else np.zeros((0, 4), np.int32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, page_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    xc = render_content(chars, boxes, std_font_path, H, page_w)

    x = page
    if aug:
        if rng.random() < 0.7:
            x = G.aug_elastic(x, amp=2.5, sigma=8.0, rng=rng)
        if rng.random() < 0.15:
            x = G.aug_morph(x, rng)
        if rng.random() < 0.4:
            x = cv2.GaussianBlur(x, (3, 3), 0)
        # 종이는 흰색 고정. 예전엔 assets/backgrounds 에서 패치를 뽑았는데 그 3장이 전부 단색
        # (246/239/255, std 0.0)이라 종이 톤 3개를 고르는 것뿐이었고 jitter(밝기 ±12)가 이미 덮는다.
        x = G.composite(x, np.full((H, page_w), 255.0, np.float32),
                        rng.uniform(0, 55), rng.uniform(0.7, 1.0))   # 잉크 농도·alpha 증강은 유지
        if rng.random() < 0.5:
            x = G.jitter(x, rng)
    return {"x": x, "xc": xc, "chars": chars, "bboxes": boxes,
            "line_id": np.array(line_id, dtype=np.int32), "page_w": page_w,
            "pitch": pitch, "line_top": line_top}


# ───────────────────────────── 레이아웃 ─────────────────────────────
def _line_base(acc: dict, line: int) -> float:
    """Δy 기준 = 이전 줄 토큰들의 평균 y1. acc = {line: [sum(y1), n]}, 없으면 0(첫 줄).

    논문 Fig 2 는 "이전 줄 baseline" 이지만 baseline 은 bbox 에서 못 뽑는다(왕복 불가). 줄 대표값이
    필요한데, max y1 은 줄 단위 ±3° 회전 증강(_rotate_strip 은 페이지 폭 중앙축)에서 가장 많이 내려간
    끝단을 집어 각도에 그대로 끌려간다 — 페이지 안 줄 간 편차 실측 5.60px(증강 off 2.37px).
    평균은 중앙 근처로 수렴해 1.78px(0.73px) 로 3.1배 안정적이고, 추론 때 예측 상자 하나가 튀어도
    기준점이 1/n 만 움직인다. 읽기 순서라 줄 L 을 쓸 시점에 줄 L−1 은 이미 완성돼 있어 왕복은 유지된다."""
    s, n = acc.get(line, (0.0, 0))
    return s / n if n else 0.0


def _line_add(acc: dict, line: int, y1: float) -> None:
    s, n = acc.get(line, (0.0, 0))
    acc[line] = (s + float(y1), n + 1)


def layout_seq(bboxes, line_id, page_w: int) -> np.ndarray:
    """bbox → [w, h, Δx, Δy]/page_w  float32[N,4] (Fig 2). ★ 토큰 순서 = 읽기 순서(줄 순, 줄 안 x 순)."""
    b = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    lid = np.asarray(line_id)
    out = np.zeros_like(b)
    x1_prev, line_y1 = 0.0, {}
    for i, (x0, y0, x1, y1) in enumerate(b):
        base = _line_base(line_y1, int(lid[i]) - 1)
        out[i] = [x1 - x0, y1 - y0, x0 - x1_prev, y0 - base]
        x1_prev = x1
        _line_add(line_y1, int(lid[i]), y1)
    return out / page_w


def layout_to_bboxes(seq, line_id, page_w: int) -> np.ndarray:
    """layout_seq 의 역변환 → float32[N,4] (페이지 px). 관측 토큰 + 예측 토큰 섞여도 순서대로 복원."""
    s = np.asarray(seq, dtype=np.float32).reshape(-1, 4) * page_w
    lid = np.asarray(line_id)
    out = np.zeros_like(s)
    x1_prev, line_y1 = 0.0, {}
    for i, (w, h, dx, dy) in enumerate(s):
        base = _line_base(line_y1, int(lid[i]) - 1)
        x0, y0 = x1_prev + dx, base + dy
        out[i] = [x0, y0, x0 + w, y0 + h]
        x1_prev = x0 + w
        _line_add(line_y1, int(lid[i]), y0 + h)
    return out


@lru_cache(maxsize=1)
def vocab() -> list[str]:
    """레이아웃 모델 vocab: ["<pad>","<unk>"] + alphabet.load_charset() (2,509자; 공백 = index 2)."""
    from custom_datasets.korean.alphabet import load_charset
    return ["<pad>", "<unk>"] + load_charset()


@lru_cache(maxsize=1)
def _vocab_index() -> dict:
    return {c: i for i, c in enumerate(vocab())}


def char_ids(chars) -> torch.Tensor:
    """글자 리스트 → LongTensor[N] (vocab 밖은 <unk>=1)."""
    idx = _vocab_index()
    return torch.tensor([idx.get(c, 1) for c in chars], dtype=torch.long)


# ───────────────────────────── 마스크 ─────────────────────────────
def rmask(P: int, rng: random.Random) -> np.ndarray:
    """R-Mask: 사각형 1~4개(변 U(P/4,3P/4)) 합집합, 16px 격자 스냅. 커버리지 [0.15,0.85] 아니면
    5회 재추첨, 그래도 실패하면 중앙 P/2 정사각형(0.25). u8[P,P], 1 = 생성."""
    for _ in range(5):
        m = np.zeros((P, P), np.uint8)
        for _ in range(rng.randint(1, 4)):
            w = int(rng.uniform(P / 4, 3 * P / 4)) // GRID * GRID
            h = int(rng.uniform(P / 4, 3 * P / 4)) // GRID * GRID
            x0 = rng.randint(0, (P - w) // GRID) * GRID
            y0 = rng.randint(0, (P - h) // GRID) * GRID
            m[y0:y0 + h, x0:x0 + w] = 1
        if 0.15 <= m.mean() <= 0.85:
            return m
    m = np.zeros((P, P), np.uint8)
    m[P // 4:3 * P // 4, P // 4:3 * P // 4] = 1
    return m


def line_mask(P: int, y_cut: int) -> np.ndarray:
    """행 ≥ y_cut 전부 1 (F-TopMask / 추론 모사). u8[P,P]."""
    m = np.zeros((P, P), np.uint8)
    m[y_cut:] = 1
    return m


# ───────────────────────────── Dataset ─────────────────────────────
def _to_tensor(u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(u8.astype(np.float32) / 127.5 - 1.0)[None]


class KoreanPageDataset(torch.utils.data.Dataset):
    """P×P 패치 {x, xc, mask, writer}. mode: rand(R-Mask+랜덤 크롭) / top(행 ≥ P/8 마스크+랜덤 크롭) /
    line(좌상단 크롭, 첫 줄 아래 전부 마스크 = one-shot 추론 모사).

    ★ 결정성: __getitem__ 마다 rng = Random(seed·1e6 + idx) 이고 gen 샘플러의 rng 도 이걸로 바꿔
      DataLoader 워커가 같은 샘플러 상태를 복제해도 샘플이 겹치지 않는다.
    ★ 페이지는 n_lines = ceil(P/pitch)+U(0,3), page_w = max(page_w, P) 라 항상 P×P 크롭이 가능하다."""

    def __init__(self, P: int = 512, page_w: int = 1024, mode: str = "rand", length: int = 100_000,
                 seed: int = 42, fonts_dir=None, sampler_cfg=None, paths=None, exclude_fonts=None,
                 aug: bool = True, std_font=STD_FONT):
        assert mode in ("rand", "top", "line"), mode
        self.P, self.page_w, self.mode, self.length = P, max(page_w, P), mode, length
        self.seed, self.aug, self.std_font = seed, aug, std_font
        fonts_dir = Path(fonts_dir or DEFAULT_FONTS_DIR)
        exc = EXCLUDE_FONTS if exclude_fonts is None else exclude_fonts
        split.ensure_font_charsets(fonts_dir)
        self.fonts = split.font_files(fonts_dir, exc)
        assert self.fonts, f"폰트 없음: {fonts_dir}"
        # 폰트별 charset(두부 방지 2차 필터). 125MB json 이지만 로드 0.6s — 1회만.
        self.charsets = json.load(open(fonts_dir / "fonts_charsets.json"))
        _, self.gen = split.build_samplers((1, 8), (1, 32), seed=seed, sampler_cfg=sampler_cfg,
                                           paths=paths, fonts_dir=fonts_dir, exclude_fonts=exc)

    def __len__(self):
        return self.length

    def _page(self, idx: int):
        """(page dict, rng, writer). 폰트 로드/렌더 실패는 log-and-resample (fontset.gen_split 방식)."""
        rng = random.Random(self.seed * 1_000_000 + idx)
        self.gen.rng = rng
        for _ in range(5):
            wi = rng.randrange(len(self.fonts))
            fp = self.fonts[wi]
            try:
                line_h = rng.uniform(40, 80)
                pitch = int(round(line_h * rng.uniform(1.05, 1.35)))
                n_lines = math.ceil(self.P / pitch) + rng.randint(0, 3)
                font = _font(str(fp), font_px(fp, line_h))
                cov = set(self.charsets.get(fp.name, ""))
                words, lines = [], []
                for _ in range(40):
                    ws = self.gen().split()
                    words += [w for w in ws if all(c in cov for c in w)] if cov else ws
                    lines = wrap_lines(words, font, self.page_w - 48, n_lines)   # 48 = left 최대 32 + 여유
                    if len(lines) >= n_lines:
                        break
                if len(lines) < n_lines:
                    raise RuntimeError(f"{n_lines}줄 못 채움 (커버 어절 부족)")
                pg = render_page(fp, lines, self.std_font, self.page_w, line_h, rng, self.aug,
                                 pitch=pitch)
                if len(pg["chars"]) == 0:
                    raise RuntimeError("빈 페이지")
                return pg, rng, wi
            except Exception as e:
                print(f"[page] {fp.name}: {e} → 재추첨")
        raise RuntimeError("페이지 렌더 5회 연속 실패")

    def __getitem__(self, idx: int) -> dict:
        pg, rng, wi = self._page(idx)
        P, H, W = self.P, pg["x"].shape[0], pg["x"].shape[1]
        if self.mode == "line":
            y0, x0 = 0, 0
            y_cut = pg["line_top"][1] if len(pg["line_top"]) > 1 else P // 8
            mask = line_mask(P, min(y_cut, P - GRID))
        else:
            y0, x0 = rng.randint(0, H - P), rng.randint(0, W - P)
            mask = rmask(P, rng) if self.mode == "rand" else line_mask(P, P // 8)
        crop = lambda a: a[y0:y0 + P, x0:x0 + P]
        return {"x": _to_tensor(crop(pg["x"])), "xc": _to_tensor(crop(pg["xc"])),
                "mask": torch.from_numpy(mask.astype(np.float32))[None], "writer": wi}


class KoreanLayoutDataset(KoreanPageDataset):
    """페이지 전체 토큰 {char_ids[N], layout[N,4], line_id[N], ref_mask[N]}. ref_mask = 첫 줄."""

    MAX_TOKENS = 1024   # ★ ≤ 모델 max_len(configs/inkspire_layout.yaml 2048). 초과분은 뒤 줄째로 버린다.
    #   2048 이면 batch 160 의 attention·FFN 활성화가 76GB 까지 가서(2026-09-10 실측) 1024 로 캡 — 초과 페이지는 pitch≈42px 극단뿐

    def __getitem__(self, idx: int) -> dict:
        pg, _, wi = self._page(idx)
        line_id, chars, boxes = pg["line_id"], pg["chars"], pg["bboxes"]
        if len(chars) > self.MAX_TOKENS:      # 작은 pitch(≈42px)×넓은 page 면 1,000+ 토큰이 나온다 — 줄 경계에서 자름
            n = int((line_id[:self.MAX_TOKENS] == line_id[self.MAX_TOKENS - 1]).argmax())
            line_id, chars, boxes = line_id[:n], chars[:n], boxes[:n]
        lid = torch.from_numpy(line_id).long()
        return {"char_ids": char_ids(chars),
                "layout": torch.from_numpy(layout_seq(boxes, line_id, pg["page_w"])),
                "line_id": lid, "ref_mask": lid == 0, "writer": wi}


def layout_collate(batch: list[dict]) -> dict:
    """가변 N → [B,Nmax] 패딩. pad_mask True = 패딩. char_ids 패딩 0(<pad>), layout 0, line_id 0."""
    B, N = len(batch), max(len(b["char_ids"]) for b in batch)
    out = {"char_ids": torch.zeros(B, N, dtype=torch.long), "layout": torch.zeros(B, N, 4),
           "line_id": torch.zeros(B, N, dtype=torch.long), "ref_mask": torch.zeros(B, N, dtype=torch.bool),
           "pad_mask": torch.ones(B, N, dtype=torch.bool), "writer": torch.tensor([b["writer"] for b in batch])}
    for i, b in enumerate(batch):
        n = len(b["char_ids"])
        for k in ("char_ids", "layout", "line_id", "ref_mask"):
            out[k][i, :n] = b[k]
        out["pad_mask"][i, :n] = False
    return out


# ───────────────────────────── self-check ─────────────────────────────
def _u8(t: torch.Tensor) -> np.ndarray:
    return ((t[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)


def _selfcheck(out: Path, n: int):
    out.mkdir(parents=True, exist_ok=True)
    P = 512
    t0 = time.time()
    ds = {m: KoreanPageDataset(P=P, mode=m, length=1000, seed=7) for m in ("rand", "top", "line")}
    print(f"init ×3: {time.time() - t0:.1f}s, fonts {len(ds['rand'].fonts)}")
    lay = KoreanLayoutDataset(P=P, length=1000, seed=7)
    times = []
    for i in range(n):
        for mode, d in ds.items():
            t = time.time(); s = d[i]; times.append(time.time() - t)
            for k in ("x", "xc", "mask"):
                assert tuple(s[k].shape) == (1, P, P), (mode, k, s[k].shape)
                assert s[k].dtype == torch.float32
            assert s["x"].min() >= -1 and s["x"].max() <= 1 and s["xc"].min() >= -1 and s["xc"].max() <= 1
            cov = s["mask"].mean().item()
            assert set(s["mask"].unique().tolist()) <= {0.0, 1.0}
            if mode == "rand":
                assert 0.15 <= cov <= 0.85, cov
            elif mode == "top":
                assert abs(cov - (1 - 1 / 8)) < 1e-6, cov
            else:
                assert 0.5 < cov < 1.0, cov
            assert s["xc"].min() < 0, "xc 에 잉크 없음"
            if mode == "rand" and i == 0 or mode == "line":
                x, xc, m = _u8(s["x"]), _u8(s["xc"]), s["mask"][0].numpy()
                ov = np.stack([x, x, x], -1); ov[m > 0, 0] = 255; ov[m > 0, 2] = (ov[m > 0, 2] * 0.5).astype(np.uint8)
                cv2.imwrite(str(out / f"{i}_{mode}_x.png"), x)
                cv2.imwrite(str(out / f"{i}_{mode}_xc.png"), xc)
                cv2.imwrite(str(out / f"{i}_{mode}_mask_overlay.png"), ov)
        # 페이지 전체(레이아웃 경로): bbox 왕복 + xc 잉크 존재 + boxes.png
        pg, _, _ = lay._page(i)
        seq = layout_seq(pg["bboxes"], pg["line_id"], pg["page_w"])
        back = layout_to_bboxes(seq, pg["line_id"], pg["page_w"])
        assert np.allclose(back, pg["bboxes"], atol=1e-3), np.abs(back - pg["bboxes"]).max()
        for ch, (x0, y0, x1, y1) in zip(pg["chars"], pg["bboxes"]):
            if ch != " ":
                assert (pg["xc"][y0:y1, x0:x1] < 128).any(), f"xc 잉크 없음: {ch!r} {(x0, y0, x1, y1)}"
        bx = cv2.cvtColor(pg["x"], cv2.COLOR_GRAY2BGR)
        for ch, (x0, y0, x1, y1) in zip(pg["chars"], pg["bboxes"]):
            cv2.rectangle(bx, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255) if ch != " " else (0, 160, 0), 1)
        cv2.imwrite(str(out / f"{i}_boxes.png"), bx)
        cv2.imwrite(str(out / f"{i}_page_xc.png"), pg["xc"])
        print(f"  [{i}] lines {len(pg['line_top'])} tokens {len(pg['chars'])} page {pg['x'].shape} "
              f"pitch {pg['pitch']} | {pg['chars'][:12]!r}")
    # collate
    items = [lay[i] for i in range(min(n, 3))]
    b = layout_collate(items)
    Nmax = max(len(it["char_ids"]) for it in items)
    assert b["char_ids"].shape == (len(items), Nmax) and b["layout"].shape == (len(items), Nmax, 4)
    for i, it in enumerate(items):
        n_ = len(it["char_ids"])
        assert (~b["pad_mask"][i]).sum() == n_ and b["pad_mask"][i, n_:].all()
        assert torch.equal(b["char_ids"][i, :n_], it["char_ids"]) and (it["char_ids"] > 0).all()
        assert torch.equal(b["ref_mask"][i, :n_], it["line_id"] == 0) and b["ref_mask"][i].any()
    print(f"vocab {len(vocab())} | collate {tuple(b['char_ids'].shape)} ok")
    print(f"__getitem__ {np.mean(times) * 1000:.0f} ms avg / {np.max(times) * 1000:.0f} ms max ({len(times)} calls)")
    print(f"self-check OK → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/kpage")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    _selfcheck(Path(a.out), a.n)
