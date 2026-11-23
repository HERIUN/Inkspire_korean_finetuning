"""InkSpire one-shot 생성기 + 뷰어 (모듈 4). 스타일 레퍼런스 한 줄 → 아래 줄들을 한 번에 생성.

캔버스 = [ref 한 줄 | 마스크된 여러 줄] (x), 표준폰트 콘텐츠 (xc), 첫 줄 아래 전부 1 (mask) →
models.inkspire.generate → 줄별 크롭. 줄 배치(bbox)는 레이아웃 모델(--layout-ckpt, sample steps=10)
또는 "std layout"(표준폰트 렌더 bbox 그대로 = Stage 2 oracle 평가 경로).

★ 기하는 학습 규약(custom_datasets/korean/page.py)에 맞춘다: LINE_H=64 는 font_px 의 line_h 로,
  '한글' 잉크높이 ≈ 0.75·64 = 48px. ref 는 잉크 트림 후 그 높이로 resize 해 표준폰트 잉크 행에 놓는다.
  PITCH=77 ≈ 1.2·LINE_H (학습 U(1.05,1.35)). 레이아웃 정규화는 학습 page_w=1024 기준(캔버스 폭과 무관).
★ 레퍼런스 레이아웃은 근사다 — 표준폰트로 style_text 를 렌더한 bbox 를 x 방향만 ref 폭에 맞춘다.
  style_text=="" 면 ref 토큰 없음(레이아웃 모델의 무레퍼런스 모드).

experiments/gen_compare.py (와 원본 repo 의 eval/htr_cer.py) 는 infer/show.py 의 디스패치(`--ckpt inkspire:<lora_dir>[,<layout_ckpt>]`)로
이 클래스를 쓴다(`gen(style_arr, style_text, target_text, seed)`).

사용 (configs/infer.yaml 의 inkspire: 섹션):
  ./inference.sh inkspire --lora-dir finetune_runs/inkspire_p512/lora_last [--layout-ckpt …/checkpoint_last.pth] \\
      --lines "첫 줄" "둘째 줄" --out _debug/inkspire_show.png
  ./inference.sh inkspire --dry-run --out _debug/inkspire_dry.png        # FLUX 없이 [x | xc | mask] 캔버스만
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import math
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from configs import loader as _cfgloader                                        # noqa: E402
from custom_datasets.korean import page                                          # noqa: E402
from custom_datasets.korean.page import (STD_FONT, char_ids, font_px, layout_seq,   # noqa: E402
                                         layout_to_bboxes, render_content, render_line)

LINE_H, PITCH, TOP, LEFT = 64, 77, 8, 16   # ★ 48/64/80 실측 CER 차이 없음(2026-09-14) — 해상도는 병목이 아니다
PAGE_W = 1024      # 레이아웃 [w,h,Δx,Δy] 정규화 기준 = 학습 page_w
MAX_W = 2048       # 캔버스 폭 상한. ★ 토큰 수는 학습(2048)보다 커질 수 있다 —
                   #   W=2048, H=176 이면 회전·연결 후 128×22 = 2816 토큰이다

def backbone(name_or_lora=None):
    """"flux1" | "klein" | LoRA 디렉토리(키 이름으로 판별) → models.inkspire 또는 models.inkspire_klein 모듈."""
    import importlib
    name = name_or_lora or "flux1"
    if name not in ("flux1", "klein"):
        from safetensors import safe_open
        with safe_open(str(Path(name) / "pytorch_lora_weights.safetensors"), "pt") as f:
            name = "klein" if any("mode_blocks" in k for k in f.keys()) else "flux1"
    return importlib.import_module("models.inkspire_klein" if name == "klein" else "models.inkspire")


def load_flux_or_exit(device="cuda", cache=None, M=None, **kw):
    """(transformer, vae, cache). 가중치/캐시가 없으면 traceback 대신 한글 안내로 종료(train/inkspire.py 공용).
    M = backbone() 모듈(기본 FLUX.1-Fill)."""
    M = M or backbone()
    cache = Path(cache or M.EMPTY_CACHE)
    if not cache.exists():
        raise SystemExit(f"[error] {cache} 없음.\n{M.HOWTO}")
    try:
        tr, vae = M.load_flux(kw.pop("repo", None) or M.REPO, device, **kw)
    except OSError as e:   # 미로그인/미수락/미다운로드
        raise SystemExit(f"[error] FLUX 로드 실패 ({str(e).splitlines()[0][:120]})\n{M.HOWTO}")
    return tr, vae, torch.load(cache)


def _word_gaps(arr, n, th=200):
    """잉크 열 투영에서 가장 넓은 빈 열 구간 n 개의 평균 폭(px) — 어절 간격 추정. 첫·끝 잉크 밖은 안 센다."""
    cols = (np.asarray(arr) < th).any(0)
    idx = np.where(cols)[0]
    if n <= 0 or len(idx) < 2:
        return None
    runs, cur = [], 0
    for c in cols[idx[0]:idx[-1] + 1]:
        if c:
            if cur: runs.append(cur)
            cur = 0
        else:
            cur += 1
    return float(np.mean(sorted(runs)[-n:])) if len(runs) >= n else None


def _fit_words(bx, ch, img, ink_h, th=200):
    """진단용 오라클: 목표 줄의 **실제 이미지**에서 어절 x 구간을 뽑아(가장 넓은 빈 열 구간 = 공백 수만큼) 표준폰트 상자를 어절 단위로 거기에 맞춘다.
    어절 안 글자 배분은 표준폰트 비율 그대로. 이미지는 잉크 높이 → ink_h 로 환산(ref 와 같은 규칙). 어절 수가 안 맞으면 None."""
    ib = _ink_box(img)
    if ib is None:
        return None
    im = img[ib[1]:ib[3], ib[0]:ib[2]]
    s, cols = ink_h / im.shape[0], (im < th).any(0)
    runs, st = [], None
    for x, c in enumerate(cols):   # 빈 열 구간 (시작, 길이) — 첫·끝 잉크 사이만
        if not c and st is None: st = x
        if c and st is not None: runs.append((st, x - st)); st = None
    n = sum(c == " " for c in ch)
    words = [[]]
    for i, c in enumerate(ch):
        if c == " ": words.append([])
        else: words[-1].append(i)
    if len(runs) < n or any(not w for w in words):
        return None
    cuts = sorted(sorted(runs, key=lambda r: -r[1])[:n])
    xs = [0] + [v for a, l in cuts for v in (a, a + l)] + [im.shape[1]]
    out = bx.copy()
    for k, w in enumerate(words):
        a0, a1, d0, d1 = bx[w[0], 0], bx[w[-1], 2], LEFT + xs[2 * k] * s, LEFT + xs[2 * k + 1] * s
        out[w, 0] = d0 + (bx[w, 0] - a0) * (d1 - d0) / max(1.0, a1 - a0)
        out[w, 2] = d0 + (bx[w, 2] - a0) * (d1 - d0) / max(1.0, a1 - a0)
    for i, c in enumerate(ch):   # 공백 상자 = 양옆 어절 사이
        if c == " ":
            out[i, 0], out[i, 2] = out[i - 1, 2], out[i + 1, 0]
    return out


def _ink_box(arr, th=250):
    ys, xs = np.where(arr < th)
    return None if len(ys) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


class InkSpireGen:
    def __init__(self, lora_dir=None, layout_ckpt=None, device="cuda", steps=20, guidance=30.0,
                 rape=True, ape=True, std_font=STD_FONT, trim_ref=True, degrade=0.0, tight_mask=False, ref_width=True, line_h=LINE_H, cfg=1.0, uncond="all", cfg_s=None, word_gap=False, cfg_t=(0.0, 1.0)):
        """lora_dir=None 이면 FLUX 를 안 올린다(prepare/--dry-run 전용). layout_ckpt=None → std layout.
        tight_mask: 마스크 = [ref 잉크 바닥 ↑16격자, 목표 상자 바닥+4 ↑16격자) — 캔버스도 거기서 끝난다.
          기본(False)은 y_cut = clip(목표 top−2, ref 바닥+1, TOP+PITCH), H = max(TOP+PITCH·(n+1), 상자 바닥+8).
        ref_width(기본 켬, 2026-10-22~): 목표 줄 상자의 x 를 ref 의 "손글씨 폭 / 표준폰트 폭" 비율로 늘이거나 줄인다(std layout 전용).
          끄면 목표 줄 폭·자간이 표준폰트 그대로라 출력이 GT 보다 ~23% 넓다(Eruku 7%) — HWD 의 큰 몫(EXPERIMENTS §9)."""
        self.device, self.steps, self.guidance, self.rape, self.ape = device, steps, guidance, rape, ape
        self.cfg = cfg   # CFG(FLUX.3-Image-Klein-Base). 1.0 = 끔. 학습 안 한 v_u 로도 듣는다(EXPERIMENTS §10 내용 CFG)
        self.uncond = uncond   # v_u 의 Xc: all = 전체 빈 이미지, gen = 생성 영역만 비움(ref 대응표 유지)
        self.cfg_t = cfg_t     # guidance 를 거는 step 비율 구간 [a, b) (models.inkspire_klein.generate)
        self.cfg_s = cfg_s     # 주면 분리 guidance — ref 대응표(필체) 쪽 배율(models.inkspire_klein.generate)
        # word_gap: ref 의 어절 간격을 목표 줄 공백에 옮긴다(ref_width 는 줄 전체 폭만 맞춰 손글씨의 넓은 띄어쓰기가 좁아진다)
        self.word_gap = word_gap
        self.trim_ref, self.degrade, self.tight_mask, self.ref_width = trim_ref, degrade, tight_mask, ref_width
        self.std_font_path = std_font          # ★ 학습(configs/inkspire.yaml data.std_font)과 같아야 한다
        # line_h: 표준폰트 '한글' 잉크 높이 기준(학습 U(40,80)). 줄 간격은 같은 비율(PITCH/LINE_H)로 늘린다
        self.std_font = page._font(str(std_font), font_px(std_font, line_h))
        self.pitch = int(round(PITCH * line_h / LINE_H))
        self.layout = None
        if layout_ckpt:
            from models.inkspire_layout import load_layout
            self.layout = load_layout(layout_ckpt, device)[0]
        self.tr = self.vae = self.cache = None
        if lora_dir:
            self.M = backbone(lora_dir)
            self.tr, self.vae, self.cache = load_flux_or_exit(device, M=self.M, lora_ckpt=lora_dir, grad_ckpt=False)
            self.tr.eval()

    def prepare(self, style_arr, style_text, lines, seed=0, ctx=(), tgt_imgs=None) -> dict:
        """캔버스 구성(FLUX 불필요). → {x, xc, mask u8[H,W], chars, boxes int32[N,4], line_id}.
        ctx: ref 아래에 보이는 줄로 더 깔 [(u8 줄 이미지, 전사[, 글자 x 상자 [N,2]]), …] — 목표 줄은 그만큼 아래로 밀린다.
          x 상자(이미지 좌표, 공백 포함 render_line 순서)를 주면 표준폰트 근사 대신 그 위치를 쓴다 — 진짜 필체 줄은
          글자 폭이 표준폰트와 달라 근사하면 X↔Xc 짝이 어긋난다(숫자·기호 많은 줄은 10% 가 반 글자 이상).
          보이는 줄은 전부 line_id 0. 레이아웃 모델의 Δy 규약과 안 맞아 std layout 전용.
        tgt_imgs: 진단용 오라클 — 줄마다 그 목표 줄의 실제 이미지(u8). 어절 위치·폭을 거기서 가져온다(_fit_words, EXPERIMENTS §20)."""
        assert not (ctx and self.layout is not None), "ctx 는 std layout 전용"
        b = self.std_font.getbbox("한글")
        ink_h, y_ref = b[3] - b[1], TOP + b[1]                       # 표준폰트 '한글' 잉크 행

        vis, chars, boxes, line_id, row = [], [], [], [], []
        sx = 1.0   # 목표 줄 x 배율 (ref_width)
        gap_add = 0.0   # word_gap: 목표 줄 공백마다 더할 px
        for r, (arr, text, *xb) in enumerate([(style_arr, style_text), *ctx]):   # 보이는 줄 r = ref(0), ctx(1..k)
            ib = _ink_box(arr)
            im = arr[ib[1]:ib[3], ib[0]:ib[2]] if ib else arr
            w = max(16, int(round(im.shape[1] * ink_h / im.shape[0])))
            vis.append(cv2.resize(im, (w, ink_h), interpolation=cv2.INTER_AREA))
            # 레퍼런스 레이아웃(근사): 표준폰트 bbox 를 x 방향으로 ref 폭에 맞춤.
            # ★ 이미지에서 글자를 직접 분할해 실제 bbox 를 주는 방식도 해봤으나 CER 0.20 → 0.61 로 폭망(2026-09-14).
            #   레이아웃 모델이 레퍼런스 토큰의 분산을 그대로 증폭한다 — 매끈한 근사값이 오히려 낫다.
            if text:
                strip, ch, bx = render_line(self.std_font, text, 4 * MAX_W, TOP + r * self.pitch, LEFT)
                if len(bx):
                    bx = bx.astype(np.float32)
                    x0, x1 = bx[:, 0].min(), bx[:, 2].max()
                    if xb:   # 이미지 좌표 → 잉크 크롭·리사이즈와 같은 변환
                        assert len(xb[0]) == len(ch), (len(xb[0]), len(ch))
                        x0 = ib[0] if ib else 0
                        bx[:, [0, 2]] = LEFT + (np.asarray(xb[0], np.float32) - x0) * w / im.shape[1]
                    else:
                        bx[:, [0, 2]] = LEFT + (bx[:, [0, 2]] - x0) * w / max(1.0, x1 - x0)
                        if r == 0 and self.ref_width:
                            sx = w / max(1.0, x1 - x0)
                        if r == 0 and self.word_gap:   # 같은 n 개 최대 빈칸을 ref 손글씨 / 표준폰트(같은 폭으로 맞춘 것) 양쪽에서
                            n = text.count(" ")
                            g_ref = _word_gaps(vis[0], n)
                            g_std = _word_gaps(strip[:, int(x0):int(x1)], n)
                            if g_ref is not None and g_std is not None:
                                gap_add = max(0.0, g_ref - g_std * w / max(1.0, x1 - x0))
                    chars += ch; boxes.append(bx); line_id += [0] * len(ch); row += [r] * len(ch)
        k = len(vis) - 1
        for i, text in enumerate(lines):   # 타깃 토큰: std layout 을 초기값/기본값으로
            _, ch, bx = render_line(self.std_font, text, 4 * MAX_W, TOP + (i + 1 + k) * self.pitch, LEFT)
            bx = bx.astype(np.float32)
            bx[:, [0, 2]] = LEFT + (bx[:, [0, 2]] - LEFT) * sx
            if gap_add:   # 공백 토큰마다 뒤 토큰들을 gap_add 만큼 민다
                sp = np.cumsum([c == " " for c in ch]).astype(np.float32)
                bx[:, [0, 2]] += (sp * gap_add)[:, None]
            if tgt_imgs is not None and tgt_imgs[i] is not None:
                fb = _fit_words(bx, ch, tgt_imgs[i], ink_h)
                self.oracle_miss = getattr(self, "oracle_miss", 0) + (fb is None)
                bx = bx if fb is None else fb
            chars += ch; boxes.append(bx); line_id += [i + 1] * len(ch); row += [-1] * len(ch)
        boxes = np.concatenate(boxes, 0) if boxes else np.zeros((0, 4), np.float32)
        line_id, row = np.array(line_id, dtype=np.int64), np.array(row, dtype=np.int64)
        if self.layout is not None and len(chars):
            from models.inkspire_layout import sample
            ref_mask = torch.from_numpy(line_id == 0)[None]
            seq = torch.from_numpy(layout_seq(boxes, line_id, PAGE_W))[None]
            out = sample(self.layout, char_ids(chars)[None], torch.from_numpy(line_id)[None], seq, ref_mask,
                         torch.zeros(1, len(chars), dtype=torch.bool), steps=10, seed=seed)
            boxes = layout_to_bboxes(out[0].cpu().numpy(), line_id, PAGE_W)
            # ★ 줄 시작 Δx(= −이전 줄 폭)는 모델이 가장 못 맞추는 값이고(학습 줄은 전부 page_w 를 채우지만
            #   추론 줄은 짧다) 학습 페이지의 모든 줄은 좌측 여백에서 시작한다 → 줄마다 첫 토큰을 LEFT 로 평행이동.
            #   줄 안의 크기·간격(w,h,Δx)은 예측값 그대로.
            for li in np.unique(line_id[line_id > 0]):
                sel = line_id == li
                boxes[sel, 0::2] += LEFT - boxes[sel, 0].min()
            if not style_text:   # 무레퍼런스: Δy 기준(첫 줄 바닥)이 없으므로 둘째 줄을 ref 아래로 평행이동
                boxes[:, [1, 3]] += (TOP + self.pitch + b[1]) - boxes[line_id == 1, 1].min()
        tgt = line_id > 0
        # ★ ref 가 목표 줄보다 길면 캔버스 오른쪽에 "생성해야 할 빈 공간"이 생긴다(학습 패치는 글자로 꽉 참).
        #   남는 ref 꼬리를 잘라 캔버스를 목표 줄 폭에 맞춘다(ref 토큰도 같이 버림).
        #   실측 2026-09-14 (held-out n=150): CER 중앙값 0.191 → 0.170, 폭주율 1.3% → 0.7%.
        if self.trim_ref and tgt.any():
            keep_w = int(boxes[tgt, 2].max() + 8 - LEFT)
            keep = tgt.copy()
            for r, im in enumerate(vis):   # ctx 줄도 같은 폭으로 자른다(캔버스 폭 = 목표 줄 폭 유지)
                if 16 <= keep_w < im.shape[1]:
                    vis[r] = im[:, :keep_w]
                keep |= (row == r) & (boxes[:, 2] <= LEFT + vis[r].shape[1])
            chars = [c for c, kk in zip(chars, keep) if kk]
            boxes, line_id, tgt = boxes[keep], line_id[keep], tgt[keep]
        Wr = vis[0].shape[1]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, MAX_W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, None)
        W = 16 * math.ceil(min(MAX_W, max(LEFT + max(v.shape[1] for v in vis), boxes[:, 2].max() if len(boxes) else 0) + 8) / 16)
        # ★ 캔버스를 학습 페이지 폭(1024)까지 흰색으로 넓혀봤자 CER 0.19 → 0.25 로 나빠진다(2026-09-14)
        if self.tight_mask and tgt.any():   # X y끝 ~ Xc y끝. 목표 아래 빈 종이(기본 방식 ~42px)를 생성시키지 않는다
            H = 16 * math.ceil((boxes[tgt, 3].max() + 4) / 16)
        else:
            H = 16 * math.ceil(max(TOP + self.pitch * (len(lines) + 1 + k), (boxes[:, 3].max() + 8) if len(boxes) else 0) / 16)
        boxes = np.round(boxes).astype(np.int32)

        xc = render_content(chars, boxes, self.std_font_path, H, W)
        x = np.full((H, W), 255, np.uint8)
        for r, v in enumerate(vis):
            y, w = y_ref + r * self.pitch, min(v.shape[1], W - LEFT)
            x[y:y + ink_h, LEFT:LEFT + w] = np.minimum(x[y:y + ink_h, LEFT:LEFT + w], v[:, :w])
        y_last = y_ref + k * self.pitch                                     # 마지막 보이는 줄의 잉크 top
        if self.degrade > 0:   # 학습 통계(블러·회색 종이)에 맞춰 캔버스를 일부러 더럽힌다 — 증강 불일치 검증용
            from custom_datasets.korean import fontset as G
            x = cv2.GaussianBlur(x, (3, 3), 0)
            x = G.composite(x, np.full(x.shape, 255.0 - 20 * self.degrade, np.float32), 30.0, 1.0)
        if self.tight_mask:   # ref 바로 아래부터, 토큰 격자(16px)에 맞춤 — 학습 R-Mask 와 같은 규칙
            # ponytail: 레이아웃 모델이 목표 줄을 y_cut 위로 끌어올리면 그 윗부분은 생성 못 한다(std layout 에선 안 생김)
            y_cut = 16 * math.ceil((y_last + ink_h + 1) / 16)
        else:
            y_cut = int(np.clip(boxes[tgt, 1].min() - 2 if tgt.any() else TOP + (k + 1) * self.pitch,
                                y_last + ink_h + 1, TOP + (k + 1) * self.pitch))
        mask = np.zeros((H, W), np.uint8); mask[y_cut:] = 1
        return {"x": x, "xc": xc, "mask": mask, "chars": chars, "boxes": boxes, "line_id": line_id,
                "ref_w": Wr, "y_cut": y_cut, "k": k}

    def _crop_line(self, out, c, i):
        """줄 i(line_id=i+1) 의 bbox 행 범위(+4px) 크롭 → 잉크 bbox 로 좌우 트림(+4px)."""
        sel = (c["line_id"] == i + 1) & np.array([ch != " " for ch in c["chars"]], dtype=bool)
        if sel.any():
            y0, y1 = int(c["boxes"][sel, 1].min()), int(c["boxes"][sel, 3].max())
        else:
            y0, y1 = TOP + (i + 1 + c["k"]) * self.pitch, TOP + (i + 2 + c["k"]) * self.pitch
        y0, y1 = max(0, y0 - 4), min(out.shape[0], y1 + 4)
        strip = out[y0:y1]
        ib = _ink_box(strip)
        if ib:
            strip = strip[:, max(0, ib[0] - 4):ib[2] + 4]
        return strip

    @torch.no_grad()
    def inpaint(self, x, xc, mask, seed=0, cfg=None, t0=0) -> np.ndarray:
        """캔버스 u8[H,W] 의 mask=1 영역을 채운 캔버스 u8. cfg 를 주면 이 호출만 그 내용 CFG 로(infer/fix.py 재인페인팅)."""
        cfg = self.cfg if cfg is None else cfg
        dev = self.device
        t = lambda a: torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0)[None, None].to(dev)
        m = torch.from_numpy(mask.astype(np.float32))[None, None].to(dev)
        with torch.no_grad():
            x_hat = self.M.generate(self.tr, self.vae, t(x), t(xc), m, self.cache, steps=self.steps,
                         guidance=self.guidance, seed=seed, rape=self.rape, ape=self.ape, **({"t0": t0} if t0 else {}),
                         **({"cfg": cfg, "uncond": self.uncond, "cfg_s": self.cfg_s, "cfg_t": self.cfg_t} if cfg != 1.0 or self.cfg_s is not None else {}))
        return ((x_hat[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

    def gen_lines(self, style_arr, style_text, lines, seed=0, ctx=(), tgt_imgs=None) -> list[np.ndarray]:
        """style_arr u8[H,W](ink~0) 한 줄 + 그 전사 → lines 각각의 생성 이미지 u8 리스트. ctx 는 prepare 참고."""
        assert self.tr is not None, "FLUX 미로드(lora_dir=None) — prepare/--dry-run 만 가능"
        c = self.prepare(style_arr, style_text, lines, seed, ctx, tgt_imgs)
        out = self.inpaint(c["x"], c["xc"], c["mask"], seed)
        return [self._crop_line(out, c, i) for i in range(len(lines))]

    def gen(self, style_arr, style_text, target_text, seed=0) -> np.ndarray:
        """한 줄 생성 (infer/show.py gen_from_style 디스패치용)."""
        return self.gen_lines(style_arr, style_text, [target_text], seed)[0]


def _vstack(arrs, gap=6):
    """폭이 다른 라인 이미지들을 세로로(흰 패딩)."""
    W = max(a.shape[1] for a in arrs)
    return np.vstack([np.pad(a, ((0, gap), (0, W - a.shape[1])), constant_values=255) for a in arrs])


def main():
    from infer.show import FONTS_DIR, cell, label_img, render_in_font
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", default=None, help="save_lora 디렉토리(lora_last). --dry-run 이면 불필요")
    ap.add_argument("--layout-ckpt", default=None, help="레이아웃 모델 ckpt. 없으면 std layout(표준폰트 bbox)")
    ap.add_argument("--std-font", default=str(STD_FONT), help="Xc 표준폰트(학습과 동일해야 함)")
    ap.add_argument("--lines-json", default=str(HERE / "data/ref_set_clean/train_lines.json"))
    ap.add_argument("--n-writers", type=int, default=4)
    ap.add_argument("--lines", nargs="+", default=["다람쥐 헌 쳇바퀴에 타고파", "한국어 손글씨 생성 2024"],
                    help="한 번에 생성할 줄들(3번째 셀에 세로 적층)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--cfg", type=float, default=1.0, help="내용 CFG. 권장 2.0(configs/infer.yaml), 1.0 = 끔")
    ap.add_argument("--uncond", default="all", choices=["all", "gen"], help="CFG v_u 에서 비울 Xc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-writers", nargs="*", default=["UlsanJunggu"])
    ap.add_argument("--cell-w", type=int, default=380)
    ap.add_argument("--cell-h", type=int, default=72, help="줄당 셀 높이(줄 수만큼 곱해짐)")
    ap.add_argument("--out", default=str(HERE / "_debug/inkspire_show.png"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry-run", action="store_true", help="FLUX 없이 첫 writer 의 [x | xc | mask] 캔버스만 --out 에 저장")
    ap.add_argument("--config", default=str(HERE / "configs/infer.yaml"))
    args = _cfgloader.parse_args(ap, default_config=str(HERE / "configs/infer.yaml"), scopes=("inkspire",))

    rows = json.loads(Path(args.lines_json).read_text(encoding="utf-8"))
    by_w = {}
    for r in rows:
        if r["person_key"].split("#")[0] not in set(args.exclude_writers or []):
            by_w.setdefault(r["person_key"], []).append(r)
    rng = random.Random(args.seed)
    writers = rng.sample(list(by_w), min(args.n_writers, len(by_w)))
    refs = [rng.choice(by_w[w]) for w in writers]

    g = InkSpireGen(None if args.dry_run else args.lora_dir, args.layout_ckpt, args.device,
                    args.steps, args.guidance, std_font=args.std_font,
                    cfg=args.cfg, uncond=args.uncond)
    if args.dry_run:
        ref = refs[0]
        arr = np.array(Image.open(ref["image_path"]).convert("L"))
        c = g.prepare(arr, ref["text"], args.lines, args.seed)
        sep = np.zeros((c["x"].shape[0], 4), np.uint8)
        img = np.hstack([c["x"], sep, c["xc"], sep, (1 - c["mask"]) * 255])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(args.out)
        print(f"[dry-run] writer {ref['person_key']} ref={ref['text']!r} → canvas {c['x'].shape} "
              f"tokens {len(c['chars'])} (ref {int((c['line_id'] == 0).sum())}) y_cut {c['y_cut']} "
              f"layout={'model' if g.layout else 'std'}\nsaved {args.out}  [x | xc | mask(검정=생성)]")
        return

    CW, CH = args.cell_w, args.cell_h * len(args.lines)
    grid = []
    for w, ref in zip(writers, refs):
        arr = np.array(Image.open(ref["image_path"]).convert("L"))
        base = w.split("#")[0]
        fp = Path(ref["font_path"]) if ref.get("font_path") else FONTS_DIR / f"{base}.ttf"
        gt = _vstack([render_in_font(fp, t) for t in args.lines]) if fp.exists() else np.full((64, 200), 230, np.uint8)
        gen = _vstack(g.gen_lines(arr, ref["text"], args.lines, args.seed))
        row = np.hstack([cell(arr, f"스타일 ref: {w}", CW, CH), cell(gt, f"정답 폰트: {base}", CW, CH, highlight=True),
                         cell(gen, f"InkSpire ({'layout' if g.layout else 'std layout'})", CW, CH)])
        grid += [row, np.full((6, row.shape[1]), 80, np.uint8)]
        print(f"  {w}: ref={ref['text'][:30]!r}")
    body = np.vstack(grid)
    title = label_img(f'InkSpire one-shot  lora={args.lora_dir}  layout={args.layout_ckpt or "std"}  '
                      f'steps={args.steps} g={args.guidance}  줄: {" / ".join(args.lines)[:60]}',
                      body.shape[1], 44, 18, bold=True, center=False, bg=255)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack([title, np.zeros((2, body.shape[1]), np.uint8), body])).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
