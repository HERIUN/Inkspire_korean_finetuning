"""실험 공용 헬퍼 (Eruku_korean_finetuning 에서 잘라옴).

원본은 VAE 재구성 실험용 헬퍼 모음이었다. eval/htr_cer.py 가 import 하는 3개만 남긴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = REPO = Path(__file__).resolve().parents[1]   # 저장소 루트 (REPO = gen_compare 호환 별칭)
sys.path.insert(0, str(HERE))


def load_line_x11(src, h=64, pad8=True):
    """라인이미지(경로 또는 grayscale 배열) → [1,1,h,W] [-1,1] (bg=+1, ink=-1).

    bilinear h-리사이즈, pad8=True 면 폭 8배수 흰(+1) 우측패딩(VAE 정확 roundtrip 조건).
    """
    if isinstance(src, (str, Path)):
        a = np.array(Image.open(src).convert("L"))
    else:
        a = np.asarray(src)
    t = torch.from_numpy(a.astype(np.float32) / 255.0)[None, None]
    w = max(1, int(round(h * a.shape[1] / a.shape[0])))
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    x = t * 2 - 1
    w8 = (w + 7) // 8 * 8
    if pad8 and w8 != w:
        x = F.pad(x, (0, w8 - w), value=1.0)
    return x


@torch.no_grad()
def roundtrip(vae, x, dev):
    """[1,1,h,W] [-1,1] → VAE encode.mode→decode → [1,1,h,W] [-1,1]."""
    z = vae.encode(x.repeat(1, 3, 1, 1).to(dev)).latent_dist.mode()
    return vae.decode(z).sample.clamp(-1, 1)[:, :1].cpu()


def to_u8(t):
    return ((t[0, 0] + 1) / 2 * 255).clamp(0, 255).byte().numpy()


def header_row(col_labels, cw, ch=None, lh=26):
    """컬럼 라벨 헤더 한 줄. 라벨 스트립만 만든다 — `cell` 을 쓰면 라벨 밑에 빈 회색
    이미지 블록(높이 ch)이 같이 붙어 montage 첫 줄이 쓸데없이 두꺼워진다.
    좌우 2px 테두리는 `cell` 과 맞춰야 열이 어긋나지 않는다(폭 = cw + 4). ch 는 하위호환용."""
    from infer.show import label_img
    return np.hstack([np.pad(label_img(lbl, cw, lh, 15, bg=245), ((2, 2), (2, 2)),
                             constant_values=0) for lbl in col_labels])


def fit_row(cells, full_w):
    """셀들을 가로로 붙이고 full_w 까지 흰 패딩."""
    row = np.hstack(cells)
    if row.shape[1] < full_w:
        row = np.hstack([row, np.full((row.shape[0], full_w - row.shape[1]), 255, np.uint8)])
    return row
