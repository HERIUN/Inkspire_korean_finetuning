"""실험 공용 헬퍼 (Eruku_korean_finetuning 에서 잘라옴).

원본은 VAE 재구성 실험용 헬퍼 모음이었다. gen_compare.py 가 쓰는 것만 남긴다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont

HERE = REPO = Path(__file__).resolve().parents[1]   # 저장소 루트 (REPO = gen_compare 호환 별칭)
sys.path.insert(0, str(HERE))


def font_can_render(fp, text):
    try:
        cm = TTFont(str(fp)).getBestCmap()
    except Exception:
        return False
    return all(ord(c) in cm for c in text if not c.isspace())


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
