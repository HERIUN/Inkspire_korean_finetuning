"""보조 HTR 입력 헬퍼 (Eruku_korean_finetuning 에서 잘라옴).

원본은 VAE 한글 적응용 `KoreanAuxDataset` 이 본체였다. eval/htr_cer.py 가 쓰는 것은
`_fit_width` 하나뿐이라 그것만 남긴다 — split.make_dataset 의존이 사라진다.
"""
from __future__ import annotations

import numpy as np
import torch.nn.functional as F

W = 768   # 보조 HTR 입력 폭(원본 상수)


def _fit_width(img):
    """[C,64,w] → [C,64,768]: w<768 오른쪽 흰(1.0) 패딩, w>768 이면 폭 스케일-투-핏(라벨 정합 유지)."""
    c, h, w = img.shape
    if w == W:
        return img
    if w < W:
        out = img.new_ones((c, h, W))
        out[:, :, :w] = img
        return out
    return F.interpolate(img.unsqueeze(0), size=(h, W), mode="bilinear", align_corners=False)[0]
