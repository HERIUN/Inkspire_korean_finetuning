"""한글 HTR 리더 로드 + CER (Eruku_korean_finetuning 에서 잘라옴).

원본은 보조 HTR 을 **학습**하는 스크립트였다(SmoothCrossEntropy·NoisyTeacherForcing·
KoreanAuxDataset). 여기서는 평가에 쓰는 로더와 거리 계산만 남긴다 — 그래야 그 셋의 의존이
사라진다. 체크포인트는 원본 repo 에서 학습한 `finetune_runs/aux_htr_ko/htr_s20000` 을 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from safetensors.torch import load_file

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))

from models.htr import HTR


def load_pretrained_htr(path: Path) -> HTR:
    """release(config.json + model.safetensors) 또는 save_pretrained 출력 둘 다 로드."""
    cfg = json.load(open(path / "config.json"))
    cfg = {k: v for k, v in cfg.items()
           if not k.startswith("_") and k not in ("architectures", "model_type", "torch_dtype", "transformers_version")}
    htr = HTR(**cfg)
    wf = path / "model.safetensors"
    if not wf.exists():
        wf = path / "diffusion_pytorch_model.safetensors"
    htr.load_state_dict(load_file(str(wf)), strict=False)
    return htr


def _lev(a, b):
    """Levenshtein 거리(문자 단위)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds, refs):
    tot_e, tot_n = 0, 0
    for p, r in zip(preds, refs):
        tot_e += _lev(p, r); tot_n += max(1, len(r))
    return tot_e / max(1, tot_n)
