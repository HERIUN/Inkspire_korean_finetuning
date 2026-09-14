"""평가 텍스트 생성 + 폰트 렌더 가능 검사 (Eruku_korean_finetuning 에서 잘라옴).

원본은 echo(스타일 재현) 정량 스크립트 전체였다. eval/htr_cer.py 와
experiments/gen_compare.py 가 쓰는 두 함수만 남긴다 — Eruku 모델 로더 의존이 사라진다.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import ImageFont
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))


EN_COHERENT = """the quick brown fox jumps over the lazy dog near the river bank at sunrise .
machine learning models can generate handwritten text in many different styles and fonts today .
she sold seashells by the seashore while the waves crashed gently against the rocks at dawn .
in the beginning there was nothing but darkness and then a brilliant light appeared over the hills .
the five boxing wizards jump quickly over the wooden fence as the summer storm approaches from the west .
scientists around the world are working hard to understand the deep mysteries of the human brain .
a journey of a thousand miles begins with a single step taken in the right direction today .
the old library on the corner holds thousands of books written by famous authors from every century .
technology continues to change the way people communicate work and live their everyday lives now .
children laughed and played in the park while their parents watched them from a nearby wooden bench .""".split("\n")


# 평가 텍스트는 **학습 코퍼스와 분리**한다. build_texts 는 라인 리스트에서 rng.choice 로 뽑으므로
# 학습 코퍼스가 바뀌면 같은 seed 라도 다른 텍스트가 나와 과거 CER 과 비교가 끊긴다.
# 이 스냅샷(2026-09-04, 중복 제거 후 941줄)을 고정해 둬야 코퍼스를 키워도 지표가 살아남는다.
EVAL_LINES_KO = "assets/corpus/eval_lines_ko.txt"


def build_texts(n, rng, english=False, coherent=False, max_words=15):
    """다양한 길이(어절 1~max_words)의 라인 n개. english=True 면 영어(pretrained 비교용).
    coherent=True 면 실제 문장에서 세그먼트 추출(OCR 신뢰성↑, CER 덜 비관적).
    max_words 올리면 더 긴 문장으로 길이 robustness 측정."""
    import re as _re
    if coherent:
        base = ([" ".join(_re.findall(r"[A-Za-z0-9']+", s)) for s in EN_COHERENT] if english
                else [ln.strip() for ln in (HERE / EVAL_LINES_KO)
                      .read_text(encoding="utf-8").splitlines() if ln.strip()])
        out = []
        for i in range(n):
            nw = 1 + (i % max_words)
            words = rng.choice(base).split()
            if len(words) <= nw:
                seg = words
            else:
                st = rng.randint(0, len(words) - nw)
                seg = words[st:st + nw]
            out.append(" ".join(seg))
        return out
    if english:
        cand = (HERE / "assets/corpus/english_words.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
        words = sorted({w.strip() for w in cand if _re.match(r"^[A-Za-z][A-Za-z'-]*$", w.strip() or "x")})
        nums = lambda: rng.choice(["2024", "98.7%", "4,500", "123", "2024-05-11", "15:30"])
    else:
        lines = (HERE / EVAL_LINES_KO).read_text(encoding="utf-8").splitlines()
        words = sorted({w for ln in lines for w in ln.split() if w.strip()})
        nums = lambda: rng.choice(["2024년", "98.7%", "4,500원", "123", "010-1234-5678", "15:30"])
    out = []
    for i in range(n):
        nw = 1 + (i % max_words)                # 어절수 1~max_words 균등 분포
        toks = [rng.choice(words) for _ in range(nw)]
        if nw >= 3 and rng.random() < 0.3:      # 가끔 숫자 섞기
            toks[rng.randrange(nw)] = nums()
        out.append(" ".join(toks))
    return out


def font_can_render(fp, text):
    try:
        cm = TTFont(str(fp)).getBestCmap()
    except Exception:
        return False
    return all(ord(c) in cm for c in text if not c.isspace())
