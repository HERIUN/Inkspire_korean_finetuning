"""한글 charset (데이터 기반) — 레이아웃 모델 vocab 의 원천(page.vocab).

구성: FONT_SQUARE_CHARSET(라틴 164) + 코퍼스 등장 한글음절(정렬).
라틴 prefix 순서는 건드리지 않는다 — 기존 체크포인트의 인덱스와 어긋난다.

한글 집합은 **코퍼스에 실제 등장하는 음절**로 짓는다(실측 ≈2,345자, 상위 2000 이 99.9% 커버).
현대 한글 전체 11,172자 중 나머지는 코퍼스에 없어 dead class → 제외. 결정된 charset 은
assets/korean_charset.json 으로 고정한다(재현성).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..upstream.constants import FONT_SQUARE_CHARSET

HERE = Path(__file__).resolve().parents[2]   # 저장소 루트
CHARSET_JSON = HERE / "assets" / "korean_charset.json"
CORPUS = HERE / "assets" / "corpus"


def _corpus_syllables() -> list[str]:
    """코퍼스(korean_lines.txt, chars.txt)에 등장하는 한글 음절(AC00–D7A3) 집합, 정렬.

    ★ `chars.txt` 가 한글 음절의 대부분을 공급한다(2,345 중 korean_lines 기여는 754 뿐).
      없으면 charset 이 2,345 → 754 로 쪼그라들어 **학습된 체크포인트의 vocab 과 어긋난다**.
      그래서 조용히 건너뛰지 않고 에러로 막는다. 샘플러는 더 이상 chars.txt 를 쓰지 않지만
      (docs/DATA_LIMITATIONS.md D2) 이 charset 경로는 계속 쓴다.
    """
    seen = set()
    for name in ("korean_lines.txt", "chars.txt"):
        fp = CORPUS / name
        if not fp.exists():
            raise FileNotFoundError(
                f"알파벳 구성에 필요한 코퍼스가 없습니다: {fp}\n"
                "  이 파일이 빠지면 한글 음절 수가 줄어 기존 체크포인트와 어긋납니다.")
        seen |= set(fp.read_text(encoding="utf-8"))
    syl = sorted(c for c in seen if 0xAC00 <= ord(c) <= 0xD7A3)
    return syl


def build_charset(save: bool = True) -> list[str]:
    """라틴(pretrained 인덱스 보존) + 코퍼스 한글음절. save=True 면 json 고정."""
    charset = list(FONT_SQUARE_CHARSET) + _corpus_syllables()
    assert len(set(charset)) == len(charset), "charset 중복"
    if save:
        CHARSET_JSON.write_text(json.dumps(charset, ensure_ascii=False), encoding="utf-8")
    return charset


def load_charset() -> list[str]:
    """고정된 json 이 있으면 로드, 없으면 build+save. (재현성 보장)"""
    if CHARSET_JSON.exists():
        charset = json.loads(CHARSET_JSON.read_text(encoding="utf-8"))
        # sanity: 앞 164 는 라틴이어야(warm-start 인덱스 정합)
        assert charset[:len(FONT_SQUARE_CHARSET)] == list(FONT_SQUARE_CHARSET), \
            "korean_charset.json 의 라틴 prefix 가 FONT_SQUARE_CHARSET 과 불일치 (warm-start 깨짐)"
        return charset
    return build_charset(save=True)


if __name__ == "__main__":
    charset = build_charset(save=True)
    n_ko = len(charset) - len(FONT_SQUARE_CHARSET)
    print(f"charset: latin={len(FONT_SQUARE_CHARSET)} + korean={n_ko} = {len(charset)}")
    print(f"latin prefix indices 0..163 preserved")
    print(f"saved: {CHARSET_JSON}")
