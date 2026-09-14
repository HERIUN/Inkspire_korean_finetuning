"""FLUX.1-Fill-dev 1회 다운로드 + 빈 프롬프트 임베딩 캐시(models/inkspire.py 용).

게이트 repo(비상업 라이선스)라 수동 선행 작업 2개가 필요하다:
  1. `hf auth login`  (https://huggingface.co/settings/tokens 의 read 토큰)
  2. https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev 에서 라이선스 수락

받는 것: transformer(24GB) / text_encoder_2 T5-XXL(9.5GB) / text_encoder CLIP / vae / tokenizer·scheduler config
≈ 34GB. 루트의 단일파일 `flux1-fill-dev.safetensors`(24GB 중복)는 받지 않는다.
그 뒤 텍스트 인코더만 올려 `encode_prompt("")` → `model_zoo/flux_fill_empty_prompt.pt`(~4MB). 학습·추론은
이 .pt 만 읽고 T5/CLIP 을 다시 로드하지 않는다. .pt 가 있으면 전부 skip.

사용:
  python tools/fetch_flux.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from huggingface_hub import snapshot_download, whoami   # noqa: E402
from huggingface_hub.errors import GatedRepoError       # noqa: E402

from models.inkspire import EMPTY_CACHE, REPO, empty_prompt_cache   # noqa: E402

PATTERNS = ["model_index.json", "transformer/*", "vae/*", "text_encoder/*", "text_encoder_2/*",
            "tokenizer/*", "tokenizer_2/*", "scheduler/*"]
HOWTO = f"""
  1) hf auth login            (https://huggingface.co/settings/tokens 에서 read 토큰 발급)
  2) https://huggingface.co/{REPO} 페이지에서 라이선스 수락
  그 뒤 다시: python tools/fetch_flux.py"""


def main():
    if EMPTY_CACHE.exists():
        print(f"[skip] {EMPTY_CACHE} already exists")
        return
    try:
        user = whoami()["name"]
    except Exception as e:   # 토큰 없음/만료
        print(f"[error] HF 로그인 안 됨 ({type(e).__name__}). FLUX.1-Fill-dev 는 게이트 repo 라 로그인이 필요합니다.{HOWTO}")
        sys.exit(1)
    print(f"[hf] logged in as {user}")
    try:
        local = snapshot_download(REPO, allow_patterns=PATTERNS)
    except GatedRepoError:
        print(f"[error] {REPO} 접근 거부 — 라이선스 미수락.{HOWTO}")
        sys.exit(1)
    print(f"[download] {local}")

    import torch
    from diffusers import FluxFillPipeline

    # transformer/vae 는 RAM 에 안 올림 — encode_prompt 는 텍스트 인코더·토크나이저만 쓴다
    pipe = FluxFillPipeline.from_pretrained(local, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    d = empty_prompt_cache(pipe, EMPTY_CACHE)
    print(f"[done] {EMPTY_CACHE}  " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in d.items()))


if __name__ == "__main__":
    main()
