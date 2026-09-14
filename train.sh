#!/usr/bin/env bash
# InkSpire 학습 진입점.
#
#   ./train.sh inkspire        [args]  이미지 모델: FLUX.1-Fill LoRA (configs/inkspire.yaml)
#   ./train.sh inkspire-layout [args]  레이아웃 모델: masked CFM (configs/inkspire_layout.yaml)
#   ./train.sh fetch-flux              FLUX.1-Fill-dev 34GB + 빈 프롬프트 캐시
#                                      (hf auth login + 라이선스 수락 선행)
#
# 환경변수:  GPU=2 (기본 0) · DRY=1 (커맨드만 출력) · PY=... (기본 ./.venv/bin/python)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"; GPU="${GPU:-0}"; cd "$ROOT"
run() { echo "+ CUDA_VISIBLE_DEVICES=$GPU ${*/#$PY/python}" >&2
        [ -n "${DRY:-}" ] || CUDA_VISIBLE_DEVICES="$GPU" "$@"; }
cmd="${1:-}"; shift || true
case "$cmd" in
  inkspire)        run "$PY" train/inkspire.py "$@" ;;
  inkspire-layout) run "$PY" train/inkspire_layout.py "$@" ;;
  fetch-flux)      run "$PY" tools/fetch_flux.py "$@" ;;
  *) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 1 ;;
esac
