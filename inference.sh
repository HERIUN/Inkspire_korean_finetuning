#!/usr/bin/env bash
# InkSpire 추론 진입점.
#
#   ./inference.sh inkspire [args...]   one-shot 뷰어 [스타일ref | 정답 | InkSpire 여러 줄]
#   ./inference.sh inkspire --dry-run   FLUX 없이 [x | xc | mask] 캔버스만
#
# 환경변수:  GPU=2 (기본 0) · DRY=1 (커맨드만 출력) · PY=... (기본 ./.venv/bin/python)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"; GPU="${GPU:-0}"; cd "$ROOT"
run() { echo "+ CUDA_VISIBLE_DEVICES=$GPU ${*/#$PY/python}" >&2
        [ -n "${DRY:-}" ] || CUDA_VISIBLE_DEVICES="$GPU" "$@"; }
cmd="${1:-}"; shift || true
case "$cmd" in
  inkspire) run "$PY" infer/inkspire.py "$@" ;;
  *) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 1 ;;
esac
