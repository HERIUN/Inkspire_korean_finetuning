#!/usr/bin/env bash
# InkSpire 평가 진입점.
#
#   ./eval.sh exp [이름] [args...]   experiments/ 스크립트 (인자 없으면 목록)
#
# ★ --ckpt 는 `inkspire:<lora_dir>[,<layout_ckpt>][,key=value…]` 형식이다.
#   key=value 로 steps · guidance · std_font · trim_ref · degrade 스윕.
#
# 환경변수:  GPU=2 (기본 0) · DRY=1 (커맨드만 출력) · PY=... (기본 ./.venv/bin/python)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"; GPU="${GPU:-0}"; cd "$ROOT"
run() { echo "+ CUDA_VISIBLE_DEVICES=$GPU ${*/#$PY/python}" >&2
        [ -n "${DRY:-}" ] || CUDA_VISIBLE_DEVICES="$GPU" "$@"; }
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
exp_run() {
  if [ -z "${1:-}" ]; then
    echo "experiments/ 스크립트 (실행: ./eval.sh exp <이름> [인자])"; echo
    for f in experiments/*.py; do b="$(basename "$f" .py)"; [ "$b" = "common" ] && continue
      printf "  %-18s %s\n" "$b" "$(sed -n '1s/^"""//p' "$f")"; done
    return
  fi
  local name="$1"; shift
  [ -f "experiments/$name.py" ] || { echo "experiments/$name.py 가 없다" >&2; exit 1; }
  run "$PY" "experiments/$name.py" "$@"
}
cmd="${1:-}"; shift || true
case "$cmd" in
  exp) exp_run "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "알 수 없는 명령: $cmd" >&2; echo >&2; usage >&2; exit 1 ;;
esac
