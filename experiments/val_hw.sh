#!/bin/bash
# 실손글씨 검증 — run 의 체크포인트별로 정제 hw_line 전쌍 380 을 cfg 1 로 생성해 PP-OCR CER 을 잰다 → <run>/val_hw.csv (step,cer).
# 학습 중 val_loss 는 테스트 폰트 64 샘플뿐이라 실손글씨 성능이 안 보인다(EXPERIMENTS §26). 학습 중에도, 끝난 뒤에도 돌릴 수 있다
# (이미 잰 step 은 건너뜀). 데이터·판독기는 비공개(Eruku repo). 사용: GPU=2 experiments/val_hw.sh finetune_runs/<run> [step ...]
set -e
I=$(cd "$(dirname "$0")/.." && pwd); E=${ERUKU:-/data/work/dgkang/Eruku_korean_finetuning}
D=$E/custom_datasets/hf_korean_handwriting_clean/hw_line
R=$(cd "$1" && pwd); shift; V=$R/val_hw; mkdir -p $V; [ -f $R/val_hw.csv ] || echo step,cer > $R/val_hw.csv
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=$I; cd $I
for L in $R/lora_step_*; do
  s=$((10#${L##*_})); [ $# -gt 0 ] && [[ " $* " != *" $s "* ]] && continue
  grep -q "^$s," $R/val_hw.csv && continue
  .venv/bin/python experiments/realhw_dump.py $V/s$s --data $D --all-pairs --ckpt "inkspire:$L,cfg=1.0,uncond=gen,word_gap=1" > $V/s$s.log 2>&1
  c=$(cd $E && .venv-paddle/bin/python eval/ppocr_rec.py $V/s$s/gen 2>/dev/null | grep -o "corpus [0-9.]*" | cut -d' ' -f2)
  [ -n "$c" ] && echo "$s,$c" | tee -a $R/val_hw.csv
done
