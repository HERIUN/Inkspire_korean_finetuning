# 실험 로그 — InkSpire 한글 재현

> `Eruku_korean_finetuning/docs/EXPERIMENTS.md` §12 를 그대로 옮겼다(2026-09-15).
> 절 번호(§12)는 원본 문서의 것이고, 본문의 상대경로는 그 repo 기준이다 —
> `finetune_runs/inkspire_*` 는 이 repo 로 옮겨왔고 `assets`/`data`/`model_zoo` 는 심링크다.
> **`./eval.sh cer` 는 이 repo 에 없다** — CER 측정 경로는 원본 repo 에만 있다. 아래 수치는 거기서 잰 것이다.

## 12. InkSpire(FLUX-Fill LoRA + 레이아웃 CFM) 한글 재현 (2026-09)

논문 *Learning to Generate Stylized Handwritten Text via a Unified Representation of Style, Content, and Noise*
(Wang et al., ICLR 2026, `docs/inkspire.pdf`) 의 두 단계 분해 p(X, Xc | C, Xs) = p(Xc | C, Xs)·p(X | Xs, Xc) 를 합성 폰트
페이지로 재현한다. ★ 논문은 Eruku/Emuru 와 직접 비교한 수치가 없다 — 아래 표가 유일한 근거다.
코드: `custom_datasets/korean/page.py` · `models/inkspire.py` · `models/inkspire_layout.py` ·
`train/inkspire*.py` · `infer/inkspire.py` · `configs/inkspire*.yaml`.

### 설정

| 항목 | 값 |
|---|---|
| 데이터 | 합성 폰트 페이지(`fonts_korean_v3/train` 12,951종, NanumGothic 제외), line_h U(40,80), P=512 패치, R-Mask |
| 이미지 모델 | FLUX.1-Fill-dev + LoRA r=32/α=32 (115.9M, 표 8 13종), R-APE, 텍스트 인코더 제거(빈 프롬프트 캐시), guidance 30 |
| 이미지 학습 | Prodigy lr 1 wd 0.01, batch 4, grad clip 1.0, 20k step, GPU 2 (H100 96GB). 실측 2.6 s/step → 20k ≈ 14.5h (계획 7~9h 과소) |
| 레이아웃 모델 | masked CFM transformer 10층/512/8h (33M) + line_emb, 정규화 버퍼, 무레퍼런스 모드 10% |
| 레이아웃 학습 | AdamW 1e-4, batch 160, 20k step |
| 추론 | one-shot: ref 한 줄(잉크 48px, pitch 77) + 마스크된 아래 줄, ODE 20 step. 레이아웃 = std layout(표준폰트 bbox) 또는 `--layout-ckpt` |
| 평가 | HTR CER **중앙값·폭주율**(평균 금지) + echo / `--style-ref-text "다람쥐 헌 쳇바퀴에 타고파"` 두 프로토콜 |

### Stage 2 — 이미지 모델 (oracle 레이아웃 val, std layout 평가)

```bash
GPU=2 ./train.sh fetch-flux                                # hf auth login + 라이선스 수락 선행, 34GB
GPU=2 ./train.sh inkspire --out finetune_runs/inkspire_smoke --max-steps 2 --batch-size 1 --P 256 \
    --num-workers 0 --val-every 1 --val-batches 1 --val-steps 2 --save-every 1
GPU=2 ./train.sh inkspire --out finetune_runs/inkspire_p512 --max-steps 2000 --val-every 500 --save-every 500
#   → finetune_runs/inkspire_p512/samples/val_s002000.png 육안 (첫 줄 스타일로 아래 줄에 읽히는 한글?)
GPU=2 ./eval.sh cer --ckpt inkspire:finetune_runs/inkspire_p512/lora_last --n 100 --coherent --binarize 200
GPU=2 ./train.sh inkspire --out finetune_runs/inkspire_p512 --resume finetune_runs/inkspire_p512/lora_step_002000 --max-steps 20000
```

| step | val loss | s/step | peak VRAM | val_s*.png 육안 | CER 중앙값 (std layout, echo, n=100) |
|---|---|---|---|---|---|
| 500 | 0.193 | 2.6 (B=4, 0.39 it/s) | 30.6GB | ✅ 아래 줄 전부 ref 폰트 스타일로 재현, 숫자·일부 자모 오류 소수 | — |
| 1000 | 0.189 | 2.6 | 30.6GB | one-shot 몽타주(`_debug/inkspire_s1000.png`) 4 writer 전부 스타일 전이, 숫자 약함 | (아래 Stage 4 표 s1000 행) |
| 2000 | 0.184 | 2.6 | 30.6GB | ✅ | — |
| 13000 | 0.171 | 3.9 (GPU 경합) | 30.6GB | ✅ 정답과 거의 구분 불가, 오류는 숫자에 몰림 | **0.190** (n=300, +layout, style-ref) |
| 20000 | 0.168 | 2.5 | 30.6GB | ✅ 동일 | 0.222 (같은 조건) |

★ **val loss 는 끝까지 내려갔지만(0.193 → 0.168) CER 은 13k 가 20k 보다 낫다**(중앙값 0.190 vs 0.222, 폭주율 1.7% vs 2.7%).
손실로 체크포인트를 고르면 안 된다. 전체 평균 0.265 it/s(GPU 경합 포함) → 20k 에 약 21h.

### Stage 3 — 레이아웃 모델

```bash
GPU=2 ./train.sh inkspire-layout --out finetune_runs/inkspire_layout_smoke --max-steps 2 --batch-size 4 \
    --num-workers 0 --norm-batches 1 --val-every 1 --val-batches 1
GPU=2 ./train.sh inkspire-layout --out finetune_runs/inkspire_layout                # val_l1_px 하강·정체 확인
```

| step | val loss | 생성 토큰 bbox L1 (px, held-out v2/test) |
|---|---|---|
| 스모크 s2 (batch 4) | 1.31 | 45.2 (미학습 기준값) |
| 2000 | 0.333 | 11.2 |
| 4000 | 0.285 | 11.2 |
| 6000 | 0.261 | 12.8 |
| 8000 | 0.255 | 11.7 |
| 20000 | TBD | TBD |

실측 메모(2026-09-10):
- 첫 배치에서 1,049 토큰 페이지가 `max_len` 1024 를 넘어 크래시 → `KoreanLayoutDataset.MAX_TOKENS` 로 줄 경계 캡, 모델 `max_len` 2048.
- batch 160 × 최대 2,048 토큰이면 attention·FFN 활성화가 **76GB** 까지 감 → MAX_TOKENS 1024 로 캡(초과는 pitch≈42px 극단뿐).
  1024 캡 후에도 이미지 모델과 동시 학습 시 GPU 2 83GB, 이미지 run 속도 0.26→0.11 it/s 로 반토막 → 레이아웃은 s8000 에서 중단, 이미지 모델 뒤에 재개.
- 속도 0.94 it/s(단독) → 20k 는 6h. 계획의 "40분"은 페이지 렌더 병목을 과소평가한 것. bbox L1 은 s2000 부터 11~12px 정체(val loss 는 계속 하강) — 20k 가 필요한지는 의문.
- 추론 관찰: 줄 시작 Δx(= −이전 줄 폭)를 가장 못 맞춰 줄이 오른쪽으로 밀림(학습 줄은 page_w 를 채우지만 추론 줄은 짧음)
  → `infer/inkspire.py` 에서 줄마다 첫 토큰을 좌측 여백으로 평행이동(학습 페이지는 전부 좌측 시작). 크기·간격은 예측값 유지.

### Stage 4 — Eruku 비교 (같은 프로토콜, n=300, coherent, binarize 200, seed 0)

```bash
BEST=finetune_runs/eruku_corpusmax/checkpoint_last.pth
INK=inkspire:finetune_runs/inkspire_p512/lora_last,finetune_runs/inkspire_layout/checkpoint_last.pth
GPU=2 ./inference.sh inkspire --lora-dir finetune_runs/inkspire_p512/lora_last \
    --layout-ckpt finetune_runs/inkspire_layout/checkpoint_last.pth --lines "첫 줄" "둘째 줄" "셋째 줄"
for F in assets/fonts_korean_v2/train assets/fonts_korean_v2/test assets/custom_hw_fonts; do for CK in $BEST $INK; do
  GPU=2 ./eval.sh cer --ckpt $CK --fonts-dir $F --n 300 --coherent --binarize 200 --seed 0 \
      --style-ref-text "다람쥐 헌 쳇바퀴에 타고파" --out finetune_runs/_eval/inkspire_$(basename $F)_${CK%%:*}_sref.txt
  GPU=2 ./eval.sh cer --ckpt $CK --fonts-dir $F --n 300 --coherent --binarize 200 --seed 0 \
      --out finetune_runs/_eval/inkspire_$(basename $F)_${CK%%:*}_echo.txt
done; done
GPU=2 ./eval.sh exp gen_compare --ckpt-before $BEST --ckpt-after $INK \
    --label-before "Eruku corpusmax s45000" --label-after "InkSpire one-shot" --out-dir finetune_runs/inkspire_p512/_eval
```

판정은 세트별 **중앙값 + 폭주율**만(평균 금지). custom_hw 는 폭주율만(리더 바닥 0.213). 레이아웃 기여는
std layout(`inkspire:<lora>`) vs `--layout-ckpt` 행의 차이로 분리한다.

| 세트 | 모델 | 프로토콜 | 평균 | 중앙값 | 폭주율 | GT 바닥 |
|---|---|---|---|---|---|---|
| seen v2/train | Eruku corpusmax s45000 | echo | 0.085 | 0.000 | 0.3% | 0.005 |
| seen v2/train | Eruku corpusmax s45000 | style-ref | 0.083 | 0.000 | 0.0% | 0.005 |
| seen v2/train | InkSpire s20000 std layout | echo | 0.247 | 0.102 | 3.0% | 0.005 |
| seen v2/train | InkSpire s20000 std layout | style-ref | 0.225 | 0.095 | 2.0% | 0.005 |
| seen v2/train | InkSpire s20000 + layout | echo | 0.325 | 0.200 | 2.7% | 0.005 |
| seen v2/train | InkSpire s20000 + layout | style-ref | 0.330 | 0.143 | 1.7% | 0.005 |
| held-out v2/test | Eruku corpusmax s45000 | echo | 0.123 | 0.029 | 0.3% | 0.026 |
| held-out v2/test | Eruku corpusmax s45000 | style-ref | 0.064 | 0.000 | 0.0% | 0.026 |
| held-out v2/test | InkSpire s20000 std layout | echo | 0.353 | 0.250 | 5.0% | 0.026 |
| held-out v2/test | InkSpire s20000 std layout | style-ref | 0.278 | 0.197 | 2.3% | 0.026 |
| held-out v2/test | InkSpire s20000 + layout | echo | 0.345 | 0.254 | 1.3% | 0.026 |
| held-out v2/test | InkSpire s20000 + layout | style-ref | 0.368 | 0.222 | 2.7% | 0.026 |
| held-out v2/test | InkSpire **s13000** + layout (최적 체크포인트) | style-ref | 0.324 | 0.190 | 1.7% | 0.026 |
| custom_hw 4 | Eruku corpusmax s45000 | echo | 0.198 | 0.056 | 0.0% | 0.209 |
| custom_hw 4 | Eruku corpusmax s45000 | style-ref | 0.083 | 0.000 | 0.0% | 0.209 |
| custom_hw 4 | InkSpire s20000 std layout | echo | 0.419 | 0.300 | 5.7% | 0.209 |
| custom_hw 4 | InkSpire s20000 std layout | style-ref | 0.346 | 0.226 | 3.3% | 0.209 |
| custom_hw 4 | InkSpire s20000 + layout | echo | 0.422 | 0.333 | 2.3% | 0.209 |
| custom_hw 4 | InkSpire s20000 + layout | style-ref | 0.384 | 0.253 | 2.3% | 0.209 |

결론: **재현은 성공했지만 Eruku 를 못 이긴다.** held-out v2/test style-ref 기준 Eruku 는 중앙값 0.000 / 폭주 0.0%,
InkSpire 최고 체크포인트는 0.190 / 1.7% 다. 세 세트·두 프로토콜 전부 같은 방향이다. 논문이 Eruku(Emuru) 와 직접 비교한
수치가 없으므로, "더 좋다"는 주장은 최소한 **한글 합성 폰트 조건에서는 우리 측정으로 확인되지 않는다**.

관찰 세 가지:
1. **레이아웃 모델이 오히려 손해다.** 6 개 비교쌍 전부 std layout(표준폰트 bbox)이 같거나 낫다
   (held-out style-ref 0.197 vs 0.222). s8000 의 bbox L1 11.7px 가 글자 위치를 흔들어 HTR 가독성을 깎는다.
   n=40 중간 점검에서는 반대로 보였는데(0.076 vs 0.141) 표본이 작아 생긴 착시였다.
2. **체크포인트 선택이 손실과 어긋난다.** val loss 최저는 s20000 이지만 CER 최저는 s13000.
3. **폰트 의존이 크다.** held-out style-ref 폰트별 중앙값이 GowunBatang 0.000 ~ NanumBrushScript 0.442(폭주 16.7%).
   획이 이어지는 붓글씨류가 가장 약하고, 숫자 오류가 전 구간에서 가장 흔하다.

몽타주: `finetune_runs/inkspire_p512/_eval/`(s20000 + 레이아웃) vs `_eval_best/`(s13000 + std layout).
후자에서 `gencmp_ko_diverse.png` 6 행 중 4 행은 Eruku 와 비슷한 품질로 스타일까지 옮기고, 나머지 2 행
(ARCHISCULPTURE, Sam3 처럼 라틴·장식 레퍼런스)은 글자가 뭉개진다 — 폭주율 2% 대의 정체가 이것이다.
전자는 레이아웃 모델 탓에 같은 행에서도 글자가 빠지거나 겹친다(표의 std layout 우세와 일치).

남은 가능성(미검증): 논문 설정인 **P=1024**(우리는 512), 레이아웃 모델 20k 완주, 실손글씨 데이터.
