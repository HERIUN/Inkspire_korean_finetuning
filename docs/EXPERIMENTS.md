# 실험 로그 — InkSpire 한글 재현

> §1~§4 는 `Eruku_korean_finetuning/docs/EXPERIMENTS.md` §12 에서 옮겼다(2026-09-15). 원본 번호는 버리고 1 부터 다시 매겼다.
> 코드 블록의 `./eval.sh cer` 와 상대경로는 원본 repo 기준이다 — **CER 측정 경로는 원본 repo 에만 있다.**
> `finetune_runs/inkspire_*` 는 이 repo 로 옮겨왔고 `assets`/`data`/`model_zoo` 는 심링크다.
> 구현 상세·진단은 [`inkspire.md`](inkspire.md).

| § | 내용 | 판정 |
|---|---|---|
| 1 | 설정 | |
| 2 | 이미지 모델 FLUX.1-Fill LoRA | 재현 성공 |
| 3 | 레이아웃 모델 | 한 줄 생성엔 std layout 이 낫다 |
| 4 | Eruku 비교 — 자체 HTR | ~~Eruku 압승~~ → 리더 착시(§5) |
| 5 | Eruku 비교 — PP-OCRv6 재측정 | Eruku 0.018 vs InkSpire 0.040 (Eruku 권장 cfg 1.5 면 0.010) |
| 6 | 논문 수치로 본 Eruku 대비 (IAM) | 논문 조건에선 InkSpire 우세 |
| 7 | 이미지 백본 FLUX.3-Image-Klein-Base | CER 0.040 → 0.026 |
| 8 | ref 보강 (inference loop) | 효과 없음 — 접음 |
| 9 | 스타일 지표 (HWD·FID) | 줄 폭 보정(`ref_width`) 후 HWD 는 InkSpire 0.93 < Eruku 0.99(cfg 1.0)~1.12(cfg 1.5). 하한(같은 폰트 다른 문구) 0.34, 폰트별 표 |
| 10 | 실손글씨 (HERIUN/korean-handwriting) — Eruku 추론 설정 정정 포함 | 문장 정확도는 Eruku 압도, HWD·지저분한 ref 견고성은 InkSpire. 원인 = 주로 분포 밖 글자 모양(자간 어긋남은 작음). HWD 하한 hw_line 0.29. 시드 3개: best-of-3 0.275 → 0.169, 폰트엔 없던 일관 오류. **내용 CFG 2.0(재학습 없음) 으로 hw_line CER 0.25 → 0.075** |
| 11 | VAE 비교 (Eruku vs InkSpire) · 획이 날아가는 이유 | VAE 는 병목 아님(Flux2 가 4배 정확). 흐린 획은 ref 농도를 베낀 것 |
| 12 | transformer 정확도 분석 — 오류는 샘플마다 무작위 | 시드 간 같은 글자 오류 0. best-of-3 상한 0.004. → Xc dropout + 내용 CFG 학습 |
| 13 | 결론 | |
| 14 | 손글씨 증강·ragged·Xc dropout 재학습 | cfg 2.0 CER 전부↑(실손글씨 0.168 → 0.144), 실손글씨 HWD↓. Klein 은 전사 없이도 정확(p512 0.0215) |
| 15 | 정제 실손글씨셋(모든 쌍·교차, CI·부호검정) | 정확도는 Eruku 유의하게 우세(hw_line 0.028 vs 0.071). 서식 InkSpire 우세는 선 잔재 견고성이었다 |

## 학습 계보 — 현재 best 까지 (2026-11-09)

각 노드 = `finetune_runs/<run>` 의 체크포인트, 괄호 = 측정·결론이 있는 절. ⭐ = 채택(릴리즈), ✗ = 기각.
이미지 모델 LoRA 는 전부 **베이스 모델에서 처음부터** 학습했다(이어 학습은 s20000 연장뿐). 새 run 은 여기에 한 줄 추가하고 결과는 해당 절에 적는다.

```
[이미지 모델 — FLUX.1-Fill-dev + LoRA r32, P=512 batch 4, Prodigy]
inkspire_p512 s13000                     재현 성공, PP CER 0.040 (§2·§4·§5)
 └─ s20000 ✗                             val loss 최저지만 CER 0.026 → 0.038 (§2·§7)

[이미지 모델 — FLUX.3-Image-Klein-Base 로 백본 교체, 같은 데이터·마스크]
inkspire_klein_p512 s13000 ⭐ 현재 best   CER 0.040 → 0.026, ref_width 켜면 0.023 · HWD 0.929 (§7·§9)
 │   추론: ref_width (§9) + 내용 CFG cfg 2.0 uncond=gen — 실손글씨 CER 0.25 → 0.07 (§10)
 │         ref 보강(inference loop) ✗ (§8) · 레이아웃 모델 대신 std layout (§3)
 └─ s20000 ✗ (resume 연장)                CER 0.034, 13k 보다 유의하게 나쁨 (§7)
inkspire_klein_xcdrop s13000 ✗           --xc-drop 0.1 — CFG 이득은 학습이 아니라 추론 몫 (§10·§12)
inkspire_klein_hwaug s13000 ✗            --hw-aug 0.5 --ragged 0.5 --ref/target-xc-drop 0.1 — 실손글씨에서 p512 보다 못함 (§14)
inkspire_klein_xcfont s13000 ✗           --xc-font-rand 0.5 — CFG 의 필체 대가 그대로 (§19)
inkspire_klein_shift3 s13000 ✗           --train-shift 3, 폰트 9종 제외 — cfg 1 CER 그대로·HWD 악화 (§25)
inkspire_klein_seed7 s13000              --seed 7, 폰트 9종 제외 — 시드만으로 실손글씨 CER −0.03 / HWD +0.17 (§26)

[레이아웃 모델 — masked CFM transformer 33M]
inkspire_layout s20000 ✗                 한 줄 생성엔 std layout(표준폰트 bbox) 이 낫다 (§3·§4)
```

공통 설정(이미지 모델): 합성 폰트 페이지(`assets/fonts/train` 12,951종), LoRA r=32/α=32, R-APE, P=512, batch 4, Prodigy lr 1, GPU 2.

논문: *Learning to Generate Stylized Handwritten Text via a Unified Representation of Style, Content, and Noise*
(Wang et al., ICLR 2026, `docs/inkspire.pdf`). 두 단계 분해 p(X, Xc | C, Xs) = p(Xc | C, Xs)·p(X | Xs, Xc) 를 합성 폰트
페이지로 재현한다. 코드: `custom_datasets/korean/page.py` · `models/inkspire.py` · `models/inkspire_klein.py` ·
`models/inkspire_layout.py` · `train/inkspire*.py` · `infer/inkspire.py` · `configs/inkspire*.yaml`.

## 1. 설정

| 항목 | 값 |
|---|---|
| 데이터 | 합성 폰트 페이지(`assets/fonts/train` 12,951종, NanumGothic 제외), line_h U(40,80), P=512 패치, R-Mask |
| 이미지 모델 | FLUX.1-Fill-dev + LoRA r=32/α=32 (115.9M, 논문 표 8 13종), R-APE, 텍스트 인코더 제거(빈 프롬프트 캐시), guidance 30 |
| 이미지 학습 | Prodigy lr 1 wd 0.01, batch 4, grad clip 1.0, 20k step, GPU 2 (H100 96GB). 실측 2.6 s/step → 20k ≈ 14.5h (계획 7~9h 과소) |
| 레이아웃 모델 | masked CFM transformer 10층/512/8h (33M) + line_emb, 정규화 버퍼, 무레퍼런스 모드 10% |
| 레이아웃 학습 | AdamW 1e-4, batch 160, 20k step |
| 추론 | one-shot: ref 한 줄(잉크 48px, pitch 77) + 마스크된 아래 줄, ODE 20 step. 레이아웃 = std layout(표준폰트 bbox) 또는 `--layout-ckpt` |
| 평가 | §4: 자체 HTR CER **중앙값·폭주율**(평균 금지). §5 이후: PP-OCRv6 corpus CER. style-ref = `"다람쥐 헌 쳇바퀴에 타고파"` |

## 2. 이미지 모델 — FLUX.1-Fill LoRA (oracle 레이아웃 val, std layout 평가)

```bash
GPU=2 ./train.sh fetch-flux                                # hf auth login + 라이선스 수락 선행, 34GB
GPU=2 ./train.sh inkspire --out finetune_runs/inkspire_smoke --max-steps 2 --batch-size 1 --P 256 \
    --num-workers 0 --val-every 1 --val-batches 1 --val-steps 2 --save-every 1
GPU=2 ./train.sh inkspire --out finetune_runs/inkspire_p512 --max-steps 2000 --val-every 500 --save-every 500
#   → finetune_runs/inkspire_p512/samples/val_s002000.png 육안 (첫 줄 스타일로 아래 줄에 읽히는 한글?)
GPU=2 ./eval.sh cer --ckpt inkspire:finetune_runs/inkspire_p512/lora_last --n 100 --coherent --binarize 200
GPU=2 ./train.sh inkspire --out finetune_runs/inkspire_p512 --resume finetune_runs/inkspire_p512/lora_step_002000 --max-steps 20000
```

| step | val loss | s/step | peak VRAM | val_s*.png 육안 | 자체 HTR CER 중앙값 (std layout, echo, n=100) |
|---|---|---|---|---|---|
| 500 | 0.193 | 2.6 (B=4, 0.39 it/s) | 30.6GB | ✅ 아래 줄 전부 ref 폰트 스타일로 재현, 숫자·일부 자모 오류 소수 | — |
| 1000 | 0.189 | 2.6 | 30.6GB | one-shot 몽타주(`_debug/inkspire_s1000.png`) 4 writer 전부 스타일 전이, 숫자 약함 | (§4 표 s1000 행) |
| 2000 | 0.184 | 2.6 | 30.6GB | ✅ | — |
| 13000 | 0.171 | 3.9 (GPU 경합) | 30.6GB | ✅ 정답과 거의 구분 불가, 오류는 숫자에 몰림 | **0.190** (n=300, +layout, style-ref) |
| 20000 | 0.168 | 2.5 | 30.6GB | ✅ 동일 | 0.222 (같은 조건) |

★ **val loss 는 끝까지 내려갔지만(0.193 → 0.168) CER 은 13k 가 20k 보다 낫다**(중앙값 0.190 vs 0.222, 폭주율 1.7% vs 2.7%).
손실로 체크포인트를 고르면 안 된다. 전체 평균 0.265 it/s(GPU 경합 포함) → 20k 에 약 21h.

## 3. 레이아웃 모델

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
| 20000 | 미완주 | — |

실측 메모(2026-09-10):
- 첫 배치에서 1,049 토큰 페이지가 `max_len` 1024 를 넘어 크래시 → `KoreanLayoutDataset.MAX_TOKENS` 로 줄 경계 캡, 모델 `max_len` 2048.
- batch 160 × 최대 2,048 토큰이면 attention·FFN 활성화가 **76GB** 까지 감 → MAX_TOKENS 1024 로 캡(초과는 pitch≈42px 극단뿐).
  1024 캡 후에도 이미지 모델과 동시 학습 시 GPU 2 83GB, 이미지 run 속도 0.26→0.11 it/s 로 반토막 → 레이아웃은 s8000 에서 중단.
- 속도 0.94 it/s(단독) → 20k 는 6h. 계획의 "40분"은 페이지 렌더 병목을 과소평가한 것. bbox L1 은 s2000 부터 11~12px 정체(val loss 는 계속 하강) — 20k 가 필요한지는 의문.
- 추론 관찰: 줄 시작 Δx(= −이전 줄 폭)를 가장 못 맞춰 줄이 오른쪽으로 밀림(학습 줄은 page_w 를 채우지만 추론 줄은 짧음)
  → `infer/inkspire.py` 에서 줄마다 첫 토큰을 좌측 여백으로 평행이동(학습 페이지는 전부 좌측 시작). 크기·간격은 예측값 유지.
- 진단(2026-10, `inkspire.md` §9.4): 짧은 단독 줄의 첫 글자 오차 142px 가 원인. 긴 줄 아래 3번째 줄로 쓰게 하면 반듯하다.

## 4. Eruku 비교 — 자체 HTR (n=300, coherent, binarize 200, seed 0)

> ★ **2026-10-20 정정:** 이 절의 자체 HTR(`aux_htr_ko/htr_s20000`)은 같은 폰트 렌더로 학습된 리더라 InkSpire 출력을 과하게 나쁘게 읽었다.
> 판정은 §5 를 따른다. 이 표는 레이아웃 모델 기여(std layout vs +layout)를 가르는 용도로만 남긴다.

```bash
BEST=finetune_runs/eruku_corpusmax/checkpoint_last.pth
INK=inkspire:finetune_runs/inkspire_p512/lora_last,finetune_runs/inkspire_layout/checkpoint_last.pth
for F in assets/fonts/ref assets/fonts/test assets/custom_hw_fonts; do for CK in $BEST $INK; do
  GPU=2 ./eval.sh cer --ckpt $CK --fonts-dir $F --n 300 --coherent --binarize 200 --seed 0 \
      --style-ref-text "다람쥐 헌 쳇바퀴에 타고파" --out finetune_runs/_eval/inkspire_$(basename $F)_${CK%%:*}_sref.txt
  GPU=2 ./eval.sh cer --ckpt $CK --fonts-dir $F --n 300 --coherent --binarize 200 --seed 0 \
      --out finetune_runs/_eval/inkspire_$(basename $F)_${CK%%:*}_echo.txt
done; done
```

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

이 리더로 본 관찰 (리더 착시를 걷어내도 방향이 유지되는 것만):
1. **레이아웃 모델이 오히려 손해다.** 6 개 비교쌍 전부 std layout 이 같거나 낫다(held-out style-ref 0.197 vs 0.222).
   단 한 줄씩 생성하는 이 프로토콜에서만 — §3 진단 참고.
2. **체크포인트 선택이 손실과 어긋난다.** val loss 최저는 s20000 이지만 CER 최저는 s13000.
3. ~~굵은 폰트 약점(BagelFatOne 0.423)~~ 은 리더 착시. 붓글씨류(NanumBrushScript) 약점과 숫자 오류는 §5 에서도 남는다.

몽타주: `finetune_runs/inkspire_p512/_eval/`(s20000 + 레이아웃), `_eval_best/`(s13000 + std layout),
`_eval_pangram/`(held-out 폰트 + 한글 pangram ref, `experiments/gencmp_split.py`) — 마지막이 공정한 조건이고 육안으로 Eruku 와 엇비슷하다.

## 5. Eruku 비교 — PP-OCRv6 재측정 (2026-10-20)

> ★ **2026-10-23 정정:** 이 절~§9 의 Eruku 는 **cfg 1.0 · 옛 디코딩**이다(§10 첫 표). 새 디코딩 + 권장 cfg 1.5 로 같은 표본을 다시 재면
> Eruku CER **0.010**(한글 0.8%, 숫자 1.9%, 기호 5.8%), HWD 1.117, FID 26.8 — cfg 1.0 새 디코딩은 0.018 / HWD 1.007 로 옛 것과 같다
> (디코딩 변경은 ref 전사를 줄 때 영향 없음). 결과 `finetune_runs/_cer/eruku_ctx_cfg{1.0,1.5}`.

독립 리더 PP-OCRv6(`HERIUN/PP-OCRv6-medium-rec-ko-printed-handwriting`, 원본 repo `.venv-paddle`). §4 와 같은 표본
(held-out 16폰트, style-ref pangram, coherent, n=300, seed 0). 경로는 `experiments/cer_dump.py` docstring:
`plan`(원본 repo env) → `gen`(여기) → `eval/ppocr_rec.py --save-preds`(원본 repo). 결과 `finetune_runs/_cer/*`. 분해·해석은 `inkspire.md` §9.6.

| 조건 | 자체 HTR 중앙값 | PP corpus CER | PP 평균 / 중앙값 | 한글 | 숫자 | 기호 |
|---|---|---|---|---|---|---|
| GT 렌더 (리더 바닥) | 0.026 | 0.006 | 0.005 / 0 | | | |
| Eruku corpusmax s45000 | 0.000 | **0.018** | 0.021 / 0 | 1.9% | 2.5% | 2.7% |
| InkSpire s13000 std layout | 0.190 | 0.040 | 0.045 / 0 | 3.5% | 12.7% | 8.9% |
| + `tight_mask` | — | 0.034 (부호검정 p=0.33) | 0.043 / 0 | 2.8% | 7.0% | 8.9% |
| + 전사 없음 (`--style-text none`) | — | 0.720 (134/300 백지) | 0.759 / 1.0 | 75.6% | 62.7% | 78.1% |
| + Xc 첫 줄에 목표 텍스트 | — | 0.355 | 0.466 / 0.409 | 40.3% | 33.5% | 40.2% |
| + Xc 첫 줄에 ref 의 PP-OCR 판독 | — | 0.041 | 0.046 / 0 | 3.6% | 12.0% | 8.9% |

- 자체 HTR 중앙값 0.190 → PP 중앙값 0. "크게 진다"는 리더 착시였지만 **corpus CER 은 여전히 Eruku 가 2배 낫다.**
- 격차는 숫자·기호·붓/펜글씨와 한글 중성(모음 짧은 획)에 몰린다.
- **ref 전사가 필수다.** 이미지 모델이 Xc ref 줄을 글자별 대응표로 쓴다 — 비우면 붕괴, 목표 텍스트로 채우면 ref 글자 모양이 섞인다.

## 6. 논문 수치로 본 Eruku 대비 (IAM Lines, 2026-10-22)

InkSpire 논문 표 2 에 Eruku/Emuru 행은 없지만 **baseline 4행이 Eruku 논문(arXiv 2510.23240, WACV 2026) 표 3 IAM Lines 와 숫자까지 같다**
(HWT FID 44.72 / HWD 2.97 / ΔCER 0.33, One-DM 43.89 / 2.83 / 0.13, DiffPen 12.89 / 2.13 / 0.03,
InkSpire 의 "VATr" 34.00 / 2.38 / 0.03 = Eruku 표의 **VATr++** 행). 같은 Pippi et al. 평가 프로토콜 수치를 옮겨 쓴 것이라 한 표로 이을 수 있다.

| IAM Lines | FID↓ | HWD↓ | ΔCER↓ | IAM 학습 |
|---|---|---|---|---|
| DiffPen | 12.89 | 2.13 | 0.03 | ✓ |
| Emuru | 13.89 | 1.87 | 0.14 | ✗ (zero-shot) |
| Eruku (γ=1.25) | 16.40 | 1.70 | 0.06 | ✗ (zero-shot) |
| **InkSpire** | **7.92** | **0.62** | **0.01** | ✓ |
| Eruku 한글 파인튜닝 (측정, 원본 `eval/iam_shtg.py` ours, γ=1.25, w/ T_s) | 20.00 | 1.62 | −0.03 | ✗ |
| InkSpire 한글 (이 repo, 측정 2026-10-26, 같은 채점기·같은 ref) | 18.58 | 1.68 | 0.33 | ✗ (zero-shot) |

- **논문 조건에서는 InkSpire 가 전 지표 우세**, 특히 HWD(스타일) 0.62 vs 1.70 로 차이가 크다.
- **같은 IAM 을 우리 두 한글 모델로 직접 재면(전체 2,925줄, `experiments/iam_dump.py`) InkSpire 의 우세는 사라진다** — HWD 1.68 vs 1.62, FID 18.58 vs 20.00 로 비슷하고
  ΔCER 0.33 vs −0.03, BFID 15.41 vs 4.73 로 Eruku 가 앞선다. 논문 InkSpire 의 우세는 IAM 학습 + 글자 bbox 주석에서 온다. 결과 `finetune_runs/_iam/inkspire_klein_lines`.
- 단 조건이 다르다: (a) InkSpire 는 IAM 으로 학습, Eruku 는 IAM 미학습(zero-shot). (b) InkSpire 는 ref 의 전사 + 글자 bbox 정답 주석을 받는다.
  (c) InkSpire 의 IAM 추론은 같은 writer 의 둘째 줄로 첫 줄을, 첫 줄로 나머지를 만든다(논문 §4.3) — 같은 문단의 실제 줄이 ref 다.
- 우리 한글 재현(§5)과 방향이 반대인 이유 후보: 학습 분포 안(IAM seen) vs 밖(held-out 폰트), FID/HWD 는 스타일 지표이고 우리는 CER 만 쟀다,
  P=1024 vs 512. 한글 HWD 비교는 §9 — 한글 조건에선 Eruku 가 앞선다.

## 7. 이미지 백본 FLUX.3-Image-Klein-Base — `black-forest-labs/flux-3-image-klein-base` (2026-10-21~)

HF [`black-forest-labs/flux-3-image-klein-base`](https://huggingface.co/black-forest-labs/flux-3-image-klein-base)(FLUX.3-Image-Klein-Base, 5.54B, 게이트 repo · 비상업 라이선스)에
같은 LoRA 학습을 얹는다. `train/inkspire.py --backbone klein`, 모듈 `models/inkspire_klein.py`(models/inkspire.py 와 같은 API).
`infer/inkspire.py backbone()` 이 LoRA 키(`mode_blocks`)로 FLUX.3-Image-Klein-Base 를 자동 판별하므로 `cer_dump.py`·`show` 는 그대로 쓴다.

| 항목 | FLUX.1-Fill-dev (§2) | FLUX.3-Image-Klein-Base |
|---|---|---|
| 조건 주입 | 채널 concat (X·(1−M), M, Xc) | 별도 `image_cond` 스트림(timestep 0): cond1 = X·(1−M)(마스크 영역 회색, l=−1), cond2 = Xc (l=−2) |
| 위치 | R-APE (회전) | 생성·조건 토큰이 같은 (h, w) id → **APE 가 기본 구조**. 회전 없음 |
| 텍스트 인코더 | 빈 프롬프트 캐시 | Qwen3-VL-4B 빈 프롬프트 캐시(`model_zoo/klein_empty_prompt.pt`) |
| 샘플링 | guidance 30 | shift 6.93, guidance 임베딩 없음(CFG 미사용). 매 step 비마스크 토큰을 (1−σ)x0+σz 로 덮어씀 |
| LoRA | r32, 115.9M | r32, 71.2M (image/image_cond mode 블록 + single 블록 attn·ff, x_embedders) |
| 학습 | 2.6 s/step, 30.6GB | 0.27 it/s(GPU 경합), 18GB, 13k step ≈ 13h |

**VAE 상한(`experiments/vae_recon.py`, GT 렌더 왕복, PP corpus CER, n=300):** VAE 는 병목이 아니다 — 둘 다 GT 와 같다.

| | GT | FLUX.1-Fill-dev VAE | FLUX.3-Image-Klein-Base VAE (AutoencoderKLFlux2) |
|---|---|---|---|
| 줄높이 64 | 0.0056 | 0.0061 | 0.0062 |
| 줄높이 32 | — | 0.0079 | 0.0079 |

환경: `Flux3Transformer2DModel` 때문에 diffusers git main(0.42.0.dev0) + transformers 5.20.2 로 올렸다. FLUX.1 출력은 bf16 잡음 수준(평균 0.5/255)만 바뀐다.

**학습** (`finetune_runs/inkspire_klein_p512`, P=512 batch 4 Prodigy, §2 와 같은 데이터·마스크):

| step | 500 | 2000 | 6000 | 10000 | 13000 |
|---|---|---|---|---|---|
| val loss | 0.119 | 0.106 | 0.097 | 0.094 | 0.092 |

latent 공간이 달라 FLUX.1 val loss 와 직접 비교 불가. s500 부터 4폰트 모두 스타일 재현, 남는 오류는 FLUX.1 과 같은 유형(모음 짧은 획·숫자:
"컬럼비아→걸럼비아", "210,679→210,6D9").

**평가 (§5 와 같은 표본·리더):**

| 조건 | PP corpus CER | PP 평균 / 중앙값 | 한글 | 숫자 | 기호 |
|---|---|---|---|---|---|
| FLUX.1-Fill-dev s13000 (§5) | 0.040 | 0.045 / 0 | 3.5% | 12.7% | 8.9% |
| FLUX.3-Image-Klein-Base s6000 | 0.039 | 0.049 / 0 | 4.2% | 1.9% | 3.6% |
| FLUX.3-Image-Klein-Base s13000 | **0.026** | 0.028 / 0 | 2.5% | 4.4% | 1.3% |
| (참고) Eruku corpusmax s45000 (§5) | 0.018 | 0.021 / 0 | 1.9% | 2.5% | 2.7% |

결과 `finetune_runs/_cer/klein_s{006000,013000}_test_sref`, 분해 `experiments/cer_analyze.py`. 폭주율 0%, GT 바닥 0.0056.

- **같은 step 수(13k)에서 FLUX.1-Fill-dev 대비 CER 0.040 → 0.026(−35%).** 개선은 숫자(12.7 → 4.4%)·기호(8.9 → 1.3%)·라틴(23.5 → 0%)에
  몰렸다 — FLUX.1 의 약점 그대로였던 곳. 기호는 Eruku(2.7%)보다 낫다.
- **한글은 3.5 → 2.5% 로 Eruku(1.9%)와 아직 격차.** 틀린 자모는 여전히 중성 41%(원→윈, 중→증, 로→르) — 모음 짧은 획 약점은 백본을 바꿔도 남는다.
- 폰트별: 붓·펜글씨가 크게 좋아졌다(NanumBrushScript 0.173 → 0.029, NanumPenScript 0.053 → 0.028). 대신 EastSeaDokdo 0.026 → 0.077,
  Dongle·BlackHanSans 0 → 0.03~0.04 로 나빠졌다(폰트당 n=11~29 라 잡음 폭 큼).
- s6000 → s13000 에 0.039 → 0.026 으로 내려가서 20k 까지 연장했다(resume, Prodigy 상태 복원) → **20k 는 나빠진다** — 아래.

**20k 연장 (2026-10-23, `ref_width` 켬, 같은 표본):** val loss 는 13k 이후 0.092 로 평평. 두 백본 모두 20k 가 13k 보다 **유의하게 나쁘다.**

| | 13k CER | 20k CER | 20k vs 13k 표본별 나음 / 나쁨 | HWD 13k → 20k |
|---|---|---|---|---|
| FLUX.3-Image-Klein-Base | **0.023** | 0.034 | 30 / 71 (p<0.001) | 0.929 → 0.909 |
| FLUX.1-Fill-dev | 0.026 | 0.038 | 34 / 70 (p<0.001) | 0.905 → 0.852 |

20k 에서 한글이 1.9 → 3.1%(Klein), 2.1 → 3.1%(FLUX.1) 로 나빠지고 HWD 는 조금 좋아진다 — 필체를 더 닮는 대신 획이 흐트러진다.
FLUX.1 에서 자체 HTR 로 본 "13k > 20k"(§2)는 리더 착시가 아니었다. **최종 체크포인트는 두 백본 모두 13k.**

## 8. ref 보강 — inference loop (2026-10-22)

### 왜

ref 는 한 줄("다람쥐 헌 쳇바퀴에 타고파", 11음절)뿐이라 목표 글자 대부분은 ref 에 없다. 가설: **ref 에 없는 글자라서 스타일 복제가
약하다.** 그렇다면 목표를 만들기 전에 보강 줄을 하나 더 깔아 필체 예시를 늘리면 나아질 것이다. 실사용에서 보강 줄은 모델이
먼저 생성한 결과로 만든다(반복 추론 = inference loop).

기존 결과가 가설을 지지했다 — FLUX.3-Image-Klein-Base s13000 에서 ref 에 **있는** 음절은 0.5%, **없는** 음절은 2.6% 틀린다(5배).
단 Eruku 도 0.3% vs 2.0% 로 같은 경향이라, ref 음절이 원래 쉬운 글자라서 생긴 차이가 섞여 있을 수 있었다.

### 무엇을 했나

`InkSpireGen.prepare(..., ctx=[(이미지, 전사[, 글자 x 상자])])` 로 ref 아래에 보이는 줄을 추가하고 목표 줄을 그만큼 내린다
(ctx 없으면 HEAD 와 픽셀 동일 확인). 실행은 `experiments/cer_dump.py gen --ctx {fixed,adapt} --ctx-src {oracle,boot}`.
보강 줄을 두 축으로 나눠 2×2 조건을 돌렸다.

| | 보강 줄 내용 |
|---|---|
| **fixed** | 목표와 무관한 고정 문장 `2024년 1~3월 원로 중 57.8%는 '꿈'을 이뤘다?! (69)` — 숫자 10종·기호·중성 오류 잦은 음절(원·로·중·는) |
| **adapt** | 목표 음절 중 ref 에 없는 것(등장 순, 중복·공백 제거). 예: "3억원이 전달된 혐의를…" → `3억원이전달된혐의를…` |

| | 보강 줄 이미지 | 의미 |
|---|---|---|
| **oracle** | 그 폰트로 렌더한 진짜 필체 + 진짜 글자 x 상자 | 실사용 불가. 보강이 완벽할 때의 **상한** |
| **boot** | 모델이 ref 만 보고 먼저 생성한 것 | 실사용 가능한 inference loop |

![입력 캔버스](img_inkspire_io/14_ctx_canvas.png)

모델 입력 [X | Xc | 마스크(검정=생성)]. 위 = 기준선, 가운데 = adapt-boot, 아래 = fixed-oracle.
캔버스 폭은 목표 줄에 맞춰 자르므로(§2 `trim_ref`, 빈 공간 생성 방지) 긴 보강 줄은 목표 폭에서 잘린다.

**기대:** oracle 은 기준선보다 확실히 낫고(특히 ref 에 없는 음절·숫자), boot 는 그 사이.
oracle 조차 효과가 없으면 이 방향은 접는다(사전 결정).

### 중간에 발견한 것 — X↔Xc 상자 어긋남

처음 oracle 은 보강 줄의 Xc 상자를 ref 처럼 **표준폰트 bbox 를 줄 폭에 맞춰 늘린 근사**로 줬는데, 기준선보다 나빴다(adapt 0.036, fixed 0.047).
진짜 필체는 글자 폭이 표준폰트와 달라 상자가 어긋난다:

| 줄 | 평균 중심 어긋남 | 반 글자 이상 어긋난 글자 |
|---|---|---|
| ref | 0.05 글자폭 | 0% |
| adapt 보강 줄 | 0.08 | 2.1% |
| fixed 보강 줄 (숫자·기호) | 0.22 | **10.4%** |

![상자 어긋남](img_inkspire_io/15_ctx_misalign.png)

어긋난 짝을 모델이 대응표로 배워 이웃 글자 오류("흘러내리다 → 흘러대리다")를 냈다. boot 는 생성이 처음부터 표준폰트 상자에 맞춰
그려지므로 어긋남이 없다. → oracle 에 진짜 글자 x 상자(`render_line` 이 주는 것 = 논문의 GT 레이아웃 자리)를 주고 다시 쟀다.
**교훈: 보이는 줄을 추가할 때는 X↔Xc 짝이 정확해야 한다 — 어긋나면 안 넣느니만 못하다.**

### 결과 (FLUX.3-Image-Klein-Base s13000, held-out 16폰트, n=300, PP-OCRv6, 시드 1개)

| 조건 | CER | 한글 | 숫자 | 기호 | 짧은 목표(≤8자) | 긴 목표 | 기준선 대비 표본별 나음/나쁨 | 부호검정 p |
|---|---|---|---|---|---|---|---|---|
| 기준선 (§7) | 0.026 | 2.5% | 4.4% | 1.3% | 0.033 | 0.025 | — | — |
| adapt-oracle | 0.024 | 2.2% | 2.5% | 4.5% | 0.019 | 0.024 | 58 / 53 | 0.64 |
| adapt-boot | 0.021 | 1.8% | 2.5% | 3.1% | 0.047 | 0.020 | 60 / 57 | 0.78 |
| fixed-oracle | 0.034 | 3.2% | 2.5% | 6.7% | 0.049 | 0.033 | 55 / **88** | **0.006** |
| fixed-boot | 0.028 | 2.6% | **0.6%** | 6.2% | 0.044 | 0.027 | 51 / 65 | 0.19 |
| (상자 근사) adapt / fixed-oracle | 0.036 / 0.047 | | | | | | 57/73 · 38/114 | 0.16 · <0.001 |

결과 `finetune_runs/_cer/klein_s013000_ctx_*` (근사 상자판은 `*_oracle_stdbox`). 각 덤프의 `ctx/` 에 보강 줄 이미지.

![조건별 차이](img_inkspire_io/16_ctx_diff.png)

조건 간 CER 차이가 큰 8표본(차이 큰 것만 골라 실제보다 극적으로 보인다). 보강 줄이 있는 셀은 회색 선 위 = 보강 줄, 아래 = 최종 생성.
fixed 열의 보강 줄은 전체 문장을 한 셀에 넣느라 작게 보일 뿐, 실제 입력에선 목표 폭에서 잘린다.

1. **상한(oracle)이 기준선과 같다.** 목표 글자의 진짜 필체·진짜 상자를 줘도 adapt-oracle 58 나음 / 53 나쁨. 그림에서 보듯 모델은
   보강 줄의 같은 글자를 **베끼고**, 진짜 필체를 베끼면 맞게 나오지만(00285 "제1야당", 00080 "되돌아가다") 원래 맞던 표본이
   새로 틀리는 경우도 비슷하게 많다.
2. **boot 는 1차 생성의 복사다.** 00285 1차 "제L야당" → 최종 "제L야당", 00109 "임무시겨인" 그대로. 최종 = 시드만 다른 1차 생성이므로
   새 정보가 없다. corpus 0.021 은 표본별로 60 / 57(p=0.78) — 긴 문장 몇 개가 만든 잡음이다.
3. **fixed 는 유의하게 손해다**(oracle p=0.006). 숫자는 좋아지지만(4.4 → 0.6~2.5%) 기호가 1.3 → 6~7% 로 나빠진다 —
   보강 줄의 따옴표·괄호·% 모양이 목표의 다른 기호로 번지는 것으로 보인다. 목표와 겹치는 글자가 없으니 베낄 것도 없다.
4. **처음 가설이 틀린 이유:** "ref 에 있는 음절 0.5% vs 없는 음절 2.6%" 는 대부분 ref 음절이 쉬워서였다(oracle 로 확인).
   남은 오류는 모음 짧은 획 같은 **획 정확도**이지 예시 커버리지가 아니다.

**판정: 접는다.** 한계 — 시드 1개. 캔버스 높이가 바뀌면 같은 시드라도 노이즈가 달라 표본의 ~1/3 이 조건과 무관하게 흔들린다.
20% 수준 효과를 가리려면 시드 3개쯤 필요하지만 상한에서 신호가 없어 그 비용을 쓸 근거가 없다. 코드(`ctx`)는 기본값이 꺼져 있어 남겨 둔다
— 다줄·문단 생성(보이는 줄이 여러 개인 캔버스)에 그대로 쓸 수 있다.

재현:
```bash
I=/data/work/dgkang/inkspire_korean; E=/data/work/dgkang/Eruku_korean_finetuning
O=$I/finetune_runs/_cer/klein_s013000_ctx_adapt_boot; mkdir -p $O; cp $I/finetune_runs/_cer/s13000_test_sref/plan.json $O/
cd $I && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$I .venv/bin/python experiments/cer_dump.py gen $O \
    --ckpt inkspire:finetune_runs/inkspire_klein_p512/lora_step_013000 --ctx adapt --ctx-src boot
cd $E && .venv-paddle/bin/python eval/ppocr_rec.py $O/gen --save-preds
python docs/_gen_inkspire_ctx.py      # 그림 14~16 (추론 없음, 덤프만 읽는다)
```

## 9. 스타일 지표 — HWD · FID · BFID · KID (2026-10-22)

### 왜

§6 에서 논문 IAM 수치로는 InkSpire 가 스타일(HWD 0.62 vs Eruku 1.70)에서 크게 앞섰고, §10 결론은 "InkSpire 를 다시 꺼낼 근거는
CER 이 아니라 스타일"이었다. 한글에서도 그런지 잰다.

### 어떻게

`experiments/style_scores.py` — 원본 repo `.venv` 의 `hwd` 패키지(Eruku 논문 표 3 과 같은 구현, 높이 32). 추론 없이 §5·§7 의 덤프를 읽는다.
real = 같은 표본의 GT 렌더, fake = 각 모델 출력, 폴더명 = 폰트(필기자) → HWD 는 폰트별 VGG16 특징 거리의 평균.
하한 기준으로 **std-font**(목표 텍스트를 표준폰트 NanumGothic 으로 렌더 = 스타일 복제 0)를 같이 잰다.

★ **정규화 필수.** 덤프마다 위아래 여백(GT 30% / InkSpire 17% / Eruku 20%)과 배경(InkSpire 249~251, 나머지 255)이 달라
높이 32 로 줄이면 글자 크기·밝기 차이를 FID 가 잡는다 — 정규화 전에는 InkSpire FID(57.7)가 std-font(59.7)와 비슷했다.
모든 이미지(real 포함)를 배경 → 255, 잉크 bbox 크롭, 잉크 높이 15% 여백으로 맞춘다(기본, `--raw` 로 끔).

### 결과 (held-out 16폰트, n=300, 정규화)

| | HWD↓ | FID↓ | BFID↓ | KID↓ | 출력 폭 vs GT (중앙값) |
|---|---|---|---|---|---|
| FLUX.3-Image-Klein-Base s13000 | 1.338 | 45.7 | 24.0 | 0.021 | +23% |
| **FLUX.3-Image-Klein-Base s13000 + `ref_width`** | **0.929** | 37.3 | 16.2 | 0.014 | **7%** |
| FLUX.1-Fill-dev s13000 | 1.306 | 48.2 | 22.3 | 0.024 | +22% |
| **Eruku s45000** | **0.989** | **26.0** | **15.7** | **0.003** | **7%** |
| std-font (스타일 복제 0) | 3.281 | 60.1 | 45.3 | 0.030 | +18% |

결과 `finetune_runs/_cer/style_scores.json`. 정규화 전(`--raw`): HWD 1.58 / 1.54 / 1.35 / 3.40, FID 57.7 / 61.6 / 34.8 / 59.7 (같은 순서).

1. **한글 합성 폰트 조건에서는 스타일도 Eruku 가 앞선다.** 논문 IAM(§6)과 방향이 반대다. 두 InkSpire 백본은 서로 비슷하다 —
   FLUX.3-Image-Klein-Base 는 CER 만 좋아졌고 스타일은 그대로다.
2. **원인 후보: 줄 기하가 표준폰트를 따른다.** std layout 은 목표 줄 글자 상자를 NanumGothic 그대로 두므로 InkSpire 출력의 줄 폭은
   GT 보다 22~23% 넓다 — 스타일 복제 0 인 std-font(18%)보다도 멀다. Eruku 는 자간·글자 폭까지 생성해 7% 다. 글자 모양은 필체를
   닮아도 줄 모양이 NanumGothic 이다. 논문은 이 자리에 레이아웃 모델을 쓰고(§3, 우리는 한 줄 생성에서 CER 손해) IAM 은 GT 레이아웃이 있다.
3. **재학습 없는 보정 `ref_width` 로 줄 폭을 맞추면 HWD 가 Eruku 를 넘는다.** ref 줄은 이미 "손글씨 폭 / 표준폰트 폭" 비율을
   안다(ref 상자를 그 비율로 늘리고 있었다). 같은 비율을 목표 줄 상자 x 에도 곱한다 — `InkSpireGen(ref_width=True)` /
   `inkspire:<lora>,ref_width=1`. **2026-10-22 부터 기본값 켬** — §2~§8 의 수치는 꺼진 상태(재현은 `,ref_width=0`). 출력 폭 오차 23% → 7%(Eruku 와 같음), **HWD 1.338 → 0.929(Eruku 0.989)**, FID 45.7 → 37.3,
   BFID 24.0 → 16.2(Eruku 15.7). FID·KID 는 여전히 Eruku 가 낫다.
   **CER 은 그대로** — 0.026 → 0.023, 표본별 54 나음 / 50 나쁨(p=0.70). 결과 `finetune_runs/_cer/klein_s013000_refwidth`.

![ref_width](img_inkspire_io/17_ref_width.png)

무작위 10표본(`experiments/montage.py --seed 3`). 가운데 두 열 = 기준선 / `ref_width`. 00184 "두리번거리다", 00244 "불완전하다" 처럼
기준선은 자간이 표준폰트만큼 벌어지고, `ref_width` 는 GT 처럼 붙는다.

**`ref_width` 는 CER 도 올린다 — FLUX.1-Fill-dev 에서 크게** (2026-10-23, 결과 `finetune_runs/_cer/{flux1,klein}_s0*_rw`):

| ref_width 켬 | CER | 한글 | 숫자 | 기호 | HWD | FID | BFID |
|---|---|---|---|---|---|---|---|
| FLUX.3-Image-Klein-Base s13000 | **0.023** (끔 0.026, p=0.70) | 1.9% | 3.2% | 2.2% | 0.929 | 37.3 | 16.2 |
| FLUX.1-Fill-dev s13000 | 0.026 (끔 0.040, **83 / 51, p=0.006**) | 2.1% | 11.4% | 8.9% | 0.905 | 37.5 | 16.0 |
| Eruku s45000 (cfg 1.0, 옛 디코딩) | 0.018 | 1.9% | 2.5% | 2.7% | 0.989 | **26.0** | **15.7** |
| **Eruku s45000 (권장 cfg 1.5, 새 디코딩, §10)** | **0.010** | **0.8%** | **1.9%** | 5.8% | 1.117 | 26.8 | 16.2 |

FLUX.1 은 표준폰트 자간 캔버스(학습 분포 밖 — 학습 Xc 상자는 늘 그 폰트의 실제 자간)에 약했던 것이고, 그 몫이 한글 3.5 → 2.1% 로 풀렸다.
남은 FLUX.1 약점은 숫자·기호·라틴(학습 한계) — 거기서 FLUX.3-Image-Klein-Base 가 낫다. Eruku 는 여전히 CER 에서 유의하게 앞선다
(Klein s13000 rw 대비 표본별 79 / 32, p<0.001). 스타일은 HWD 는 InkSpire, FID·KID 는 Eruku.

한계: 비율은 ref 한 줄 전체의 평균이라 글자별 폭 차이(예: 숫자·라틴만 좁은 폰트)는 못 맞춘다 — 그건 레이아웃 모델의 몫이다.
GT 가 폰트 렌더라 "줄 폭 = 폰트 고유 자간" 이 정확히 정해진 조건이다. 실손글씨는 줄마다 자간이 흔들려 이득이 줄 수 있다.

### HWD 하한 — 폰트별 (2026-10-28, `experiments/style_floor.py font`)

std-font 는 상한(스타일 복제 0)이고, 하한 = **같은 폰트로 다른 문구**를 같은 표본 수만큼 렌더해 GT 와 잰 HWD(같은 사람이 다른 글을 쓴 거리).
같은 문구 재렌더 = 0.000 으로 렌더 경로가 GT 와 같음을 확인했다. 모델은 FLUX.3-Image-Klein-Base s13000 + `ref_width`, Eruku cfg 1.5 새 디코딩.

| 폰트 | n | 하한 | InkSpire | Eruku | std-font |
|---|---|---|---|---|---|
| NanumBrushScript | 24 | 0.198 | 1.493 | **1.361** | 3.590 |
| Dokdo | 29 | 0.210 | **1.256** | 1.862 | 4.128 |
| Gaegu | 22 | 0.277 | 1.138 | **0.917** | 3.609 |
| Youth | 24 | 0.285 | **0.458** | 0.542 | 1.561 |
| DoHyeon | 13 | 0.289 | **0.938** | 1.543 | 2.421 |
| SongMyung | 19 | 0.293 | **0.624** | 0.735 | 3.134 |
| BlackHanSans | 15 | 0.295 | **0.669** | 1.473 | 3.426 |
| KirangHaerang | 21 | 0.335 | **1.091** | 1.105 | 3.359 |
| NanumPenScript | 21 | 0.341 | 1.309 | **1.060** | 3.877 |
| BagelFatOne | 11 | 0.361 | **0.714** | 1.501 | 3.524 |
| CuteFont | 17 | 0.368 | **0.772** | 1.044 | 3.985 |
| EastSeaDokdo | 17 | 0.371 | **0.835** | 1.129 | 3.986 |
| GowunBatang | 20 | 0.392 | 0.816 | **0.589** | 1.248 |
| HiMelody | 15 | 0.445 | **0.986** | 1.182 | 3.490 |
| Sunflower | 13 | 0.468 | **0.670** | 0.890 | 2.888 |
| Dongle | 19 | 0.481 | 1.100 | **0.943** | 4.277 |
| **평균** | 300 | **0.338** | **0.929** | 1.117 | 3.281 |

- 하한 0.34(폰트별 0.20~0.48). [하한, 상한] 을 0~1 로 보면 InkSpire 0.20 / Eruku 0.27 — 둘 다 하한까지 0.6 이상 남았다.
- 폰트별 InkSpire 11 / Eruku 5 승. InkSpire 는 굵은 디스플레이체(BlackHanSans·BagelFatOne·DoHyeon), Eruku 는 붓·펜 손글씨체
  (NanumBrushScript·NanumPenScript·Gaegu)와 가는 명조(GowunBatang)에서 낫다. InkSpire 가 하한에서 가장 먼 것도 붓·펜체(+0.97~1.30).
- ★ 측정 함정: 폰트 안에서 GT 를 한 칸 밀어 fake 로 쓰면 같은 이미지 집합이라 평균 특징이 같아 0 이 나온다.
  split-half(폰트별 GT 짝/홀 절반끼리)는 0.541 — 절반 n 의 표본 잡음만큼 높다. 실손글씨 하한(§10)은 이 방식이라 그만큼 감안해서 읽는다.
결과 `finetune_runs/_cer/style_floor_{font,split}.json`.

## 10. 실손글씨 — HF `HERIUN/korean-handwriting` 미러링 · 교차 생성 · 자유 문구 (2026-10-23)

### Eruku 추론 설정 정정 (먼저 읽을 것)

이 절과 §5·§9 의 Eruku 수치는 처음에 **원본 repo 권장 추론과 다른 설정**으로 쟀다. 원본 README·`experiments/mirror_compare.py` 기준:

| | 원본 권장 | 처음 쓴 것 |
|---|---|---|
| cfg | 한글 **1.5~2.0** (1.0 = CFG 끔 → 흐림·백지) | 1.0 (`eval/htr_cer.py` 평가 기본값) |
| 실손글씨 ref 크롭 | 공식 `ink_bbox_crop`(solid 테두리 벗기기 + 밀도 bbox, 여백 4px) — 여백이 latent std 를 6~29배 튀겨 runaway/백지 | 자체 크롭 + 잉크 높이 15% 여백 |
| VAE 디코딩 | `d3918e0`(2026-10-23) `gen_only="ctx"` — style+gen 을 같이 디코딩하고 미학습 SOG latent 만 뺀다 | 폰트 렌더 덤프(10-20)는 옛 디코딩 |

체크포인트(corpusmax s45000 = HF 릴리즈), `drop_text=True`, `style_text`, SOG 기준 gen 크롭, 64px bilinear style 텐서는 맞았다.
**아래는 전부 공식 설정(공식 크롭, 새 디코딩, Eruku cfg 1.5)** 이다. 이전 덤프는 `finetune_runs/_realhw/*_{raw,crop,prep}`, `eruku_cfg{1.0,2.0}` 에 보관.

### 무엇을

데이터: 비공개 338장 — `hw_line` 20(한 사람의 문장 줄), `seosik_hw` 286(서식 손글씨 칸 크롭, 73% 숫자·기호만), `sanrim_hw_pseudo` 32.
`experiments/realhw_dump.py`(`infer.show` 공통 API 만 써서 두 repo env 에서 같은 스크립트) 세 모드:
- **미러링(echo):** 실제 이미지 + 전사를 ref 로 같은 전사를 쓰게 한다 — ref 에 같은 글자가 있어 베끼기 상한.
- **교차(`--cross`):** 같은 필기자의 **다른** 표본 문구를 쓰게 하고 그 표본의 실제 이미지가 GT(논문 IAM 프로토콜과 같은 꼴) — 실제 생성 능력.
- **자유(`--free`):** 데이터에 없는 문구(한글 5 + 영어 2)를 ref 7개로 — 정답 없음, 육안.
필기자 키 = hw_line 한 명, 서식 = 문서. 모델 = InkSpire(FLUX.3-Image-Klein-Base s13000 + `ref_width`) vs Eruku s45000.
**개인정보가 섞인 재배포 금지 데이터라 그림은 문서에 넣지 않는다** — 몽타주는 `finetune_runs/_realhw/montage_{,cross_}{hw_line,random}.png`, `free_grid.png`.
PP-OCRv6 리더가 이 이미지로 학습됐을 수 있어(데이터 README) "실제 이미지" CER 은 낙관적이다.

### Eruku cfg (미러링, 공식 크롭)

| subset | InkSpire | Eruku cfg 1.0 | **cfg 1.5** | cfg 2.0 |
|---|---|---|---|---|
| hw_line | 0.185 | 0.066 | **0.069** | 0.403 |
| seosik_hw | **0.365** | 2.027 (백지 30·폭주 49) | 0.897 (백지 7·폭주 19) | 0.692 (백지 1·폭주 16) |
| 전체 | **0.349** | 1.676 | 0.755 | 0.653 |

cfg 1.5 = 문장 최적이자 README 권장 → 이후 고정. 2.0 은 서식 폭주는 줄지만 문장이 무너진다.

### 결과 (공식 설정)

| | subset | 실제 이미지 | InkSpire CER | Eruku CER | 표본별 InkSpire / Eruku | HWD InkSpire / Eruku |
|---|---|---|---|---|---|---|
| 미러링 | hw_line | 0.133 | 0.185 | **0.069** | — | **1.41** / 1.45 |
| | seosik_hw | 0.129 | **0.365** | 0.897 | — | **0.99** / 1.79 |
| | 전체 | 0.136 | **0.349** | 0.755 | — | **1.06** / 1.77 (FID 53.2 / 53.1) |
| **교차** | hw_line | 0.133 | 0.246 | **0.014** | 0 / **19** | **1.35** / 1.51 |
| | seosik_hw | 0.129 | **0.417** | 0.771 (백지 7·폭주 11) | 134 / 101 | **1.20** / 1.83 |
| | 전체 | 0.136 | **0.398** | 0.667 | 145 / 136 | **1.31** / 1.81 (FID 59.6 / 65.8) |

1. **깨끗한 문장은 Eruku 가 압도적이다** — 다른 문구를 써도 CER 0.014(20장 중 19장 승). InkSpire 는 모음·받침 오류(넓은 → 넒은)가 남는다.
2. **필체 거리(HWD)는 모든 subset 에서 InkSpire 가 낫다.** 몽타주로 보면 InkSpire 는 가는 펜 획·농담을, Eruku 는 더 진하고 또렷하게
   정리된 글씨를 낸다(원본 README 의 "VAE 가 필압·농담을 못 살린다"와 같은 관찰).
3. **서식 크롭(짧은 숫자·표 선 잔재·흐린 획)에선 Eruku 가 공식 설정에서도 5% 가량 백지·폭주** — InkSpire 는 백지 0~2. corpus CER 은 InkSpire
   가 낫지만 표본별로는 134 / 101 로 차이가 corpus 만큼 크지 않다(Eruku 실패가 폭주라 corpus 를 크게 키운다).
4. **자유 문구:** 깨끗한 ref·숫자 ref("700,000")에서 Eruku 는 한글·날짜·금액·영어 전부 거의 완벽하고 ref 의 밑줄 습관까지 옮긴다.
   지저분한 ref 3개에선 흐리거나 백지. InkSpire 는 모든 ref 에서 쓰지만 글자 오류가 더 많다. 영어는 둘 다 쓴다.

**판정:** 실사용 정확도는 Eruku(권장 설정). InkSpire 의 장점은 필체 질감 재현(HWD)과 지저분한 ref 견고성으로 좁혀진다.

### InkSpire 는 왜 실손글씨에서 획 정확도가 무너지나 (진단)

CER 이 폰트 렌더 0.023 → 실손글씨 문장(교차) 0.246 으로 10배가 된다. Eruku 는 0.010 → 0.014 로 그대로다.

| 후보 | 실험 | 결과 | 판정 |
|---|---|---|---|
| 실제 획이 가늘고 연하다 | hw_line 교차 20장, ref 획을 3×3 erode 로 굵히고 잉크를 진하게(`realhw_dump.py --stroke 3`) | 0.246 → 0.285 | ✗ |
| 출력이 흐려 OCR 이 불리하다 | 출력·실제 이미지를 Otsu 이진화 후 재채점 | hw_line 0.246 → 0.276, 서식 0.417 → 0.434, 폰트 렌더 0.023 → 0.027 (Eruku 0.014 → 0.011, 0.010 → 0.012) | ✗ |
| 읽기 어려운 필체를 물려받는다 | 표본별 ref 실제 이미지 CER 과 출력 CER 의 Spearman (교차 338장) | InkSpire +0.06, Eruku +0.04. ref 가 잘 읽히는 표본에서도 InkSpire hw_line 0.26 | ✗ |
| **불규칙한 자간 → X↔Xc 대응표 어긋남** | 폰트 렌더 ref 의 획·필체는 그대로, 글자 간격만 advance·U(−j, j) 로 흔든다(`experiments/diag_spacing.py`, §5 표본 100개) | j = 0 / 0.3 / 0.6 → **0.024 / 0.037 / 0.076** | △ 있지만 작다 — 과하게 흔든 ±60% 도 격차(+0.22)의 ~1/4. 실제 hw_line 어긋남은 대부분 반 글자 미만이라 기여는 더 작다 |
| **실제 글자 모양이 학습 분포(폰트 12,951종) 밖** | (위를 빼고 남는 몫) | 자간을 크게 흔든 폰트 ref 도 0.076 — 실손글씨 0.246 에 못 미친다 | ✓ **주원인(격차의 ~3/4 이상, 소거법 추정 — 직접 측정은 실손글씨 학습이 필요)** |

![자간 흔들기](img_inkspire_io/18_diag_spacing.png)

각 3행 = 자간 흔들기 0 / ±30% / ±60% 의 [ref | 생성]. 흔들수록 "2개월도 → 2개올도 → 재월도" 처럼 이웃 글자가 섞이고 ref 의 불규칙한 자간이
출력으로 옮겨 간다. 실제 hw_line ref 를 Xc 상자와 겹쳐 보면 띄어쓰기 위치가 표준폰트와 다른 곳("새벽 첫 기차를")에서 상자가 반 글자쯤 밀린다.

**흐릿함이 HWD 를 부풀리는가 — 아니다.** 실제 이미지·두 출력을 모두 이진화(농담·흐림 제거, 획 모양·배치만 남김)해도 InkSpire 가 앞선다:

| HWD↓ | 원본 InkSpire / Eruku | 이진화 InkSpire / Eruku | 이진화 FID InkSpire / Eruku |
|---|---|---|---|
| 폰트 렌더 | 0.93 / 1.12 | **0.77** / 1.06 | 16.2 / 16.3 |
| 실손글씨 문장(교차) | 1.35 / 1.51 | **1.18** / 1.37 | 84.2 / **74.2** |
| 실손글씨 서식(교차) | 1.20 / 1.83 | **1.15** / 1.88 | **33.3** / 49.3 |

결과 `finetune_runs/_bin/`, `finetune_runs/_cer/diag_spacing_j*`, `finetune_runs/_realhw/diag_ink_stroke*`.

**해석:** 같은 성질의 양면이다. InkSpire 는 Xc ref 줄을 글자별 대응표로 써서 ref 의 글자 **모양을 베낀다**(§8) — 그래서 형태(HWD)를
더 닮지만, ref 모양이 학습 분포 밖이면(주원인) — 또는 자간이 불규칙해 짝이 어긋나면(작은 몫) — 틀린 모양도 그대로 옮긴다. Eruku 는 ref 를 학습된 글씨체 공간으로
끌어와 다시 쓰므로 정확·선명하지만 형태가 평균화된다(HWD↑).
개선 방향(미검증): 실손글씨 학습 데이터, ref 의 글자 bbox 를 실제로 추정해 대응표를 맞추기(단 §3 의 이미지 분할 bbox 는 실패했다).

### HWD 하한 — 실손글씨 (2026-10-28, `experiments/style_floor.py split`)

실손글씨는 다시 렌더할 수 없어 **필기자(서식 = 문서)별 GT 를 겹치지 않는 두 절반으로 나눈 HWD** 를 하한으로 쓴다(교차 338장의 GT).
절반 n 이라 표본 잡음만큼 높게 나온다 — 같은 방식으로 폰트(§9)를 재면 0.541(제대로 잰 하한 0.338). 모델 값은 위 결과 표(전체 n).

| subset | 필기자 | 하한 (split-half) | InkSpire | Eruku |
|---|---|---|---|---|
| hw_line | 1 (20장) | **0.29** | 1.35 | 1.51 |
| seosik_hw | 14 | 1.07 (문서별 0.44~2.13 — 4~5장 문서가 1.9~2.1 로 끌어올림) | 1.20 | 1.83 |
| sanrim_hw_pseudo | 3 | 1.58 | — | — |

- 문장(hw_line)은 하한 0.29 에 비해 InkSpire 도 +1.06 — "스타일은 하한에 가깝다" 가 아니다.
- 서식은 하한 자체가 표본 잡음에 묻혀 판정 불가. InkSpire 가 그 근처, Eruku 는 확실히 멀다.
- ★ 측정 함정: `--cross` 덤프의 `ref/` 를 fake 로 쓰면 안 된다. ref 338장 전부가 다른 행 GT 와 **같은 원본**(103장은 픽셀 동일, 나머지는 크롭만 다름)이라
  거의 같은 이미지 집합끼리 재는 셈이다(그렇게 재면 0.339 / hw_line 1.00 / 서식 0.29 — 의미 없음).
결과 `finetune_runs/_realhw/style_floor_split.json`.

### 실손글씨 오류는 무작위인가 — 시드 3개 (2026-10-28, `experiments/seed_errors.py`)

§12 는 폰트에서 "오류가 시드마다 다른 글자" 였다. hw_line 교차 20장(284자)을 시드 0/1/2 로 다시 생성(최종 설정, `realhw_dump.py --cross --subset hw_line --seed s`):

| | 실손글씨 hw_line | (참고) 폰트 §12 |
|---|---|---|
| 한 번 생성 CER (공백 무시) | 0.275 (PP corpus 0.246 / 0.251 / 0.251) | 0.020 |
| **정답 아는 best-of-3 (상한)** | **0.169 (−39%)** | 0.0039 (−80%) |
| 틀린 자리가 1 / 2 / 3 시드에서 틀림 | 72 / 41 / **27** | 72 / 9 / **0** |
| 실제 이미지 판독 CER (리더 바닥) | **0.133** | 0.007 |

![시드 몽타주](../finetune_runs/_realhw/seeds_hw_line/montage_seeds.png) ← 개인정보 데이터라 저장소 밖(gitignore) 경로. 열 = ref | GT | seed0 | seed1 | seed2.

1. **무작위 몫이 있다** — best-of-3 만으로 0.275 → 0.169. CFG·best-of-N 이 먹을 여지.
2. **폰트엔 없던 일관 오류가 있다** — 세 시드 모두 같은 글자를 틀린 곳 27. 그중 9(33%)는 **실제 이미지도 리더가 오판독**("마셨다 → 마였다")
   — 읽기 어려운 필체 모양을 충실히 베낀 것이라 스타일 재현과 정면으로 부딪힌다.
3. **기준점:** 리더가 실제 이미지조차 0.133 틀린다. best-of-3 상한 0.169 는 그 바닥 근처 — 무작위 몫을 다 없애도 "원본만큼 읽힘" 이 한계이고,
   Eruku(0.014)는 원본보다 읽기 쉬운 정형 글씨로 다시 쓰기 때문에 바닥보다 낮다.
4. 육안: 짧은 줄은 세 시드 모두 대체로 멀쩡하고, 긴 줄(00014 "할머니께서 정성껏…", 00015 "창밖으로 참새…")은 세 시드 모두 중간이 뭉개진다.
   "긴 ref 가 가로로 눌려 대응표가 촘촘해진다" 는 가설은 **틀렸다** — `prepare` 는 비율 유지로 잉크 높이만 맞추고(캔버스 ≤ 656 < MAX_W 2048),
   글자당 ref 폭(29~37px)·목표 상자 폭(23~28px)이 줄 길이와 무관하다. 긴 줄이 무너지는 이유는 미확인.
결과 `finetune_runs/_realhw/seeds_hw_line/s{0,1,2}`(gt 판독 포함).

### 내용 CFG — 재학습 없이 실손글씨 CER 0.25 → 0.075 (2026-10-28)

`inkspire:<lora>,cfg=s` — v = v_u + s·(v_c − v_u), v_u = **Xc 전체 빈 이미지**(`models/inkspire_klein.xc_uncond`). 같은 hw_line 교차 20장, 시드 3개.

| LoRA · cfg | PP CER 시드 0/1/2 | 한 번(공백 무시) | best-of-3 | 3시드 공통 오류 | HWD↓ (seed0) |
|---|---|---|---|---|---|
| 최종 s13000 (cfg 1) | 0.246 / 0.251 / 0.251 | 0.275 | 0.169 | 27 | 1.35 |
| xcdrop s13000, cfg 1.0 | 0.268 / 0.293 / 0.268 | 0.306 | 0.218 | 35 | 1.32 |
| xcdrop s13000, cfg 1.5 | 0.130 / 0.119 / 0.135 | 0.127 | 0.067 | 12 | 1.43 |
| xcdrop s13000, cfg 2.0 | 0.086 / 0.088 / 0.102 | 0.078 | 0.035 | 8 | **1.48** |
| **최종 s13000, cfg 2.0** | **0.075 / 0.077 / 0.072** | **0.066** | 0.035 | **7** | 1.58 |
| Eruku (cfg 1.5) | 0.014 | | | | 1.51 |
| 실제 이미지 (리더 바닥 / HWD 하한) | 0.133 | | | | 0.29 |

![CFG 몽타주](../finetune_runs/_realhw/xcdrop_hw_line/montage_cfg.png) ← 저장소 밖. 열 = ref | GT | 최종 | xcdrop cfg 1.5 | xcdrop cfg 2.0 | Eruku.

1. **CFG 가 InkSpire 실손글씨 정확도를 1/3 로 줄인다 — 리더 바닥(0.133)보다 낮다.** 무작위·일관 오류가 둘 다 준다(공통 오류 27 → 7).
   §12 의 "InkSpire 에서 CFG 가 CER 을 낮춘다는 측정은 없다" 를 뒤집는다.
2. **xcdrop 학습은 필요 없었다.** 그 LoRA 는 생성 영역 Xc 만 비워 학습해 v_u(Xc 전체 빈 이미지)와 규약도 안 맞는데, dropout 없는 최종 LoRA 도
   같은 cfg 에서 오히려 CER 이 낮다. "빈 Xc 일 때의 예측" 은 학습에 없던 입력이지만 그 반대 방향으로 미는 것만으로 Xc(내용)를 더 세게 따른다.
   cfg 1.0 의 xcdrop 은 최종보다 약간 나쁘다 — 개선은 학습이 아니라 CFG 몫.
3. **대가는 스타일 약간 — 시드 잡음보다 크다.** 시드 3개 HWD: 최종 1.35 / 1.42 / 1.46(평균 1.41) → cfg 2.0 1.58 / 1.54 / 1.59(1.57, +0.16),
   xcdrop cfg 2.0 1.48 / 1.45 / 1.55(1.49, +0.08). 하한 0.29 ~ std-font 3.76 눈금으로 0.32 → 0.37(xcdrop 0.35) — Eruku(1.51, 0.35)와 같은 수준.
   FID 는 오히려 114~118 → 110. 몽타주에선 필체·펜 질감이 그대로다. 추론 2배(조건·무조건 한 배치).
   결과 `style_scores_seeds.json`.
4. **폰트 렌더(§5 n=300, 2026-10-30): Eruku 를 두 지표 모두 앞선다.** `inkspire:<s13000>,ref_width=1,cfg=2.0,uncond=gen`:
   CER 0.023 → **0.0045**(GT 렌더 바닥 0.0056 수준, Eruku cfg 1.5 0.010), HWD 0.929 → 1.015(Eruku 1.117), FID 37.3 → 37.2, BFID 16.2 → 15.9.
   결과 `finetune_runs/_cer/klein_s013000_rw_cfg2gen`, 몽타주 `finetune_runs/_cer/montage_best_vs_eruku_font.png`(폰트당 1, 열 = ref | GT | CFG 없음 | cfg 2 | Eruku).
   **실손글씨 전체 교차 338장(2026-10-30, 같은 설정):** corpus CER 0.398 → **0.168**(문장 0.246 → 0.069 / 서식 0.417 → **0.180** / sanrim 0.444 → 0.208),
   표본별 vs Eruku 204 / 62 승(서식 183 / 43, 문장 2 / 12), 백지 4 → 1. HWD 1.31 → 1.46(문장 1.35 → 1.63 / 서식 1.20 → 1.37, Eruku 1.81).
   서식 몽타주: InkSpire 는 ref 에 남은 **표 선·밑줄도 잉크로 보고 그린다**.
   ★ Eruku 정정: 현재 `finetune_runs/_realhw/eruku_cross` 덤프를 다시 판독하면 서식 0.711 / 전체 0.620 — 위 결과 표의 0.771 / 0.667 은 재현되지 않는다
   (같은 날 덤프를 덮어쓰기 전 판독으로 추정). README 는 재현되는 값으로 바꿨다.
   결과 `finetune_runs/_realhw/inkspire_cross_cfg2gen`, `style_scores_cfg2gen_{all,hw_line,seosik_hw}.json`.
5. **cfg 를 더 올려도 무너지지 않는다 — Eruku 와 다르다**(Eruku 는 문장이 cfg 2.0 에서 0.069 → 0.403). 같은 20장, `uncond=gen`, 시드 3개 평균:

   | cfg | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 |
   |---|---|---|---|---|---|---|
   | PP CER | 0.249 | 0.107 | 0.071 | 0.063 | 0.055 | 0.054 |
   | HWD↓ | 1.41 | 1.57 | 1.60 | 1.65 | 1.67 | 1.68 |

   CER 은 3.0 근처에서 0.055 로 포화, HWD 는 천천히 계속 오른다. 폭주·백지 0, 몽타주상 cfg 2~4 출력은 거의 같다(과포화 흔적 없음).
   캔버스·글자 상자가 고정이라 자기회귀처럼 누적·폭주할 자리가 없다(해석). → 권장값은 2.0 유지(3.0 은 CER −0.016 에 HWD +0.07).
   결과 `finetune_runs/_realhw/cfg_sweep_hw_line/`.
6. **추론 때 표준폰트(Xc)만 바꾸면 — 무너지진 않지만 손해다**(2026-10-30, 학습은 NanumGothic-Regular 그대로, §5 앞 100장, cfg 2.0 · `uncond=gen`,
   `inkspire:<lora>,…,std_font=<ttf>`):

   | 표준폰트 | CER | HWD↓ |
   |---|---|---|
   | NanumGothic-Regular (학습과 같음) | 0.0061 | **1.03** |
   | NanumGothic-Bold | **0.0039** | 1.10 |
   | GothicA1 (다른 산세리프) | 0.0174 | 1.11 |
   | HANBatang (명조) | 0.0109 | 1.09 |
   | (참고) Eruku cfg 1.5 | 0.0091 | |

   - 100장(글자 ~2,000)이라 CER 차이 0.002 는 글자 4개 — Bold 의 이득은 잡음 수준. HWD 는 셋 다 0.06~0.08 나빠진다.
   - GothicA1 오류는 한글이 아니라 **기호·띄어쓰기**다 — 따옴표를 그 폰트의 굽은 모양(″ ’)으로 그리고, 공백 폭이 달라 어절이 붙는다
     (몽타주 `finetune_runs/_cer/stdfont/montage_worst.png`). Xc 글리프 모양이 출력으로 새어 나온다는 뜻.
   → 학습 폰트(NanumGothic-Regular) 그대로 쓴다. 결과 `finetune_runs/_cer/stdfont/`.

**v_u 를 바꿔 HWD 손실을 줄일 수 있나 — 안 된다** (같은 20장, 시드 3개 평균, `inkspire:<lora>,cfg=…,uncond=gen|all,cfg_s=…`):

| LoRA · guidance | PP CER | HWD↓ |
|---|---|---|
| 최종, cfg 1 (CFG 없음) | 0.249 | **1.41** |
| 최종, cfg 2.0, uncond=all (Xc 전체 빈) | 0.075 | 1.57 |
| 최종, cfg 2.0, uncond=gen (생성 영역만 빈, ref 대응표 유지) | **0.071** | 1.60 |
| 최종, cfg 1.5, uncond=gen | 0.107 | 1.57 |
| 최종, cfg 2.0, 분리 guidance cfg_s 3.0 | 0.082 | 1.51 |
| 최종, cfg 2.0, 분리 guidance cfg_s 4.0 | 0.091 | 1.47 |
| xcdrop, cfg 2.0, uncond=all / gen | 0.092 / 0.085 | 1.49 / 1.52 |
| xcdrop, cfg 1.5, uncond=gen | 0.121 | 1.47 |

- `uncond=gen`(v_c − v_u 에서 ref 대응표 몫을 빼 내용만 키움)은 all 과 잡음 차이 — HWD 손실은 ref 대응표를 같이 키워서가 아니라
  **내용 guidance 자체**에서 온다. 목표 Xc 가 NanumGothic 글리프라 "Xc 를 세게 따르기" = 정형 글자 쪽으로 당기기(해석, 미측정).
- 분리 guidance v = v_all + cfg_s·(v_gen − v_all) + cfg·(v_c − v_gen) 의 cfg_s 를 올리면 HWD 가 내려가지만 CER 도 같이 오른다(같은 절충선 위).
  1.41 아래로는 못 간다. → **추론 guidance 는 CER↔HWD 절충선 위의 이동일 뿐**, 둘 다 낮추려면 학습(분포·데이터)을 바꿔야 한다.
결과 `finetune_runs/_realhw/{uncond_gen,cfgs}_hw_line/`.
결과 `finetune_runs/_realhw/xcdrop_hw_line/{c1.0,c1.5,c2.0,base_c2.0}_s{0,1,2}`, `style_scores*.json`.

## 11. VAE 비교 — Eruku vs InkSpire, 그리고 InkSpire 획이 날아가는 이유 (2026-10-26)

### 구조

| | Eruku VAE | InkSpire VAE (FLUX.3-Image-Klein-Base 의 Flux2) |
|---|---|---|
| 클래스 | `AutoencoderKL` (Emuru VAE + 한글 파인튜닝 가중치, corpusmax ckpt 의 `vae.*`) | `AutoencoderKLFlux2` (범용) |
| 입력 → 출력 | RGB 3ch → **흑백 1ch** | RGB 3ch → RGB 3ch |
| latent | **1ch** × H/8 × W/8 (64px 줄 → [1, 8, W/8]) | **32ch** × H/8 × W/8 (patchify 후 16px 당 128) |
| 압축 (픽셀 : 숫자) | **64 : 1** | **2 : 1** |
| 학습 목표 | 스타일 입힌 RGB → 깨끗한 흑백(정규화를 학습) + 한글 HTR loss | 항등 재구성 |
| 생성 모델 학습 중 | 동결(`train/core.py:402`). 그 전에 `train/vae_korean.py` 로 따로 한글 적응 | 동결. 파인튜닝 없음 — LoRA(transformer)만 학습 |

### 한글 획 복원력 (인코딩 → 디코딩만, 64px, `experiments/vae_recon.py --vae flux2|eruku`)

| | PP-OCR CER 원본 → 왕복 | 잉크 IoU 원본↔왕복 | 획 평균 밝기 원본 → 왕복 (0=검정) |
|---|---|---|---|
| 한글 폰트 300장(§5 표본) | 0.0056 → Flux2 **0.0062** / Eruku 0.0232 | Flux2 **0.973** / Eruku 0.879 | 50 → 53 / 33 |
| 실손글씨 문장 20장 | 0.113 → Flux2 **0.102** / Eruku 0.188 | Flux2 **0.958** / Eruku 0.688 | 137 → 141 / 41 |
| IAM 100줄 | (영어 — PP-OCR 불가) | Flux2 **0.958** / Eruku 0.752 | 121 → 123 / 44 |

![VAE 왕복](img_inkspire_io/19_vae_compare.png)

세 줄씩 = 원본 / Flux2 왕복 / Eruku 왕복(위부터 폰트 2 · 실손글씨 문장 2 · IAM 2).

- **Flux2 는 거의 항등이다** — 연한 회색 획까지 그대로. **Eruku VAE 는 획을 진하고 굵게 다시 그리며 모양을 바꾼다**("책을 → 적을", "학교 → 학모",
  IAM 영어 단어 밑에 한글 자모 같은 얼룩 — 한글 HTR 적응의 흔적으로 보인다). 한글 복원 CER 은 Flux2 가 4배 낫다.
- ★ 측정 규약: 원본 `htr_cer --vae-only` 는 폭 > 768 줄을 768 로 가로 압축해 Eruku VAE 를 0.044 로 깎는다(긴 줄 157개). 위 표는 ≤768 은 학습처럼
  768 흰 패딩, > 768 은 압축 없이 8 배수 패딩.
- Eruku **생성** CER(0.010)이 자기 VAE 왕복(0.023)보다 낮다 — 왕복은 임의 폰트 모양을 1ch latent 에 담아야 하지만, 생성은 VAE 가 잘 디코딩하는
  정형 글자를 고르면 된다. 저용량·정규화형 VAE 가 Eruku 의 "정확하지만 평균화된 필체"(HWD↑)를, 고용량·항등형 VAE 가 InkSpire 의 "모양·농담 보존"
  (HWD↓)을 만든다. **InkSpire 의 글자 정확도 문제는 VAE 가 아니라 transformer 쪽이다.**

### InkSpire 획이 끊기고 흐린(날아가는) 이유 — ref 농도를 베낀다

실손글씨·IAM 은 가는 펜 획이라 잉크 높이 48px 로 줄이면 1~2px 회색 선이 된다. 캔버스에 놓인 ref 의 획 평균 밝기: 폰트 33~46 / 실손글씨 137~142 /
IAM 114~156 (진한 획(<128) 비율 84~89% / 39~44% / 15~61%). 학습 ref 는 잉크 농도 0.75~1 의 진한 획뿐이었다.
같은 hw_line 교차 20장에서 ref 획만 3×3 erode 로 굵고 진하게 하면 **출력도 그대로 진해진다** — 출력 획 평균 밝기 151 → 64, 진한 획 24% → 81%.
CER 은 0.246 → 0.285 로 그대로 — **흐림(농도)과 글자 오류(모양)는 별개 문제다.**

![ref 농도](img_inkspire_io/20_stroke_darkness.png)

표본 3개 × [원본 ref | 생성 | 진하게 한 ref | 생성].

## 12. transformer 정확도 분석 — 오류는 샘플마다 무작위다 (2026-10-26)

VAE 는 병목이 아니다(§11). 그럼 FLUX.3-Image-Klein-Base + LoRA 가 왜 틀리나. 대상: §5 표본(test 폰트), 최종 설정(s13000 + `ref_width`).

### 데이터 분석 (300장, Eruku 대조)

| 오류율 | InkSpire | Eruku cfg 1.5 | InkSpire 만의 특징? |
|---|---|---|---|
| 중성이 ref 에 있음 / 없음 | 1.3% / 2.4% | 0.4% / 1.0% | ✗ 둘 다 같은 비율 — ref 에 없는 모음(ㅣㅡㅜㅕ…)이 원래 어렵다 |
| 받침 없음 / 있음 | 1.7% / 2.1% | 0.3% / 1.3% | ✗ (Eruku 가 받침에 더 약함) |
| 단모음 / 겹모음(ㅘㅝㅟㅢ…) | 1.7% / 3.7% | 0.7% / 0.9% | △ 300장에선 InkSpire 만 2.2배, 아래 100장 시드 실험에선 재현 안 됨(1.1~2.2%) — 표본 잡음 수준 |
| 줄 안 위치(앞 → 뒤 5등분) | 2.0 / 1.7 / 2.2 / 2.1 / 1.4% | 고름 | ✗ |

목표 줄 Xc 상자에는 정답 글자(NanumGothic)가 정확히 그려져 있다 — 정보가 없어서가 아니라 **Xc 구조를 그 필체의 획으로 옮기는 변환이 부정확**하다.

### 생성 실험 (100장, `experiments/cer_dump.py --seed-offset` · `inkspire:<lora>,steps=…,line_h=…`)

| 실험 | CER | 판정 |
|---|---|---|
| 기준 (seed 0, 20 step, line_h 64) | 0.019 | |
| seed +1 / +2 | 0.022 / 0.019 | |
| ODE 50 step | 0.021 | ✗ 적분 정밀도 아님 |
| line_h 80 (학습 범위 최대, 글자당 토큰 ~1.6배) | 0.021 | ✗ 해상도 아님 |
| **학습 폰트(seen, `assets/fonts/train` 에서 plan)** | **0.0096** (GT 바닥 0.0070) | ✓ 학습 폰트는 거의 완벽 — 처음 보는 필체에서 2배 |

**결정적: 오류가 시드마다 다른 글자에서 난다.** 시드 3개의 틀린 글자 24 / 36 / 30 개 중 **세 시드 모두 같은 글자는 0개**, 두 시드 이상 9개,
한 시드만 72개. seed 0 오류 중 seed 1 에서도 틀린 것은 4%. 모델이 특정 글자를 잘못 아는 게 아니라 **샘플마다 글자당 ~2% 확률로 획을 삐끗한다.**

| 시드 3개 중 고르기 | CER |
|---|---|
| 한 번 생성(평균) | 0.020 |
| **정답을 알고 best-of-3 (상한)** | **0.0039** — Eruku(0.010)보다 낫고 GT 바닥 수준 |
| OCR 없이 medoid(다른 둘과 픽셀 L1 이 가장 가까운 것) | 0.020 ✗ |

### 해석과 다음 실험

처음 보는 필체에서 p(X | Xs, Xc) 가 넓다 — 틀린 획에도 확률이 남는다(학습 폰트에선 좁고, 실손글씨처럼 더 먼 분포 밖에선 더 넓다 — §10 과 같은 현상).
Eruku 가 정확한 이유도 여기다: `drop_text` 학습 + CFG 로 내용 조건 쪽으로 분포를 날카롭게 한다(cfg 1.0 → 1.5 에서 CER 0.018 → 0.010, §5).
**InkSpire 엔 그 장치가 없다** — 텍스트는 빈 프롬프트 캐시라 CFG 걸 대상이 없고, Xc 는 학습 내내 항상 주어져 무조건부 예측을 배운 적이 없다.

→ **목표 Xc dropout + 내용 CFG**(구현·학습 완료, **미평가로 보관** — 아래 메모):
- 학습 `train/inkspire.py --xc-drop p` — 확률 p 로 샘플의 **생성 영역 Xc 를 흰색으로 비운다**(ref 줄의 Xc 대응표는 남김) = 내용 무조건부.
- 추론 `InkSpireGen(cfg=s)` / `inkspire:<lora>,cfg=s` — 조건(Xc 있음)·무조건(`models/inkspire_klein.xc_uncond`)을 한 배치로 예측해
  v = v_u + s·(v_c − v_u). cfg=1 은 기존과 비트 동일(smoke 단언).
- run: `finetune_runs/inkspire_klein_xcdrop` — s13000 과 같은 레시피에 `--xc-drop 0.1` 만, 13k step. 2026-10-27 완료.
- ★ **보관 메모(2026-10-27):** 이 학습은 **생성 영역** Xc 를 비운 것이라 style text(ref 전사)를 버린 학습이 아니다 — 전사 없는 추론에는 쓸 수 없다.
  cfg 평가는 시작 직후 중단해 결과가 없고, **InkSpire 에서 CFG 가 CER 을 낮춘다는 측정은 없다**(근거는 Eruku 의 cfg 1.0 → 1.5 뿐).
  다음 후보는 ref 전사 dropout — 보이는 영역 Xc(ref 줄)를 비우는 학습(docs/TODO.md).
- ★ **2026-10-28 평가:** CFG 는 효과가 크다 — 실손글씨 hw_line CER 0.25 → 0.075(cfg 2.0). **xcdrop 없이 최종 LoRA 에 걸어도 된다**(§10 내용 CFG).

다른 선택지(미착수): best-of-N + 판독기(상한 0.004, 외부 모델 의존·N배 느림), 실손글씨 학습 데이터(분포 폭의 근본 원인).

## 13. 결론

**Eruku 는 원본 권장 추론(cfg 1.5, 새 디코딩, 공식 ref 크롭)으로 비교해야 한다(§10 첫 표).** 그 기준으로:

| | Eruku | InkSpire (FLUX.3-Image-Klein-Base s13000 + `ref_width`) |
|---|---|---|
| 글자 정확도 — 폰트 렌더 (PP CER) | **0.010** | 0.023 |
| 글자 정확도 — 실손글씨 문장, 다른 문구 | **0.014** | 0.246 |
| 필체 거리 HWD — 폰트 렌더 / 실손글씨 문장 / 서식 | 1.12 / 1.51 / 1.83 | **0.93 / 1.35 / 1.20** |
| 지저분한 ref(서식 크롭) 백지·폭주 | ~5% | **~0.5%** |
| ref 전사 없이 | 동작 | 동작하지만 필체가 사라짐(§14) — 붕괴(§5)는 옛 FLUX.1 백본 |

- **실사용 정확도는 Eruku 가 크게 앞선다.** InkSpire 의 장점은 필체 형태 재현(HWD — 이진화해도 유지)과 지저분한 ref 견고성으로 좁혀진다.
  둘 다 "ref 모양을 베끼는" 같은 구조에서 나오고, 실손글씨 정확도 하락(주로 분포 밖 글자 모양, 자간 어긋남은 작음)도 같은 구조 탓이다(§10 진단).
- 논문 조건(IAM)의 InkSpire 우세(§6)는 IAM 학습·글자 bbox 주석이 있는 조건이라 우리 결론과 충돌하지 않는다.
- InkSpire 개선 중 남길 것: FLUX.3-Image-Klein-Base 백본(§7, 0.040 → 0.026), `ref_width`(§9, HWD 1.34 → 0.93), 13k 체크포인트(20k 는 나쁨).
  효과 없음: ref 보강(§8).
- ~~"공백을 빼면 Eruku 와 같다", "깨끗한 문장도 경쟁력 있다"~~ — Eruku cfg 1.0·자체 크롭에서 나온 착시였다(철회).

생성 결과 비교: 저장된 덤프를 합치기만 하는 `experiments/montage.py`(추론 없음, `--pick random|worst|diff`, `--font`, `--runs`, `--show-ctx`).

## 14. 손글씨 증강 + ragged + Xc dropout 재학습 — `inkspire_klein_hwaug` (2026-11-02)

p512 와 같은 레시피(13k, 처음부터)에 네 가지를 한꺼번에 바꿨다(각 효과는 분리 안 됨):
`--hw-aug 0.5`(글자별 크기 ×U(0.85,1)·기준선 σ4%·자간 ×U(1,1.25)·획 굵기·흐린 잉크, Xc 상자도 같이 옮김 — 회전·기울기는 뺐다)
· `--ragged 0.5` · `--ref-xc-drop 0.1` · `--target-xc-drop 0.1`(추론 `uncond=gen` 규약). val loss 13k 0.097.

| | 폰트 n=300 CER | 폰트 HWD↓ | 실손글씨 338 CER | 문장 / 서식 / sanrim CER | 실손글씨 HWD↓ 전체 / 문장 / 서식 |
|---|---|---|---|---|---|
| p512 cfg 1 | 0.023 | **0.929** | 0.398 | 0.246 / 0.417 / 0.444 | **1.31** / **1.35** / **1.20** |
| hwaug cfg 1 | 0.032 | 0.932 | 0.393 | 0.224 / 0.417 / 0.428 | 1.48 / 1.47 / 1.39 |
| p512 cfg 2.0 (`uncond=gen`) | 0.0045 | 1.015 | 0.168 | 0.069 / 0.180 / 0.208 | 1.46 / 1.63 / 1.37 |
| **hwaug cfg 2.0** | **0.0032** | 0.971 | **0.144** | **0.052** / **0.154** / **0.188** | 1.62 / 1.64 / 1.54 |
| Eruku cfg 1.5 | 0.010 | 1.117 | 0.620 | 0.014 / 0.711 / 0.672 | 1.81 / 1.51 / 1.83 |

- **cfg 2.0 정확도는 모든 조건에서 나아졌다** — 실손글씨 0.168 → 0.144(표본별 93 / 52), 문장 0.069 → 0.052, 폰트 0.0045 → 0.0032(리더 바닥 0.0056 아래).
  폰트 HWD 도 1.015 → 0.971 로 좋아졌다. cfg 1 은 폰트 CER 이 나빠졌다(0.023 → 0.032) — 목표 Xc dropout 으로 cfg 없이 쓰는 힘이 약해진 몫으로 보인다(미분리).
- **대가: 실손글씨 HWD 가 나빠졌다** — cfg 2.0 서식 1.37 → 1.54, 전체 1.46 → 1.62(문장은 1.63 → 1.64 로 같음). cfg 1 도 1.31 → 1.48.
  해석(미검증): 증강으로 "불규칙한 ref" 를 많이 보면서 ref 의 불규칙함을 베끼기보다 정리해서 쓰게 됐다 — 정확도↑ 와 같은 방향.
- **ref 전사 없이도 된다 — p512 부터 이미 그랬다.** 폰트 n=300 `--style-text none`:

  | | CER | HWD↓ |
  |---|---|---|
  | p512 cfg 1, 전사 없음 | 0.0215 (전사 있음 0.023) | 1.49 (전사 있음 0.93) |
  | hwaug cfg 1, 전사 없음 | 0.043 | — |
  | hwaug cfg 2.0, 전사 없음 | **0.0030** | 1.38 |

  §5 의 "전사 없으면 붕괴(0.720)" 는 **FLUX.1-Fill-dev** 측정이다 — FLUX.3-Image-Klein-Base 는 전사 없이도 글자는 정확하다.
  대신 필체가 크게 약해진다(HWD 0.93 → 1.49, 줄 폭 보정 `ref_width` 도 전사가 있어야 작동). 실손글씨에서의 전사 없음은 미측정.
  몽타주 `finetune_runs/_cer/montage_notext.png`.

  **실손글씨 교차 338장, 전사 없음**(`realhw_dump.py --no-style-text`):

  | | CER 전사 있음 → **없음** | 문장 / 서식 CER (없음) | HWD↓ 전사 있음 → **없음** |
  |---|---|---|---|
  | p512 cfg 1 | 0.398 → 0.206 | 0.210 / 0.205 | 1.31 → **1.76** |
  | p512 cfg 2.0 | 0.168 → **0.033** | 0.066 / **0.025** | 1.46 → **2.05** |
  | hwaug cfg 2.0 | 0.144 → 0.073 | 0.130 / 0.062 | 1.62 → **2.03** |
  | (참고) Eruku · std-font | 0.620 · — | | 1.81 · 2.62 |

  전사를 빼면 **정확도는 크게 좋아지고 필체는 거의 사라진다** — 출력이 표준폰트에 가까운 정형 글씨가 되고 ref 의 표 선·흘림도 안 옮긴다
  (몽타주 `finetune_runs/_realhw/montage_notext.png`). 즉 InkSpire 가 실손글씨 필체를 따라 하는 통로가 바로 ref 전사로 만든 Xc 대응표이고,
  실손글씨 오류의 대부분도 그 통로로 들어온다 — §10 "분포 밖 모양을 베낀다" 의 직접 증거. 전사 없음은 HWD 가 Eruku 보다 나빠 쓸 이유가 없다.

  **전사 없음에서 hwaug(학습 때 ref 전사 10% 제거) vs p512** — 전사 없이 학습한 게 전사 없는 추론에 도움이 되나:

  | 전사 없음 | 폰트 300 CER / HWD↓ | 실손글씨 338 CER / HWD↓ |
  |---|---|---|
  | p512 cfg 1 | 0.0215 / 1.49 | **0.206** / **1.76** |
  | p512 cfg 2.0 | **0.0010** / 1.63 | **0.033** / 2.05 |
  | hwaug cfg 1 | 0.043 / **1.30** | 0.283 / 1.84 |
  | hwaug cfg 2.0 | 0.0030 / 1.38 | 0.073 / 2.03 |

  폰트에서만 필체가 낫고(HWD 1.49 → 1.30) 실손글씨는 어느 cfg 에서도 p512 가 같거나 낫다. 몽타주
  `finetune_runs/_cer/montage_notext_p512_vs_hwaug.png`, `finetune_runs/_realhw/montage_notext_p512_vs_hwaug.png`.

**판정: hwaug 는 채택하지 않는다 — 릴리즈는 p512 유지.** 나은 곳은 전사 있는 cfg 2.0 정확도(필체와 맞바꿈)와 폰트 전사 없음 필체뿐이다.

결과 `finetune_runs/_cer/{hwaug_*,klein_s013000_rw_notext}`, `finetune_runs/_realhw/hwaug_cross_*`, `style_scores_hwaug*.json`.

## 15. 정제 실손글씨셋으로 다시 — Eruku 가 정확도에서 앞선다 (2026-11-04)

§10 의 서식 비교(InkSpire 0.18 vs Eruku 0.71)는 **표 선이 남은 크롭**을 ref 로 줬을 때다. Eruku repo 의 정제셋
(`Eruku_korean_finetuning/custom_datasets/hf_korean_handwriting_clean`, 비공개 — hw_line 20줄 + 서식 41크롭, 긴 격자·밑줄 제거)으로
같은 세 모델을 다시 잰다: InkSpire(`inkspire_klein_p512` s13000, cfg 1 / cfg 2.0 `uncond=gen`) · Eruku corpusmax s45000(cfg 1.5).
`realhw_dump.py --all-pairs`(같은 페이지 안 모든 순서쌍) / `--cross-k 5`(페이지 무관, 크롭마다 목표 5개 — CER 전용),
통계 `experiments/realhw_stats.py`(corpus CER + 표본 부트스트랩 95% CI, 실제 이미지 대비 ΔCER, 표본별 부호검정). PP-OCRv6.

| 세트 | n(글자) | 실제 이미지 | InkSpire cfg 1 | InkSpire cfg 2.0 | **Eruku** | cfg 2.0 vs Eruku 표본별 |
|---|---|---|---|---|---|---|
| hw_line 모든 쌍 | 380 (6,878) | 0.133 | 0.240 [0.226, 0.254] | 0.071 [0.065, 0.077] | **0.028** [0.023, 0.032] | 51 / 230 (p=3e-28) |
| 서식 같은 페이지 모든 쌍 | 148 (1,385) | 0.160 | 0.474 | 0.267 [0.243, 0.291] | **0.087** [0.071, 0.104] | 9 / 116 (p=8e-25) |
| 서식 교차 41×5 | 205 (1,413) | 0.141 | 0.375 | 0.139 [0.117, 0.162] | **0.107** [0.082, 0.137] | 34 / 74 (p=2e-4) |

| HWD↓ (같은 페이지 쌍) | InkSpire cfg 1 | InkSpire cfg 2.0 | Eruku | std-font |
|---|---|---|---|---|
| hw_line | **1.43** | 1.64 | 1.52 | 3.76 |
| 서식 | **1.76** | 1.85 | 1.84 | 3.57 |

- **정확도는 세 세트 모두 Eruku 가 유의하게 앞선다** — 실제 손글씨보다도 잘 읽힌다(ΔCER −0.03~−0.11). InkSpire cfg 2.0 은 실제 수준(ΔCER −0.06~+0.11).
- **필체(HWD)는 InkSpire cfg 1 만 Eruku 보다 0.08~0.09 낮다**(CI 없음 — 확정 아님). 베스트 설정 cfg 2.0 은 Eruku 와 같거나 나쁘다.
- "서식에선 InkSpire" 는 **지저분한 ref(선 잔재) 견고성**이었다 — 선을 지우면 Eruku 가 회복한다(0.71 → 0.09~0.11). 정제셋에서도 Eruku 는
  밑줄이 남은 ref 하나(교차 00051)에서 줄무늬로 무너진다.
- InkSpire 의 실패는 **흐린 연필 ref**(서식 "임산물 판매실적" 페이지, hw_line 의 가장 긴 줄)에 몰린다 — 출력도 흐리고 획이 엉킨다(§11 농도 복사).
  같은 페이지 쌍 148개 중 132개가 이 페이지(12크롭)라 서식 같은-페이지 수치는 사실상 이 한 필기자다 — 교차 세트가 더 대표적.
- ref 하나로 고정한 Eruku 프로토콜(hw_line 가장 긴 줄 → 19)에선 InkSpire cfg 2.0 이 0.131 이었다 — 그 줄이 가장 흐린 ref 라서다. 모든 쌍 평균은 0.071.

몽타주 `finetune_runs/_realhw_clean/montage_{ap_hw_line,ap_seosik_clean,ck_seosik}.png`(열 = ref | GT | InkSpire cfg 1 | cfg 2.0 | Eruku).

**학습량 — cfg 2.0 에서도 13k 가 최선** (같은 klein_p512 run 의 s13000/16000/20000, `uncond=gen`, 결과 `finetune_runs/_steps/`):

| cfg 2.0 | 폰트 300 CER / HWD | 정제 hw_line 380 CER [CI] / HWD | 정제 서식 교차 205 CER [CI] |
|---|---|---|---|
| **s13000** | **0.0045** / 1.015 | **0.071** [0.065, 0.077] / 1.64 | **0.139** [0.117, 0.162] |
| s16000 | 0.0071 / 1.009 | 0.093 [0.086, 0.101] / 1.62 | 0.162 [0.138, 0.188] |
| s20000 | 0.0044 / 0.994 | 0.081 [0.074, 0.088] / 1.67 | 0.151 [0.127, 0.176] |
| Eruku s45000 | 0.010 / 1.117 | 0.028 [0.023, 0.032] / 1.52 | 0.107 [0.082, 0.137] |

hw_line 표본별 13k 가 16k 를 139 / 41(p=1e-13), 20k 를 115 / 70(p=0.001) 로 이긴다. HWD 는 step 간 ±0.03 으로 차이 없음.
같은 데이터·손실로 더 학습해도 실손글씨 정확도는 안 오른다 — 손실에 "읽히는가" 항이 없다(flow matching MSE 뿐, §7).

**어절 간격 보정 `word_gap` — 채택 후보** (`inkspire:<lora>,…,word_gap=1`, `InkSpireGen(word_gap=True)`, 기본 꺼짐).
InkSpire 오답의 상당수가 획이 아니라 **띄어쓰기 누락**이었다(공백 무시 CER: hw_line 0.071 → 0.062, 서식 0.139 → 0.059 — 서식은 Eruku 0.096 보다 낮다).
목표 줄 상자는 표준폰트 간격 × `ref_width`(줄 전체 폭) 라 손글씨의 넓은 어절 간격이 좁아진다. ref 손글씨와 같은 폭으로 맞춘 표준폰트 렌더에서
잉크 열 투영의 최대 빈칸(공백 수만큼) 평균을 재 그 차이를 목표 줄 공백마다 더한다(`infer/inkspire._word_gaps`).

| s13000 cfg 2.0 | 폰트 300 CER / HWD | 정제 hw_line 380 CER [CI] / HWD | 정제 서식 교차 205 CER [CI] |
|---|---|---|---|
| 기존 | 0.0045 / 1.015 | 0.071 [0.065, 0.077] / 1.64 | 0.139 [0.117, 0.162] |
| **+ word_gap** | 0.0045 / **0.998** | 0.068 [0.063, 0.073] / **1.46** | **0.112** [0.093, 0.133] |
| Eruku s45000 | 0.010 / 1.117 | 0.028 / 1.52 | 0.107 |

hw_line CER 은 같고(표본별 95 / 102, p=0.67) HWD 가 1.64 → 1.46 으로 **cfg 2.0 에서 처음으로 Eruku(1.52) 아래**. 서식 CER 유의 개선(27 / 5, p=1e-4),
Eruku 와 거의 같아짐(35 / 58, p=0.02). 폰트는 손해 없음. 결과 `finetune_runs/_wordgap/`.

## 16. ref 에 있는 글자 · 없는 글자 — 정확도와 필체 (2026-11-06)

`finetune_runs/_wordgap/{hwline,seosik}`(s13000 cfg 2.0 `uncond=gen` `word_gap=1`)를 글자 단위로 나눠 본다. 공백 제외, 정답↔PP-OCR 정렬은 difflib.

**ref 에 있는 글자는 그대로 옮겨 그린다 — 오류는 ref 글자의 독특함 때문이지 결함이 아니다.**

OCR 이 틀리게 읽은 비율(목표 한글 음절 기준). "실제" = 같은 필기자가 실제로 쓴 같은 목표 줄 — 글자 난이도 기준선(서식 교차는 목표가 다른 필기자라 없음).

| 한글 음절 | hw_line n | hw_line 생성 | hw_line 실제 | 서식 교차 n | 서식 교차 생성 |
|---|---|---|---|---|---|
| ref 에 있음 · ref 원본을 OCR 이 틀림 | 26 | 7.7% | 7.7% | 10 | 50% |
| ref 에 있음 · ref 원본을 OCR 이 맞힘 | 803 | 4.1% | 5.1% | 50 | 4.0% |
| ref 에 없음 | 4187 | 2.9% | 13.5% | 774 | 5.0% |

- hw_line 의 ref-있음 오류 35건 중 16건이 문장 끝 "다" — ref "…서점을 찾았다" 의 ㄷ·ㅏ 가 떨어진 "다" 를 거의 똑같이 옮겨 그렸고 OCR 이 새 문장에서 "바·냐·나" 로 읽는다.
  사람에겐 읽힌다. 서식은 7건 중 5건이 ref 원본부터 OCR 이 틀린 글자("잣" → "짓·잘")다. Eruku 는 같은 조건 1.6% — 덜 베낀다.
  (서식 50% = ref 원본에서 OCR 이 틀린 글자가 목표에 나온 10자 중 생성에서도 틀린 5자 — 대부분 "잣". n 이 작아 사례 수준)
- 규모는 작다(hw_line 약 5,000자 중 ~10자). 남은 정확도 과제는 띄어쓰기(§15)와 ref 에 없는 글자다.
- 몽타주(위 ref 원본, 아래 생성) ← 개인정보 데이터라 저장소 밖(gitignore) 경로:

  ![ref 복사 hw_line](../finetune_runs/_wordgap/refcopy_hwline.png)
  ![ref 복사 서식](../finetune_runs/_wordgap/refcopy_seosik.png)

**ref 에 없는 글자의 필체 — 한글은 유지, 숫자·영문은 폰트 쪽으로.** hw_line 20줄을 ref 로, ref 마다 5문구 × 조건 3개(조건당 100장,
두 모델 같은 문구): A = ref 음절만 무작위 조합, B = ref 에 없는 코퍼스 상위 1000 음절, C = 숫자·영문. real = hw_line 20줄. 결과 `finetune_runs/_unseen/`.

| HWD↓ | InkSpire (cfg 2.0 + word_gap) | Eruku s45000 | std-font |
|---|---|---|---|
| A: ref 음절 | 1.341 | 1.346 | 3.674 |
| B: ref 에 없는 한글 | **1.361** (+0.02) | 1.513 (+0.17) | 3.746 |
| C: 숫자·영문 | 1.978 | **1.916** | 2.677 |

- 한글은 처음 보는 음절도 자모·획을 ref 에서 가져와 필체가 안 빠진다 — Eruku 보다 확실히 낫다.
- 숫자·영문은 한글과 공유하는 획이 없어 학습 폰트 모양(깔끔한 고딕)으로 돌아간다. Eruku 는 손글씨 느낌이 남는다(몽타주 `montage.png`, 칸마다 위 InkSpire · 아래 Eruku).
  C 의 HWD 는 real 이 한글뿐이라 내용 차이가 섞인다 — 모델 비교용. CI 없음.
- CER(PP-OCR) A / B / C: InkSpire 0.111 / 0.070 / 0.032, Eruku 0.143 / 0.085 / 0.026.

**ref 에 숫자가 있으면** (서식 정제셋에서 숫자 3종 이상인 크롭 7장, ref 마다 3문구, ref 에 있는 숫자만 / 없는 숫자만, `montage_digit.png`):
ref 숫자가 크고 또렷하면 InkSpire 가 그 모양(흘린 1·4, 꺾인 2)을 그대로 쓰고, ref 에 없는 숫자도 손글씨 느낌이 난다.
긴 주소 줄처럼 숫자가 작으면 여전히 폰트 모양이고, 알아보기 힘든 ref 숫자("150g")는 그대로 옮겨 깨진다.
CER(각 21장, 숫자열이라 OCR 자체가 약함): InkSpire 0.248 / 0.190, Eruku 0.239 / 0.285. 영문은 정제셋에 "g" 뿐이라 못 쟀다.

## 17. guidance 구간 제한 `cfg_t0`/`cfg_t1` — 절충선은 그대로 (2026-11-12)

가설: 글자 구조는 앞쪽 step, 획 질감은 뒤쪽 step 에서 정해지니 내용 CFG 를 앞쪽에만 걸면 CER 은 유지하고 HWD 손실은 줄 것이다.
`generate(cfg_t=(a, b))` — step 비율 [a, b) 에서만 guidance, 밖은 조건 예측만(`inkspire:<lora>,cfg=2.0,cfg_t0=…,cfg_t1=…`).
정제 hw_line 모든 쌍 380(§15 와 같은 plan), s13000 `uncond=gen` `word_gap=1`, 20 step, 시드 1개. HWD 는 CI 없음.

| 설정 | CER [95% CI] | HWD↓ |
|---|---|---|
| cfg 1 | 0.223 [0.210, 0.236] | **1.29** |
| cfg 2.0 앞 25% | 0.195 [0.183, 0.208] | 1.31 |
| cfg 2.0 앞 50% | 0.149 [0.139, 0.159] | 1.37 |
| cfg 3.0 앞 50% | 0.115 [0.107, 0.123] | 1.44 |
| cfg 2.0 뒤 50% | 0.076 [0.071, 0.082] | 1.44 |
| cfg 2.0 전 구간 (§15) | **0.068** [0.063, 0.073] | 1.46 |
| Eruku s45000 | 0.028 [0.023, 0.032] | 1.52 |

- **가설 기각 — 반대였다.** 글자 정확도는 **뒤쪽 step** 의 guidance 가 만든다: 뒤 50% 만으로 전 구간 이득의 95%(0.223 → 0.076), 앞 50% 는 절반(0.149).
  앞쪽을 세게(cfg 3.0) 걸어도 뒤 50% 보다 CER 이 나쁘고 HWD 는 같다.
- **CER↔HWD 는 여섯 점이 한 줄로 선다** — 구간을 바꿔도 절충선 위 이동일 뿐(§10 의 `uncond`·`cfg_s` 와 같은 결론).
- 쓸모는 속도뿐: 뒤 50% 는 transformer 호출 40 → 30 회(−25%)로 CER +0.008(표본별 18 / 67, p=8e-8), HWD −0.02. 기본값은 안 바꾼다.

결과 `finetune_runs/_cfgt/{stats.md,style.json}`, 실행 `run.sh`.

## 18. 판독기로 틀린 어절만 재인페인팅 — PP-OCR 로는 절충선을 넘지만 독립 리더로는 아니다 (2026-11-12)

guidance 는 분포를 당기니(§10·§17) 당기지 않고 **고르면** 절충선 밖으로 나갈 수 있나. `experiments/realhw_fix.py`: 생성 줄을 PP-OCR 로 읽어
목표와 다른 어절만 마스크 → 새 시드로 그 부분만 다시 채움(맞은 어절은 보이는 채), 3 라운드, 나빠진 표본은 되돌림.
정제 hw_line 모든 쌍 380(§15·§17 과 같은 plan), s13000 `uncond=gen` `word_gap=1`. A = cfg 1 → 재생성 cfg 1, B = cfg 1 → cfg 2.0, C = cfg 2.0 → cfg 2.0.
독립 리더 = 원본 repo 한글 HTR(`experiments/htr_dirs.py`, 폰트 학습이라 실손글씨에 약함 — 실제 이미지 0.417).

| | PP-OCR CER [95% CI] (고르는 데 쓴 리더) | **HTR CER (독립)** | HWD↓ |
|---|---|---|---|
| cfg 1 | 0.224 [0.211, 0.237] | 0.388 | 1.29 |
| cfg 2.0 앞 25% / 앞 50% (§17) | 0.195 / 0.149 | 0.347 / 0.276 | 1.31 / 1.37 |
| cfg 2.0 뒤 50% (§17) | 0.076 | 0.119 | 1.44 |
| cfg 2.0 전 구간 | 0.068 [0.064, 0.073] | 0.095 | 1.46 |
| A: cfg 1 → 1 | 0.096 [0.088, 0.106] | 0.326 | 1.34 |
| B: cfg 1 → 2.0 | 0.049 [0.046, 0.053] | 0.177 | 1.40 |
| C: cfg 2.0 → 2.0 | 0.044 [0.041, 0.048] | 0.088 | 1.48 |
| Eruku s45000 | 0.028 [0.023, 0.032] | 0.145 | 1.52 |

- **PP-OCR 로만 보면 성공처럼 보인다** — B 가 cfg 2.0 을 CER·HWD 둘 다 앞선다(표본별 115 / 17). 그러나 고른 리더로 채점한 값이다.
- **독립 리더로는 절충선 위다.** A(0.326 / 1.34)는 앞 25%~50% 사이, B(0.177 / 1.40)는 앞 50%~뒤 50% 사이, C 는 전 구간과 같다.
  PP-OCR 이득의 대부분은 "그 리더가 맞다고 읽는 표본" 을 고른 것 — 리더 과적합. 실제 이득은 C 의 0.095 → 0.088 정도.
- 비용도 싸지 않다: 어절만 마스크해도 캔버스 전체를 다시 돌린다. 라운드마다 75~99% 줄이 재생성 대상(리더가 실제 손글씨도 13% 틀리므로)이라 ~4배.
- HTR 리더 자체가 실손글씨에 약해(실제 이미지 0.417, Eruku 0.145 > InkSpire cfg 2.0 0.095 로 PP-OCR 과 순위가 뒤집힘) 절대값은 못 믿는다 — 같은 모델 안 비교용.
- **판정: 미확정 — 채택 보류.** ★ 정정(같은 날): 여기 쓴 "독립 리더" 는 §4·inkspire.md §9.6 에서 **InkSpire 출력을 과하게 나쁘게 읽는 착시 리더**로
  이미 판정된 그 HTR 이다. 그 리더로 PP-OCR 결과를 뒤집을 수 없다 — "PP-OCR 이득은 리더 과적합" 은 확인된 게 아니라 가능성이다.
  PP-OCR 로 고르고 PP-OCR 로 채점한 편향도 실제라, B(0.049 / 1.40)가 진짜 이득인지는 제3의 리더(VLM OCR 등)나 육안으로 가려야 한다.
- **육안 판정(몽타주, PP-OCR 이 r0 에서 틀리고 최종에서 맞게 읽은 어절 무작위 40개씩, `finetune_runs/_fix/montage_words.py`):**
  B 는 대부분 실제로 고쳐졌다 — 재생성 전은 뭉개진 획("위에", "오래된", "깊은", "넓은"), 후는 또렷하다(40개 중 30개 이상, 나머지는 마침표·애매).
  A(cfg 1 재생성)는 진짜 고침과 "전후가 비슷한데 판독만 바뀐" 경우가 섞여 있다(대략 반반). 즉 B 의 PP-OCR 이득은 대체로 진짜, A 는 리더 운이 섞였다.
  B 의 재생성 어절은 전보다 진하고 반듯하다 — HWD 1.29 → 1.40 의 대가가 눈에도 보인다. 한 사람의 육안이라 확정은 아님.
  몽타주 `finetune_runs/_fix/montage_words_{A,B}.png`(저장소 밖).
- 표준폰트 렌더까지의 HWD(낮을수록 고딕에 가깝다, 시드 1개): 실제 3.76 · cfg 1 3.62 · A 3.60 · B 3.56 · cfg 2.0 3.49 · Eruku 3.27 —
  CFG 는 출력을 표준폰트 쪽으로 옮기고(−0.13) 그만큼 실제에서 멀어진다(+0.17). Xc 폰트 무작위화(§19) 가설의 방향성 근거.

결과 `finetune_runs/_fix/{stats.md,style.json}`, 실행 `run.sh`.

## 19. Xc 폰트 무작위화 재학습 — `inkspire_klein_xcfont` · 기각 (2026-11-12)

**가설**: CFG 가 출력을 표준폰트 쪽으로 당기는 건(§18 말미) Xc 가 늘 같은 폰트라 "Xc 글자 모양 = 내용 신호"로 배웠기 때문. 학습 때 Xc 폰트를
절반 확률로 다른 폰트로 그리면(`--xc-font-rand 0.5`) Xc 에서 모양이 아닌 글자 정체만 읽게 되어 CFG 의 필체 대가가 준다.
p512 와 같은 설정으로 13k step. 추론 Xc 는 그대로 NanumGothic.

**미리 정한 기각 기준**: cfg 1→2.0 의 HWD 상승폭이 p512 의 절반 이하로 줄지 않거나, cfg 2.0 CER 이 유의하게 나빠지면 기각.

정제 hw_line 380 (PP-OCR corpus CER / §20 고친 채점 HWD):

| | cfg 1 CER | cfg 1 HWD | cfg 2.0 CER | cfg 2.0 HWD | HWD 상승폭 |
|---|---|---|---|---|---|
| p512 s13000 | 0.224 | 0.520 | 0.068 | 0.712 | +0.19 |
| xcfont s13000 | 0.210 | 0.602 | 0.065 | 0.761 | +0.16 |

- 상승폭 +0.19 → +0.16: 절반 기준에 한참 못 미침. 게다가 **두 cfg 모두 HWD 절대값이 나빠졌다**(+0.08 / +0.05).
- CER: cfg 1 은 조금 낫고(부호검정 p=0.003), cfg 2.0 은 차이 없음(p=0.19). 정제 서식 교차 205 cfg 2.0: 0.112 → 0.117 (p=0.36).
- 폰트 300: CER cfg 1 0.023 → 0.031, cfg 2.0 0.0045 → 0.0048. HWD 0.928 → 0.909, 1.013 → 0.979 (소폭 개선, 시드 잡음 수준).

**판정: 기각.** CFG 의 필체 대가는 "Xc 폰트 모양을 베껴서"가 아니다 — Xc 폰트를 흔들어도 대가가 그대로다. 현 권장은 p512 s13000 유지.
코드(`xc_font_rand`, 기본 0)는 남겨 두되 쓰지 않는다. 산출물: `finetune_runs/_xcfont/`.

**교훈**: 7시간 학습 전에 값싼 검증(§18 의 표준폰트 거리 측정)은 "방향"만 보여 줬지 원인을 가리지 못했다. CFG 대가를 줄이는 길은
학습 쪽보다 추론 쪽(§18 B: cfg 1 로 만들고 틀린 어절만 cfg 2.0 — HWD 0.62 / CER 0.049)이 지금까지 유일하게 통했다.

## 20. HWD 분해 — 실손글씨 HWD 는 지표 버그로 ~0.8 부풀어 있었다 (2026-11-12)

질문: 같은 필체면 HWD 가 0.6 쯤(논문 IAM 0.62)이어야 하는데 왜 1.3~1.5 인가. 하한 0.29 까지의 ~1.0 을 학습 없이 분해했다(정제 hw_line, s13000 `uncond=gen` `word_gap=1`).

**★ 원인은 채점이었다.** `style_scores.norm` 이 "잉크(<200)가 하나라도 있는 bbox" 로 잘랐는데, 실손글씨 스캔은 잡티 한 점이 bbox 를 이미지 전체(100×1348)로
넓힌다(hw_line 20줄 중 절반 — 글씨는 60~70×600~800). real 쪽만 글씨가 작게 잡혀 모든 모델의 HWD 가 부풀었다. 실제 이미지를 잉크 높이 48 로 **줄이기만 한 것**이
원본 대비 1.15 로 나와서 드러났다. → `norm` 에 밀도 기반 크롭(`realhw_dump.eruku_crop`, 모델 ref 에 쓰는 것)을 넣었다. 폰트 수치는 불변(0.929 → 0.928, 1.015 → 1.013, Eruku 1.117).
**§10·§14~§18 과 README 의 실손글씨 HWD(1.2~2.0 대)는 전부 옛 채점이다 — 순위는 대체로 같지만 절대값·격차가 다르다. 재채점 필요.**

| 정제 hw_line 모든 쌍 380 | HWD↓ 옛 채점 | **HWD↓ 고친 채점** | PP CER |
|---|---|---|---|
| 실제 이미지를 잉크 높이 48 로 축소만 (파이프라인 바닥) | 1.15 | **0.27** | |
| 〃 + Flux2 VAE 왕복 | 1.18 | **0.30** | |
| InkSpire cfg 1 + 목표 실제 이미지의 어절 위치(오라클) | 1.17 | 0.44 | 0.300 |
| InkSpire cfg 1 | 1.29 | **0.52** | 0.224 |
| A (§18, cfg 1 → 1 재인페인팅) | 1.34 | 0.56 | 0.096 |
| B (§18, cfg 1 → 2.0 재인페인팅) | 1.40 | 0.62 | 0.049 |
| InkSpire cfg 2.0 + 어절 위치 오라클 | 1.34 | 0.64 | 0.114 |
| InkSpire cfg 2.0 뒤 50% (§17) | 1.44 | 0.69 | 0.076 |
| InkSpire cfg 2.0 | 1.46 | 0.71 | 0.068 |
| C (§18, cfg 2.0 → 2.0) | 1.48 | 0.74 | 0.044 |
| Eruku s45000 | 1.52 | **1.03** | 0.028 |
| std-font | 3.76 | 3.65 | |

| 같은 문구(ref = 목표) vs 다른 문구, 20줄 × 시드 3 (고친 채점) | HWD↓ 시드 0 / 1 / 2 (평균) | PP CER 평균 |
|---|---|---|
| cfg 1 같은 문구 | 0.52 / 0.55 / 0.47 (0.51) | 0.162 |
| cfg 1 다른 문구 | 0.49 / 0.49 / 0.55 (0.51) | 0.219 |
| cfg 2.0 같은 문구 | 0.57 / 0.61 / 0.59 (0.59) | 0.064 |
| cfg 2.0 다른 문구 | 0.64 / 0.66 / 0.68 (0.66) | 0.061 |

- **InkSpire 는 이미 0.5~0.7 이다** — 바닥 0.30 까지 cfg 1 은 0.22, cfg 2.0 은 0.41 남았다.
- **Eruku 와의 필체 격차는 훨씬 크다** — 옛 채점 0.06~0.23 → 고친 채점 0.32~0.51. Eruku 는 바닥에서 0.73.
- **CFG 의 필체 대가(+0.20)는 작지 않다** — 바닥까지 남은 거리를 거의 두 배로 만든다. B 는 그 절반쯤(+0.10)만 내고 CER 은 더 낮다(단 고른 리더로 채점, §18).
- **어절 위치 오라클은 HWD −0.08, CER 은 나빠진다**(0.224 → 0.300, 0.068 → 0.114) — 실제 어절 폭에 표준폰트 비율로 글자를 구겨 넣기 때문. 어절 단위 줄 모양은 큰 몫이 아니다.
- **글자가 ref 에 다 있어도 cfg 1 필체는 같다**(0.51 vs 0.51). cfg 2.0 에서만 같은 문구가 0.07 낫다(시드 폭 ±0.03 — 약한 신호).
- VAE 왕복 몫은 0.03.

코드: `InkSpireGen.prepare(tgt_imgs=)`·`_fit_words`(진단 전용), `realhw_dump.py --oracle-words`, self-check `experiments/test_fit_words.py`. 결과 `finetune_runs/_decomp/{style_fixed.json,cer.txt}`, 실행 `run.sh`.

### 20-b. 옛 실손글씨 HWD 재채점 (2026-11-13)

기존 덤프를 고친 채점으로 다시 쟀다(생성 없음, `finetune_runs/_rescore/run.sh`). 옛 채점 → 고친 채점, InkSpire cfg 1 / cfg 2.0 / Eruku:

| 세트 | 옛 채점 | 고친 채점 | 쓰인 곳 |
|---|---|---|---|
| 옛 교차 · 문장 hw_line 20 | 1.35 / 1.63 / 1.51 | **0.60 / 0.93 / 1.02** | README 실손글씨 표, §10 |
| 옛 교차 · 서식 seosik_hw 286 | 1.20 / 1.37 / 1.83 | 1.13 / 1.31 / 1.91 | 〃 |
| 옛 교차 · 전체 338 | 1.31 / 1.46 / 1.81 | 1.21 / 1.38 / 1.84 | 〃 |
| 옛 교차 · 전체, 전사 없음(cfg 1 / 2.0) | 1.76 / 2.05 | 1.68 / 1.98 | README 한계 표, §14 |
| 정제 hw_line longest-style 19 | 1.26 / 1.48 / 1.43 | 1.06 / 1.10 / 1.01 | §15 |
| 정제 hw_line 전쌍 380 | 1.29 / 1.46 / 1.52 | **0.52 / 0.71 / 1.03** | §15~§18 |
| 정제 서식 longest-style | 2.67 / 2.50 / 2.40 | 2.58 / 2.39 / 2.31 | §15 |
| 정제 서식 전쌍 | 1.76 / 1.85 / 1.84 | 1.70 / 1.82 / 1.78 | §15 |
| 정제 서식 교차 k=5 205 | — | 1.41 / 1.52 / 1.49 | §15 |

- **부풀림은 문장(hw_line) 세트에만 컸다**(0.4~0.8). 서식은 0.05~0.1 로 거의 그대로 — §20 에서 "서식이 더 클 수 있다"고 한 추측은 틀렸다.
  서식 칸 크롭은 이미 글씨에 붙어 있고, 문장 스캔은 1348×100 줄 이미지에 잡티가 있어서다.
- 결론의 방향은 어디서도 안 뒤집힌다. 달라지는 건 문장 세트의 크기뿐: InkSpire cfg 1 의 Eruku 대비 필체 우위가 0.16 → 0.42(옛 교차), 0.23 → 0.51(정제 전쌍).
  단 ref 하나로 19줄을 쓰는 longest-style 에선 셋이 1.0~1.1 로 같다.
- 추가 재채점(`run2.sh`): cfg 표 시드 3개 평균 cfg 1 / 1.5 / 2.0 / 2.0+cfg_s 4 = 1.41 / 1.57 / 1.60 / 1.47 → **0.66 / 0.84 / 0.89 / 0.75**. 하한(절반 나눔) 0.29 → 0.41.
  §16 미등장 글자 InkSpire / Eruku: A 1.34 / 1.35 → 0.62 / 0.92, B 1.36 / 1.51 → 0.86 / 1.30, C 1.98 / 1.92 → 1.78 / 1.86 — "처음 보는 한글도 필체가 안 빠진다(+0.02)"는 **+0.24 로 정정**(Eruku +0.38 보다는 작다).
- README · HF 모델 카드는 고친 수치로 갱신(2026-11-13). 미재채점: §10 의 나머지 스윕, §14 hwaug — 문장 세트 수치는 부풀어 있다고 읽는다.

## 21. "긴 줄은 뒷부분이 무너진다" 재확인 — 재현 안 됨, 대신 끝 마침표 (2026-11-13)

README 한계의 "긴 실손글씨 줄(18자 이상)은 뒷부분이 무너지는 경향"은 §10 진단 4 에서 옛 20줄 세트의 긴 줄 2개를 눈으로 본 것이었다.
정제 hw_line 전쌍 380(목표 20문장 × ref 19, 11~27자)의 PP-OCR 판독으로 다시 쟀다. 생성 없음 — `finetune_runs/_longline/pos_err.py`.

목표 글자별 오류율(공백 제외):

| | 길이 ≤12 / 15~16 / 18~20 / 22~27 | 끝 마침표 | 뒤⅓(마침표 제외) 긴 줄 / 짧은 줄 |
|---|---|---|---|
| 실제 이미지 | 0.085 / 0.087 / 0.149 / 0.154 | 0.150 | 0.175 / 0.100 |
| InkSpire cfg 1 | 0.216 / 0.228 / 0.223 / 0.229 | **0.871** | 0.196 / 0.213 |
| InkSpire cfg 2.0 | 0.090 / 0.077 / 0.064 / 0.059 | **0.768** | 0.038 / 0.042 |
| B (§18) | 0.082 / 0.065 / 0.042 / 0.036 | **0.776** | 0.008 / 0.024 |
| Eruku | 0.015 / 0.026 / 0.022 / 0.038 | 0.126 | 0.025 / 0.000 |

- **길이 효과 없음.** cfg 1 은 길이와 무관(0.22~0.23), cfg 2.0 은 긴 줄이 오히려 낮다. 실제 이미지는 긴 줄이 더 어렵게 읽히는데도(0.15 vs 0.09) 그렇다.
- **위치 효과도 없음** — 끝 마침표만 빼면 뒤⅓ 이 나머지와 같다(cfg 2.0: 뒤⅓ 0.038, 긴 줄 전체 0.037).
- **끝 마침표가 InkSpire 에서만 77~87% 안 읽힌다**(20문장 모두 마침표로 끝남). 줄 끝 몽타주(`tails.png`, 비공개)로 보면 점은 대부분 찍혀 있지만
  작고 기준선 아래로 처진다. Eruku 는 굵고 크게 찍는다. 재인페인팅(B)으로도 안 고쳐진다(0.776).
- **CER 을 끝 마침표 빼고 다시 재면** (전체 / ≤16자 / ≥18자):

| | 원래 | 끝 마침표 제외 |
|---|---|---|
| 실제 이미지 | 0.133 / 0.086 / 0.152 | 0.132 / 0.082 / 0.151 |
| InkSpire cfg 1 | 0.224 / 0.222 / 0.225 | 0.186 / 0.161 / 0.196 |
| InkSpire cfg 2.0 | 0.068 / 0.084 / 0.061 | **0.027** / 0.018 / 0.031 |
| B | 0.049 / 0.075 / 0.039 | 0.006 / 0.008 / 0.006 (고를 때 쓴 판독기 — 낙관적) |
| Eruku | 0.028 / 0.020 / 0.031 | 0.022 / 0.008 / 0.027 |

  → cfg 2.0 과 Eruku 의 정확도 격차(0.068 vs 0.028)는 **대부분 마침표 한 글자**였다. 마침표 밖에서는 0.027 vs 0.022.
  §15 "Eruku 가 정확도에서 앞선다"는 이 세트에선 "끝 마침표에서 앞선다"에 가깝다. 서식 세트는 마침표로 끝나지 않아 해당 없음.
- 미확인: 점이 작은 게 원인인지(Xc 의 NanumGothic 마침표 상자가 아주 작다), 판독기가 작은 점을 버리는 건지. 점 크기를 키워 다시 읽혀 보면 가를 수 있다.

## 22. ref 를 여러 줄로 — 실손글씨 문장 (2026-11-13)

§8 은 폰트에서 보강 줄 하나의 CER 만 쟀다(효과 없음). 실손글씨에서 같은 사람의 줄을 ref 로 더 깔면 필체(HWD)와 정확도가 어떻게 되나.
정제 hw_line 전쌍 380, p512 s13000 + word_gap. 추가 줄 = 같은 필기자의 다른 줄(목표·첫 ref 제외, 표본 번호로 고정), 전사 포함,
Xc 상자는 첫 ref 와 같은 근사(표준폰트 bbox 를 줄 폭에 맞춤). `experiments/realhw_multiref.py`, 결과 `finetune_runs/_multiref/`.
K=1 은 기존 덤프(`_fix/c{1,2}_r0`)와 같은 조건임을 확인(픽셀 차 0.1/255).

| ref 줄 수 | cfg 1 CER | cfg 1 HWD | cfg 2.0 CER | cfg 2.0 HWD |
|---|---|---|---|---|
| 1 (기존) | 0.224 | 0.520 | 0.068 | 0.712 |
| 2 | 0.232 | 0.506 | 0.074 | 0.684 |
| 3 | 0.240 | 0.476 | 0.077 | 0.693 |
| 5 | 0.256 | 0.477 | 0.083 | 0.689 |

- **CER 은 줄을 늘릴수록 조금씩 나빠진다** — 단조. 1줄 대비 표본별 부호검정: cfg 1 은 5줄에서 유의(203 나쁨 / 121 나음, p=6e-6),
  cfg 2.0 은 3줄부터(p=0.003), 5줄 p=2e-6. 2줄은 둘 다 유의하지 않다.
- **HWD 는 조금 내려간다** — cfg 1 0.52 → 0.48(3줄에서 포화), cfg 2.0 0.71 → 0.69. 시드 폭(±0.03, §20)과 같은 크기라 cfg 2.0 은 신호라 하기 어렵다.
- 즉 줄 수도 **같은 절충선 위**다(필체 조금 ↔ 정확도 조금). 비용은 캔버스가 커지는 만큼 든다(5줄 ≈ 2.5배 시간).
- §20 의 "ref 에 글자가 다 있어도(echo) cfg 1 HWD 는 그대로"와 맞는다 — 필체 재현의 병목은 예시 수가 아니다.
- 한계: 추가 줄의 Xc 상자가 근사라 X↔Xc 어긋남이 줄마다 쌓인다(§8 교훈). CER 손해가 이 어긋남 때문인지 예시가 늘어서인지는 못 갈랐다. 시드 1개.

**판정: 기본은 1줄 유지.** 필체가 최우선이고 cfg 1 로 쓸 때만 2~3줄이 약간 이득.

## 23. 학습 폰트 판독성 감사 — 획 구조가 어긋난 장식 폰트 찾기 (2026-11-13)

계기: `ARCHISCULPTURE.ttf`(학습 풀에 있음) — E 가로획 4개, A·M·ㅍ 모양이 표준과 다르다. 이런 폰트는 "틀린 획 = 정답"으로 학습된다.
학습 폰트 12,951개마다 고정 문구(한글 3줄·영문 3줄·숫자 2줄)를 48px 로 렌더해 PP-OCR 로 읽고 문자종별 CER 을 냈다.
`finetune_runs/_fontaudit/audit.py` → `audit.json`, 의심 목록 `suspects.tsv`, 견본 `sheet_ko.png`·`sheet_en.png`.

| | CER = 0 | > 0.1 | > 0.2 | > 0.5 |
|---|---|---|---|---|
| 한글 | 10,608 | 86 | 22 | 8 |
| 영문 | 8,807 | 106 | 29 | 8 |
| 숫자 | 3,390 | 105 | 45 | 10 |

한글 또는 영문 CER > 0.1 인 폰트 159개(1.2%). 상위를 눈으로 보면 네 부류다:
- **글자 자체가 틀림**: `IdeaHwegwan`(풀어쓰기 — 음절을 자모로 늘어놓음, CER 1.15), `TTSamlipCreamyWhiteR`, `Hungeul.otf`(영문 자리에 다른 기호).
- **외곽선·훼손 장식**: `MoiraiOne`, `KERISKEDU_Line`, `HakgyoansimPuzzleTTF-Outline`, `BMEuljirooraeorae`, `UhBeekairumischief`, `WagleWagle`.
- **소문자가 대문자 모양(small caps)**: `LeferiPoint-WhiteObliqueA`, `Lasthighteen-Bold`, `BoldDungGeunMo` — Xc 는 소문자인데 글리프는 대문자.
- **정상인데 판독기가 약한 것**: 픽셀 폰트(Galmuri·dalmoori·Mulmaru…), 영문이 필기체인 손글씨 폰트(Ownglyph 계열) — 빼면 안 된다.

**한계 — 계기가 된 폰트는 이 방법으로 안 걸린다.** ARCHISCULPTURE 는 한글 0.04(702위) · 영문 0.10(104위): 판독기가 어긋난 획을 뚫고 읽는다.
"읽히지만 획 구조가 틀린" 폰트는 OCR CER 로는 못 찾는다. 규모도 작다 — 확실히 틀린 건 십여 개(0.1%)라 실손글씨 획 오류(§10)의 원인으로 보긴 어렵다.
조치(2026-11-13): 9개(IdeaHwegwan · UhBeekairumischief · TTSamlipCreamyWhiteR · MoiraiOne · Hungeul · KERISKEDU_Line · BMEuljirooraeorae · HakgyoansimPuzzleTTF-Outline · WagleWagle)를
`configs/inkspire.yaml` · `inkspire_layout.yaml` 의 `exclude_fonts` 에 추가(학습 풀 12,949 → 12,940). 다음 학습부터 적용 — 현재 모델(p512 s13000)은 이 폰트들을 본 상태다.

## 24. 재학습 사전 확인 — 글자 오류는 어느 노이즈 구간에서 정해지나 · AI 손글씨 폰트 (2026-11-13)

**질문**: 학습 σ 는 shift(U(0,1)), shift 6.93 이라 높은 노이즈에 몰린다(σ>0.82 가 60%, σ<0.27 은 5%). 획이 낮은~중간 σ 에서 정해진다면 그 구간을 더 학습시킬 근거가 된다.

**방법**(`finetune_runs/_sigma/restart.py`, 생성만): p512 s13000 cfg 1 결과(`_fix/c1_r0`, 정제 hw_line 190쌍)를 step t0 의 노이즈로 되돌려
새 시드로 나머지 step 만 다시 돈다(`generate(t0=)`, t0=0 은 기존과 동일). 원래 틀린 글자가 그대로 틀린 비율(지속률)을 본다 — 1 이면 이미 굳은 것.

| 되돌린 σ (t0) | 1.00 (0) | 0.91 (8) | 0.82 (12) | 0.75 (14) | 0.63 (16) | 0.55 (17) | 0.44 (18) | 0.27 (19) |
|---|---|---|---|---|---|---|---|---|
| 틀림 → 틀림 | 0.51 | 0.53 | 0.57 | 0.60 | 0.70 | 0.78 | 0.84 | 0.91 |
| 맞음 → 틀림 | 0.17 | 0.15 | 0.13 | 0.12 | 0.09 | 0.08 | 0.06 | 0.05 |
| CER | 0.213 | 0.206 | 0.207 | 0.204 | 0.212 | 0.225 | 0.229 | 0.234 |

- **실손글씨 오류의 절반은 시드를 바꿔도 남는다**(σ=1 지속률 0.51) — 폰트에서의 "시드 간 공통 오류 0"(§12)과 다르다. 이 절반은 σ 분포로 못 건드린다.
- 나머지 절반이 정해지는 곳: 지속률 0.51 → 1.0 의 변화량을 구간별로 나누면 σ>0.82 12% · 0.63~0.82 27% · 0.44~0.63 27% · 0.27~0.44 15% · <0.27 19%(판독 잡음 포함 상한).
  학습 표본 비율은 같은 구간 순서로 60% · 20% · 10% · 5% · 5%. **결정의 ~70% 가 σ 0.27~0.82 에서 나는데 학습은 35% 만 거기 쓴다.**
- 판정: 가설(학습 σ 를 중간 쪽으로)은 **살아 있다 — 단 근거는 간접적**. "어디서 정해지나"이지 "거기를 더 학습하면 준다"가 아니고,
  ODE 50 step 이 무효였던 것(§12)처럼 그 구간의 예측 오차가 데이터(분포 밖 모양) 탓이면 효과 없다. 후보 변경: 학습 shift 6.93 → 3(σ 0.27~0.82 비율 35% → 49%), 추론은 그대로.

**AI 손글씨 폰트 4종** `Eruku repo assets/ai_fonts_261113/*-Plus.otf`(이전판 `ai_fonts_261109`, 배포 금지). 코퍼스 150줄 렌더(`finetune_runs/_aifont/`):

| | PP CER 이전 → 새 | 글자당 잡티 이전 → 새 | 실제 hw_line 과 HWD |
|---|---|---|---|
| dgkang | 0.042 → 0.051 | 0.83 → 0.27 | 1.76 |
| mykwak | 0.062 → 0.067 | 0.15 → 0.07 | 1.17 |
| shkim | 0.070 → 0.053 | 1.83 → 0.54 | 1.29 |
| sthong | 0.057 → 0.038 | 0.14 → 0.08 | 1.42 |
| (대조) 일반 Ownglyph 폰트 | 0.005 | 0.001 | 3.37 |

- 잡티는 1/2~1/3 로 줄었지만 일반 폰트보다 훨씬 많고, 글리프의 ~5% 가 판독 안 됨. 획 2px(48px 렌더)로 가장 가는 축.
- hw_line 필기자 = dgkang 이지만 그 폰트는 실제 글씨와 네 폰트 중 가장 멀다(1.76) → 학습에 넣어도 평가 오염 우려는 작다. 네 폰트 모두 일반 폰트(3.37)보다는 실손글씨에 훨씬 가깝다.
- Eruku 쪽 기록: 이전판을 10~30% 가중 → HWD 불변·획 오류 증가로 기각, 단 가는 획을 지우는 erosion 버그와 섞여 미확정. InkSpire 증강엔 그 버그 없음(가늘게 p 0.075).
- 균등 추출로 넣으면 0.03%(13k step 에 ~16회) — 효과도 해도 측정 불가. 넣으려면 비중을 줘야 한다(미구현).

## 25. 학습 노이즈 분포 shift 3 재학습 — `inkspire_klein_shift3` · 기각 (2026-11-14)

**가설**(§24): 글자의 ~70% 가 σ 0.27–0.82 에서 정해지는데 학습 샘플은 35% 만 그 구간에 떨어진다(shift 6.93). 학습 σ 추출만 shift 3 으로
바꾸면(`--train-shift 3`, 그 구간 49%) cfg 1 획 오류가 준다. 추론 스케줄(shift 6.93, 20 step)은 그대로. p512 레시피 13k step.
같이 바뀐 것: §23 의 폰트 9종 제외(풀의 0.07% — 영향 없다고 봄).

**미리 정한 기각 기준**: cfg 1 CER 이 p512 보다 유의하게(380쌍 부호검정) 낮아지지 않거나, cfg 1 HWD 가 0.05 넘게 나빠지면 기각.

정제 hw_line 380 (PP-OCR corpus CER / 고친 채점 HWD):

| | cfg 1 CER | cfg 1 HWD | cfg 2.0 CER | cfg 2.0 HWD |
|---|---|---|---|---|
| p512 s13000 | 0.224 | 0.520 | 0.068 | 0.712 |
| shift3 s13000 | 0.227 | 0.670 | 0.064 | 0.764 |

- cfg 1 CER 차이 없음(나음 134 / 124, p=0.58), cfg 1 HWD +0.15. **두 기준 모두 불합격.**
- cfg 2.0: CER 0.068 → 0.064 (p=0.027), HWD +0.05 — 절충선 위 이동. 정제 서식 교차 205 cfg 2.0: 0.112 → 0.107 (p=0.77).
- 폰트 300: CER cfg 1 0.024, cfg 2.0 0.0035 (p512 0.023 / 0.0045), HWD 0.933 / 1.009 (p512 0.928 / 1.013) — 같음.
- 검증 손실(기본 shift 로 측정) 13k 0.0935.

σ 재시작 진단(§24 와 같은 190쌍, 새 모델로 원본부터 다시 생성) — 틀림→틀림 지속률:

| σ | 1.00 | 0.91 | 0.82 | 0.75 | 0.63 | 0.55 | 0.44 | 0.27 |
|---|---|---|---|---|---|---|---|---|
| p512 | 0.51 | 0.53 | 0.57 | 0.60 | 0.70 | 0.78 | 0.84 | 0.91 |
| shift3 | 0.52 | 0.55 | 0.57 | 0.62 | 0.67 | 0.74 | 0.84 | 0.91 |

글자가 정해지는 구간은 옮겨 가지 않았고, 중간 구간 오류도 줄지 않았다(원본 CER 0.231 → 0.229).

**판정: 기각.** 중간 σ 구간의 학습 샘플 부족은 cfg 1 획 오류의 원인이 아니다 — 그 구간 샘플을 1.4배로 늘려도 CER 이 그대로다.
점진 스케줄(낮은 s → 높은 s)도 같은 가설 위라 하지 않는다. 현 권장은 p512 s13000 유지. `--train-shift`(기본 없음)는 남겨 두되 쓰지 않는다.
산출물: `finetune_runs/_shift3/`, `finetune_runs/_sigma/restart_shift3.*`.

**관찰**: 재학습 둘(§19 xcfont 0.60, shift3 0.67) 모두 cfg 1 HWD 가 p512(0.52)보다 나쁘다. p512 의 0.52 가 학습 시드 운일 수 있다 —
같은 레시피 재학습의 HWD 편차는 재 본 적 없다. 이후 재학습의 HWD 기준은 이 점을 감안해야 한다.

## 26. 같은 레시피를 시드만 바꿔 재학습 — `inkspire_klein_seed7` · 학습 시드 편차가 지금까지의 판정 폭보다 크다 (2026-11-16)

§25 의 관찰(재학습마다 cfg 1 HWD 가 p512 보다 나쁨)을 확인하려고 p512 레시피를 `--seed 7` 로 13k step 다시 학습.
p512(seed 42)와 다른 점: 시드, §23 의 폰트 9종 제외(풀의 0.07%).

| | hw_line cfg 1 CER / HWD | hw_line cfg 2.0 CER / HWD | 서식 교차 cfg 2.0 CER | 폰트 cfg 1 CER / HWD | 폰트 cfg 2.0 CER / HWD |
|---|---|---|---|---|---|
| p512 (seed 42) | 0.224 / 0.520 | 0.068 / 0.712 | 0.112 | 0.023 / 0.928 | 0.0045 / 1.013 |
| seed7 | 0.192 / 0.692 | 0.056 / 0.814 | 0.090 | 0.021 / 0.922 | 0.0021 / 1.000 |

- 실손글씨에서 **시드만으로 CER −0.03 / HWD +0.17**(cfg 1). 쌍별 부호검정으로는 "유의"하다(cfg 1 p=3e-9, cfg 2.0 p=7e-8, 서식 p=0.002) —
  즉 부호검정은 평가 쌍의 잡음만 보지 학습 시드 편차는 못 본다. 폰트 평가는 거의 안 움직인다.
- 방향은 또 절충선: CER 이 낮은 시드가 HWD 가 높다. p512 의 HWD 0.52 는 운이라기보다 절충선의 한쪽 끝.
- 검증 손실 13k 0.0925 (p512 와 같은 수준) — 검증 손실로는 이 차이가 안 보인다.

**지난 판정 다시 보기** (모두 단일 시드 비교였다):
- §19 xcfont(0.210 / 0.60), §25 shift3(0.227 / 0.67): 둘 다 p512↔seed7 범위 안. "효과 없음/기각"은 유지되지만 근거는
  "시드 편차와 구별 안 됨"이다. xcfont 의 "cfg 1 CER 조금 낫다(p=0.003)"는 시드 잡음으로 봐야 한다.
- 이후 재학습의 판정 기준: 실손글씨 cfg 1 에서 CER 0.03 · HWD 0.17 안쪽 차이는 단일 run 으로 판정 불가. 절충선 밖(CER·HWD 동시 개선)인지로 본다.

seed7 은 CER 로는 세 평가 모두 p512 보다 낫지만 HWD 가 나빠 릴리스 교체는 보류. 산출물: `finetune_runs/_seed7/`.

### 26-b. 검증 확대 · 체크포인트별 실손글씨 CER (2026-11-17)

- 학습 중 val: `val_batches` 2 → 16(64 샘플), 배치마다 다른 시드. 전에는 배치마다 같은 시드를 다시 걸어 σ 가 4개뿐이었다.
  이후 run 의 val_loss 는 이전 run 과 직접 비교 불가.
- `experiments/val_hw.sh <run>`: 체크포인트마다 정제 hw_line 380 을 cfg 1 로 생성 → PP-OCR CER → `<run>/val_hw.csv`. (비공개 데이터·판독기 필요)

기존 run 후반 체크포인트(cfg 1 CER):

| step | 9k | 10k | 11k | 12k | 13k | 평균 |
|---|---|---|---|---|---|---|
| p512 (seed 42) | 0.202 | 0.218 | 0.246 | 0.233 | 0.224 | 0.225 |
| seed7 | 0.196 | 0.206 | 0.192 | 0.219 | 0.192 | 0.201 |
| shift3 | — | — | 0.279 | 0.218 | 0.227 | |

- **같은 run 안에서 1,000 step 사이에 0.02–0.06 움직인다**(표준편차 ~0.015). 체크포인트 하나끼리의 비교는 이 폭 안에서 의미 없다 —
  seed7 12k(0.219) 와 p512 10k(0.218) 는 같고, p512 9k(0.202) 는 seed7 평균과 같다.
- 5개 평균으로는 seed7 이 0.024 낮다. 시드 차이가 조금은 있어 보이나 run 2개로는 단정 못 한다.
- 앞으로 재학습 비교는 **후반 체크포인트 여러 개의 평균**(9k–13k)으로 한다. 단일 13k 비교(§19·§25 포함)는 ±0.03 잡음을 안고 있다.
- seed7 13k 재측정 0.1916 (평가 스크립트 0.1919) — 측정 자체는 재현된다. 흔들림은 가중치에서 온다.
