# TODO

## 진행 중 / 대기

### ref 전사(T_s) dropout 학습 — 다음 후보, 미착수

- **왜:** InkSpire 는 ref 전사가 필수다 — Xc 의 ref 줄(고딕으로 쓴 ref 전사)을 비우면 붕괴(CER 0.720, 300장 중 134장 백지, docs/inkspire.md §9.6).
  Eruku 는 `style_text_dropout=0.10` 으로 전사 없이도 동작한다.
- **무엇:** 학습 때 확률 p 로 **보이는 영역(마스크 밖)의 Xc** 를 흰색으로 비운다 → "전사 모르는 ref 의 필체로 목표 글자를".
- **평가:** `cer_dump.py --style-text none` CER 이 0.720 에서 얼마나 회복되나 + 전사 있을 때 0.023 유지되나.
- **코드(2026-10-27, 미학습):** Eruku 와 같은 짝으로 두 dropout 을 둔다(샘플마다 독립).

  | 옵션 | 지우는 것 | 용도 | Eruku |
  |---|---|---|---|
  | `--ref-xc-drop p` | 보이는 영역 Xc(ref 전사) | 전사 없는 추론 | `style_text_dropout` (0.1) |
  | `--xc-drop p` | **Xc 전체**(빈 이미지) | 추론 `cfg` 의 무조건부 v_u(`xc_uncond`) | `text_dropout` (0.05) |

  **결정(2026-10-27): 1차는 목표만** — `--ref-xc-drop 0.1 --xc-drop 0`. 학습 시점은 미정(지시 대기).

- **2차(1차 성능 보고 고려):** `--ref-xc-drop 0.1 --xc-drop 0.05` + 추론 `cfg` 1.0 vs 1.5/2.0 비교(100장).
  - 기대: 샘플마다 무작위인 획 오류 감소 — 시드 3개 공통 오류 0, best-of-3 상한 0.004(§12) → 조건을 더 세게 따르면 줄 수 있다.
  - 근거는 Eruku cfg 1.0 → 1.5 의 CER 0.018 → 0.010 뿐, InkSpire 측정 없음. Xc 는 이미 위치까지 맞춘 강한 조건이라 효과 없을 수 있음.
  - 비용: HWD 악화 가능, 추론 2배. 1차와 따로 돌려야 어느 dropout 효과인지 분리된다.

### 내용 CFG 확대 평가 — 다음

2026-10-28: 최종 LoRA(dropout 없음) + `cfg=2.0` 이 실손글씨 hw_line CER 0.25 → 0.075, HWD 1.35 → 1.58(EXPERIMENTS §10 내용 CFG).
- 폰트 렌더 §5 n=300 (`cer_dump.py` + `inkspire:<lora>,cfg=…`)과 seosik_hw 에서 CER·HWD.
- cfg 1.5 / 2.0 / 2.5 / 3.0 스윕 — CER·HWD 절충점. 괜찮으면 `cfg` 기본값·HF Space 반영.
- **재학습 대기 중(2026-10-28):** `finetune_runs/inkspire_klein_refxc_xc` — `--ref-xc-drop 0.1 --xc-drop 0.05`(위 "2차" 를 1차 없이 바로, Eruku 짝),
  p512 와 같은 레시피 13k, 처음부터. 기존 xcdrop 은 생성 영역만 비운 옛 규약이라 v_u(Xc 전체 빈 이미지)와 안 맞았다.
  `waiter.sh` 가 다른 프로세스 없는 GPU(2 우선)를 잡으면 시작. 평가: hw_line 교차 cfg 1.5/2.0 × 시드 3(기존 두 LoRA 와 CER·HWD) + `--style-text none` CER(0.720 에서 회복?).

### (보관) 목표 Xc dropout — 용도 불일치로 미평가 → 2026-10-28 평가함(위)

★ 이 LoRA 는 생성 영역 Xc 만 비워 학습했다 — 지금 `xc_uncond`(Xc 전체 빈 이미지)와 규약이 달라 cfg 평가에 그대로 쓸 수 없다.

`finetune_runs/inkspire_klein_xcdrop`(`--xc-drop 0.1`, 2026-10-27 13k 완료)은 **생성 영역** Xc 를 비운 것(내용 CFG 용)이라
style text 를 버린 학습이 아니다 — 원했던 ref 전사 dropout 이 아님. cfg 평가는 시작 직후 중단, 결과 없음.
InkSpire 에서 CFG 가 CER 을 낮춘다는 측정은 없다(근거는 Eruku cfg 1.0 → 1.5 의 0.018 → 0.010 뿐).

## 구현 상태 (2026-10-27 커밋)

`--xc-drop`(Xc 전체 dropout) / `--ref-xc-drop`(ref 전사 dropout) / `cfg` 구현(train/inkspire.py · models/inkspire_klein.py · infer/inkspire.py · infer/show.py · configs/inkspire.yaml),
`line_h` 옵션, `cer_dump.py --seed-offset`, EXPERIMENTS §12. smoke 통과.

## 디스크 정리 (2026-10-27)

- 삭제: `finetune_runs/inkspire_p512`(FLUX.1-Fill-dev 전 체크포인트 + `_eval*` 몽타주, 39G), `finetune_runs/inkspire_layout_smoke`(1.6G).
  문서의 FLUX.1 수치·그림은 남아 있지만 그 LoRA 로 재생성은 불가. `docs/_gen_inkspire_wobble_ablate.py`·`_gen_inkspire_layout_line_len.py` 는 그 경로를 참조.
- 남은 정리 후보: `inkspire_klein_p512`·`inkspire_klein_xcdrop` 의 중간 step·`trainer.pt`, `inkspire_layout` 의 step 체크포인트.

## 보류 (근거와 함께 접은 것)

- ref 보강 inference loop — 상한(oracle)이 기준선과 같음(§8).
- ragged 학습 — 남은 오류가 빈 공간이 아니라 획 정확도(docs/inkspire.md §9.5).
- best-of-N + 판독기 — 상한 0.004 지만 외부 모델 의존·N배 느림(§12).
