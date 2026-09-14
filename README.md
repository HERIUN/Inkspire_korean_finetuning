# InkSpire 한글 재현

논문 *Learning to Generate Stylized Handwritten Text via a Unified Representation of Style,
Content, and Noise* (Wang et al., ICLR 2026, OpenReview `FBPuLChGNX`) 의 한글 재현 구현.
코드 미공개라 논문만 보고 구현했다. `Eruku_korean_finetuning` 에서 분리해 나왔다.

- 설계·이슈·측정값: [`docs/inkspire.md`](docs/inkspire.md)
- 논문 PDF 는 `.gitignore` 로 제외했다(12MB, 저작권). 필요하면 `docs/inkspire.pdf` 로 직접 넣는다.

## 구조

두 단계 분해 `p(X, Xc | C, Xs) = p(Xc | C, Xs) · p(X | Xs, Xc)`.

| 경로 | 역할 |
|---|---|
| `custom_datasets/korean/page.py` | 페이지 렌더러 + 두 데이터셋. 글자별 bbox 를 렌더 시점에 공짜로 얻는다 |
| `models/inkspire_layout.py` | ① 레이아웃 — masked CFM transformer 10층/512/8h, v-pred L1, 10-step ODE |
| `models/inkspire.py` | ② 이미지 — FLUX.1-Fill-dev + LoRA r=32 (115.9M, 논문 표 8), R-APE, 텍스트 인코더 제거 |
| `train/inkspire_layout.py`, `train/inkspire.py` | 각 트레이너 |
| `infer/inkspire.py` | one-shot 생성기 + 뷰어 (`InkSpireGen.gen`) |
| `tools/fetch_flux.py` | FLUX 가중치 1회 다운로드 + 빈 프롬프트 임베딩 캐시 |
| `configs/*.yaml` | 조절 가능한 전부 |

지원 모듈(`configs/loader.py`, `custom_datasets/korean/{fontset,split,alphabet}.py`,
`train/core.py`, `infer/show.py`)은 원본 repo 에서 가져왔고, `split.py`·`train/core.py`·
`infer/show.py` 는 **여기서 쓰는 심볼만 남기고 잘라냈다** — 그래야 Eruku 모델과
`custom_datasets/upstream/font_square/*` 의존이 사라진다. 각 파일 docstring 에 무엇을 왜
버렸는지 적어 뒀다.

## 실행

```bash
uv sync                      # 또는 pip install -e .
ln -s <경로> assets          # 아래 「필요한 자산」 참고
ln -s <경로> data            # 뷰어의 style ref 세트(선택)

# 의존 없는 self-check
python models/inkspire.py --smoke              # tiny 모델, 다운로드 없음
python models/inkspire_layout.py --smoke       # 마스크 규칙·과적합·ckpt 왕복
python custom_datasets/korean/page.py --n 4    # bbox 왕복·shape·마스크·collate + PNG 덤프

# ① 레이아웃 (FLUX 불필요)
GPU=2 ./train.sh inkspire-layout

# ② 이미지
GPU=2 ./train.sh fetch-flux                    # hf auth login + 라이선스 수락 선행, 34GB
GPU=2 ./train.sh inkspire

# 추론
GPU=2 ./inference.sh inkspire --lora-dir finetune_runs/inkspire_p512/lora_last \
    --lines "첫 줄" "둘째 줄" [--layout-ckpt finetune_runs/inkspire_layout/checkpoint_last.pth]
GPU=2 ./inference.sh inkspire --dry-run        # FLUX 없이 [x | xc | mask] 캔버스만
```

## 필요한 자산 (전부 `.gitignore` — 심링크 권장)

| 경로 | 내용 |
|---|---|
| `assets/fonts_korean_v3/train` | 스타일 폰트 풀 12,951종 (+ `fonts_charsets.json`) |
| `assets/fonts_korean_v2/test` | held-out 폰트 16종 (val) |
| `assets/fonts_label/NanumGothic-Regular.ttf` | Xc 표준폰트. 스타일 풀에서는 `exclude_fonts` 로 제외 |
| `assets/corpus/{korean_lines,english_words}.txt` | 어절 공급 |
| `assets/backgrounds` | 종이 배경 패치 |
| `model_zoo/flux_fill_empty_prompt.pt` | 빈 프롬프트 임베딩 캐시 (`./train.sh fetch-flux` 가 만든다) |
| `data/ref_set_clean/train_lines.json` | 뷰어용 style ref 세트 (선택) |

## 원본 repo 에 남긴 것

**Eruku 비교 평가** — `eval/htr_cer.py`, `experiments/gen_compare.py`, `infer/show.py` 의
`inkspire:<lora_dir>[,<layout_ckpt>]` 디스패치. 이건 Eruku 체크포인트와 한글 HTR 리더가 같이
있어야 의미가 있어서 `Eruku_korean_finetuning` 에 둔다. 측정 결과는 그쪽 `docs/EXPERIMENTS.md` §12.

## 알려진 이슈

- **기존 레이아웃 체크포인트는 무효다.** Δy 기준을 이전 줄 max y1 → 평균 y1 로 바꿔서
  (회전 증강 대응, 줄 간 편차 5.60px → 1.78px) 학습 규약이 달라졌다. 재학습해야 한다.
  이미지 LoRA 는 영향 없다.
- 논문 대비 남은 편차와 미검증 가설은 [`docs/inkspire.md`](docs/inkspire.md) §5 참고.

## 라이선스 주의

FLUX.1-Fill-dev 는 게이트 repo이자 **비상업 라이선스**다. 가중치를 받으려면 `hf auth login` 과
모델 페이지에서 라이선스 수락이 선행돼야 한다.
