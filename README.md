# InkSpire 한글 재현

논문 *Learning to Generate Stylized Handwritten Text via a Unified Representation of Style,
Content, and Noise* (Wang et al., ICLR 2026, OpenReview `FBPuLChGNX`) 의 한글 재현 구현.
코드 미공개라 논문만 보고 구현했다. [Eruku_korean_finetuning](https://github.com/HERIUN/Eruku_korean_finetuning) 에서 분리해 나왔다.

- 설계·이슈·측정값: [`docs/inkspire.md`](docs/inkspire.md)


## 구조

두 단계 분해 `p(X, Xc | C, Xs) = p(Xc | C, Xs) · p(X | Xs, Xc)`.

| 경로 | 역할 |
|---|---|
| `custom_datasets/korean/page.py` | 페이지 렌더러(X,Xc) + 글자별 bbox |
| `models/inkspire_layout.py` | ① masked CFM transformer(페이지내의 글자들 bbox 예측 모델), v-pred L1, 10-step ODE |
| `models/inkspire.py` | ② 이미지 — FLUX.1-Fill-dev + LoRA r=32 (115.9M, 논문 표 8), R-APE, 텍스트 인코더 제거 |
| `train/inkspire_layout.py`, `train/inkspire.py` | 각 트레이너 |
| `infer/inkspire.py` | one-shot 생성기 + 뷰어 (`InkSpireGen.gen`) |
| `experiments/gen_compare.py` | 두 체크포인트 생성 비교 몽타주 |
| `tools/fetch_flux.py` | FLUX 가중치 1회 다운로드 + 빈 프롬프트 임베딩 캐시 |
| `configs/*.yaml` | 조절 가능한 전부 |

지원 모듈은 원본 repo 에서 가져왔다. `configs/loader.py` · `korean/{fontset,alphabet}.py` ·
`upstream/constants.py` 는 그대로 복사했고, 아래 넷은 **여기서 쓰는 심볼만 남기고 잘라냈다** —
그래야 Eruku 모델(`models/eruku.py`)과 `upstream/font_square/*` 의존이 사라진다.
각 파일 docstring 에 무엇을 왜 버렸는지 적어 뒀다.

| 파일 | 줄 수 | 끊어낸 의존 |
|---|---|---|
| `custom_datasets/korean/split.py` | 471 → 175 | `upstream/font_square/*` |
| `train/core.py` | 733 → 97 | `models/eruku.py`, `korean/handb.py` |
| `infer/show.py` | 368 → 118 | `models/eruku.py` (InkSpire 백엔드만 남긴 `load_model`/`gen_from_style`) |
| `experiments/common.py` | 142 → 39 | VAE 실험 헬퍼 |

**CER 평가는 이 repo 에 없다.** HTR 리더와 `eval/htr_cer.py` 경로는 원본
[Eruku_korean_finetuning](https://github.com/HERIUN/Eruku_korean_finetuning) 에만 있다 —
CER 을 재려면 거기서 잰다.

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
    --lines "첫 줄" "둘째 줄"       # --layout-ckpt 생략 = std layout (실측상 이쪽이 낫다)
GPU=2 ./inference.sh inkspire --dry-run        # FLUX 없이 [x | xc | mask] 캔버스만

# 생성 비교 몽타주
GPU=2 ./eval.sh exp gen_compare --ckpt-after inkspire:finetune_runs/inkspire_p512/lora_last
```

`--ckpt` 뒤에 `,key=value` 로 `steps`·`guidance`·`std_font`·`trim_ref`·`degrade` 를 스윕할 수 있다.
정량 평가(CER)는 원본 repo 에서 한다. 지난 측정은 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## 필요한 자산 (전부 `.gitignore` — 심링크 권장)

| 경로 | 내용 |
|---|---|
| `assets/fonts_korean_v3/train` | 스타일 폰트 풀 12,951종 (+ `fonts_charsets.json`) |
| `assets/fonts_korean_v2/test` | held-out 폰트 16종 (val) |
| `assets/fonts_label/NanumGothic-Regular.ttf` | Xc 표준폰트. 스타일 풀에서는 `exclude_fonts` 로 제외 |
| `assets/corpus/{korean_lines,english_words}.txt` | 어절 공급 |
| `assets/backgrounds` | 종이 배경 패치 |
| `model_zoo/flux_fill_empty_prompt.pt` | 빈 프롬프트 임베딩 캐시 (`./train.sh fetch-flux` 가 만든다) |
| `data/ref_set_clean/train_lines.json` | 뷰어·평가용 style ref 세트 |

## 원본 repo 에 남긴 것

**Eruku 본체와 그 학습·평가 경로 전부** — 한글 HTR 리더와 CER 측정 스크립트를 포함한다.
이 repo 는 InkSpire 생성만 다룬다. CER 비교는 원본 repo 에서 같은 프로토콜로 재서
숫자를 맞춘다 — [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) 의 표가 그 방식으로 만들어졌다.

## 논문에서 우리 조건으로 못 옮기는 것

재현을 막는 것은 모델이 아니라 **데이터에 없는 라벨**이다.

| 논문 | 우리 | 왜 |
|---|---|---|
| 추론 레퍼런스의 **GT 레이아웃** — IAM 어절 / CASIA 글자 단위 주석(§4.1.1), 같은 문단의 다른 줄을 레퍼런스로(§4.3) | 표준폰트 렌더 bbox 를 ref 폭에 맞춘 **근사** | 스타일 이미지에 bbox 라벨이 없다. 이미지에서 글자를 분할해 실제 bbox 를 뽑아본 시도는 CER 0.20 → 0.61 로 더 나빴다(2026-09-14) |
| 실손글씨 다중 writer 페이지 (IAM 496 / CASIA 1,019 writer) | 합성 폰트 페이지 12,951종 | 한글 실손글씨 **페이지 + 레이아웃 주석** 데이터가 없다 |
| 패치 P=1024, A100 4장 | P=512, H100 1장 | 자원. 미검증 편차로 남아 있다 |

### 결과: Stage 1(레이아웃 CFM)은 학습되지만 추론에서 쓰면 손해다

레이아웃 모델 자체는 정상 학습된다 — 토큰별 L1 은 w 2.08 / h 1.93 / Δx 9.71 / Δy 10.08px 이고
w·h·Δy 는 논문 표 1 보다 낫다. 무너지는 건 추론이고, 이유는 둘이다.

1. **조건이 근사값이다.** 학습·검증에서는 레퍼런스 줄로 GT bbox 를 받지만 추론에서는 위 표의
   근사 bbox 를 받는다. 모델은 레퍼런스 토큰의 분산을 증폭한다.
2. **차분 표현이 줄을 따라 적분된다.** 타깃이 `[w, h, Δx, Δy]` 이고 `layout_to_bboxes` 가
   `x0 = x1_prev + Δx` 로 푼다. 글자당 9.71px 오차가 10글자 뒤엔 **~40px** 이 된다(평균 글자 폭
   23.7px 의 1.5~2배). `render_content` 가 표준폰트 글리프를 그 상자에 resize 해 박으므로
   Xc 에서 글자가 겹치거나 빠지고, 이미지 모델은 그 Xc 를 충실히 따라 그린다.

그래서 **기본 경로는 std layout**(표준폰트 렌더 bbox 그대로, 줄 간격 77px 고정)이다. 필체 적응은
없지만 누적 오차가 0 이고, 측정에서 6쌍 전부 같거나 나았다(held-out style-ref CER 0.197 vs 0.222).
`--layout-ckpt` 를 생략하면 이 경로를 탄다. 수치는 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) §12,
분해는 [`docs/inkspire.md`](docs/inkspire.md) §9.1.

레이아웃 모델을 살리려면 **재학습이 아니라** (a) 추론에서 진짜 레퍼런스 레이아웃을 줄 수 있거나
(b) 적분 누적을 안 타는 표현(절대 좌표 / 줄 단위 재정렬)으로 바꾸거나, 둘 중 하나가 필요하다.

## 알려진 이슈

- **레이아웃 체크포인트와 코드의 Δy 규약이 어긋나 있다(치명적이진 않다).** `page._line_base` 가
  이전 줄 max y1 → 평균 y1 로 바뀌었는데(회전 증강에서 max 는 끝단에 끌려간다, 줄 간 편차
  5.60px → 1.78px) `finetune_runs/inkspire_layout` 은 구규약으로 학습됐다. 실측(2026-09-15):
  encode/decode 가 같은 규약이라 왕복은 맞고 레퍼런스 줄은 Δy 기준 0 이라 동일 — 어긋남은
  **예측 줄의 상수 세로 오프셋**으로만 나타난다. 3줄 생성 기준 −2.1 / −4.6 / −7.0px 이고,
  평가 프로토콜인 2줄 캔버스에서는 2.1px 로 이 모델 자신의 Δy L1(10.15px) 안이다. 그대로 써도 된다.
  resume 도 가능하다(정규화 버퍼가 구규약이라 Δy 중심이 0.6σ 어긋난 채 이어질 뿐).
- **레이아웃 재학습은 보류한다.** 추론이 실제로 쓰는 오차는 w 2.08 / h 1.93 / Δx 9.71 / Δy 10.08px
  으로 어느 한 차원이 지배하지 않는다(줄 시작 Δx 446px 은 `infer/inkspire.py` :126 이 덮어쓴다 —
  차원 분해 `docs/inkspire.md` §9.1). Δy 규약만 고쳐 다시 돌려봐야 얻을 게 없고, 레이아웃 모델은
  그 전에 std layout 한테 진다(held-out CER 0.222 vs 0.197). 다음에 돌린다면 **레이아웃 데이터셋
  증강을 끄는 것**(레이아웃 타깃에서 aug 가 바꾸는 건 줄 회전뿐)이 가장 싸지만 그것도 Δy −17%,
  전체 −10% 이고 Δx 는 −3% 로 안 움직인다 — 즉 격차를 메울 카드는 아직 없다. 재학습하는 날
  새 Δy 규약은 공짜로 딸려온다.
- **`aug=false` 는 레이아웃 트레이너에만 해당한다.** 이미지 LoRA 는 증강을 끄면 깨끗한 폰트 렌더를
  학습한다(docs §8). 애초에 이미지 LoRA 는 `layout_seq` 를 안 써서 Δy 규약과도 무관하다 —
  09-10 run 의 val 샘플을 현재 코드로 재생성해 xc 는 픽셀 단위 동일, x 는 최대 6/255 차이로 확인했다.
- 논문 대비 남은 편차와 미검증 가설은 [`docs/inkspire.md`](docs/inkspire.md) §5 참고.

## 라이선스 주의

FLUX.1-Fill-dev 는 게이트 repo이자 **비상업 라이선스**다. 가중치를 받으려면 `hf auth login` 과
모델 페이지에서 라이선스 수락이 선행돼야 한다.
