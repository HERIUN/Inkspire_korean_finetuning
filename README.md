# InkSpire 한글 재현

논문 *Learning to Generate Stylized Handwritten Text via a Unified Representation of Style,
Content, and Noise* (Wang et al., ICLR 2026, OpenReview `FBPuLChGNX`) 의 한글 재현 구현.
코드 미공개라 논문만 보고 구현했다.

- 설계·이슈·측정값: [`docs/inkspire.md`](docs/inkspire.md)
- 논문 PDF 는 `.gitignore` 로 제외했다(12MB, 저작권). 필요하면 `docs/inkspire.pdf` 로 직접 넣는다.

## 구성

| 경로 | 역할 |
|---|---|
| `custom_datasets/korean/page.py` | 페이지 렌더러 + 두 데이터셋(P×P 패치 / 페이지 토큰). 글자별 bbox 를 렌더 시점에 얻는다 |
| `models/inkspire.py` | 이미지 모델 — FLUX.1-Fill-dev + LoRA r=32 (115.9M, 논문 표 8), R-APE, 텍스트 인코더 제거 |
| `models/inkspire_layout.py` | 레이아웃 모델 — masked CFM transformer 10층/512/8h |
| `train/inkspire.py`, `train/inkspire_layout.py` | 각 트레이너 |
| `infer/inkspire.py` | one-shot 생성기 + 뷰어 (`InkSpireGen.gen`) |
| `tools/fetch_flux.py` | FLUX 가중치 1회 다운로드 + 빈 프롬프트 임베딩 캐시 |
| `configs/inkspire*.yaml` | 조절 가능한 전부 |

self-check (외부 의존 없는 것만):

```bash
python models/inkspire.py --smoke          # tiny 모델, 다운로드 없음
python models/inkspire_layout.py --smoke   # 마스크 규칙·과적합·ckpt 왕복
```

## ★ 아직 standalone 이 아니다

`Eruku_korean_finetuning` 에서 갈라져 나온 스냅샷이라, 아래는 **아직 원본 repo 에 있고 여기 없다.**
`models/*.py` 두 개만 저장소-로컬 import 가 없어 그대로 돌아간다.

**코드**

| 원본 경로 | 여기서 쓰는 심볼 |
|---|---|
| `configs/loader.py` | `parse_args`, `PATH_KEYS`(inkspire 키 4개 포함) |
| `train/core.py` | `PATH_ARGS`, `data_paths_of`, `log_run_config`, `require_optim_state`, `load_rng_state`, `rng_state` |
| `custom_datasets/korean/fontset.py` | `aug_elastic`, `aug_morph`, `bg_patch`, `composite`, `jitter` |
| `custom_datasets/korean/split.py` | `ensure_font_charsets`, `font_files`, `build_samplers`, `data_paths`, `DEFAULT_EXCLUDE_FONTS` |
| `custom_datasets/korean/alphabet.py` | `load_charset` (vocab 2,509자) |
| `infer/show.py` | `FONTS_DIR`, `cell`, `label_img`, `render_in_font` (뷰어 전용) + `inkspire:` 디스패치 |
| `eval/htr_cer.py`, `experiments/gen_compare.py` | 평가 진입점 |
| `train.sh` / `inference.sh` / `eval.sh` | `inkspire`, `inkspire-layout`, `fetch-flux` 서브커맨드 |

`train/core.py` 와 `custom_datasets/korean/split.py` 는 각각 `models/eruku.py`,
`custom_datasets/upstream/font_square/*` 를 끌어온다 — 그래서 그냥 복사하면 Eruku 본체가 딸려 온다.
분리를 끝내려면 위 6개 헬퍼를 작은 모듈로 뽑아내고 `infer/show.py` import 를 지연 로드로 바꿔야 한다.

**자산** (전부 `.gitignore`)

`assets/fonts_korean_v3/train`(12,951종) · `assets/fonts_korean_v2/test`(held-out) ·
`assets/fonts_label/NanumGothic-Regular.ttf`(Xc 표준폰트) · `assets/corpus/*` ·
`assets/backgrounds` · `model_zoo/flux_fill_empty_prompt.pt`

**의존성**: `diffusers>=0.38`, `peft>=0.18`, `prodigyopt>=1.1`, torch, opencv, pillow

## 라이선스 주의

FLUX.1-Fill-dev 는 게이트 repo이자 **비상업 라이선스**다. 가중치를 받으려면 `hf auth login` 과
모델 페이지에서 라이선스 수락이 선행돼야 한다.
