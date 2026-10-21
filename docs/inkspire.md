# InkSpire 한글 재현 — 구현 내용 · 이슈 · 코드 위치

논문: *Learning to Generate Stylized Handwritten Text via a Unified Representation of Style, Content, and Noise*
(Wang et al., ICLR 2026, OpenReview `FBPuLChGNX`, 사본 `docs/inkspire.pdf`). 코드 미공개라 논문만 보고 구현했다.
브랜치 `inkspire` (2026-09-10~). 측정값·재현 커맨드는 [`EXPERIMENTS.md` §12](EXPERIMENTS.md#12-inkspireflux-fill-lora--레이아웃-cfm-한글-재현-2026-09),
설계 근거는 계획 파일(`~/.claude/plans/functional-inventing-lovelace.md`).

★ 논문 표 2·3 은 HWT/VATr/One-DM/DiffPen/TGC-Diff 와만 비교한다. **Eruku/Emuru 와의 직접 비교 수치는 논문에 없다.**
"Eruku 보다 좋은가"는 §12 표(같은 HTR CER 프로토콜)로만 판정한다.
★ 2026-10-20: 그 자체 HTR 리더가 InkSpire 출력을 과하게 나쁘게 읽었다 — 독립 리더(PP-OCRv6) 재측정은 §9.6. **판정은 §9.6 을 따른다.**
★ 2026-10-21: **최종 결론 §10 — 한 줄 생성은 Eruku 를 주력으로, InkSpire 는 현 상태로 보류.**

---

## 1. 논문 요약 (구현에 필요한 것만)

두 단계 분해 p(X, Xc | C, Xs) = p(Xc | C, Xs) · p(X | Xs, Xc).

| 단계 | 내용 |
|---|---|
| ① 레이아웃 p(Xc \| C, Xs) | 글자마다 bbox b=[w, h, Δx, Δy] (페이지 폭 정규화). Masked Modeling + Conditional Flow Matching 이 최선(표 1). 10층 transformer, hidden 512, 8 heads, v-prediction L1, 10-step ODE. 마스킹 = 연속 구간 / 토큰별 20% 각 50%. 추론 시 첫 줄 = 레퍼런스. 예측 bbox 에 **표준 폰트**로 콘텐츠 이미지 Xc 렌더. |
| ② 이미지 p(X \| Xs, Xc) | **FLUX.1-Fill-dev** + LoRA r=32/α=32 (115.9M, 표 8). 텍스트 인코더 제거. I = X ⓒ Xc(공간 연결), 마스크 영역만 CFM 손실(식 10-11). **R-APE**: X·Xc 를 90° 회전 후 연결, Xc 토큰의 RoPE 위치 id = 대응 X 토큰과 동일. 학습 = 페이지에서 P×P 패치 + R-Mask(랜덤 다중 사각형). Prodigy lr 1, batch 4, 20k iter. 추론 20 ODE step. |
| 추론 | one-shot: 스타일 한 줄을 첫 줄에 두고 아래 줄을 마스크, Xc 에 목표 텍스트 → 여러 줄 동시 생성. 편집 = 마스크만 변경. |

---

## 2. 우리 설정에서의 설계 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 데이터 | Eruku 와 같은 **합성 폰트 페이지**(`assets/fonts/train` 12,951종). 글자별 bbox 는 렌더 시 공짜 | 실손글씨 페이지+레이아웃 라벨이 없음 |
| 패치 P | 기본 512 (1024 옵션) | 단일 GPU. 64px 라인 기준 512 에 6~8줄 |
| 콘텐츠 폰트(Xc) | **NanumGothic-Regular**(`assets/fonts/label/NanumGothic-Regular.ttf`). `configs/inkspire.yaml data.std_font` / `configs/infer.yaml inkspire.std_font` 로 노출, 학습·추론이 같아야 한다. 스타일 폰트 풀에서는 `exclude_fonts: [NanumGothic]` 으로 제외 | 라벨 렌더에 이미 쓰던 자산(신규 자산 없음). 표준폰트가 스타일로 다시 나오면 과제가 복사로 퇴화 |
| 텍스트 인코더 | 빈 프롬프트 임베딩 1회 캐시 → 학습·추론에 T5-XXL/CLIP 미탑재 | 논문: 텍스트 인코더 제거. 0 벡터가 아니라 실제 빈 프롬프트 임베딩이어야 백본이 분포 안에서 시작 |
| guidance | 30 고정(학습=추론) | Fill-dev 인페인팅 동작점. 텍스트가 없으니 상수 조건 스칼라, train/infer 동일성만 중요. 논문 미기재 |
| 레이아웃 모델 | Masked+CFM 만 구현(AR/Masked 변형 생략) | 표 1 에서 열세 |
| 논문 편차 | `line_emb` 추가, additive t(adaLN 아님), t~U(0,1), 10% 무레퍼런스 마스킹 | 코드 내 `ponytail:` 주석 참고. 무레퍼런스 모드는 레이아웃 라벨 없는 실손글씨 추론용 |
| 단계 게이트 | 데이터 → 이미지 모델(oracle 레이아웃) → 레이아웃 모델 → 통합 추론·Eruku 비교 | 이미지 모델 성패를 레이아웃 오차와 분리 |

---

## 3. 코드 위치

### 신규 파일

| 파일 | 역할 | 핵심 심볼 |
|---|---|---|
| `custom_datasets/korean/page.py` | 페이지 렌더러 + 두 데이터셋 | `render_line` :91 (글자별 bbox, 공백=h 0 토큰) · `render_content` :136 (표준폰트 글리프를 bbox 에 resize 합성 = Xc, 추론 공용) · `render_page` :173 (줄 단위 ±3° 회전 → 페이지 단위 elastic/morph/blur/배경/jitter, `fontset.py` 재사용) · `layout_seq` :229 / `layout_to_bboxes` :243 (Fig 2 파라미터, bbox 만으로 역변환) · `vocab` :259 / `char_ids` :270 (2,511 = pad, unk + `alphabet.load_charset()`) · `rmask` :277 (1~4 사각형 합집합, 16px 격자 스냅, 커버리지 [0.15,0.85]) · `line_mask` :295 · `KoreanPageDataset` :307 (`mode=rand|top|line` → `{x, xc, mask}` [1,P,P]) · `KoreanLayoutDataset` :384 (`MAX_TOKENS` 캡) · `layout_collate` :402 |
| `models/inkspire.py` | FLUX-Fill LoRA 이미지 모델 | `LORA_TARGETS` :49 (표 8 13종, full-match 정규식) · `load_flux` :87 · `empty_prompt_cache` :101 · `build_inputs` :129 (R-APE 회전, X\|Xc 연결, VAE 인코딩, FluxFill 마스크 패킹, APE id 타일, `loss_mask`) · `cfm_loss` :165 (식 10-11, logit-normal σ + 해상도 shift) · `generate` :179 (Euler ODE → X 절반 역회전 → 비마스크 픽셀 보존) · `save_lora` :95 |
| `models/inkspire_layout.py` | 레이아웃 CFM 모델 | `LayoutCFM` :33 (char/pos/line/obs 임베딩 + layout_proj + t_mlp, `mean/std` 버퍼 → `set_norm` :57 필수) · `make_gen_mask` :83 · `cfm_loss` :109 (마스크 토큰 v-pred L1) · `sample` :128 (마스크 토큰만 Euler, 관측 토큰 bit-identical) · `save_ckpt` :147 / `load_layout` :154 |
| `train/inkspire.py` | 이미지 LoRA 트레이너 | `make_parser` :45 → `configs/inkspire.yaml` · `main` :91 (Prodigy/AdamW, grad-accum, clip, bf16 autocast, val = `mode=line` 고정 배치 + `generate` 그리드 PNG, `lora_step_%06d/{pytorch_lora_weights.safetensors, trainer.pt}` + `lora_last` 심링크, `--resume <lora_dir>`) |
| `train/inkspire_layout.py` | 레이아웃 트레이너 | `make_parser` :37 → `configs/inkspire_layout.yaml` · `validate` :77 (val loss + `sample` 의 bbox L1 px) · `main` :91 (step 0 `set_norm`, AdamW, `checkpoint_step_%06d.pth`/`checkpoint_last.pth`) |
| `infer/inkspire.py` | one-shot 생성기 + 뷰어 | `InkSpireGen` :71 — `prepare` :86 (ref 잉크 48px 로 정규화, 레퍼런스 bbox = 표준폰트 렌더를 ref 폭으로 스케일, 레이아웃 모델 `sample` 또는 std layout, 줄마다 좌측 여백 고정, 2줄+ 캔버스·Xc·마스크 구성) · `gen_lines` :153 · `gen` :166 (Eruku `gen_from_style` 과 같은 계약) · CLI `main` :177 (`--lines a b c` 다중 줄, `--dry-run` 캔버스만) |
| `tools/fetch_flux.py` | FLUX 34GB 1회 다운로드 + 빈 프롬프트 캐시 `model_zoo/flux_fill_empty_prompt.pt` | 로그인/라이선스 미완이면 한글 안내 후 exit 1 |
| `configs/inkspire.yaml`, `configs/inkspire_layout.yaml` | 조절 가능한 전부(paths 4키는 `base.yaml` 상속 불가라 복제) | |

### 수정 파일

| 파일 | 변경 |
|---|---|
| `infer/show.py` :137, :197 | `load_model("inkspire:<lora_dir>[,<layout_ckpt>]")` 디스패치(지연 import) + `gen_from_style` 이 `.gen` 있는 모델은 그대로 호출 → `experiments/gen_compare.py` (및 원본 repo 의 `eval/htr_cer.py`) 무수정으로 두 백엔드 비교 |
| `configs/loader.py` :46 | `PATH_KEYS` 에 `empty_cache`, `lora_dir`, `layout_ckpt` |
| `configs/eval.yaml`, `configs/infer.yaml` | `inkspire:` 섹션 |
| `train/core.py` | `log_run_config` 추출(두 신규 트레이너가 import) |
| `train.sh` / `inference.sh` / `eval.sh` | `inkspire`, `inkspire-layout`, `fetch-flux` / `inkspire` / 헤더 안내 |
| `pyproject.toml` | `peft>=0.18`, `prodigyopt>=1.1` |
| `README.md`, `docs/EXPERIMENTS.md` | repo 구조 트리, §12 |

### self-check (전부 통과, 2026-09-10)

```bash
.venv/bin/python custom_datasets/korean/page.py --out /tmp/kpage --n 4        # bbox 왕복·shape·마스크·xc 잉크·collate, PNG 덤프 (~33s)
.venv/bin/python models/inkspire.py --smoke                                     # tiny 모델, 다운로드 없음 (~5s)
GPU=2 .venv/bin/python models/inkspire.py --smoke --with-weights                # trainable == 115,912,704, peak VRAM
.venv/bin/python models/inkspire_layout.py --smoke                              # 마스크 규칙·샘플러·과적합·ckpt 왕복
GPU=2 ./train.sh inkspire-layout --out finetune_runs/inkspire_layout_smoke --max-steps 2 --batch-size 4 --num-workers 0 --norm-batches 1 --val-every 1 --val-batches 1 --save-every 1
GPU=2 ./train.sh inkspire --out finetune_runs/inkspire_smoke --max-steps 2 --batch-size 1 --P 256 --num-workers 0 --val-every 1 --val-batches 1 --val-steps 2 --save-every 1
GPU=2 ./inference.sh inkspire --dry-run --layout-ckpt <ckpt> --lines "첫 줄" "둘째 줄" --out _debug/inkspire_dry.png
```

---

## 4. 검증된 사실 (구현 중 확인)

- 표 8 의 LoRA 대상 이름은 diffusers 모듈명 그대로(`ff.net.2`, `norm1.linear`, `norm.linear`). r=32 로 계산한 합이 **115,912,704 = 논문 115.91M** 과 정확히 일치 → 매핑은 항등. 단, suffix 매칭이면 최상위 `transformer.proj_out` 까지 잡히므로 full-match 정규식이어야 한다.
- diffusers 0.38: `load_lora_adapter` 는 `use_safetensors=True` 와 `adapter_name="default"` 를 명시해야 resume 후 `save_lora_adapter` 가 동작한다(없으면 `default_0` 로 등록되어 저장 실패).
- transformer 는 timestep·guidance 를 내부에서 ×1000 → σ∈(0,1] 그대로 넘긴다. `img_ids` 는 2-D [L,3].
- APE 검증 포인트: 토큰은 행 우선이라 X/Xc 가 행마다 교차한다. `ids.view(hh, 2ww, 3)[:, :ww] == [:, ww:]` 가 맞는 불변식이고 `ids[:L/2]==ids[L/2:]` 는 아니다.
- grad checkpointing 게이트는 `torch.is_grad_enabled()` (train() 여부 무관). `generate` 는 no_grad.
- 실측: P=512 B=4 → 2,048 토큰(+txt 512), **2.6 s/step, peak 30.6GB** → 20k step ≈ 14.5h (계획 7~9h 과소, 실제 run 은 23h). P=256 B=1 fwd+bwd 23.5GB.
- **P=1024 도 VRAM 은 여유다**(2026-09-15 실측, 한 GPU 에 다른 작업 34GB 와 공유한 채로):
  P=512 B=4 30.5GB / 5.0 s·step · P=1024 B=2 34.9GB / 10.0 s·step · **P=1024 B=4 45.4GB / 20.0 s·step**.
  토큰이 4배(2,048 → 8,192)라 step 시간이 정확히 4배다. 즉 논문 설정(P=1024, batch 4)은
  배치를 줄이지 않고 H100 한 장에 올라가고, 제약은 VRAM 이 아니라 **시간(20k ≈ 4일)** 뿐이다.

---

## 5. 이슈 로그

| # | 증상 | 원인 | 조치 |
|---|---|---|---|
| 1 | OpenReview pdf/forum/API 전부 403 | 이 서버 → Cloudflare 브라우저 검증 | 사용자가 PDF 를 `docs/inkspire.pdf` 로 넣음 |
| 2 | FLUX.1-Fill-dev 다운로드 불가 | 게이트 repo + **비상업 라이선스**, 미로그인 | `hf auth login --token …` + 모델 페이지 라이선스 수락 → `./train.sh fetch-flux`(34GB). 릴리즈 시 라이선스 주의 |
| 3 | 레이아웃 학습 첫 배치 `AssertionError: N=1049 > max_len=1024` | pitch≈42px·page_w 1024 페이지가 1,000+ 토큰 | `KoreanLayoutDataset.MAX_TOKENS` 로 줄 경계 캡, 모델 `max_len` 2048 |
| 4 | 레이아웃 학습이 GPU **76GB** 점유 | batch 160 × 최대 2,048 토큰의 attention·FFN 활성화 | `MAX_TOKENS` 1024 로 캡(초과는 극단 pitch 뿐). 그래도 이미지 run 과 동시 실행 시 83GB·이미지 속도 반토막 → **s8000 에서 중단**, 이미지 학습 후 `--resume` |
| 5 | 레이아웃 학습 속도 0.94 it/s (계획 "40분" → 실제 6h) | 페이지 렌더가 CPU 병목 | val bbox L1 이 s2000 부터 11~12px 정체(val loss 는 하강) → 20k 필요성 의문, s8000 사용 중 |
| 6 | 추론 시 생성 줄이 오른쪽으로 누적 밀림 | 줄 시작 Δx(= −이전 줄 폭)를 모델이 가장 못 맞춤. 학습 줄은 page_w 를 채우지만 추론 줄은 짧음 | `infer/inkspire.py prepare`: 줄마다 첫 토큰을 좌측 여백으로 평행이동(학습 페이지는 전부 좌측 시작). 크기·간격은 예측값 유지 |
| 7 | 배경 대기 셸이 "low on memory" 로 반복 종료 | 하네스 자체 제한(머신 RAM 은 685GB 여유) | `Monitor` 로 대체. 학습 프로세스엔 영향 없음 |
| 8 | `pkill -f <패턴>` 이 대기 셸 자신까지 죽임(exit 144) | 패턴이 자기 커맨드라인에도 매치 | `pgrep -f "^/data/.*python train/…"` 처럼 실행 경로로 앵커 |
| 9 | `_with_weights` 스모크 OOM | 당시 레이아웃 트레이너가 GPU 2 의 76GB 점유(이슈 4) | 레이아웃 중단 후 재실행 → 통과 |

미해결/관찰:
- **숫자가 가장 약하다**(s1000 몽타주에서 "2026" → "2020"/판독불가). 학습 진행에 따라 재확인.
- 레퍼런스 레이아웃은 근사다(표준폰트 렌더 bbox 를 ref 폭으로 스케일). 실제 글리프 위치와 다르므로 결과가 나쁘면 std layout(`--layout-ckpt` 생략) 수치와의 차이로 원인을 분리한다.
- 평가 텍스트(`eval_lines_ko.txt`)와 학습 코퍼스의 어절 단위 중복은 기존 repo 관례 그대로.
- `docs/inkspire.pdf`(12MB) 커밋 여부 미결정.

---

## 6. 진행 상태 (2026-09-11 16:00 기준)

| 항목 | 상태 |
|---|---|
| 이미지 LoRA `finetune_runs/inkspire_p512` | **완료** 20,000 step (val loss 0.193 → 0.168, 평균 0.27 it/s, peak 30.6GB). `lora_last` = `lora_step_020000` |
| 레이아웃 `finetune_runs/inkspire_layout` | s8000 에서 중단(val bbox L1 11.7px). **재개할 이유가 약함** — 아래 결과에서 레이아웃 모델이 std layout 보다 나쁨 |
| Stage 4 평가 | 3 세트 × {std layout, +layout} × {echo, style-ref} n=300 완료 → `finetune_runs/_eval/inkspire_s20000_*.txt`, 표는 EXPERIMENTS §12 |
| 판정 | **최종: §10 — 보류.** ~~**Eruku 우세.** held-out style-ref 중앙값 Eruku 0.000 / 폭주 0.0% vs InkSpire 최고(s13000) 0.190 / 1.7%~~ → **자체 HTR 착시 (§9.6).** PP-OCRv6 로는 중앙값 둘 다 0, corpus CER Eruku 0.018 vs InkSpire 0.040(tight_mask 0.034) — Eruku 가 여전히 앞서지만 격차는 숫자·기호·붓글씨류에 몰려 있다 |
| 최적 체크포인트 | **s13000**(0.190) > s20000(0.222). val loss 는 s20000 이 최저 — 손실로 고르면 틀린다 |
| 레이아웃 모델 기여 | **음(−)**. 6 쌍 전부 std layout 이 같거나 나음. n=40 중간 점검의 반대 결과는 표본 착시였음 |
| 약점 | (PP-OCRv6, §9.6) 숫자 12.7%·기호 8.9% 오류(Eruku 2.5%·2.7%), 붓·펜글씨(NanumBrushScript 중앙값 0.173, NanumPenScript 0.053), 한글은 모음(중성) 치환 비중이 큼. ~~굵은 폰트 약점~~(BagelFatOne 0.423 등)은 HTR 착시 |

### 남은 선택지 (하려면)

| 후보 | 근거 | 비용 |
|---|---|---|
| P=1024 재학습 | 논문 설정. 지금 가장 큰 편차이고 논문 A.5.4 가 해상도 축소의 손해를 지적 | 20k step, **배치 4 그대로**(45.4GB), 약 90h |
| 레이아웃 20k 완주 | bbox L1 이 s2000 부터 정체라 기대값 낮음. 지금은 std layout 이 더 나음 | 6h |
| 숫자·붓글씨 보강 | 코퍼스 숫자 비중 상향, 붓글씨 폰트 가중 샘플링 | 데이터만 수정, 재학습 필요 |

재현 커맨드는 `docs/EXPERIMENTS.md` §12 에 있다.

---

## 7. 모델 입출력 — 무엇이 들어가 무엇이 나오고 손실이 어떻게 계산되나

그림은 실제 코드 경로(`KoreanPageDataset` → `build_inputs`, `KoreanLayoutDataset` → `layout_collate`)를 그대로 돌려 만든 것이다.
재생성: `PYTHONPATH=. .venv/bin/python docs/_gen_inkspire_io.py` → `docs/img_inkspire_io/` + `numbers.json`(아래 수치의 출처).

### 7.1 Stage 1 — 레이아웃 CFM (`models/inkspire_layout.py`)

![레이아웃 토큰](img_inkspire_io/01_layout_tokens.png)

한 페이지가 곧 한 샘플이다. 위 그림의 페이지는 9줄·240 토큰(공백 포함), 932×1024 px, pitch 98 px 였다.
글자 하나 = 토큰 하나이고, 토큰의 타깃은 bbox 를 페이지 폭으로 나눈 4 개 값 `[w, h, Δx, Δy]` 다
(`Δx` = 이전 글자 오른쪽 끝과의 차, `Δy` = 이전 줄 baseline 과의 차, `layout_seq` :229).

| 글자 | bbox(px) | `[w, h, Δx, Δy]` |
|---|---|---|
| 븃 | 8, 38, 55, 100 | 0.0459, 0.0605, 0.0078, 0.0371 |
| 하 | 52, 42, 94, 93 | 0.0410, 0.0498, −0.0029, 0.0410 |
| (공백) | 172, 83, 193, 85 | 0.0205, 0.0020, −0.0020, 0.0811 |

배치(`layout_collate`)는 가변 길이를 `Nmax` 로 패딩한다. 위 그림의 2 샘플 배치는 `N=281` 이었다.

| 텐서 | shape | 뜻 |
|---|---|---|
| `char_ids` | [B, N] long | vocab 2,511 (`<pad>`, `<unk>` + charset 2,509) |
| `layout` | [B, N, 4] f32 | 타깃 `[w,h,Δx,Δy]` (page_w 정규화) |
| `line_id` | [B, N] long | 줄 번호 0…, 0 = 레퍼런스 줄 |
| `ref_mask` | [B, N] bool | `line_id == 0` = 관측(조건) 토큰 |
| `pad_mask` | [B, N] bool | True = 패딩 |

한 step:

```
x0      = (layout − mean) / std            # 버퍼 mean/std, step 0 에 20 배치로 set_norm (★ 필수)
gen     = make_gen_mask(ref, pad)          # 연속구간 45% / 토큰별 20% i.i.d. 45% / 전체마스크 10%
t ~ U(0,1)[B],  z ~ N(0,I)[B,N,4]
x_t     = (1−t)·x0 + t·z
입력    = char_ids, line_id, where(gen, x_t, x0), gen, t, pad_mask
```

토큰 벡터는 6 개 임베딩의 합(512-d): `char_emb + pos_emb + line_emb + layout_proj(4→512) + obs_emb(관측/마스크) + t_mlp(sinusoid(t))`.
이걸 10층 Transformer encoder(8 head, FFN 2048, pre-LN)에 넣고 `LayerNorm → Linear(512,4)` 로 내보낸다.

- **출력** `v [B, N, 4]` — 속도 예측. pad 위치 출력은 버린다.
- **손실**(`cfm_loss` :109) — 마스크 토큰만의 v-prediction **L1**: `Σ |v − (z − x0)| · gen / (Σgen · 4)`.
- **추론**(`sample` :128) — `gen = ~ref & ~pad` 만 노이즈로 시작해 10-step Euler(t: 1→0.1), 마지막에 `denormalize`. 관측 토큰은 입력값 그대로 나온다.
- 모델 33.77M 파라미터. 배치 160(우리 실행은 128), 페이지당 최대 1,024 토큰(`KoreanLayoutDataset.MAX_TOKENS`).

### 7.2 Stage 2 — 이미지 (FLUX.1-Fill-dev + LoRA, `models/inkspire.py`)

![패치](img_inkspire_io/02_patch.png)

데이터셋은 페이지에서 P×P(=512) 패치를 뜯어 3 장을 준다. 전부 `[1,512,512] f32 ∈ [−1,1]`(mask 만 0/1).

| 텐서 | 뜻 |
|---|---|
| `x` | 스타일 폰트로 렌더한 손글씨 패치 (생성 타깃 + 스타일 조건을 겸함) |
| `xc` | 같은 bbox 에 표준폰트 글리프를 채운 콘텐츠 이미지 |
| `mask` | 1 = 생성할 영역. **학습은 `rand`=R-Mask 만 쓴다**(1~4 사각형 합집합, 커버리지 0.15~0.85, 논문 §3.3.1). `line`(첫 줄 아래 전부)은 val 전용이고 추론 마스크를 흉내낸 것이다(논문 §3.4). `top`(상단 1/8 만 보임)은 논문 §4.4.2 의 F-TopMask 어블레이션용 옵션 — 기본값 아님 |

**학습 (R-Mask, 실제 학습 경로)**

![R-APE 학습](img_inkspire_io/03_rape_train.png)

**추론 모사 (첫 줄만 보임, val 배치와 `infer/inkspire.py` 가 만드는 마스크)**

![R-APE 추론](img_inkspire_io/04_rape_infer.png)

**APE / R-APE 가 무엇을 바꾸나**

![APE 설명](img_inkspire_io/06_ape.png)

토큰은 행 우선으로 펴지므로 X 와 Xc 를 가로로 붙이면 두 종류가 행마다 번갈아 나오고, 짝까지의 좌표 차이가
그림 폭에 따라 달라진다(학습 패치 +32칸, 추론 캔버스 +39칸). APE 는 Xc 토큰에 짝의 좌표를 복사해 그 차이를 0 으로 만든다.
★ 따라서 **R-APE 의 회전은 RoPE 거리를 더 줄이는 게 아니다** — APE 로 이미 0 이다. 회전이 바꾸는 것은
(a) 캔버스·시퀀스에서의 물리적 거리(추론 캔버스 39칸 → 11칸), (b) 이음매에서 w 좌표가 되감기는 폭(38 → 10),
(c) 캔버스 종횡비. 논문은 메커니즘을 설명하지 않고 표 4·5 의 수치(FID 9.31 → 7.92, ΔCER 0.05 → 0.01)로만 주장한다.
우리는 R-APE 로만 학습했고 ablation 은 안 했다(각 14h).
조건인지 생성 대상인지는 위치가 아니라 마스크 채널이 구분한다. 그림 재생성: `docs/_gen_inkspire_ape.py`.

`build_inputs` :129 가 하는 일 (그림이 각각 I, I_m, I_i):

```
rot90 시계방향 → I  = [rot(x) | rot(xc)]        [B,1,512,1024]     ← R-APE
                 I_m = [rot(mask) | 0]           (Xc 절반 마스크 0 = 항상 보임)
                 I_i = I ⊗ (1 − I_m)
VAE 인코딩       (I) → [B,16,64,128] → pack(2×2) → x0_packed   [B,2048,64]
                 (I_i) →              → pack     → masked      [B,2048,64]
mask 픽셀언셔플  I_m → [B,64,64,128]  → pack     → mask 토큰    [B,2048,256]
cond_packed = [masked ⊙ mask 토큰]                              [B,2048,320]
img_ids     = X 절반 격자(32×32)를 폭 방향으로 2회 타일           [2048,3]   ← Xc 토큰 id == 대응 X 토큰 id
loss_mask   = mask 토큰이 하나라도 1 인 토큰                       [B,2048] bool
              (Xc 절반은 항상 0 → 위 예시 R-Mask 패치 29.6%, 추론 모사 패치 39.1%)
```

transformer 입력은 `hidden_states = [x_t ⊙ cond_packed]` = **[B, 2048, 384]** (노이즈 64 + 마스크된 latent 64 + 마스크 256).
텍스트 자리에는 빈 프롬프트 캐시가 그대로 들어간다: `encoder_hidden_states [1,512,4096]`, `pooled_projections [1,768]`, `txt_ids [512,3]`.
`guidance = 30` 고정 스칼라. 출력은 **`v [B, 2048, 64]`**.

한 step (`cfm_loss` :165, 식 10-11):

```
u ~ logit-normal(0,1)                        # compute_density_for_timestep_sampling
σ = time_shift(mu, 1, u),  mu = calculate_shift(L=2048) = 0.8033
x_t = (1−σ)·x0 + σ·z,   z ~ N(0,I)
v   = transformer([x_t ⊙ cond], timestep=σ, guidance=30, 빈 프롬프트, img_ids)
loss = Σ_tok mean_ch (v − (z − x0))² · loss_mask / Σ loss_mask
```

즉 마스크된 latent 토큰에서만 v-prediction MSE 를 재고, Xc 절반과 보이는 영역은 손실에 안 들어간다. perceptual/CTC 같은 추가 손실 없음.

추론(`generate` :179)은 σ 를 `linspace(1, 1/steps, steps)` 로 놓고 20-step Euler → `unpack` → VAE 디코딩 → **X 절반만 잘라** 역회전 → `x·(1−mask) + gen·mask` 로 합성. 반환 `[B,1,H,W] ∈ [−1,1]`. 보이는 픽셀은 입력 그대로 보존된다.

### 7.3 추론 한 줄 생성 — 실제 폰트 예시

![추론 예시](img_inkspire_io/05_infer_example.png)

held-out 폰트 Gaegu 로 스타일 한 줄을 주고 다른 문장 한 줄을 받은 것이다(`lora_step_013000`, std layout, 20 step).
캔버스는 176×624 이고 레퍼런스가 첫 줄을 차지한다(★ 스타일 인코더가 없으므로 레퍼런스는 캔버스 위에 있어야 한다).
마스크는 그 아래 전부. 생성 후 목표 줄의 상자 범위만 잘라 리더에 넘긴다 — `gen_compare` 가 받는 것도 이 한 줄이다.
재생성: `PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 .venv/bin/python docs/_gen_inkspire_infer_example.py`

**shape 요약(P=512, B=1)**

| 단계 | shape |
|---|---|
| 패치 x / xc / mask | [1,1,512,512] |
| I (회전·연결) | [1,1,512,1024] |
| VAE latent | [1,16,64,128] |
| x0_packed / cond_packed | [1,2048,64] / [1,2048,320] |
| transformer 입력 / 출력 | [1,2048,384] / [1,2048,64] |
| img_ids | [2048,3] (앞뒤 절반 동일) |

---

## 8. 논문 수치 vs 구현 수치 (차이 목록)

**일치**: LoRA r=32/α=32 및 대상 모듈 → trainable 정확히 115,912,704 (논문 "약 115.9M", 표 8) · Prodigy lr 1 / wd 0.01 / batch 4 / 20,000 iter ·
이미지 추론 20 ODE step · 레이아웃 10층 / hidden 512 / 8 head / AdamW 1e-4 / 10-step ODE · F-TopMask 비율(논문 1024 중 상단 128 = 1/8, 우리 512 중 64 = 1/8) ·
마스크 토큰만의 손실(레이아웃 L1 v-pred, 이미지 L2) · 추가 손실 없음 · 텍스트 인코더 미사용.

| 항목 | 논문 | 구현 | 이유 / 영향 |
|---|---|---|---|
| 패치 P | **1024** (§4.1.4) | **512** | ★ VRAM 이 아니라 **시간** 때문이다 — P=1024 B=4 는 45.4GB 로 H100 한 장에 들어가지만 step 이 4배 느리다(20.0 vs 5.0 s, 2026-09-15 실측). 토큰 L 이 8192 → 2048 로 줄어 한 패치가 담는 줄 수가 절반. 논문 A.5.4 는 (패치 크기가 아니라 레퍼런스 줄 높이 정규화 얘기지만) 해상도를 줄이면 스타일 단서를 잃는다고 주장 |
| GPU / 유효 배치 | A100 40GB **4장**, batch 4 | H100 1장, batch 4, accum 1 | 논문이 batch 4 를 GPU 당인지 전체인지 안 밝힘. GPU 당이면 우리 유효 배치는 1/4. (참고: A100 40GB 로는 P=1024 B=4 의 45.4GB 가 안 들어간다 → 논문의 batch 4 는 GPU 당 1~2 였을 가능성) |
| 학습 데이터 | IAM(영문 13,353줄/496 writer) + CASIA-HWDB2.0-2.2(중문 52,230줄/1,019 writer), 다국어 통합. 레이아웃 라벨은 영문 어절·중문 글자 단위 | 한글 **합성 폰트 페이지** 12,951종, 글자 단위 라벨 | 한글 실손글씨 페이지+레이아웃 라벨이 없음. 논문 수치와 직접 비교 불가 |
| 데이터 증강 | **언급 없음**(실제 손글씨라 기울기·번짐이 원래 들어 있음. 논문에서 rotate 는 R-APE 의 90° 뿐) | 줄 단위 ±3° 회전 p0.5, elastic p0.7, morph p0.15, blur p0.4, 배경 합성, jitter p0.5 | 합성 폰트는 너무 깨끗해 그대로 두면 폰트 렌더를 학습한다. 부작용: 회전 bbox 는 외접사각형, elastic ≤3px 는 라벨 미반영 → 레이아웃 타깃 정확도 손해. `--aug false` 로 해제 |
| guidance | 미기재 | **30** 고정(학습=추론) | Fill-dev 동작점. ★ 3.5·10 으로 낮추면 CER 0.19 → 0.53·0.42 로 폭망(2026-09-14 실측) |
| 레이아웃 값 정규화 | 언급 없음 | 차원별 mean/std(step 0 에 20 배치) | 원값이 0.02~0.06 이라 노이즈 N(0,1) 대비 SNR 붕괴. 없으면 학습 안 됨 |
| 레이아웃 토큰 입력 | char + pos + 조건 토큰 + 노이즈 토큰 | + **`line_emb`**, `obs_emb` | `line_emb` 는 논문에 없음(줄 시작 Δx 이봉 타깃 분리용). `obs_emb` 는 Fig 10(c) 의 reference/masked 구분에 해당 |
| 레이아웃 t 분포 / 조건 부여 | 미기재 | `t ~ U(0,1)`, additive t-embedding(adaLN 아님) | 4-d 회귀엔 logit-normal·adaLN 이 과함 |
| 마스킹 모드 | 연속구간 / 토큰별 20% (각 50%) | + **전체 마스크 10%** | 레퍼런스 없는 실손글씨 추론용. 나머지 90% 는 45/45 로 논문과 동일 비율 |
| 레이아웃 학습량 | 미기재(구조만) | 20k + EMA(2026-09-14 진행) | 8k 중단본은 std layout 보다 나빴다 |
| 페이지 토큰 상한 | 미기재 | 1,024 (`MAX_TOKENS`), 모델 `max_len` 2048 | 좁은 pitch 페이지가 1,000+ 토큰을 만들어 VRAM 초과 |
| R-Mask 세부 | "랜덤 크기 다중 영역" | 1~4 사각형, 변 U(P/4, 3P/4), 16px 스냅, 커버리지 [0.15, 0.85] | 논문에 수치 없음 — 추정 |
| 이미지 σ 샘플링 | 미기재(식 10-11 만) | logit-normal + FLUX `time_shift(calculate_shift(L))` | FLUX 학습 관례. 추론 스케줄과 일치 |
| grad clip | 미기재 | 1.0 | dreambooth 관례. `--clip-grad 0` 으로 해제 |
| 텍스트 조건 | "텍스트 인코더 제거" | 빈 프롬프트 임베딩 캐시(0 벡터 아님) | 0 을 넣으면 백본이 학습 분포 밖에서 시작 |
| 추론 레퍼런스 레이아웃 | 데이터셋의 **GT 레이아웃**(§4.1.1 글자/어절 단위 주석), 같은 문단의 다른 줄을 레퍼런스로(§4.3) | 표준폰트 렌더 bbox 를 ref 폭에 맞춘 근사 + 줄마다 좌측 여백 고정 | 스타일 이미지에 bbox 라벨이 없음. ★ 이미지에서 글자를 분할해 실제 bbox 를 넣어봤으나 CER 0.20 → 0.61 로 더 나빴다(2026-09-14) |
| 추론 해상도 | 원본 해상도 유지(A.5.4) | 줄 높이 64px 고정 | ref set·HTR 입력 규약. ★ 48/64/80 실측 CER 차이 없음 — 우리 조건에선 무관 |
| 평가 지표 | FID / KID / HWD / ΔCER / CR·AR / Swin 스타일 분류기 | **HTR CER 중앙값·폭주율**만 | repo 규약. 지표 라이브러리 미설치 |

---

## 9. 성능 개선 조사 (2026-09-14) — 논문 재독 + 진단 실험

### 9.1 진단 실험 결과 (원인 분해)

| 실험 | 결과 | 해석 |
|---|---|---|
| FLUX VAE 한글 왕복 CER (줄높이 48/64/80) | **0.000 / 0.000 / 0.000** (Emuru VAE 는 0.308 / 0.037 / 0.000) | **VAE 는 병목이 아니다.** Eruku 의 한글 상한([[vae-korean-capacity]])이 InkSpire 엔 없다 — 잠재 16채널 |
| 추론 줄 높이 48 / 64 / 80 (held-out, std layout, n=150) | 중앙값 **0.188 / 0.191 / 0.189** (폭주 0.7 / 1.3 / 2.7%) | **추론 해상도는 병목이 아니다.** 높일수록 폭주만 늘어 원래 값 64 유지 |
| 오라클 레이아웃 one-shot CER (스타일 폰트 GT bbox, 페이지 캔버스, held-out 6폰트) | 중앙값 **0.088** (n=18) | 이미지 모델 자체의 상한. GT 바닥 0.026 |
| 실제 추론 (std layout, 좁은 2줄 캔버스) | 중앙값 **0.191~0.197** (n=150/300) | 오라클 대비 **+0.10 은 레이아웃·캔버스 탓** |
| 레이아웃 모델 사용 | 0.222 (더 나쁨) | 추론 시 레퍼런스 토큰이 근사값이라 조건이 틀림(아래 9.2 ①) |
| **차분 오차의 적분** (EMA s20000, held-out, 줄 평행이동 적용 후 \|x0 예측−정답\|, 2026-09-15) | 줄 안 1~3번째 **142.3px** / 4~8 43.8 / 9~15 36.3 / 16+ 46.2 px | 토큰별 Δx L1 은 9.71px 인데 `layout_to_bboxes` 가 `x0 = x1_prev + Δx` 로 **줄을 따라 적분**해 절대 위치 오차가 ~40px 이 된다. 평균 글자 폭 23.7px 의 1.5~2배 — `render_content` 가 글리프를 그 상자에 resize 해 박으므로 Xc 에서 글자가 겹치거나 빈다. **std layout 이 이기는 이유가 이것**(누적 오차 0). 차분 지표가 시각적 오차를 크게 과소평가한다 |
| 체크포인트 s13000 vs s20000 | 0.190 vs 0.222 | val loss 최저 ≠ CER 최저 |
| 레이아웃 L1 **차원 분해** (EMA s20000, held-out, 생성토큰 79,465, 2026-09-15) | 전체 w 2.11 / h 1.94 / Δx 21.71 / Δy 10.15 px | Δx 가 60% 로 보이지만 **그 56.5% 가 줄 시작 토큰(2.8%, 개당 446px)** 이다 — 읽기 순서상 줄 첫 Δx = LEFT − 이전 줄 오른쪽 끝 ≈ −(이전 줄 폭). 학습 줄은 `wrap_lines(page_w−48)` 로 −960 에 몰려 있고 추론 줄은 짧아 분포 밖 |
| 같은 분해, **줄 시작 토큰 제외** | w 2.08 / h 1.93 / **Δx 9.71** / Δy 10.08 px (평균 5.95) | `infer/inkspire.py` :126 이 줄마다 첫 토큰을 LEFT 로 평행이동해 **줄 시작 Δx 를 이미 덮어쓴다** → 추론에 실제로 쓰이는 오차는 이쪽이다. Δx 와 Δy 가 비슷하고 어느 한 차원이 지배하지 않는다. ×10³ 로 논문 표 1 과 비교하면 w 2.03(논문 4.74) · h 1.88(4.94) · Δy 9.84(**14.39**) 은 우리가 낫고 Δx 9.48(**4.74**)만 2배 나쁘다 (단 논문은 IAM/CASIA 어절·글자 라벨이라 직접 비교는 아님). ★ 위 괄호는 **IAM 열(영문 어절)**이다 — 우리는 글자 단위라 **ICDAR2013 열(중문 글자)** 과 맞추는 게 옳다: Δx 5.13 · Δy 14.85 · w 12.43 · h 8.74 → Δx 는 여전히 ~1.8배, 나머지는 우리가 낮다(합성 폰트가 실손글씨보다 규칙적인 탓일 가능성이 크다) |

즉 **전체 오차 0.19 ≈ 레이아웃/캔버스 0.10 + 이미지 모델 0.06 + 리더 바닥 0.03**. 이미지 모델보다 레이아웃 쪽 손실이 크다.

### 9.2 논문 재독으로 새로 확인한 차이 (§8 표에 없던 것)

① **추론 레퍼런스의 레이아웃이 논문은 GT, 우리는 근사.** 논문 §4.3 은 같은 문단의 다른 줄을 레퍼런스로 쓰고(둘째 줄로 첫 줄을, 그 다음엔 첫 줄로 나머지를), IAM·CASIA 는 어절/글자 단위 레이아웃 주석을 제공한다(§4.1.1). 우리는 스타일 이미지에 bbox 라벨이 없어 **표준폰트 렌더 bbox 를 ref 폭에 맞춰 늘린 값**을 레퍼런스 토큰으로 준다. 레이아웃 모델이 std layout 보다 나빴던 원인으로 가장 유력하다(학습·검증에서는 GT 레퍼런스를 받으므로 bbox L1 11.7px 로 멀쩡했다).

② **레이아웃 정확도 자체도 논문보다 나쁘다 — 단 줄 내부 Δx 하나 때문이다.** 논문 표 1 의 CFM 은 L1×10³ 로 Δx 4.74 / Δy 14.39 / Δw 4.74 / Δh 4.94 (평균 **7.2e-3**). 우리 s8000 은 11.7px / page_w 1024 = **11.4e-3** — 약 1.6배였다. 20k+EMA 완주 후 8.75px = 8.5e-3 으로 줄었다. 차원 분해(§9.1)에서 추론이 실제로 쓰는 값(줄 시작 토큰 제외)만 보면 평균 5.95px = 5.8e-3 이고, w·h·Δy 는 논문보다 낫고 **줄 내부 Δx 만 2배 나쁘다**.

③ **논문은 추론에서 해상도를 건드리지 않는다.** A.5.4 는 레퍼런스를 고정 높이로 줄이면 스타일 단서가 사라진다고 명시한다. 우리는 ref 잉크를 48px, 줄 높이 64px 로 강제한다(HTR 리더·ref set 규약).

④ **유효 배치가 논문의 1/4 일 수 있다.** 논문은 "batch size 4 … on four A100" — GPU 당 4 라면 전체 16. 우리는 단일 GPU batch 4, accum 1.

⑤ 논문 실패 사례(A.5.6)는 실데이터의 취소선 흔적 문제로 우리와 무관. 인코더 어블레이션(A.5.5)은 사전학습 VAE 가 최선이라는 결론이라 우리 구성과 같다.

### 9.3 우선순위 (기대 효과 / 비용)

| 순위 | 할 일 | 근거 | 비용 |
|---|---|---|---|
| 1 | **추론 레퍼런스 레이아웃을 이미지에서 직접 추출** (수직 투영·연결요소로 글자 분할 → 실제 bbox). 실패 시 무레퍼런스 모드(`ref_mask` 전부 False) 사용 | 9.2 ① — 레이아웃 모델을 쓸 수 있게 만드는 전제 | 코드 반나절, 재학습 없음 |
| 2 | **레이아웃 데이터셋의 증강을 끈다** — `KoreanLayoutDataset` 은 bbox 만 먹는데 `aug=True` 가 기본이고 트레이너에 `--aug` 플래그도 없다. ★ **Δy 대책이지 Δx 대책이 아니다** | 레이아웃 타깃에서 aug 가 바꾸는 것은 `_rotate_strip` 뿐이다(elastic/morph/blur/배경/jitter 는 `xc`·boxes 확정 후 `x` 에만 적용). 실측(2026-09-15, 기존 ckpt 를 깨끗한 val 에 평가): 평균 L1 5.95 → 5.38px, **Δy 10.08 → 8.37(−17%)**, Δx 9.71 → 9.44(−3%), 타깃 Δx std 는 137.61 로 불변. ±3° 회전은 폭 1000px 줄을 세로로만 움직인다 | 인자 1개 + 재학습 |
| 3 | 학습에 짧은 줄·가변 page_w 를 넣어 줄 시작 Δx 분포를 추론과 맞춘다 | §9.1 — 줄 시작 Δx 개당 446px. 다만 `infer/inkspire.py` :126 의 LEFT 평행이동이 이미 이 값을 버리므로 실익은 "지표가 정직해진다" 쪽 | 데이터 코드 + 재학습 |
| ? | **줄 내부 Δx 9.71px(논문 4.74 의 2배) 은 원인 미상** | 증강과 무관하고(위 −3%) 줄 시작 분포 문제도 아니다. 남은 후보: 학습량/용량, 또는 한글은 글자폭이 균일해 Δx 의 정보량 자체가 적음. **먼저 원인을 재고 손댈 것** | 진단 먼저 |
| ~~2 이전~~ | ~~20k 완주 + EMA, Δy 가중~~ | 20k+EMA 는 2026-09-14 완료(L1 9.64 → 8.75px). Δy 는 이미 논문보다 나으므로 **Δy 가중은 폐기** | — |
| 3 | 추론 캔버스를 학습 페이지에 맞추기(폭 1024 패딩, pitch 학습 분포) | 오라클 0.088 vs 실추론 0.191 의 나머지 절반 | 코드 1시간 + 평가 |
| 4 | ODE step 20 → 30~50, guidance 스윕 | 가장 싼 노브. 논문 미기재 | 평가 시간만 |
| 5 | **P=1024 재학습** | 논문 설정, 지금 가장 큰 구조적 편차. VRAM 은 문제가 아니었다(B=4 45.4GB) | 약 90h |
| 6 | 유효 배치 16 (grad_accum 4) | 9.2 ④ | 20k 에 4배 시간 |
| 7 | 학습 데이터에 짧은 줄·불규칙 우측 정렬·다양한 페이지 폭 추가 | 추론 캔버스가 학습 분포 밖 | 데이터 코드 + 재학습 |

체크포인트는 앞으로도 **val loss 가 아니라 CER 로** 고른다(§6).

### 9.4 생성 글씨가 "흔들리는" 원인 분리 (2026-10-20)

`_eval/gencmp_*.png` 의 InkSpire 열은 글자가 위아래로 출렁이고 겹치거나 깨진다. 학습 문제인지 추론 문제인지
같은 샘플로 조건을 하나씩 바꿔 갈랐다. **학습이 아니라 추론 조건 문제다.**

![조건별 분리](img_inkspire_io/08_wobble_ablate.png)

gencmp ko_short 6행과 같은 ref·목표·seed 0 (`docs/_gen_inkspire_wobble_ablate.py`).

| 열 | 조건 | 결과 |
|---|---|---|
| ① | s20000 **+ 레이아웃 모델**, gencmp ref (ref_set 혼합 스크립트 줄) | 기존 gencmp 와 동일하게 재현. 기준선이 출렁이고 글자 겹침·누락 |
| ② | ①에서 레이아웃 모델만 끔 (std layout) | **흔들림 소멸** — 기준선 일정, 획 매끈. 영문 위주 ref 2행(Yeongdo, IncheonEducationSotong)만 깨짐/자간 과다 |
| ③ | ②에서 s13000 | ②와 거의 같다 |
| ④ | ③에서 ref 를 **같은 폰트로 렌더한 한글 pangram** 으로 | 6행 중 5~6행이 정답과 거의 같다 |

원인 (영향 큰 순):

1. **레이아웃 모델의 Δx·Δy 적분 오차** (§9.1). 상자를 앞 글자 기준 상대값으로 예측해 더해 가므로 토큰별 9.7px
   오차가 줄을 따라 ~40px 로 쌓인다. 이미지 모델은 Xc(표준폰트 글리프를 그 상자에 박은 밑그림)를 충실히 따르므로
   틀어진 상자가 그대로 출렁이는 글씨가 된다. std layout 은 표준폰트 렌더 상자라 기준선이 일직선이고 누적 오차가 0.
   추론 때 레퍼런스 토큰이 근사값(§9.2 ①)이라 학습 때보다 오차가 더 크다.
2. **혼합 스크립트 ref.** gencmp 는 `ref_set_clean` 에서 줄을 무작위로 골라 "lavalier. 후배에게", "153: 207.17",
   "심을," 같은 영문·숫자·구두점 줄이 ref 가 된다. 레퍼런스 bbox 근사(나눔고딕 렌더를 ref 폭에 맞춰 늘림)는
   라틴·구두점에서 실제 글자 위치와 크게 어긋나고, 숫자뿐인 ref 는 한글 스타일 단서가 없다.
3. 체크포인트(s13000 vs s20000)는 육안 차이가 작다. 레이아웃 체크포인트(s8000 / s20000 / EMA s20000)도 셋 다 같은 모양으로 흔들린다.

**단, 레이아웃 모델이 늘 망가지는 건 아니다 — "ref 바로 아래 짧은 한 줄"에서만 망가진다.**
`_debug/inkspire_s13000.png`(s13000 + layout s8000, 긴 문장 3줄 동시 생성)는 혼합 스크립트 ref 로도 꽤 반듯하다.
같은 레이아웃 모델·같은 짧은 목표로 줄 위치만 바꿔 확인했다:

![줄 길이·위치](img_inkspire_io/10_layout_line_len.png)

열: 짧은 줄 **단독**(gencmp·CER 프로토콜) | 긴 줄 2개를 먼저 쓰게 했을 때의 1번째 줄 | 그 아래 **3번째 줄로 쓴 같은 짧은 목표** (`docs/_gen_inkspire_layout_line_len.py`).
단독이면 흔들리고, 3번째 줄이면 반듯하다(인공지능 모델·새벽 공기는 std layout 보다도 낫다).
§9.1 의 위치별 오차(줄 안 1~3번째 글자 **142px**, 4번째부터 ~40px)와 맞는다 — 짧은 줄은 오차가 큰 첫 글자들이
전부다. 또 단독 줄의 Δy 기준은 **근사 bbox 인 ref 줄**이지만 3번째 줄의 기준은 모델이 스스로 예측한 2번째 줄이라
자기 일관적이다. 학습 페이지도 폭을 채운 여러 줄이라 "ref 아래 짧은 한 줄"은 분포 밖이다.
→ EXPERIMENTS §12 의 "레이아웃 모델 6쌍 전부 열세"는 **한 줄씩 생성하는 CER 프로토콜에서의 결론**이다.
여러 줄을 한 번에 쓰는 용도에서는 레이아웃 모델이 쓸 만할 수 있다(CER 미측정).

**학습된 이미지 모델 자체는 멀쩡하다.** held-out 16폰트에서 pangram ref 면 std layout 으로도 흔들림이 없다.
남는 차이는 손글씨류(Dongle, Dokdo, EastSeaDokdo, NanumBrushScript)에서 **자간이 넓고 글자가 작은 것** —
std 상자가 나눔고딕 폭이기 때문이고, 스타일 폰트의 실제 bbox(오라클)를 주면 정답에 가까워진다.
학습 증강도 원인이 아니다: page elastic(`amp=2.5, sigma=8`)은 균등 노이즈를 σ=8 로 블러한 뒤 곱해서
실제 변위가 std 0.05px · max 0.26px 로 사실상 꺼져 있다.

![오라클 vs std](img_inkspire_io/09_oracle_vs_std.png)

held-out 16폰트, s13000, pangram ref (`docs/_gen_inkspire_oracle_vs_std.py`). 열: 정답 | A std layout(현재 추론) | B 스타일 폰트 실제 bbox + 학습 page 빌더(`render_page`, aug off).

한계: 육안 비교(6행·16폰트), CER 은 다시 재지 않았다. 레이아웃 모델 쪽 방향은 EXPERIMENTS §12 표(6쌍 전부 std layout ≥)와 일치한다.

**권장 추론 설정:** 한 줄씩 생성이면 레이아웃 모델 끔(std layout), **s13000**, ref 는 한글 문장.

**Eruku 비교 몽타주 재생성** → `finetune_runs/inkspire_p512/_eval_pangram/gencmp_*.png`
(`experiments/gencmp_split.py` — 두 repo 환경에서 따로 생성 후 조립. 재현 커맨드는 그 docstring).
held-out 16폰트, pangram ref, InkSpire s13000 std layout, seed 0. 육안으로는 **엇비슷하다**:

- ko_short·ko_long: 둘 다 대부분 읽히고 스타일도 옮긴다. 긴 줄은 둘 다 정답과 거의 같다.
  InkSpire 는 NanumPenScript 처럼 가는 손글씨 긴 줄에서 글자가 작고 흐려진다.
- ko_diverse: Eruku 가 더 정확하다 — InkSpire 는 Dongle 에서 "AI → 41", "GPT → GDT". 반대로 Eruku 는
  GowunBatang "−3.5℃" 를 "놈3.5…" 로 바꿨고 InkSpire 는 −·℃ 를 빼먹었다(표준폰트 Xc 에 글리프가 없는 기호).
- 기존 `_eval/`(s20000 + layout, 혼합 ref) 의 InkSpire 열이 유난히 나빴던 건 위 원인 1·2 가 겹친 결과다.
  CER 표(Eruku 우세)는 그대로 유효하다 — 이 몽타주는 n=36 육안이다.

### 9.5 학습 페이지를 추론 캔버스에 맞추기 — `ragged` (2026-10-20, 구현)

**왜.** 학습 페이지는 모든 줄이 `wrap_lines(page_w−48)` 로 폭을 꽉 채우고, 패치는 그 안의 랜덤 512 크롭이다.
추론 캔버스는 **짧은 줄 + 오른쪽 여백**이다 — 이미지 모델은 "줄이 일찍 끝나고 오른쪽이 빈" 장면을 거의 못 봤다.
오라클 0.088 vs 실추론 0.19 의 차이(§9.1) 중 캔버스 몫을 학습 쪽에서 줄이려는 시도.

**왜 line mask 가 아니라 페이지인가.** 논문 §4.4.2(표 6·7, A.5.2)는 추론을 흉내 낸 F-TopMask(상단 1/8 만 보임) 학습을
R-Mask 와 비교해 R-Mask 가 낫다고 보고한다(IAM ΔCER 0.07 vs 0.01) — 고정 마스크는 국소 글자 패턴 다양성이 줄어
구조 오류가 늘어난다는 해석. 그래서 **마스크는 R-Mask 그대로** 두고 페이지 기하만 바꾼다.

**무엇.** `KoreanPageDataset(ragged=p)` / `configs/inkspire.yaml data.ragged` / `--ragged`:
페이지를 확률 p 로 "줄마다 최대 폭 `U(0.15, 1.0)·(page_w−48)`" 로 만들고, 그 페이지의 크롭은 `x0 = 0`(y0 는 랜덤).

- x0=0 은 줄 시작을 보이려는 게 아니라 **빈 종이 패치 방지**다 — 짧은 줄은 왼쪽에 몰려 있다.
  추론 캔버스의 줄 시작을 16px → 304px 로 밀어도 16폰트 중 13개는 결과가 같았다(가는 손글씨 3개만 약간 흐트러짐).
- **잘린 글자는 그대로 둔다.** 기존 패치의 글자 14.7% 가 가장자리에서 잘리지만 Xc 도 같이 잘려 감독이 모순되지 않고,
  실사용 ref 도 잘려 들어올 수 있어 견고성으로 본다. 나머지 1−p 페이지는 기존 그대로라 이 분포가 남는다.
- `ragged=0`(기본)은 rng 소비까지 기존과 같다 — HEAD 코드와 `rand`/`line` 24샘플 비트 동일 확인. val 셋은 항상 ragged=0.
- 레이아웃 데이터셋도 같은 `_page` 라, 나중에 레이아웃 모델을 재학습하면 줄 시작 Δx 분포 문제(§9.3 3번)도 같이 풀린다.
- self-check: `python custom_datasets/korean/page.py` 에 ragged 단언 추가(줄 끝 편차 ≥ 페이지 폭 1/4, 패치 좌측 64px 에 잉크).

![ragged 패치](img_inkspire_io/11_ragged_patches.png)

`ragged=0.5` 학습 패치 8장(시안 = R-Mask 생성 영역). 400페이지 중 ragged 비율 0.48.

**보류 (2026-10-20).** 재측정(§9.6) 결과 남은 문제는 빈 공간이 아니라 **획 정확도**(숫자·기호·모음·붓글씨)라 ragged 는 과녁이 아니다. 기본값 0 이라 코드는 영향 없이 남겨 둔다.

### 9.6 독립 리더(PP-OCRv6)로 재측정 — "Eruku 우세" 는 리더 착시였다 (2026-10-20)

**리더.** 원본 repo `eval/ppocr_rec.py` — HF `HERIUN/PP-OCRv6-medium-rec-ko-printed-handwriting`, `.venv-paddle`(paddle **3.3.1** 전용).
자체 HTR(`aux_htr_ko/htr_s20000`)은 같은 폰트 렌더로 학습된 리더라 독립적이지 않다.

**경로.** 이 repo 엔 리더가, 원본 repo 엔 InkSpire 코드가 없어 셋으로 나눴다 — 재현 커맨드는 `experiments/cer_dump.py` docstring.
`cer_dump.py plan`(원본 repo env, `htr_cer.py` 와 같은 rng 순서 → 폰트 분포가 §12 표와 정확히 일치 확인) →
`cer_dump.py gen`(여기) → `ppocr_rec.py --save-preds`(원본 repo) → `experiments/cer_analyze.py`(폰트·문자 종류·자모 위치 분해).
Eruku 는 원본 repo `htr_cer.py --dump-dir`. 결과: `finetune_runs/_cer/*`.

held-out 16폰트, style-ref pangram, coherent, n=300, seed 0 (§12 와 같은 표본):

| 조건 | 자체 HTR 중앙값 | PP corpus CER | PP 평균 / 중앙값 | 한글 | 숫자 | 기호 |
|---|---|---|---|---|---|---|
| GT 렌더 (리더 바닥) | 0.026 | 0.006 | 0.005 / 0 | | | |
| Eruku corpusmax s45000 | 0.000 | **0.018** | 0.021 / 0 | 1.9% | 2.5% | 2.7% |
| InkSpire s13000 std layout | **0.190** | 0.040 | 0.045 / 0 | 3.5% | 12.7% | 8.9% |
| + `tight_mask` (§ 아래) | — | **0.034** | 0.043 / 0 | 2.8% | 7.0% | 8.9% |
| + 전사 없음 (`--style-text none`) | — | 0.720 | 0.759 / **1.0** | 75.6% | 62.7% | 78.1% |
| + Xc 첫 줄에 목표 텍스트 (`--style-text target`) | — | 0.355 | 0.466 / 0.409 | 40.3% | 33.5% | 40.2% |
| + Xc 첫 줄에 ref 의 PP-OCR 판독 (`--style-text ocr`) | — | 0.041 | 0.046 / 0 | 3.6% | 12.0% | 8.9% |

1. **자체 HTR 이 InkSpire 를 과하게 깎았다.** 중앙값 0.190 → 0. 폰트별 "굵은 폰트 약점"(BagelFatOne 0.423, BlackHanSans 0.333)도
   PP 로는 0.028 / 0.000 — 리더 착시. §12 표와 §9.1 의 HTR 기반 추론 노브 결론(줄 높이 48/64/80, guidance, trim_ref, 캔버스 폭)도
   **PP 로 다시 재기 전까지 미확정**이다.
2. **남은 격차는 획 정확도.** 숫자·기호가 가장 크고(−·℃ 는 표준폰트 NanumGothic 에 글리프가 없어 Xc 에 안 그려진다),
   붓·펜글씨(NanumBrushScript 0.173 / NanumPenScript 0.053, 나머지 폰트는 0). 한글 1:1 치환의 틀린 자모는 InkSpire 가 중성 31~42%
   (원→윈, 은→을, 이→어) — Eruku 는 초성 76%(이→미, 들→틀: 리더 혼동에 가까움). **InkSpire 는 모음의 짧은 획을 놓친다.**
3. **`tight_mask`** (`InkSpireGen(tight_mask=True)` / `inkspire:<lora>,tight_mask=1`): 마스크 = ref 잉크 바닥 ↑16격자 ~ 목표 상자 바닥+4 ↑16격자,
   캔버스도 거기서 끝. 기존 매직 상수(`TOP+PITCH` 상한, `−2`, `+8`, H=`TOP+PITCH·(n+1)`)를 데이터로 대체하고, 목표 아래 빈 종이 41px → 9px,
   마스크 경계를 토큰 격자(16px = VAE 8 × patchify 2)에 맞춘다. 편집거리 합 280 → 239(−15%), 숫자 12.7 → 7.0%.
   **단 짝지은 부호검정 81 나음 / 68 나쁨, p=0.33** — 캔버스 높이가 바뀌면 같은 시드라도 초기 노이즈가 달라져 샘플 변동이 크다.
   방향은 좋으나 시드를 늘려 확인 전까지 기본값은 그대로.
4. **전사 없음 = 붕괴.** ref 이미지는 그대로 두고 전사만 비우면(Xc 첫 줄 빈칸) 300개 중 134개가 **백지**다 — 크롭 문제가 아니라
   생성 자체가 흰 종이다(아래 그림, 왼쪽 Xc · 오른쪽 생성, 위 = 전사 있음 / 아래 = 없음). 학습에서 Xc 는 페이지 전체(보이는 영역 포함)를
   항상 채웠으므로 "X 에 잉크, Xc 는 빈칸" 은 분포 밖이다. 이미지 모델은 ref 줄의 **X↔Xc 대응을 핵심 조건으로 쓴다.**
   Eruku 는 학습 `style_text_dropout=0.10` 덕에 전사가 선택이지만(그래도 있으면 CER 0.197 vs 없으면 0.474, 원본 README) InkSpire 는 필수다.
   실사용(손글씨 사진)에서는 전사 입력·OCR(PP-OCRv6 로 자동 전사) 또는 **이미지 모델 학습에 "Xc ref 영역 비우기" p_drop** 이 필요하다.

![전사 없음 캔버스](img_inkspire_io/12_nostext_canvas.png)

5. **전사 대신 목표 텍스트로 Xc 첫 줄을 채우면** 백지는 사라지지만(134 → 6) 글자가 틀린다(중앙값 0.409).
   틀리는 방식이 진단적이다 — 목표 글자 자리에 **ref 글자 모양이 섞여 나온다**("동네 학교에서" → "다람쥐 헌 쳇바퀴" 비슷한 획,
   "3억원이" → "귀비 쥐비"). Xc 첫 줄에서 "'시' 의 표준폰트 = X 의 '다' 손글씨" 라는 잘못된 짝을 배우고 목표 줄에 그대로 적용한다.
   즉 이미지 모델은 Xc ref 줄을 "글자가 있다" 신호가 아니라 **글자별 대응표**로 쓴다. 전사를 모를 때 쓸 수 있는 건
   OCR 로 실제 전사를 얻는 것뿐이고(오인식 글자는 같은 방식으로 오염될 것), 아니면 Xc ref 영역 dropout 학습이 필요하다.

![목표 텍스트로 채움](img_inkspire_io/13_tgttext_vs_ref.png)

샘플마다 위 = 실제 전사, 아래 = 목표 텍스트로 채움.

6. **OCR 전사로 채우면 실제 전사와 같다.** ref(모델에 들어가는 64px 줄)를 PP-OCRv6 로 읽어 Xc 첫 줄에 쓰면(`cer_dump.py ref` → `ppocr_rec.py <out>/ref --save-preds`
   → `gen --style-text ocr`) corpus 0.041 vs 실제 전사 0.040. ref 판독 CER 0.0095 — 300개 중 40개가 한 글자 오독(헌→현 29, 람→락 11)인데
   그 40개도 0.056 → 0.059 로 거의 영향 없다. **한 글자 틀린 정도는 대응표가 버틴다.** 단 ref 가 폰트 렌더라 OCR 에 쉬운 조건이다 —
   실제 손글씨 사진이면 오독이 늘 것이고, 그때 얼마나 버티는지는 미측정.

**다음 후보 (획 정확도):** (a) PP 로 추론 노브 재측정 — 줄 높이 80(글자당 토큰↑), ODE step, guidance > 30, tight_mask 시드 추가.
(b) 데이터: 숫자·기호 비중(`sampler.weights.num` 0.20), 표준폰트에 없는 기호 처리. (c) 전사 없는 실사용이 목표면 Xc ref 영역 dropout 학습.

---

## 10. 결론 (2026-10-21) — InkSpire 는 현 상태로 보류

held-out 16폰트 · pangram ref · n=300 · PP-OCRv6 corpus CER 기준.

| | Eruku corpusmax s45000 | InkSpire s13000 |
|---|---|---|
| ref 전사 있음 | **0.018** (숫자 2.5% · 기호 2.7%) | 0.040 (숫자 12.7% · 기호 8.9%), tight_mask 0.034(p=0.33) |
| ref 전사 없음 | 동작 (전사는 선택, `style_text_dropout=0.10`) | **붕괴** 0.720, 134/300 백지 (§9.6-4) |
| 전사 대용 | — | 목표 텍스트 0.355 (ref 글자 모양 오염, §9.6-5) / ref OCR 0.041 (§9.6-6, 외부 모델 의존) |

1. **판독성은 전사를 줘도 Eruku 가 앞선다.** 격차는 숫자·기호·붓/펜글씨와 한글 중성(모음 짧은 획)에 몰려 있다.
2. **InkSpire 는 ref 전사가 필수다.** 이미지 모델이 Xc ref 줄을 글자별 대응표로 쓰기 때문이다. 논문도 ref 의 전사 +
   글자 bbox 정답 주석이 주어지는 설정(IAM/ICDAR, §4.1.1·§4.3)만 다루고 전사 없는 경우는 언급이 없다.
   외부 OCR 로 메우면 수치는 회복되지만 다른 모델 의존이 생기고, 자체 해결(Xc ref-dropout 재학습)은 비용 대비
   Eruku 를 넘을 근거가 없다 — **하지 않는다.**
3. **InkSpire 를 다시 꺼낼 이유가 생긴다면 CER 이 아니라** (a) 다줄·문단 동시 생성, 글자 단위 편집처럼 Eruku 가 구조적으로
   못 하는 기능, 또는 (b) 스타일 충실도(미측정)다. 그때는 재학습보다 **스타일 비교 평가를 먼저** 한다.

보류된 작업: `ragged` 학습(§9.5, 코드는 기본값 0 으로 남김), PP 기반 추론 노브 재측정, 숫자·기호 데이터 보강, Xc ref-dropout.
