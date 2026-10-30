"""HF 릴리즈 — 최종 LoRA 를 모델 repo 로, gradio 데모를 Space 로 발행 (Eruku_korean_finetuning tools/hf_upload.py 대응).

    HERIUN/inkspire_korean        LoRA(pytorch_lora_weights.safetensors) + 빈 프롬프트 캐시 + FLUX Kommunity 라이선스·고지
    HERIUN/eruku_korean_demo      Space(ZeroGPU) — Eruku 데모에 InkSpire 를 합쳐 모델 선택(추론 코드 최소 묶음 + app.py)

베이스(black-forest-labs/flux-3-image-klein-base)는 gated·FLUX Kommunity(비상업)라 **재배포하지 않는다** —
Space 는 실행 때 받으므로 Space secret HF_TOKEN(베이스 약관 수락한 계정)이 필요하다.

  .venv/bin/python tools/hf_release.py model            # dry-run: _release/model 에 스테이징만
  .venv/bin/python tools/hf_release.py space --push     # 업로드 (토큰 = HF_TOKEN env 또는 hf auth login 캐시)
"""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MODEL_REPO = "HERIUN/inkspire_korean"
SPACE_REPO = "HERIUN/eruku_korean_demo"   # Eruku 데모에 합쳤다(모델 선택). 단독 Space 는 비PRO 계정에서 ZeroGPU 배정 실패
BASE = "black-forest-labs/flux-3-image-klein-base"
LORA = HERE / "finetune_runs/inkspire_klein_p512/lora_step_013000/pytorch_lora_weights.safetensors"
EMPTY = HERE / "model_zoo/klein_empty_prompt.pt"
ERUKU_EXAMPLES = "HERIUN/eruku_korean_demo"   # 기존 Eruku 데모의 예시 그대로
# Space 가 import 하는 최소 코드(InkSpireGen 경로). 폰트는 Xc 표준폰트 하나.
SPACE_CODE = ["configs/__init__.py", "configs/loader.py", "custom_datasets/__init__.py",
              "custom_datasets/korean/__init__.py", "custom_datasets/korean/page.py",
              "custom_datasets/korean/fontset.py", "custom_datasets/korean/split.py",
              "custom_datasets/korean/alphabet.py", "models/__init__.py", "models/inkspire.py",
              "models/inkspire_klein.py", "infer/inkspire.py", "assets/fonts/label/NanumGothic-Regular.ttf"]

NOTICE = """This FLUX Model is licensed by Black Forest Labs Inc. under the FLUX Kommunity License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

Modified: this repository contains a LoRA adapter (fine-tune) of black-forest-labs/flux-3-image-klein-base trained by HERIUN for Korean styled handwriting generation. It is not an official product of, nor endorsed by, Black Forest Labs Inc.
"""

MODEL_CARD = f"""---
license: other
license_name: flux-kommunity-license
license_link: LICENSE.md
base_model: {BASE}
library_name: diffusers
pipeline_tag: image-to-image
language: [ko]
tags: [lora, handwriting, handwriting-generation, korean, inkspire, flux]
---

# InkSpire Korean — 한글 손글씨 스타일 생성 LoRA

[InkSpire](https://openreview.net/forum?id=FBPuLChGNX) (Wang et al., ICLR 2026) 를 한글로 재현한 LoRA.
**FLUX.3-Image-Klein-Base + LoRA r32 (71.2M), 13k step.** 스타일 손글씨 한 줄(+ 그 전사)을 보고 같은 필체로 새 문장을 쓴다.

- 학습 코드·실험 기록: https://github.com/HERIUN/Inkspire_korean_finetuning
- 데모(Eruku 와 모델 선택): https://huggingface.co/spaces/{SPACE_REPO}
- 같은 데이터로 학습한 자기회귀 모델 비교 대상: [HERIUN/eruku_korean](https://huggingface.co/HERIUN/eruku_korean)

## 입력 규약

캔버스 X(위 = ref 손글씨, 아래 = 빈 칸) · 콘텐츠 이미지 Xc(NanumGothic 으로 ref 전사와 목표 문구를 글자 상자에 그린 것) · 마스크.
Xc 를 글자별 대응표로 쓰므로 **ref 전사(style text)는 필수**다 — 비우면 붕괴한다(CER 0.72, 절반 백지).
목표 줄 폭은 ref 의 "손글씨 폭 / 표준폰트 폭" 비율로 맞춘다(`ref_width`, 기본).

## 성능 (ref 전사를 줄 때, PP-OCRv6 판독 CER)

InkSpire 는 **내용 CFG(`cfg=2.0, uncond="gen"`, 권장)** 와 끔(`cfg=1.0`, 기본) 두 줄로 적는다. Eruku 는 권장 cfg 1.5.

| 평가 | InkSpire cfg 2.0 | InkSpire cfg 1.0 | Eruku Korean |
|---|---|---|---|
| 한글 폰트 held-out 16종 n=300 — CER↓ / HWD↓ | **0.0045** / 1.02 | 0.023 / **0.93** | 0.010 / 1.12 |
| 실손글씨 문장 교차 생성 n=20 — CER↓ / HWD↓ | 0.071 / 1.60 | 0.246 / **1.41** | **0.014** / 1.51 |
| 실손글씨 서식 칸 n=286 — CER↓ / HWD↓ | **0.180** / 1.37 | 0.417 / **1.20** | 0.711 (백지 7) / 1.83 |
| IAM lines 2,925 — HWD↓ / FID↓ | — | 1.68 / **18.58** | **1.62** / 20.00 |

- 폰트 조건에선 cfg 2.0 이 정확도·필체 모두 Eruku 를 앞선다. **실제 사람 손글씨 문장은 정확도가 Eruku 에 뒤진다** — 폰트로만 학습해 분포 밖 글자 모양을 그대로 베낀다.
- CFG 는 정확도 ↔ 필체(HWD) 절충이다. 지저분한 ref(서식 칸·표 선 잔재)에선 InkSpire 가 백지·폭주가 거의 없다.
- 한계 전체와 사용 팁: GitHub README 「먼저 읽을 것」. 측정 근거: `docs/EXPERIMENTS.md` §9·§10.

![한글 폰트 — 열 = ref | GT | InkSpire cfg 1 | InkSpire cfg 2 | Eruku](samples/font_render.png)
![IAM](samples/iam_lines.png)

## 사용법

추론 코드는 GitHub repo 에 있다(베이스 가중치는 {BASE} 약관 수락 후 자동 다운로드).

```python
import numpy as np, torch
from huggingface_hub import snapshot_download
from PIL import Image
from infer.inkspire import InkSpireGen          # GitHub repo 루트에서

lora = snapshot_download("{MODEL_REPO}")        # pytorch_lora_weights.safetensors + klein_empty_prompt.pt
# klein_empty_prompt.pt 는 model_zoo/ 에 둔다(빈 프롬프트 텍스트 임베딩 캐시 — 텍스트 인코더를 안 올린다)
g = InkSpireGen(lora, device="cuda", cfg=2.0, uncond="gen")   # 권장. cfg=1.0 = CFG 끔(필체 최대, 오타↑)
ref = np.asarray(Image.open("style_line.png").convert("L"))   # 손글씨 한 줄(스캔 테두리·여백은 미리 잘라 둔다)
out = g.gen_lines(ref, "스타일 이미지의 전사", ["생성할 문장"], seed=0)[0]   # u8 줄 이미지
```

- 전사는 글자 그대로 정확히 — Xc 대응표의 원천이다(비우면 붕괴).
- 틀린 글자가 보이면 seed 만 바꿔 다시 — 오류가 시드마다 다른 글자에서 난다.
- 흐린 연필 ref 는 출력도 흐리다(잉크 농도를 베낀다). 진하게 원하면 ref 대비를 올려 넣는다.

## 라이선스

[FLUX Kommunity License](LICENSE.md) (베이스 모델과 동일, 비상업 용도) — [NOTICE](NOTICE.md) 참고.
학습 데이터는 공개 한글 폰트 렌더이며, 비교 평가에 쓴 실손글씨 데이터는 포함하지 않았다.
"""

SPACE_README = f"""---
title: Eruku · InkSpire Korean 한글 손글씨 생성
emoji: 🖋️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.20.0
python_version: "3.12.12"   # ZeroGPU 지원: 3.12.12 · 3.10.13 만 (diffusers 0.42 Flux3 는 ≥3.11)
app_file: app.py
pinned: false
license: other
short_description: 스타일 손글씨로 한글 문장 생성 — Eruku(자기회귀) / InkSpire(FLUX.3 LoRA) 선택
models:
- HERIUN/eruku_korean
- HERIUN/emuru_vae_korean
- blowing-up-groundhogs/eruku
- blowing-up-groundhogs/emuru_vae
- {MODEL_REPO}
- {BASE}
---

# Eruku · InkSpire Korean — 한글 손글씨 스타일 생성

같은 입력(스타일 이미지 + 전사 + 생성 문구)으로 두 모델을 골라 비교하는 데모.

| 모델 | repo | 학습 코드 |
|---|---|---|
| Eruku Korean (자기회귀, T5 + 한글 적응 VAE) | [HERIUN/eruku_korean](https://huggingface.co/HERIUN/eruku_korean) | https://github.com/HERIUN/Eruku_korean_finetuning |
| InkSpire Korean (확산, FLUX.3-Image-Klein-Base + LoRA) | [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) | https://github.com/HERIUN/Inkspire_korean_finetuning |

실제 사람 손글씨 문장의 글자 정확도는 Eruku, 필체 형태(HWD)·지저분한 스캔 ref·폰트 조건 정확도는 InkSpire(내용 CFG 2.0) 쪽이 낫다. InkSpire 는 스타일 전사가 필수다.

라이선스: Eruku 는 Apache-2.0. InkSpire 의 베이스 {BASE} 는 FLUX Kommunity License(비상업)이며
그 약관을 수락한 계정 토큰(secret `HF_TOKEN`)으로 실행 때 받는다. 생성 이미지는 AI 로 만든 것이다.

**하드웨어**: ZeroGPU.
"""

APP = r'''"""Eruku · InkSpire Korean — 한글 손글씨 스타일 생성 데모 (모델 선택).
Eruku: HERIUN/eruku_korean (자기회귀) / InkSpire: HERIUN/inkspire_korean (FLUX.3-Image-Klein-Base + LoRA)
"""
import os
os.environ.setdefault("GRADIO_SSR_MODE", "0")  # gradio SSR 렌더 버그(unhashable dict) 회피

import json
import shutil
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image

try:
    import spaces
    GPU = spaces.GPU
except Exception:  # spaces 없거나 ZeroGPU 아님
    def GPU(fn=None, **_):
        return fn if fn else (lambda f: f)

HERE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ERUKU_ID, INKSPIRE_ID = "HERIUN/eruku_korean", "%(model)s"

# ── InkSpire: ZeroGPU 권장대로 모듈 레벨에서 cuda 로 올린다(실제 GPU 는 @GPU 안에서 붙는다)
lora = snapshot_download(INKSPIRE_ID)
(HERE / "model_zoo").mkdir(exist_ok=True)
if not (HERE / "model_zoo/klein_empty_prompt.pt").exists():   # models.inkspire_klein.EMPTY_CACHE
    shutil.copyfile(Path(lora) / "klein_empty_prompt.pt", HERE / "model_zoo/klein_empty_prompt.pt")
from infer.inkspire import InkSpireGen   # noqa: E402
inkspire = InkSpireGen(lora, device="cuda", cfg=2.0, uncond="gen")   # 내용 CFG — 실손글씨 CER 0.25 → 0.07

# ── Eruku: 기존 데모와 같은 지연 로드
eruku = None


def load_eruku():
    global eruku
    if eruku is None:
        from transformers import AutoModel
        eruku = AutoModel.from_pretrained(ERUKU_ID, trust_remote_code=True).eval()
    return eruku


def ink_bbox_crop(a, thr=180, bthr=215, dens_col=0.03, dens_row=0.006, solid=0.55, pad=4):
    """스캔 여백·테두리 제거(Eruku 릴리즈 ink_bbox_crop 과 같은 값) — InkSpire 경로용."""
    bink = a < bthr
    y0, y1, x0, x1 = 0, a.shape[0], 0, a.shape[1]
    while y0 < y1 and bink[y0].mean() > solid: y0 += 1
    while y1 > y0 and bink[y1 - 1].mean() > solid: y1 -= 1
    while x0 < x1 and bink[:, x0].mean() > solid: x0 += 1
    while x1 > x0 and bink[:, x1 - 1].mean() > solid: x1 -= 1
    a = a[y0:y1, x0:x1]
    ink = a < thr
    cols, rows = np.where(ink.mean(0) > dens_col)[0], np.where(ink.mean(1) > dens_row)[0]
    if len(cols) == 0 or len(rows) == 0:
        return a
    return a[max(0, rows.min() - pad):rows.max() + pad + 1, max(0, cols.min() - pad):cols.max() + pad + 1]


def vstack(arrs, gap=8):
    W = max(a.shape[1] for a in arrs)
    return np.vstack([np.pad(a, ((0, gap), (0, W - a.shape[1])), constant_values=255) for a in arrs])


def run_eruku(style_image, style_text, lines, cfg_scale, bbox_crop):
    m = load_eruku()
    m.to(DEVICE)
    # ★ 먼저 h=64 로 줄인다(모델 내부 ink bbox 크롭은 그 축소본에 적용) — 여백 큰 입력 6장 CER 0.131 vs 원본 그대로 0.202
    style_img = style_image.convert("RGB")
    w, h = style_img.size
    style_img = style_img.resize((max(int(64 * w / h), 64), 64), Image.LANCZOS)
    outs = []
    with torch.inference_mode():
        for t in lines:
            r = m.generate_handwriting(style_image=style_img, gen_text=t, style_text=style_text or "",
                                       cfg_scale=float(cfg_scale), max_new_tokens=512, bbox_crop=bool(bbox_crop))
            outs.append(np.asarray(r.convert("L")))
    return outs


def run_inkspire(style_image, style_text, lines, ink_cfg, steps, seed, bbox_crop):
    ref = np.asarray(style_image.convert("L"))
    if bbox_crop:
        ref = ink_bbox_crop(ref)
    inkspire.steps, inkspire.cfg = int(steps), float(ink_cfg)
    return inkspire.gen_lines(ref, style_text.strip(), lines, seed=int(seed))


@GPU(duration=120)   # InkSpire cfg ≠ 1 은 조건·무조건 2배 배치
def generate(model_name, style_image, style_text, gen_text, cfg_scale, ink_cfg, steps, seed, bbox_crop):
    lines = [s.strip() for s in (gen_text or "").splitlines() if s.strip()][:8]
    if style_image is None:
        return None, "❌ 스타일 이미지를 업로드하세요"
    if not lines:
        return None, "❌ 생성할 텍스트를 입력하세요"
    if isinstance(style_image, np.ndarray):
        style_image = Image.fromarray(style_image)
    try:
        if model_name == "InkSpire":
            if not (style_text or "").strip():
                return None, "❌ InkSpire 는 스타일 전사가 필수입니다(없으면 붕괴)"
            outs = run_inkspire(style_image, style_text, lines, ink_cfg, steps, seed, bbox_crop)
        else:
            outs = run_eruku(style_image, style_text, lines, cfg_scale, bbox_crop)
        return Image.fromarray(vstack(outs)), f"✅ {model_name} · {len(outs)}줄 생성"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 오류: {e}"


def load_examples():
    f = HERE / "examples/samples.json"
    if not f.exists():
        return None
    return [[s["style_image"], s.get("style_text", ""), s.get("gen_text", "안녕하세요")]
            for s in json.loads(f.read_text(encoding="utf-8")) if (HERE / s["style_image"]).exists()] or None


custom_css = """
.gradio-container { max-width: 1100px !important; }
footer { visibility: hidden; }
"""

with gr.Blocks(css=custom_css, title="Eruku · InkSpire Korean — 한글 손글씨 생성") as demo:
    gr.HTML("""
    <div style="text-align:center; margin-bottom:16px;">
      <h1>🖋️ Eruku · InkSpire Korean</h1>
      <h3 style="color:#888; font-weight:normal;">한글 손글씨 스타일 생성 — 모델을 골라 비교하세요</h3>
      <p>
        <a href="https://huggingface.co/HERIUN/eruku_korean" target="_blank">🤗 Eruku</a> &nbsp;|&nbsp;
        <a href="https://huggingface.co/HERIUN/emuru_vae_korean" target="_blank">🤗 한글 VAE</a> &nbsp;|&nbsp;
        <a href="https://huggingface.co/%(model)s" target="_blank">🤗 InkSpire</a> &nbsp;|&nbsp;
        <a href="https://github.com/HERIUN/Eruku_korean_finetuning" target="_blank">💻 Eruku 코드</a> &nbsp;|&nbsp;
        <a href="https://github.com/HERIUN/Inkspire_korean_finetuning" target="_blank">💻 InkSpire 코드</a>
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            model_radio = gr.Radio(["Eruku", "InkSpire"], value="Eruku", label="🧠 모델",
                                   info="Eruku = 글자 정확도↑ · 전사 없어도 동작 / InkSpire = 필체 형태(HWD)↑ · 전사 필수")
            style_image_input = gr.Image(label="📷 스타일 이미지 (손글씨 참조, 한 줄)", type="pil", height=200)
            style_text_input = gr.Textbox(label="📝 스타일 전사 (Eruku 선택 · InkSpire 필수)",
                                          placeholder="스타일 이미지에 적힌 글자 그대로", lines=2)
            gen_text_input = gr.Textbox(label="✍️ 생성할 텍스트 (줄마다 한 줄씩, 최대 8줄)", lines=3,
                                        value="한국어 손글씨 생성 테스트")
            cfg_slider = gr.Slider(label="🎚️ CFG scale (Eruku) — 1.0 = 원문 충실도 최고, 1.5~2.0 = 스타일 강조",
                                   minimum=1.0, maximum=3.0, step=0.1, value=1.5)
            ink_cfg_slider = gr.Slider(label="🎚️ CFG (InkSpire) — 1.0 = 끔(필체 최대·오타↑), 2.0 = 권장(오타↓, 필체 약간 정형화)",
                                       minimum=1.0, maximum=3.0, step=0.1, value=2.0)
            with gr.Row():
                steps_slider = gr.Slider(label="steps (InkSpire)", minimum=10, maximum=50, step=1, value=20)
                seed_input = gr.Number(label="seed (InkSpire)", value=0, precision=0)
            bbox_checkbox = gr.Checkbox(label="✂️ 스타일 이미지 여백 자동 제거 (스캔 사진이면 켜두세요)", value=True)
            generate_btn = gr.Button("🚀 생성", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="🎨 생성 결과", type="pil")
            status_text = gr.Textbox(label="상태", lines=1, interactive=False)

    examples = load_examples()
    if examples:
        gr.Examples(examples=examples, inputs=[style_image_input, style_text_input, gen_text_input],
                    label="📚 예시 스타일", examples_per_page=4, cache_examples=False)

    inputs = [model_radio, style_image_input, style_text_input, gen_text_input, cfg_slider, ink_cfg_slider, steps_slider,
              seed_input, bbox_checkbox]
    for trigger in (generate_btn.click, gen_text_input.submit):
        trigger(fn=generate, inputs=inputs, outputs=[output_image, status_text])

    gr.Markdown("""
---
### 📖 사용법
1. **모델 선택** → **스타일 이미지 업로드**(따라 할 손글씨 **한 줄**, 크기는 알아서 맞춥니다)
2. **스타일 전사** — Eruku 는 선택(넣으면 정확도↑), **InkSpire 는 필수**(글자 위치 대응표를 전사로 만듭니다)
3. **생성할 텍스트** → 생성. 예제 앞 3개는 실제 사람 손글씨 스캔, 나머지는 폰트 렌더입니다.

### 📊 비교 (ref 전사를 줄 때, PP-OCRv6 CER↓ / HWD↓, InkSpire = CFG 2.0)
| 평가 | Eruku | InkSpire |
|---|---|---|
| 한글 폰트 held-out 16종 (n=300) | 0.010 / 1.12 | **0.0045** / **1.02** |
| 실손글씨 문장 교차 생성 (n=20) | **0.014** / **1.51** | 0.069 / 1.63 |
| 실손글씨 서식 칸 (n=286) | 0.711 (백지 7) / 1.83 | **0.180** / **1.37** |

- **Eruku**: 한글 적응 VAE([HERIUN/emuru_vae_korean](https://huggingface.co/HERIUN/emuru_vae_korean))와 짝. 여백 있는 스캔은 «여백 자동 제거» 를 반드시 켜세요
  (실측 CER 0.119 켬 vs 1.571 끔). 짧은 문장이 가장 어렵고, 잉크를 선명한 검정으로 스냅합니다.
- **InkSpire**: FLUX.3-Image-Klein-Base(FLUX Kommunity License, 비상업) + LoRA. **CFG 2.0 권장** — 1.0 으로 내리면 필체는 더 닮지만 오타가 3배 이상 늘어납니다.
  틀린 글자는 시드마다 다른 곳에서 나므로 이상하면 seed 를 바꿔 보세요. 흐린 연필 ref 는 출력도 흐리고, ref 에 남은 표 선·밑줄도 따라 그립니다.
- 생성 이미지는 AI 가 만든 것입니다.

### 인용 (base models)
```bibtex
@inproceedings{zaccagnino2026eruku,
  title={Autoregressive Styled Text Image Generation, but Make it Reliable},
  author={Zaccagnino, Carmine and Quattrini, Fabio and Pippi, Vittorio and Cascianelli, Silvia and Tonioni, Alessio and Cucchiara, Rita},
  booktitle={WACV}, year={2026}
}
@inproceedings{wang2026inkspire,
  title={Learning to Generate Stylized Handwritten Text via a Unified Representation of Style, Content, and Noise},
  author={Wang et al.}, booktitle={ICLR}, year={2026}
}
```
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
'''

REQUIREMENTS = """spaces
torch
torchvision
transformers
diffusers>=0.42
peft>=0.18
einops
pillow
numpy
accelerate
sentencepiece
safetensors
huggingface_hub
opencv-python-headless
fonttools
pyyaml
"""


def _push(folder, repo_id, repo_type, msg):
    from huggingface_hub import HfApi
    api = HfApi()
    kw = {"space_sdk": "gradio", "space_hardware": "zero-a10g"} if repo_type == "space" else {}
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True, **kw)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type=repo_type, commit_message=msg)
    print(f"pushed → https://huggingface.co/{'spaces/' if repo_type == 'space' else ''}{repo_id}")


def stage_model(out):
    from huggingface_hub import hf_hub_download
    (out / "samples").mkdir(parents=True, exist_ok=True)
    shutil.copy(LORA, out / LORA.name)
    shutil.copy(EMPTY, out / EMPTY.name)
    shutil.copy(hf_hub_download(BASE, "LICENSE.md"), out / "LICENSE.md")
    (out / "NOTICE.md").write_text(NOTICE, encoding="utf-8")
    (out / "README.md").write_text(MODEL_CARD, encoding="utf-8")
    for f in ("font_render", "iam_lines"):   # real_hw_* 는 비공개 데이터셋 크롭이라 모델 카드엔 안 올린다
        shutil.copy(HERE / f"docs/img_readme/{f}.png", out / f"samples/{f}.png")


def stage_space(out):
    from huggingface_hub import snapshot_download
    for f in SPACE_CODE:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(HERE / f, out / f)
    ex = Path(snapshot_download(ERUKU_EXAMPLES, repo_type="space", allow_patterns=["example_images/*", "examples/*"]))
    shutil.copytree(ex / "example_images", out / "example_images", dirs_exist_ok=True)
    shutil.copytree(ex / "examples", out / "examples", dirs_exist_ok=True)
    (out / "app.py").write_text(APP % {"model": MODEL_REPO}, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model", "space"])
    ap.add_argument("--out", default=None, help="스테이징 폴더(기본 _release/<what>)")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--msg", default="InkSpire Korean s13000 release", help="HF 커밋 메시지")
    a = ap.parse_args()
    out = Path(a.out or HERE / "_release" / a.what)
    shutil.rmtree(out, ignore_errors=True)
    {"model": stage_model, "space": stage_space}[a.what](out)
    print(f"staged → {out}")
    if a.push:
        _push(out, MODEL_REPO if a.what == "model" else SPACE_REPO, "model" if a.what == "model" else "space",
              a.msg)
