"""HF 릴리즈 — 최종 LoRA 를 모델 repo 로, gradio 데모를 Space 로 발행 (Eruku_korean_finetuning tools/hf_upload.py 대응).

    HERIUN/inkspire_korean        LoRA(pytorch_lora_weights.safetensors) + 빈 프롬프트 캐시 + FLUX Kommunity 라이선스·고지
    HERIUN/inkspire_korean_demo   Space(ZeroGPU) — 이 repo 의 추론 코드 최소 묶음 + app.py

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
MODEL_REPO, SPACE_REPO = "HERIUN/inkspire_korean", "HERIUN/inkspire_korean_demo"
BASE = "black-forest-labs/flux-3-image-klein-base"
LORA = HERE / "finetune_runs/inkspire_klein_p512/lora_step_013000/pytorch_lora_weights.safetensors"
EMPTY = HERE / "model_zoo/klein_empty_prompt.pt"
ERUKU_EXAMPLES = "HERIUN/eruku_korean_demo"   # 같은 예시로 두 데모를 나란히 비교할 수 있게 재사용
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
- 데모: https://huggingface.co/spaces/{SPACE_REPO}
- 같은 데이터로 학습한 자기회귀 모델 비교 대상: [HERIUN/eruku_korean](https://huggingface.co/HERIUN/eruku_korean)

## 입력 규약

캔버스 X(위 = ref 손글씨, 아래 = 빈 칸) · 콘텐츠 이미지 Xc(NanumGothic 으로 ref 전사와 목표 문구를 글자 상자에 그린 것) · 마스크.
Xc 를 글자별 대응표로 쓰므로 **ref 전사(style text)는 필수**다 — 비우면 붕괴한다(CER 0.72, 절반 백지).
목표 줄 폭은 ref 의 "손글씨 폭 / 표준폰트 폭" 비율로 맞춘다(`ref_width`, 기본).

## 성능 (ref 전사를 줄 때, PP-OCRv6 판독 CER)

| 평가 | InkSpire (이 모델) | Eruku Korean |
|---|---|---|
| 한글 폰트 held-out 16종 n=300 — CER↓ / HWD↓ | 0.023 / **0.93** | **0.010** / 1.12 |
| 실손글씨 문장 교차 생성 n=20 — CER↓ | 0.246 | **0.014** |
| 실손글씨 서식 칸 n=286 — CER↓ | **0.417** | 0.771 (백지·폭주) |
| IAM lines 2,925 — HWD↓ / FID↓ | 1.68 / **18.58** | **1.62** / 20.00 |

글자 정확도는 Eruku, 필체 형태(HWD)와 지저분한 ref 에 대한 견고성은 InkSpire 쪽이 낫다.
남은 오류는 샘플마다 무작위인 획 오류다(시드 3개 공통 오류 0) — 상세는 GitHub `docs/EXPERIMENTS.md`.

![한글 폰트](samples/font_render.png)
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
g = InkSpireGen(lora, device="cuda")
ref = np.asarray(Image.open("style_line.png").convert("L"))
out = g.gen_lines(ref, "스타일 이미지의 전사", ["생성할 문장"], seed=0)[0]   # u8 줄 이미지
```

## 라이선스

[FLUX Kommunity License](LICENSE.md) (베이스 모델과 동일, 비상업 용도) — [NOTICE](NOTICE.md) 참고.
학습 데이터는 공개 한글 폰트 렌더이며, 비교 평가에 쓴 실손글씨 데이터는 포함하지 않았다.
"""

SPACE_README = f"""---
title: InkSpire Korean 한글 손글씨 생성
emoji: ✒️
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: other
short_description: 손글씨 한 줄 + 전사로 같은 필체의 한글 문장 생성 (FLUX.3 Klein LoRA)
models:
- {MODEL_REPO}
- {BASE}
---

# InkSpire Korean

[{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) 데모 — 학습 코드 https://github.com/HERIUN/Inkspire_korean_finetuning

베이스 모델 {BASE} 는 FLUX Kommunity License(비상업) 이며, 이 Space 는 그 약관을 수락한 계정 토큰(secret `HF_TOKEN`)으로 베이스를 받는다.
생성 이미지는 AI 로 만든 것이다.
"""

APP = r'''"""InkSpire Korean — 한글 손글씨 스타일 생성 데모 (FLUX.3-Image-Klein-Base + LoRA)."""
import os
os.environ.setdefault("GRADIO_SSR_MODE", "0")

import json
import shutil
from pathlib import Path

import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

try:
    import spaces
    GPU = spaces.GPU
except Exception:
    def GPU(fn=None, **_):
        return fn if fn else (lambda f: f)

MODEL_ID = "%(model)s"
HERE = Path(__file__).resolve().parent
lora = snapshot_download(MODEL_ID)
(HERE / "model_zoo").mkdir(exist_ok=True)
shutil.copy(Path(lora) / "klein_empty_prompt.pt", HERE / "model_zoo/klein_empty_prompt.pt")   # models.inkspire_klein.EMPTY_CACHE

from infer.inkspire import InkSpireGen   # noqa: E402

model = InkSpireGen(lora, device="cuda")   # ZeroGPU: import 때 cuda 로 올려도 된다(실제 GPU 는 @GPU 안에서 붙는다)


def ink_bbox_crop(a, thr=180, bthr=215, dens_col=0.03, dens_row=0.006, solid=0.55, pad=4):
    """스캔 여백·테두리 제거(Eruku 릴리즈 ink_bbox_crop 과 같은 값, 실손글씨 평가 기본)."""
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


@GPU(duration=90)
def generate(style_image, style_text, gen_text, steps, seed, bbox_crop):
    lines = [s.strip() for s in (gen_text or "").splitlines() if s.strip()]
    if style_image is None:
        return None, "❌ 스타일 이미지를 올리세요"
    if not (style_text or "").strip():
        return None, "❌ 스타일 전사를 적으세요 — InkSpire 는 ref 전사가 필수입니다(없으면 붕괴)"
    if not lines:
        return None, "❌ 생성할 텍스트를 입력하세요"
    ref = np.asarray(style_image.convert("L"))
    if bbox_crop:
        ref = ink_bbox_crop(ref)
    model.steps = int(steps)
    outs = model.gen_lines(ref, style_text.strip(), lines[:8], seed=int(seed))
    return Image.fromarray(vstack(outs)), f"✅ {len(outs)}줄 생성" + (" (앞 8줄만)" if len(lines) > 8 else "")


def examples():
    f = HERE / "examples/samples.json"
    if not f.exists():
        return None
    return [[s["style_image"], s["style_text"], s["gen_text"]] for s in json.loads(f.read_text(encoding="utf-8"))
            if (HERE / s["style_image"]).exists()] or None


with gr.Blocks(title="InkSpire Korean — 한글 손글씨 생성") as demo:
    gr.HTML("""
    <div style="text-align:center">
      <h1>✒️ InkSpire Korean</h1>
      <h3 style="color:#888;font-weight:normal">손글씨 한 줄 + 그 전사 → 같은 필체의 한글 문장</h3>
      <p><a href="https://huggingface.co/%(model)s" target="_blank">🤗 Model</a> &nbsp;|&nbsp;
         <a href="https://github.com/HERIUN/Inkspire_korean_finetuning" target="_blank">💻 학습 코드</a> &nbsp;|&nbsp;
         <a href="https://huggingface.co/spaces/HERIUN/eruku_korean_demo" target="_blank">비교: Eruku Korean 데모</a></p>
    </div>""")
    with gr.Row():
        with gr.Column():
            style_image = gr.Image(label="📷 스타일 이미지 (손글씨 한 줄)", type="pil", height=200)
            style_text = gr.Textbox(label="📝 스타일 전사 (필수)", placeholder="스타일 이미지에 적힌 글자 그대로", lines=1)
            gen_text = gr.Textbox(label="✍️ 생성할 텍스트 (줄마다 한 줄씩, 최대 8줄)", lines=3, value="한국어 손글씨 생성 테스트")
            with gr.Row():
                steps = gr.Slider(label="steps", minimum=10, maximum=50, step=1, value=20)
                seed = gr.Number(label="seed", value=0, precision=0)
            bbox = gr.Checkbox(label="✂️ 스캔 여백·테두리 자동 제거", value=True)
            btn = gr.Button("🚀 생성", variant="primary", size="lg")
        with gr.Column():
            out = gr.Image(label="🎨 생성 결과", type="pil")
            status = gr.Textbox(label="상태", interactive=False)
    ex = examples()
    if ex:
        gr.Examples(examples=ex, inputs=[style_image, style_text, gen_text], label="📚 예시 (Eruku 데모와 같은 예시)")
    btn.click(generate, [style_image, style_text, gen_text, steps, seed, bbox], [out, status])
    gr.Markdown("""
---
- **스타일 전사는 필수**입니다. 글자 위치 대응표(Xc)를 전사로 만들기 때문에, 틀리게 적으면 필체가 어긋납니다.
- 성능(ref 전사를 줄 때, PP-OCRv6 CER): 한글 폰트 held-out **0.023**(HWD 0.93) / 실손글씨 문장 0.246 / 서식 칸 0.417.
  글자 정확도는 Eruku 가, 필체 형태(HWD)와 지저분한 스캔 ref 는 InkSpire 가 낫습니다.
- 남은 오류는 시드마다 다른 획 오류입니다 — 이상하면 seed 를 바꿔 보세요.
- 베이스: FLUX.3-Image-Klein-Base (FLUX Kommunity License, 비상업). 생성 이미지는 AI 가 만든 것입니다.
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
'''

REQUIREMENTS = """spaces
torch
diffusers>=0.42
peft>=0.18
transformers
accelerate
safetensors
huggingface_hub
opencv-python-headless
fonttools
pyyaml
pillow
numpy
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
    samples = json.loads((ex / "examples/samples.json").read_text(encoding="utf-8"))
    samples = [s for s in samples if any("가" <= c <= "힣" for c in s["gen_text"])]   # 한글 전용 학습 — 영어 예시는 뺀다
    (out / "examples").mkdir(exist_ok=True)
    (out / "examples/samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "app.py").write_text(APP % {"model": MODEL_REPO}, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["model", "space"])
    ap.add_argument("--out", default=None, help="스테이징 폴더(기본 _release/<what>)")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    out = Path(a.out or HERE / "_release" / a.what)
    shutil.rmtree(out, ignore_errors=True)
    {"model": stage_model, "space": stage_space}[a.what](out)
    print(f"staged → {out}")
    if a.push:
        _push(out, MODEL_REPO if a.what == "model" else SPACE_REPO, "model" if a.what == "model" else "space",
              "InkSpire Korean s13000 release")
