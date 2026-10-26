"""VAE 복원 상한 — GT 렌더를 encode→decode 만 해서 PP-OCR 로 읽는다(생성 모델이 넘을 수 없는 획 바닥).

cer_dump.py 출력의 gt/ 를 줄 높이 h 로 줄여(style_tensor 와 같은 bilinear) 흰 여백으로 16 배수 패딩 → 복원 → 크롭.
  python experiments/vae_recon.py finetune_runs/_cer/s13000_test_sref/gt <out> --vae flux1|flux2 [--h 64]
  cd $E && PYTHONPATH=$E .venv/bin/python $I/experiments/vae_recon.py <src> <out> --vae eruku   # Eruku 체크포인트 속 한글 VAE
--vae eruku: Emuru VAE(latent 1ch, /8, RGB→흑백)에 corpusmax ckpt 의 vae.* 를 얹은 것. 높이 64 고정이고, 폭 ≤768 은 학습과
  같이 768 까지 흰 패딩, 768 초과는 **압축하지 않고** 8 배수 패딩(원본 htr_cer --vae-only 는 768 로 가로 압축해 긴 줄을 손해 본다).
  cd $E && .venv-paddle/bin/python eval/ppocr_rec.py <out> --save-preds
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

VAES = {"flux1": ("AutoencoderKL", "black-forest-labs/FLUX.1-Fill-dev"),
        "flux2": ("AutoencoderKLFlux2", "black-forest-labs/flux-3-image-klein-base"),
        "eruku": (None, "finetune_runs/eruku_corpusmax/checkpoint_last.pth")}   # 원본 repo env·cwd 에서


@torch.no_grad()
def main(a):
    cls, repo = VAES[a.vae]
    if cls is None:
        from infer.show import load_model   # 원본 repo 의 것(PYTHONPATH=$E)
        vae = load_model(repo, torch.device("cuda"))[0].vae.float().eval()
    else:
        import diffusers
        vae = getattr(diffusers, cls).from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32).cuda().eval()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = json.loads((src / "transcriptions.json").read_text())
    for k in tr:
        im = Image.open(src / k).convert("L")
        w, h = im.size
        if a.h:
            im = im.resize((max(1, round(w * a.h / h)), a.h), Image.BILINEAR); w, h = im.size
        H, W = -(-h // 16) * 16, -(-w // 16) * 16
        if a.vae == "eruku":
            H, W = h, (768 if w <= 768 else -(-w // 8) * 8)
        x = np.full((H, W), 255, np.float32); x[:h, :w] = np.asarray(im)
        t = torch.from_numpy(x / 127.5 - 1)[None, None].repeat(1, 3, 1, 1).cuda()
        y = vae.decode(vae.encode(t).latent_dist.mode()).sample[0].mean(0)[:h, :w]
        Image.fromarray(((y.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()).save(out / k)
    shutil.copy(src / "transcriptions.json", out / "transcriptions.json")
    print(f"{a.vae} h={a.h} {len(tr)} → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--vae", choices=list(VAES), required=True)
    ap.add_argument("--h", type=int, default=64, help="줄 높이로 축소(0 = 원본 크기)")
    main(ap.parse_args())
