"""덤프 gen/ 폴더들을 원본 repo 한글 HTR(PP-OCR 과 독립인 리더)로 채점 — 원본 repo 에서 그 .venv 로 돌린다.
  cd $E && PYTHONPATH=$E .venv/bin/python $I/experiments/htr_dirs.py <dump>/gen …    (실손글씨엔 약하다: 실제 이미지 0.417)"""
import json, re, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from eval.htr_cer import to_htr_input, htr_read
from train.aux_htr import load_pretrained_htr
norm = lambda s: re.sub(r"\s+", " ", s).strip()
def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]
dev = torch.device("cuda")
htr = load_pretrained_htr(Path("finetune_runs/aux_htr_ko/htr_s20000")).to(dev).eval()
for d in map(Path, sys.argv[1:]):
    gt = json.loads((d / "transcriptions.json").read_text())
    e = n = 0
    for k, t in gt.items():
        with torch.no_grad():
            h = htr_read(htr, to_htr_input(np.asarray(Image.open(d / k).convert("L")), dev), dev)[0]
        e += ed(norm(h), norm(t)); n += len(norm(t))
    print(f"{e / n:.4f}\t{d}", flush=True)
