"""InkSpire 를 SHTG-IAM lines(Eruku 논문 표 3 프로토콜)로 잰다 — 원본 repo eval/iam_shtg.py 의 채점기를 그대로 쓴다.

hwd 패키지는 원본 repo .venv 에만, InkSpire 는 이 repo .venv 에만 있어 셋으로 나눈다.
  1) export (원본 env): IAMLines(num_style_samples=1) 항목 → <out>/style/{i:05d}.png + items.json
     (style 은 hwd 가 고정 순서로 고른다 — Eruku 결과와 같은 ref)
  2) gen (이 env): InkSpire 로 <out>/fake/<author>/<gen_id>.png. w/ T_s = ref 전사를 준다(InkSpire 는 필수).
     --shard k/N 으로 병렬.
  3) 채점 (원본 env): iam_shtg.py 는 fake/ 가 다 차 있으면 생성을 건너뛰고 채점만 한다(--ckpt 는 라벨).
  4) view (이 env): montage.py 용 plan.json + {fake,real,eruku}/gen/{i:05d}.png 심볼릭 링크

  I=/data/work/dgkang/inkspire_korean; E=/data/work/dgkang/Eruku_korean_finetuning; O=$I/finetune_runs/_iam/inkspire_klein_lines
  cd $E && .venv/bin/python $I/experiments/iam_dump.py export $O
  cd $I && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$I .venv/bin/python experiments/iam_dump.py gen $O --shard 0/2   (와 1/2)
  cd $E && CUDA_VISIBLE_DEVICES=2 .venv/bin/python eval/iam_shtg.py --ckpt inkspire_klein_s13000_rw --split lines --out $O
  cd $I && .venv/bin/python experiments/iam_dump.py view $O --eruku $E/finetune_runs/_iam/corpusmax_lines_cfg1.25_ctxeog
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
CKPT = "inkspire:finetune_runs/inkspire_klein_p512/lora_step_013000"


def export(out):
    from hwd.datasets.shtg import IAMLines
    ds = IAMLines(num_style_samples=1, load_gen_sample=False)
    (out / "style").mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(len(ds)):
        it = ds[i]
        it["style_imgs"][0].convert("L").save(out / "style" / f"{i:05d}.png")
        items.append({"i": i, "author": it["author"], "dst": it["dst_path"], "gen_text": it["gen_text"],
                      "style_text": it["style_imgs_text"][0], "style_id": it["style_ids"][0]})
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"export {len(items)} → {out}")


def gen(out, ckpt, shard):
    import torch
    sys.path.insert(0, str(HERE))
    from infer.show import load_model
    k, n = map(int, shard.split("/"))
    items = json.loads((out / "items.json").read_text(encoding="utf-8"))[k::n]
    todo = [it for it in items if not (out / "fake" / it["dst"]).exists()]   # 이어 돌리기
    model = load_model(ckpt, torch.device("cuda"))[0]
    for j, it in enumerate(todo):
        ref = np.asarray(Image.open(out / "style" / f"{it['i']:05d}.png").convert("L"))
        g = model.gen(ref, it["style_text"], it["gen_text"], it["i"])
        (out / "fake" / it["dst"]).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(g).save(out / "fake" / it["dst"])
        if (j + 1) % 100 == 0:
            print(f"  shard {shard}: {j + 1}/{len(todo)}", flush=True)
    print(f"gen shard {shard} {len(todo)} → {out / 'fake'}")


def view(out, eruku):
    """montage.py 가 읽는 형식: plan.json(rows: i, text, font=필기자, ref_image) + <run>/gen/{i:05d}.png."""
    items = json.loads((out / "items.json").read_text(encoding="utf-8"))
    runs = {"view_inkspire": out / "fake", "view_real": out / "real"} | ({"view_eruku": Path(eruku) / "fake"} if eruku else {})
    for name, src in runs.items():
        d = out / name / "gen"
        d.mkdir(parents=True, exist_ok=True)
        for it in items:
            link = d / f"{it['i']:05d}.png"
            if not link.exists() and (src / it["dst"]).exists():
                link.symlink_to((src / it["dst"]).resolve())
    rows = [{"i": it["i"], "text": it["gen_text"], "font": it["author"], "ref_image": str(out / "style" / f"{it['i']:05d}.png")}
            for it in items]
    (out / "plan.json").write_text(json.dumps({"seed": 0, "style_ref_text": None, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"view → {out}/view_*/gen, plan.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "gen", "view"])
    ap.add_argument("out")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--eruku", default=None, help="view: 비교할 Eruku iam_shtg 결과 폴더(fake/ 를 쓴다)")
    a = ap.parse_args()
    out = Path(a.out)
    {"export": lambda: export(out), "gen": lambda: gen(out, a.ckpt, a.shard), "view": lambda: view(out, a.eruku)}[a.mode]()
