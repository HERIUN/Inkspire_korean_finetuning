"""실손글씨 미러링(echo) 덤프 — HF HERIUN/korean-handwriting(비공개, 338장) 이미지를 ref 로, 그 전사를 그대로 다시 쓰게 한다.

ref = 실제 손글씨 줄 + 그 전사, 목표 = 같은 전사 → 출력과 실제 이미지를 비교(CER·HWD). 두 모델에 같은 ref 를 준다.
infer.show 의 load_model / gen_from_style / style_tensor 만 써서 **이 repo(InkSpire)와 원본 repo(Eruku) 양쪽 env 에서 돈다.**
출력 <out>/{gen,gt}/{i:05d}.png + transcriptions.json + plan.json — cer_dump 와 같은 형식이라 ppocr_rec.py·style_scores.py·
montage.py 를 그대로 쓴다(plan 의 "font" 칸 = 필기자 키: hw_line 은 한 명, 서식 크롭은 문서 이름, "ref_image" = ref 경로).

  I=/data/work/dgkang/inkspire_korean; E=/data/work/dgkang/Eruku_korean_finetuning; D=$I/finetune_runs/_realhw
  cd $I && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$I .venv/bin/python experiments/realhw_dump.py $D/inkspire \\
      --ckpt inkspire:finetune_runs/inkspire_klein_p512/lora_step_013000
  cd $E && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$E .venv/bin/python $I/experiments/realhw_dump.py $D/eruku \\
      --ckpt finetune_runs/eruku_corpusmax/checkpoint_last.pth
  cd $E && .venv-paddle/bin/python eval/ppocr_rec.py $D/inkspire/gt $D/inkspire/gen $D/eruku/gen --save-preds
--cross: 다른 문구 생성. 같은 필기자(hw_line 은 다음 줄, 서식은 같은 문서의 다음 칸)의 다른 표본 문구를 쓰게 하고
  그 표본의 실제 이미지를 GT 로 둔다(논문 IAM 프로토콜과 같은 꼴). 출력 형식은 echo 와 같다(plan 행에 ref_text 추가).
--free "문구" … --refs 0 25 …: 데이터에 없는 문구를 고른 ref 들로 생성 → <out>/free/{ref:05d}_{문구번호}.png (정답 없음, 육안용).
  격자 몽타주는 이 repo env 에서: python experiments/realhw_dump.py <out.png> --grid "InkSpire=<dir>" "Eruku=<dir>"
데이터는 개인정보가 섞인 내부 자료다 — 결과물을 외부에 올리지 않는다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "finetune_runs/_realhw/dataset"


def writer(r):
    """필기자 키. 소비자(style_scores·montage)가 Path(font).stem 을 쓰므로 '.' 을 뺀다."""
    return "hw_line" if r["subset"] == "hw_line" else Path(r["file_name"]).stem.rsplit("_p", 1)[0].replace(".", "_")


def eruku_crop(a, thr=180, bthr=215, dens_col=0.03, dens_row=0.006, solid=0.55, pad=4):
    """원본 repo 공식 실손글씨 ref 크롭(experiments/mirror_compare.py bbox_crop_gray = HF 릴리즈 ink_bbox_crop 과 같은 값).
    ① 가장자리 solid 테두리(밀도 > solid 인 바깥 행·열) 벗기기 ② 밀도 기반 bbox(희소 노이즈 무시), 여백 4px.
    ★ 기본값. Eruku 는 여백이 latent std 를 6~29배 튀겨 runaway/백지가 된다(원본 README) — 여백을 붙이는 prep 은 불리했다."""
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


def thick(a, k):
    if k <= 0:
        return a
    import cv2   # 두 env 모두에 있다
    a = cv2.erode(a, np.ones((k, k), np.uint8))   # grayscale erode = 어두운 획을 넓힌다
    return np.where(a < 200, (a * 0.3).astype(np.uint8), a).astype(np.uint8)


def prep(arr):
    """ref 전처리(두 모델 공통): 배경 중앙값 → 255, 표 선 제거, 잉크(<200) bbox 크롭, 잉크 높이 15% 여백.
    ★ 없으면 Eruku 가 무너진다 — hw_line 은 폭 1348 에 글씨가 왼쪽 45% 뿐이고 획이 연해서 20장 전부 백지,
      서식 크롭은 빈 영역·표 선 때문에 줄무늬 반복(CER 1.96, 폭주 18.6%). Eruku 는 늘 잉크에 맞춘 ref 로 학습됐다.
      InkSpire 는 prepare 가 잉크 bbox 를 따로 잘라 영향이 작다(0.42).
    ★ 표 선 제거: 가장자리 세로 여백선·서식 칸 선이 있으면 잉크 bbox 가 선까지 잡혀 글씨가 작게 남는다(hw_line 20장 중
      절반, 크롭만 했을 때 Eruku 백지 6장). 높이(폭)의 60% 이상 잉크인 열(행)을 지운다 — 글자 획은 그만큼 길지 않다."""
    a = np.clip(arr.astype(np.float32) * 255.0 / max(1.0, float(np.median(arr))), 0, 255)
    ink = a < 200
    a[:, ink.mean(0) > 0.6] = 255
    a[ink.mean(1) > 0.6, :] = 255
    ys, xs = np.where(a < 200)
    if len(ys):   # 잉크 좌표 0.5~99.5 백분위 — 가장자리 잡티·짧은 선 조각이 bbox 를 끌고 가지 않게(+잉크 높이 5% 여유)
        y0, y1 = np.percentile(ys, [0.5, 99.5]); x0, x1 = np.percentile(xs, [0.5, 99.5])
        m = 0.05 * (y1 - y0)
        y0, y1 = int(max(0, y0 - m)), int(min(a.shape[0], y1 + m + 1))
        x0, x1 = int(max(0, x0 - m)), int(min(a.shape[1], x1 + m + 1))
        a = a[y0:y1, x0:x1]
    p = max(2, int(round(0.15 * a.shape[0])))
    return np.pad(a, p, constant_values=255).astype(np.uint8)


def grid(out, runs):
    """--free 덤프들 → 행 = ref, 열 = 문구, 셀 안에 모델별 출력을 세로로. 맨 왼쪽 열 = ref."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 이 repo 의 infer.show(이 repo env 전용)
    from infer.show import fit, label_img
    runs = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in runs]
    meta = json.loads((runs[0][1] / "free.json").read_text(encoding="utf-8"))
    CW, CH = 380, 52
    lab = lambda t, w, bold=False: np.pad(label_img(t, w, 22, 13, bold=bold, center=False, bg=235), ((0, 0), (0, 0)))
    head = [lab("ref", CW // 2)] + [lab(t[:28], CW, True) for t in meta["texts"]]
    rows = [np.hstack([np.pad(h, 1) for h in head])]
    for ri in meta["refs"]:
        refc = np.vstack([fit(np.asarray(Image.open(runs[0][1] / "ref" / f"{ri:05d}.png").convert("L")), CW // 2, CH * len(runs))])
        cells = [np.vstack([lab("", CW // 2), refc])]
        for ti in range(len(meta["texts"])):
            parts = []
            for name, d in runs:
                parts += [lab(name, CW), fit(np.asarray(Image.open(d / "free" / f"{ri:05d}_{ti}.png").convert("L")), CW, CH)]
            cells.append(np.vstack(parts))
        H = max(c.shape[0] for c in cells)
        rows.append(np.hstack([np.pad(np.pad(c, ((0, H - c.shape[0]), (0, 0)), constant_values=255), 1) for c in cells]))
    W = max(r.shape[1] for r in rows)
    Image.fromarray(np.vstack([np.pad(r, ((0, 3), (0, W - r.shape[1])), constant_values=120) for r in rows])).save(out)
    print(f"grid {len(meta['refs'])}×{len(meta['texts'])} → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ckpt", help="생성 모드에 필수(--grid 엔 불필요)")
    ap.add_argument("--data", default=str(DATA), help="snapshot_download 한 HERIUN/korean-handwriting")
    ap.add_argument("--subset", nargs="*", default=None, help="hw_line / seosik_hw / sanrim_hw_pseudo (기본 전부)")
    ap.add_argument("--cfg", type=float, default=1.5, help="Eruku 전용 — 원본 README 한글 권장 1.5~2.0(1.0 = CFG 끔, 흐림·백지). InkSpire 는 무시")
    ap.add_argument("--max-new", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", choices=["eruku", "prep", "raw"], default="eruku",
                    help="ref 전처리: eruku = 원본 repo 공식 크롭(기본) / prep = 표 선 제거+백분위+15%% 여백 / raw = 원본 그대로")
    ap.add_argument("--stroke", type=int, default=0,
                    help="진단용: 크롭 뒤 ref 획을 k×k 최솟값 필터로 굵히고 잉크를 검정(0)으로 진하게 — 가늘고 연한 실제 획이 원인인지")
    ap.add_argument("--cross", action="store_true", help="같은 필기자의 다른 표본 문구를 생성(docstring)")
    ap.add_argument("--free", nargs="*", default=None, help="데이터에 없는 문구들 — --refs 와 함께")
    ap.add_argument("--refs", nargs="*", type=int, default=[], help="--free 에 쓸 ref 표본 번호(metadata 순서)")
    ap.add_argument("--grid", nargs="*", default=None, help="라벨=--free 덤프 디렉토리 … → out 을 PNG 경로로 보고 격자만 만든다")
    a = ap.parse_args()
    if a.grid:
        return grid(a.out, a.grid)

    import torch
    from infer.show import gen_from_style, load_model, style_tensor
    data, out = Path(a.data), Path(a.out)
    rows = [json.loads(l) for l in (data / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not a.subset or r["subset"] in a.subset]
    model = load_model(a.ckpt, torch.device("cuda"))[0]
    gen = lambda ref, st, tt, seed: gen_from_style(model, style_tensor(ref), st, tt, a.cfg, a.max_new, torch.device("cuda"), seed=seed)
    load = lambda r: np.asarray(Image.open(data / r["file_name"]).convert("L"))
    if a.free:
        (out / "free").mkdir(parents=True, exist_ok=True); (out / "ref").mkdir(exist_ok=True)
        for ri in a.refs:
            arr = load(rows[ri]); ref = thick({"eruku": eruku_crop, "prep": prep, "raw": lambda x: x}[a.crop](arr), a.stroke)
            Image.fromarray(ref).save(out / "ref" / f"{ri:05d}.png")
            for ti, t in enumerate(a.free):
                Image.fromarray(gen(ref, rows[ri]["text"], t, a.seed * 1_000_003 + ri)).save(out / "free" / f"{ri:05d}_{ti}.png")
        (out / "free.json").write_text(json.dumps({"refs": a.refs, "texts": a.free, "ckpt": a.ckpt}, ensure_ascii=False), encoding="utf-8")
        print(f"free {len(a.refs)}×{len(a.free)} → {out / 'free'}")
        return
    nxt = {}   # --cross: 같은 필기자 안에서 다음 표본(순환)
    by_w = {}
    for i, r in enumerate(rows):
        by_w.setdefault(writer(r), []).append(i)
    for idx in by_w.values():
        for k, i in enumerate(idx):
            nxt[i] = idx[(k + 1) % len(idx)]
    for sub in ("gen", "gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    plan, tr = [], {}
    for i, r in enumerate(rows):
        name = f"{i:05d}.png"
        arr = load(r)
        ref = thick({"eruku": eruku_crop, "prep": prep, "raw": lambda x: x}[a.crop](arr), a.stroke)
        t = rows[nxt[i]] if a.cross else r   # 목표 표본(echo 면 자기 자신)
        g = gen(ref, r["text"], t["text"], a.seed * 1_000_003 + i)
        Image.fromarray(g).save(out / "gen" / name)
        Image.fromarray(load(t)).save(out / "gt" / name)   # 목표 표본의 실제 이미지(전처리 전) — CER 리더 바닥·HWD real
        tr[name] = t["text"]
        plan.append({"i": i, "text": t["text"], "ref_text": r["text"], "font": writer(r), "subset": r["subset"],
                     "ref_image": str(data / r["file_name"])})
        if a.crop != "raw":
            (out / "ref").mkdir(exist_ok=True); Image.fromarray(ref).save(out / "ref" / name)
            plan[-1]["ref_image"] = str(out / "ref" / name)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    for sub in ("gen", "gt"):
        (out / sub / "transcriptions.json").write_text(json.dumps(tr, ensure_ascii=False), encoding="utf-8")
    (out / "plan.json").write_text(json.dumps({"seed": a.seed, "style_ref_text": None, "rows": plan},
                                              ensure_ascii=False, indent=0), encoding="utf-8")
    (out / "ckpt.txt").write_text(f"{a.ckpt} cfg={a.cfg}" + f" crop={a.crop}" + (" cross" if a.cross else " echo") + "\n")
    print(f"gen {len(tr)} → {out}")


if __name__ == "__main__":
    main()
