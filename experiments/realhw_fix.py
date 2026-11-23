"""판독기로 틀린 어절만 다시 인페인팅 — realhw_dump 덤프의 plan(ref·목표·시드)을 그대로 받아 라운드 단위로 돈다.

guidance 는 분포 전체를 정형 글자 쪽으로 당겨 CER↔HWD 절충선 위에서만 움직인다(EXPERIMENTS §10·§17). 여기선 당기지 않고
**고른다**: 생성 줄을 판독기(PP-OCR, 원본 repo env)로 읽어 목표와 다른 어절만 마스크하고 새 시드로 다시 채운다. 맞은 어절은 그대로 보인다.
판독기가 다른 env 라 라운드마다 프로세스를 나눈다(셸에서 생성 → ppocr_rec.py --save-preds → 다음 라운드).

  python experiments/realhw_fix.py <out>/r0 --src <dump> --ckpt inkspire:…           # 라운드 0: 캔버스째 생성(gen/ + canvas/)
  <ppocr> <out>/r0/gen --save-preds
  python experiments/realhw_fix.py <out>/r1 --src <dump> --ckpt … --prev <out>/r0    # 틀린 어절만 재생성
  …
  python experiments/realhw_fix.py <out>/final --src <dump> --prev <out>/r3 --final  # 생성 없이 고르기만
--prev 가 재생성 결과(base.json 있음)면 표본마다 그 전 결과와 편집거리를 비교해 **나빠진 건 되돌린 뒤** 이어 간다.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from infer.fix import demo, ed, norm, wrong_spans   # 핵심은 infer/fix.py(사용자용 gen_fix)와 공유


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--src", help="realhw_dump 덤프(plan.json · ref/ · gt/)")
    ap.add_argument("--ckpt")
    ap.add_argument("--prev", help="앞 라운드 디렉토리(gen/ppocr_preds.json 필요). 없으면 라운드 0")
    ap.add_argument("--final", action="store_true", help="생성 없이 되돌리기 판정만 해서 최종 결과를 만든다")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()
    src, out = Path(a.src), Path(a.out)
    P = json.loads((src / "plan.json").read_text(encoding="utf-8"))
    rows = P["rows"]
    for s in ("gen", "canvas"):
        (out / s).mkdir(parents=True, exist_ok=True)
    rd, base, preds = 0, {}, {}
    if a.prev:
        prev = Path(a.prev)
        rd = json.loads((prev / "round.json").read_text())["round"] + 1
        preds = json.loads((prev / "gen/ppocr_preds.json").read_text(encoding="utf-8"))
        old = json.loads((prev / "base.json").read_text()) if (prev / "base.json").exists() else {}
        for r in rows:   # base = 이번 라운드가 출발할 결과: prev, 단 prev 가 그 출발점보다 나빠졌으면 출발점으로 되돌린다
            k = f"{r['i']:05d}.png"
            T, b = norm(r["text"]), {"dir": str(prev), "pred": preds[k]}
            if k in old and ed(T, norm(old[k]["pred"])) < ed(T, norm(preds[k])):
                b = old[k]
            base[k] = b
    if not a.final:
        import torch
        from infer.show import load_model, style_tensor
        g = load_model(a.ckpt, torch.device("cuda"))[0]
        t = lambda x: torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)[None, None].to("cuda")
    n_regen = n_words = 0
    for r in rows:
        k = f"{r['i']:05d}.png"
        T = norm(r["text"])
        spans = wrong_spans(T, norm(base[k]["pred"])) if a.prev else [(0, len(T) - 1)]
        if a.final or not spans:
            for s in ("gen", "canvas"):
                shutil.copy(Path(base[k]["dir"]) / s / k, out / s / k)
            continue
        seed = P["seed"] * 1_000_003 + r["i"]
        ref = np.asarray(Image.open(r["ref_image"]).convert("L"))
        ref = ((style_tensor(ref)[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)   # realhw_dump(gen_from_style)와 같은 왕복
        c = g.prepare(ref, r["ref_text"], [r["text"]], seed)
        x, mask = c["x"], c["mask"]
        if a.prev:
            idx = np.where(c["line_id"] == 1)[0]
            if "".join(c["chars"][i] for i in idx) != T:   # 표준폰트에 없는 글자 등 — 상자와 글자 번호가 안 맞으면 줄 전체
                spans = [(0, len(idx) - 1)]
            x = np.asarray(Image.open(Path(base[k]["dir"]) / "canvas" / k).convert("L"))
            m = np.zeros_like(mask)
            for s0, s1 in spans:
                m[c["y_cut"]:, max(0, c["boxes"][idx[s0], 0] - 3):c["boxes"][idx[s1], 2] + 3] = 1
            mask = m
            n_words += len(spans)
        with torch.no_grad():
            xh = g.M.generate(g.tr, g.vae, t(x), t(c["xc"]), torch.from_numpy(mask.astype(np.float32))[None, None].to("cuda"), g.cache,
                              steps=g.steps, guidance=g.guidance, seed=seed + 7919 * rd, rape=g.rape, ape=g.ape,
                              **({"cfg": g.cfg, "uncond": g.uncond, "cfg_s": g.cfg_s, "cfg_t": g.cfg_t} if g.cfg != 1.0 else {}))
        o = ((xh[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(o).save(out / "canvas" / k)
        Image.fromarray(g._crop_line(o, c, 0)).save(out / "gen" / k)
        n_regen += 1
        if n_regen % 50 == 0:
            print(f"  regen {n_regen}", flush=True)
    (out / "gen/transcriptions.json").write_text(json.dumps({f"{r['i']:05d}.png": r["text"] for r in rows}, ensure_ascii=False), encoding="utf-8")
    shutil.copy(src / "plan.json", out / "plan.json")
    (out / "round.json").write_text(json.dumps({"round": rd, "regen": n_regen, "spans": n_words, "ckpt": a.ckpt}))
    if base:
        (out / "base.json").write_text(json.dumps(base, ensure_ascii=False))
    print(f"round {rd}: regen {n_regen}/{len(rows)} 줄, 구간 {n_words} → {out}")


if __name__ == "__main__":
    main()
