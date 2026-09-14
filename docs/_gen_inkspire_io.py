"""docs/inkspire.md §7 용 그림 — InkSpire 두 단계에 "뭐가 들어가서 뭐가 나오는가" 를 실제 텐서로.

실제 코드 경로(KoreanPageDataset → build_inputs / KoreanLayoutDataset → layout_collate)를 그대로
돌리고 중간 텐서를 이미지로 저장한다. FLUX 가중치는 필요 없다 — 소형 VAE(models.inkspire._tiny,
8× 압축·16ch 로 실물과 shape 동일)로 shape 만 뽑는다.

재현:  PYTHONPATH=. .venv/bin/python docs/_gen_inkspire_io.py
출력:  docs/img_inkspire_io/*.png + numbers.json (md 본문 수치의 출처)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from custom_datasets.korean.page import (KoreanLayoutDataset, KoreanPageDataset,   # noqa: E402
                                         layout_collate, layout_seq, vocab)
from infer.show import label_img                                                   # noqa: E402
from models import inkspire                                                        # noqa: E402

OUT = HERE / "img_inkspire_io"
SEED, P = 7, 512


def _bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _lab(text, w, h=26):
    return _bgr(label_img(text, w, h, 15, center=False, bg=240))


def _row(cells, pad=8):
    """[(라벨, gray)] → 라벨 붙인 가로 결합 BGR."""
    h = max(c.shape[0] for _, c in cells)
    blocks = []
    for name, g in cells:
        g = np.pad(g, ((0, h - g.shape[0]), (0, 0), (0, 0))[:g.ndim], constant_values=255)
        b = g if g.ndim == 3 else _bgr(g)
        blocks += [np.vstack([_lab(name, b.shape[1]), b]), np.full((h + 26, pad, 3), 255, np.uint8)]
    return np.hstack(blocks[:-1])


def _stack(rows, pad=10):
    w = max(r.shape[1] for r in rows)
    out = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    return np.vstack([x for r in out for x in (r, np.full((pad, w, 3), 255, np.uint8))][:-1])


def _u8(t):
    return ((t[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nums = {}

    # ── ① 레이아웃 단계: 페이지 토큰 ──────────────────────────────────
    lay = KoreanLayoutDataset(P=P, length=100, seed=SEED)
    pg, _, _ = lay._page(0)
    box = _bgr(pg["x"])
    for ch, (x0, y0, x1, y1) in zip(pg["chars"], pg["bboxes"]):
        cv2.rectangle(box, (int(x0), int(y0)), (int(x1), int(y1)),
                      (0, 0, 255) if ch != " " else (0, 160, 0), 1)
    cv2.imwrite(str(OUT / "01_layout_tokens.png"),
                _stack([_row([("x: 스타일 폰트 페이지 + 글자 bbox (빨강=글자, 초록=공백 토큰)", box)]),
                        _row([("xc: 같은 bbox 에 표준폰트(NanumGothic) 글리프를 resize 합성 = 콘텐츠 이미지", pg["xc"])])]))
    b = layout_collate([lay[i] for i in range(2)])
    seq = layout_seq(pg["bboxes"], pg["line_id"], pg["page_w"])
    nums["layout"] = {"vocab": len(vocab()), "N_page": len(pg["chars"]), "lines": len(pg["line_top"]),
                      "page": list(pg["x"].shape), "pitch": pg["pitch"],
                      "collate": {k: list(v.shape) for k, v in b.items()},
                      "sample_tokens": [{"ch": c, "box": [int(v) for v in bb], "layout": [round(float(v), 4) for v in s]}
                                        for c, bb, s in zip(pg["chars"][:6], pg["bboxes"][:6], seq[:6])],
                      "seq_mean": [round(float(v), 4) for v in seq.mean(0)],
                      "seq_std": [round(float(v), 4) for v in seq.std(0)]}

    # ── ② 이미지 단계: P×P 패치 ───────────────────────────────────────
    rows, samp = [], {}
    for mode, who in (("rand", "학습"), ("line", "val·추론 모사")):
        s = KoreanPageDataset(P=P, mode=mode, length=100, seed=SEED)[0]
        samp[mode] = s
        x, xc, m = _u8(s["x"]), _u8(s["xc"]), s["mask"][0].numpy()
        ov = _bgr(x); ov[m > 0, 0] = 255; ov[m > 0, 2] = (ov[m > 0, 2] * 0.5).astype(np.uint8)
        rows.append(_row([(f"x  [1,{P},{P}]  mode={mode} ({who})", x), ("xc  [1,512,512]", xc),
                          (f"mask 1=생성 (커버리지 {m.mean():.2f})", ov)]))
    cv2.imwrite(str(OUT / "02_patch.png"), _stack(rows))

    # ── ③ R-APE: 90° 회전 후 [X|Xc] 연결 — 학습(R-Mask)·추론(첫 줄만 보임) 둘 다 ──
    rot = lambda t: inkspire._rot(t, -1)
    packed = {}
    for mode, (fname, who) in (("rand", ("03_rape_train.png", "학습 R-Mask")),
                               ("line", ("04_rape_infer.png", "추론 모사: 첫 줄만 보임"))):
        x, xc, mk = (samp[mode][k][None] for k in ("x", "xc", "mask"))
        I = torch.cat([rot(x), rot(xc)], -1)
        Im = torch.cat([rot(mk), torch.zeros_like(mk)], -1)
        cv2.imwrite(str(OUT / fname), _stack([
            _row([(f"[{who}]  I = rot90(X) ⓒ rot90(Xc)   [1,1,512,1024]  ← VAE 는 이걸 그대로 본다", _u8(I[0]))]),
            _row([("I_m: 마스크 (흰색=생성). Xc 절반은 항상 0 = 항상 보임", _u8(Im[0] * 2 - 1))]),
            _row([("I_i = I ⊗ (1−I_m): 실제 조건 입력 (생성 영역은 지워짐)", _u8((I * (1 - Im))[0]))])]))
        packed[mode] = (x, xc, mk)
    x, xc, mk = packed["rand"]                      # shape 수치는 학습 경로 기준

    # ── ④ shape: 소형 VAE 로 build_inputs 실행(실물과 동일 격자) ───────
    tr, vae, cache = inkspire._tiny("cpu")
    with torch.no_grad():
        inp = inkspire.build_inputs(x, xc, mk, vae)
        frac_line = float(inkspire.build_inputs(*packed["line"], vae)["loss_mask"].float().mean())
    ids = inp["img_ids"]
    L = inp["x0_packed"].shape[1]
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift
    nums["image"] = {"patch": [1, 1, P, P], "I": list(I.shape), "latent": [1, 16, 512 // 8, 1024 // 8],
                     "x0_packed": list(inp["x0_packed"].shape), "cond_packed": list(inp["cond_packed"].shape),
                     "hidden_states": [1, L, 64 + 320], "img_ids": list(ids.shape), "L": L,
                     "ids_halves_equal": bool(torch.equal(*ids.view(512 // 16, 2 * 512 // 16, 3).chunk(2, 1))),
                     "loss_mask_frac_rand": round(float(inp["loss_mask"].float().mean()), 4),
                     "loss_mask_frac_line": round(frac_line, 4),
                     "mu_calculate_shift": round(float(calculate_shift(L)), 4),
                     "empty_cache": {k: list(v.shape) for k, v in
                                     (torch.load(inkspire.EMPTY_CACHE).items() if Path(inkspire.EMPTY_CACHE).exists()
                                      else [("prompt_embeds", torch.zeros(1, 512, 4096)),
                                            ("pooled", torch.zeros(1, 768)), ("text_ids", torch.zeros(512, 3))])}}
    (OUT / "numbers.json").write_text(json.dumps(nums, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(nums, ensure_ascii=False, indent=2))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
