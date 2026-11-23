"""판독기(OCR)로 틀린 어절만 다시 인페인팅 — 필체를 덜 깎고 글자 오류를 잡는다(EXPERIMENTS §18 의 B).

내용 CFG 는 줄 전체를 정형 글자 쪽으로 당겨 필체를 깎는다. 여기선 cfg 1 로 먼저 쓰고, 판독기가 목표와 다르게 읽은 **어절만**
마스크해 새 시드 · 내용 CFG(기본 2.0)로 다시 채운다. 맞은 어절은 그대로 남는다. 판독 결과가 나빠진 라운드는 버린다.

  g = InkSpireGen("<lora_dir>")                       # cfg=1.0(기본) — 첫 패스는 CFG 끔
  read = PPOCRReader("<paddle env>/bin/python", "<Eruku repo>/eval")   # 또는 아무 함수: u8[h,w] 줄 이미지 → 문자열
  img, pred = gen_fix(g, ref, "ref 전사", "생성할 문장", read, rounds=2)

판독기는 `read(u8[h,w]) -> str` 이면 무엇이든 된다. PP-OCRv6 는 paddlepaddle-gpu==3.3.1 전용 env 가 따로 필요해서
(원본 repo eval/ppocr_rec.py 머리말) PPOCRReader 가 그 env 의 파이썬을 자식 프로세스로 띄워 둔다.
  python infer/fix.py   # 의존 없는 self-check
"""
import difflib
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

norm = lambda s: re.sub(r"\s+", " ", s).strip()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def wrong_spans(T, pred):
    """목표 T 와 판독 pred 가 다른 글자를 어절로 넓힌 구간 [(a, b)] (T 글자 번호, 양끝 포함). 공백 오류는 양옆 어절을 묶는다."""
    bad = set()
    for op, i1, i2, _, _ in difflib.SequenceMatcher(None, T, pred, autojunk=False).get_opcodes():
        if op != "equal":
            bad.update(range(i1, i2) if i2 > i1 else {max(i1 - 1, 0), min(i1, len(T) - 1)})   # 삽입 = 그 자리 양옆
    mark = [False] * len(T)
    for i in bad:
        l = r = i
        while l > 0 and T[l - 1] != " ": l -= 1
        while r < len(T) - 1 and T[r + 1] != " ": r += 1
        mark[l:r + 1] = [True] * (r - l + 1)
    spans, a = [], None
    for i, m in enumerate(mark + [False]):
        if m and a is None: a = i
        if not m and a is not None: spans.append((a, i - 1)); a = None
    return spans


def gen_fix(g, style_arr, style_text, text, read, seed=0, rounds=2, cfg=2.0):
    """한 줄 생성 + 틀린 어절 재인페인팅 최대 rounds 번. g = InkSpireGen, read = u8[h,w] → str.
    → (줄 이미지 u8, 그 판독 문자열). 비용 = 1 + (틀린 줄에 한해) 라운드마다 cfg 패스(전방 2회)."""
    T = norm(text)
    c = g.prepare(style_arr, style_text, [text], seed)
    idx = np.where(c["line_id"] == 1)[0]
    aligned = "".join(c["chars"][i] for i in idx) == T   # 표준폰트에 없는 글자 등 — 상자와 글자 번호가 안 맞으면 줄 전체를 다시
    best = None
    for rd in range(rounds + 1):
        x, mask = c["x"], c["mask"]
        if rd:
            x, mask = best[1], np.zeros_like(c["mask"])
            for s0, s1 in (wrong_spans(T, best[3]) if aligned else [(0, len(idx) - 1)]):
                mask[c["y_cut"]:, max(0, c["boxes"][idx[s0], 0] - 3):c["boxes"][idx[s1], 2] + 3] = 1
        out = g.inpaint(x, c["xc"], mask, seed + 7919 * rd, cfg=cfg if rd else None)
        line = g._crop_line(out, c, 0)
        pred = norm(read(line))
        d = ed(T, pred)
        if best is None or d < best[0]:
            best = (d, out, line, pred)
        if best[0] == 0:
            break
    return best[2], best[3]


class PPOCRReader:
    """다른 env 의 PP-OCRv6(원본 repo eval/ppocr_rec.py)를 자식 프로세스로 띄워 두고 줄 이미지를 읽는다."""
    SRV = ("import sys; sys.path.insert(0, sys.argv[1]); import ppocr_rec as P; m = P.load()\n"
           "print('ready', flush=True)\n"
           "for l in sys.stdin: print(P.read(*m, l.strip()).replace('\\n', ' '), flush=True)")

    def __init__(self, python, eval_dir):
        self.p = subprocess.Popen([str(python), "-c", self.SRV, str(eval_dir)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert self.p.stdout.readline().strip() == "ready", "PP-OCR 로드 실패 — env 의 paddlepaddle-gpu==3.3.1 확인"
        self.f = Path(tempfile.mkdtemp(prefix="ppocr_")) / "line.png"

    def __call__(self, img):
        from PIL import Image
        Image.fromarray(img).save(self.f)
        self.p.stdin.write(f"{self.f}\n"); self.p.stdin.flush()
        return self.p.stdout.readline().rstrip("\n")


def demo():
    T = "새벽 첫 기차를 탔다"
    assert wrong_spans(T, T) == []
    assert wrong_spans(T, "새벽 첫 기자를 탔다") == [(5, 7)]            # 한 글자 → 그 어절
    assert wrong_spans(T, "새벽 첫기차를 탔다") == [(3, 7)]             # 공백 누락 → 양옆 어절
    assert wrong_spans(T, "새벽 첫 기차를 탔다가") == [(9, 10)]          # 끝 삽입 → 마지막 어절
    assert wrong_spans(T, "") == [(0, len(T) - 1)]

    class G:   # 가짜 생성기: 마스크된 칸에 시드를 적는다 — 틀린 어절만 다시 그려지는지, 나빠진 라운드가 버려지는지
        def prepare(self, *a):
            return {"x": np.zeros((2, 10), np.uint8), "xc": None, "mask": np.ones((2, 10), np.uint8), "y_cut": 0,
                    "line_id": np.ones(len(T), int), "chars": list(T), "boxes": np.array([[i, 0, i + 1, 2] for i in range(len(T))])}
        def inpaint(self, x, xc, mask, seed, cfg=None):
            calls.append((int(mask[0].sum()), cfg)); return np.where(mask > 0, len(calls), x).astype(np.uint8)
        def _crop_line(self, out, c, i): return out
    calls, reads = [], iter(["새벽 첫 기자를 탔다", "새벽 첫 가자를 탔다", T])
    img, pred = gen_fix(G(), None, "", T, lambda im: next(reads), rounds=3)
    assert pred == T and [c[1] for c in calls] == [None, 2.0, 2.0]
    assert calls[1][0] == calls[2][0] < calls[0][0]            # 2·3번째는 '기차를' 칸만(±3px), 나빠진 2번째는 버리고 같은 칸을 다시
    assert img[0, 0] == 1 and img[0, 6] == 3                   # 맞은 어절은 첫 패스 그대로, 틀린 어절은 마지막 패스
    print("[demo] ok")


if __name__ == "__main__":
    demo()
