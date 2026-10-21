"""cer_dump.py + eval/ppocr_rec.py --save-preds 결과 분석 — 획 오류가 어디서 나는가.

폰트별 중앙값 · 문자 종류별 오류율(한글/숫자/라틴/기호) · 한글 치환의 자모 위치(초성/중성/종성) · 잦은 치환쌍.
  python experiments/cer_analyze.py finetune_runs/_cer/s13000_test_sref [finetune_runs/_cer/eruku_test_sref ...]
eruku 덤프(htr_cer.py --dump-dir)는 plan.json 이 없으므로 폰트별은 첫 인자의 plan.json 을 같이 쓴다(같은 표본).
"""
import collections
import difflib
import json
import re
import sys
from pathlib import Path

import numpy as np

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def jamo(c):
    o = ord(c) - 0xAC00
    return (CHO[o // 588], JUNG[o % 588 // 28], JONG[o % 28]) if 0 <= o < 11172 else None


def kind(c):
    return "한글" if jamo(c) else "숫자" if c.isdigit() else "라틴" if c.isascii() and c.isalpha() else "기호"


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def analyze(d, plan):
    gt = json.loads((d / "gen/transcriptions.json").read_text())
    pr = json.loads((d / "gen/ppocr_preds.json").read_text())
    font = {f"{r['i']:05d}.png": Path(r["font"]).stem for r in plan["rows"]}
    per, byf = [], collections.defaultdict(list)
    tot, err = collections.Counter(), collections.Counter()
    pos, pairs = collections.Counter(), collections.Counter()
    for k, t in gt.items():
        t, p = norm(t), norm(pr[k])
        c = edit(p, t) / max(1, len(t)); per.append(c); byf[font.get(k, "?")].append(c)
        for ch in t.replace(" ", ""):
            tot[kind(ch)] += 1
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, t, p, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            for ch in t[i1:i2].replace(" ", ""):
                err[kind(ch)] += 1
            if op == "replace" and i2 - i1 == j2 - j1:          # 1:1 치환만 자모 비교
                for a, b in zip(t[i1:i2], p[j1:j2]):
                    ja, jb = jamo(a), jamo(b)
                    if ja and jb:
                        pos[tuple(n for n, x, y in zip(("초", "중", "종"), ja, jb) if x != y)] += 1
                    pairs[(a, b)] += 1
    per = np.array(per)
    print(f"\n## {d}  (n={len(per)})  중앙값 {np.median(per):.3f}  평균 {per.mean():.3f}  폭주율 {(per > 1).mean():.1%}")
    print("문자 종류별 오류율: " + "  ".join(f"{k} {err[k] / tot[k]:.1%}(n={tot[k]})" for k in ("한글", "숫자", "라틴", "기호") if tot[k]))
    n = sum(pos.values()) or 1
    print("한글 1:1 치환의 틀린 자모: " + "  ".join(f"{'+'.join(k) or '동일?'} {v / n:.0%}" for k, v in pos.most_common(6)) + f"  (n={n})")
    print("잦은 치환(정답→읽음): " + "  ".join(f"{a}→{b}×{v}" for (a, b), v in pairs.most_common(15)))
    return {f: np.median(v) for f, v in byf.items()}, {f: len(v) for f, v in byf.items()}


if __name__ == "__main__":
    dirs = [Path(x) for x in sys.argv[1:]]
    plan = json.loads((dirs[0] / "plan.json").read_text())
    res = [analyze(d, plan) for d in dirs]
    print("\n폰트별 중앙값  " + "  ".join(d.name for d in dirs))
    for f in sorted(res[0][0], key=lambda f: -res[0][0][f]):
        print(f"{f[:16]:16s} " + "  ".join(f"{r[0].get(f, float('nan')):.3f}" for r in res) + f"  (n={res[0][1][f]})")
