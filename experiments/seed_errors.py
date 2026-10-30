"""같은 표본을 시드 여러 개로 생성한 덤프들의 PP-OCR 판독(ppocr_rec.py --save-preds) → best-of-N 상한과
틀린 정답 글자 위치가 몇 시드에서 겹치나(무작위 오류 vs 일관된 오류). EXPERIMENTS §10.
  python experiments/seed_errors.py finetune_runs/_realhw/seeds_hw_line s0 s1 s2
"""
import json, sys, difflib
from pathlib import Path
D = Path(sys.argv[1]); seeds = sys.argv[2:]
norm = lambda s: "".join(s.split())
rows = json.loads((D / seeds[0] / "plan.json").read_text())["rows"]
P = {s: json.loads((D / s / "gen/ppocr_preds.json").read_text()) for s in seeds}
def ed(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]
def errpos(gt, pr):  # 틀린 정답 글자 인덱스
    bad = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gt, pr, autojunk=False).get_opcodes():
        if op in ("replace", "delete"): bad |= set(range(i1, i2))
        if op == "insert": bad.add(min(i1, len(gt) - 1))
    return bad
N = tot = best = 0; cnt = {}
for r in rows:
    k = f"{r['i']:05d}.png"; gt = norm(r["text"]); N += len(gt)
    e = [ed(gt, norm(P[s][k])) for s in seeds]; tot += sum(e) / len(e); best += min(e)
    for p in set().union(*(errpos(gt, norm(P[s][k])) for s in seeds)):
        c = sum(p in errpos(gt, norm(P[s][k])) for s in seeds); cnt[c] = cnt.get(c, 0) + 1
print(f"한 번(평균) {tot/N:.4f}   best-of-{len(seeds)} 상한 {best/N:.4f}   (글자 {N})")
print("틀린 글자 위치가 몇 시드에서 틀렸나:", dict(sorted(cnt.items())))
