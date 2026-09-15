"""docs/inkspire.md §7.2 용 APE/R-APE 설명 그림 — 토큰 격자와 위치 id 를 직접 그린다.

재생성: PYTHONPATH=. .venv/bin/python docs/_gen_inkspire_ape.py → docs/img_inkspire_io/06_ape.png"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
F = str(HERE.parent / "assets/fonts/label/NanumGothic-Regular.ttf")
big, mid, small = (ImageFont.truetype(F, s) for s in (19, 15, 12))
CELL, GAP = 46, 3
X_BG, C_BG, TXT = (214, 231, 247), (252, 232, 214), (40, 40, 40)


def grid(d, x0, y0, rows, cols, ids, bg):
    """ids[r][c] 문자열을 셀에 찍는다."""
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * (CELL + GAP), y0 + r * (CELL + GAP)
            d.rectangle([x, y, x + CELL, y + CELL], fill=bg, outline=(150, 150, 150))
            t = ids[r][c]
            b = d.textbbox((0, 0), t, font=small)
            d.text((x + (CELL - b[2]) / 2, y + (CELL - b[3]) / 2), t, font=small, fill=TXT)
    return x0 + cols * (CELL + GAP), y0 + rows * (CELL + GAP)


def seq_strip(d, x0, y0, blocks, w=58, h=30):
    """[라벨, 색] 리스트를 1-D 토큰 열로 그린다."""
    for i, (t, bg) in enumerate(blocks):
        x = x0 + i * (w + 2)
        d.rectangle([x, y0, x + w, y0 + h], fill=bg, outline=(150, 150, 150))
        b = d.textbbox((0, 0), t, font=small)
        d.text((x + (w - b[2]) / 2, y0 + (h - b[3]) / 2), t, font=small, fill=TXT)
    return x0 + len(blocks) * (w + 2)


W, H = 1180, 1150
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
R, C = 3, 4          # 손글씨 절반: 3행 4열 토큰

# ── ① baseline: 순차 id ──
d.text((20, 16), "1) 순진한 연결 (baseline) — 이어붙인 캔버스에 좌표를 순서대로 매긴다", font=big, fill=TXT)
d.text((20, 46), "X (손글씨, 생성 대상)", font=mid, fill=(20, 70, 130))
d.text((20 + 4 * (CELL + GAP) + 24, 46), "Xc (콘텐츠, 항상 보임)", font=mid, fill=(150, 80, 20))
ids_x = [[f"{r},{c}" for c in range(C)] for r in range(R)]
ids_c = [[f"{r},{c + C}" for c in range(C)] for r in range(R)]
x_end, y_end = grid(d, 20, 70, R, C, ids_x, X_BG)
grid(d, x_end + 20, 70, R, C, ids_c, C_BG)
d.text((20, y_end + 10), "행 우선으로 펴면 두 종류가 행마다 번갈아 나온다:", font=mid, fill=TXT)
seq_strip(d, 20, y_end + 36, [("X 0행", X_BG), ("Xc 0행", C_BG), ("X 1행", X_BG),
                              ("Xc 1행", C_BG), ("X 2행", X_BG), ("Xc 2행", C_BG)])
d.text((20, y_end + 76), "★ 짝까지 거리 = +4칸(=폭). 학습 패치는 +32칸, 추론 캔버스는 +39칸 — 그림마다 달라져서 규칙으로 못 배운다",
       font=mid, fill=(170, 30, 30))

# ── ② APE ──
y = 340
d.text((20, y), "2) APE — Xc 토큰에 짝인 X 토큰의 좌표를 그대로 복사", font=big, fill=TXT)
x_end, y_end = grid(d, 20, y + 30, R, C, ids_x, X_BG)
grid(d, x_end + 20, y + 30, R, C, ids_x, C_BG)          # ★ 같은 id
for r in range(R):                                       # 짝 화살표
    ya = y + 30 + r * (CELL + GAP) + CELL // 2
    d.line([x_end - 6, ya, x_end + 20 + 6, ya], fill=(60, 140, 60), width=2)
d.text((20, y_end + 10), "★ 거리 0. 폭이 얼마든 짝을 바로 안다. 조건인지 생성 대상인지는 위치가 아니라 "
       "마스크 채널이 알려준다(Xc 절반은 항상 0)", font=mid, fill=(30, 120, 30))

# ── ③ R-APE ──
y = 640
d.text((20, y), "3) R-APE — 90° 돌려 연결하면 캔버스·시퀀스에서의 짝 거리와 이음매 불연속이 줄어든다 (위치 id 는 APE 로 이미 동일)", font=big, fill=TXT)
d.text((20, y + 34), "돌리기 전: 2행 × 6열 → 이어붙이면 12열. 짝은 캔버스에서 6칸 떨어지고, 이음매에서 w 좌표가 5→0 으로 튄다", font=mid, fill=TXT)
ids_w = [[f"{r},{c}" for c in range(6)] for r in range(2)]
x_end, y2 = grid(d, 20, y + 58, 2, 6, ids_w, X_BG)
grid(d, x_end + 12, y + 58, 2, 6, ids_w, C_BG)
d.text((20, y2 + 10), "돌린 뒤: 6행 × 2열 → 이어붙여도 4열. 짝은 2칸, 이음매 점프도 1→0 으로 작다", font=mid, fill=TXT)
ids_r = [[f"{r},{c}" for c in range(2)] for r in range(6)]
x_end, y3 = grid(d, 20, y2 + 36, 6, 2, ids_r, X_BG)
grid(d, x_end + 12, y2 + 36, 6, 2, ids_r, C_BG)
d.text((x_end + 12 + 2 * (CELL + GAP) + 30, y2 + 60),
       "실측(추론 캔버스 176×624):\n  회전 없이 → 11행 78열, 캔버스 짝 거리 39칸\n  회전 후   → 39행 22열, 11칸\n"
       "※ RoPE id 거리는 APE 로 이미 0",
       font=mid, fill=(30, 60, 140))
im.save(HERE / "img_inkspire_io/06_ape.png")
print("saved", HERE / "img_inkspire_io/06_ape.png")
