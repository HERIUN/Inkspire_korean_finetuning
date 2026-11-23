"""_fit_words self-check: 잉크 덩어리 3개(어절 3개) 이미지 → 어절 상자가 그 x 구간으로 옮겨지나.  python experiments/test_fit_words.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer.inkspire import LEFT, _fit_words

img = np.full((24, 300), 255, np.uint8)
for x0, x1 in ((0, 40), (100, 180), (220, 300)):
    img[:, x0:x1] = 0
ch = list("가나 다 라마")
bx = np.array([[10, 0, 20, 9], [20, 0, 30, 9], [30, 0, 35, 9], [35, 0, 45, 9], [45, 0, 50, 9], [50, 0, 60, 9], [60, 0, 70, 9]], np.float32)
o = _fit_words(bx, ch, img, 48)   # 잉크 높이 24 → 48: x 2배
assert np.allclose(o[[0, 1], 0], [LEFT, LEFT + 40]) and np.isclose(o[1, 2], LEFT + 80), o
assert np.allclose(o[3, [0, 2]], [LEFT + 200, LEFT + 360]) and np.allclose(o[[5, 6], 2], [LEFT + 520, LEFT + 600]), o
assert np.allclose(o[2, [0, 2]], [LEFT + 80, LEFT + 200])   # 공백 = 어절 사이
assert _fit_words(bx, list("가 나 다 라"), img, 48) is None   # 어절 수 불일치
print("ok")
