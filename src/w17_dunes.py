"""Dunes — paper-cut sand dunes, each with a lit face and a shaded slip face."""
import numpy as np
from common import *
from paper import *

x, y = grid()

sky = vgradient(y, [(0, "#f1e6d8"), (0.35, "#f3e2cb"), (0.55, "#f2d6b6"), (1, "#f2d6b6")])
s = Sheet(sky)

# sun low on the left, so the steep lee sides fall into shade on the right
sx, sy = W * 0.30, H * 0.405
s.add(shapes(circles=[(sx, sy, 250)]), hex2lin("#f5dcb6"), shadow=0.12, off=8, soft=16, rim=0.04, alpha=0.8)
s.add(shapes(circles=[(sx, sy, 150)]), hex2lin("#efb778"), shadow=0.22, off=10, soft=16, rim=0.05)


def bez(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)
    return [tuple(p) for p in pts]


def dune_layer(x0, base, dunes):
    """dunes: (windward length, lee length, height) left to right, starting at x0.
    Returns the silhouette polygon and a soft mask of the shaded lee faces."""
    line = []
    shade = np.zeros((H, W), np.float32)
    vx = x0
    for wl, wr, h in dunes:
        v0, c, v1 = (vx, base), (vx + wl, base - h), (vx + wl + wr, base)
        # windward: a long rise that still climbs at the brink; lee: steep, easing out at the foot
        up = bez(v0, (c[0] - wl * 0.30, c[1] + h * 0.10), c)
        down = bez(c, (c[0] + wr * 0.45, v1[1] - h * 0.02), v1)
        line += up + down[1:]
        # the lee face: from the brink its edge sweeps down and to the left, open below
        d = np.clip(y - c[1], 0, None)
        left = c[0] - wr * 0.65 * (d / h) ** 0.85
        right = v1[0] + np.clip(d - h, 0, None) * 0.5
        shade = np.maximum(shade, np.clip(x - left + 0.5, 0, 1) * np.clip(right - x + 0.5, 0, 1) * (y > c[1]))
        vx = v1[0]
    poly = [(-10, H + 10), (-10, base)] + line + [(W + 10, base), (W + 10, H + 10)]
    return poly, shade


layers = [
    # start x, base (frac of H), dunes (windward, lee, height), lit, shade
    (-260, 0.600, [(560, 200, 70), (520, 190, 90), (470, 180, 60)], "#ecd0a8", "#d9b389"),
    (-420, 0.665, [(620, 230, 95), (600, 220, 115), (560, 210, 85)], "#e3bd8c", "#ca9b6d"),
    (-120, 0.745, [(700, 260, 130), (640, 250, 110)], "#d7a873", "#b78152"),
    (-500, 0.840, [(760, 290, 160), (720, 280, 150)], "#c7915d", "#a0683c"),
    (-200, 0.945, [(820, 320, 190), (700, 300, 150)], "#b47a4a", "#87522e"),
]
for i, (x0, base, dunes, lit, shade) in enumerate(layers):
    poly, sh = dune_layer(x0, base * H, dunes)
    body = shapes([poly])
    col = mix(np.broadcast_to(hex2lin(lit), (H, W, 3)), hex2lin(shade), sh * body)
    s.add(body, col, shadow=0.35, off=10 + 3 * i, soft=12 + 3 * i, rim=0.10)

s.finish("../wallpapers/17-dunes.png", seed=17)
