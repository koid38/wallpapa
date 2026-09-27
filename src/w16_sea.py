"""Sea — paper-cut waves under a coral sun, two clouds and a small sailboat."""
import numpy as np
from common import *
from paper import *

x, y = grid()

sky = vgradient(y, [(0, "#efe5d6"), (0.4, "#f1e4d0"), (0.52, "#efd9bf"), (1, "#efd9bf")])
s = Sheet(sky)

hz = H * 0.53  # horizon

# sun, sitting just above the horizon, and its broken reflection on the water
sx, sy, sr = W * 0.36, H * 0.475, 175
s.add(shapes(circles=[(sx, sy, sr)]), hex2lin("#e48462"), shadow=0.22, off=10, soft=16, rim=0.06)


def cloud(cx, cy, w):
    """Puffs on a flat base."""
    r = w / 5
    circles = [(cx - w * 0.28, cy, r * 1.05), (cx - w * 0.05, cy - r * 0.7, r * 1.45),
               (cx + w * 0.22, cy - r * 0.2, r * 1.15), (cx + w * 0.38, cy + r * 0.25, r * 0.8)]
    base = [(cx - w * 0.28, cy), (cx + w * 0.38, cy + r * 0.25), (cx + w * 0.38, cy + r * 1.05), (cx - w * 0.28, cy + r * 1.05)]
    m = shapes([base], circles)
    return m * (y < cy + r * 1.05)


s.add(cloud(W * 0.70, H * 0.33, 420), hex2lin("#fbf6ee"), shadow=0.22, off=12, soft=16, rim=0.0)
s.add(cloud(W * 0.18, H * 0.255, 280), hex2lin("#fbf6ee"), shadow=0.20, off=10, soft=14, rim=0.0)


def waves(base, amp, k, phase, sharp=1.6):
    """Trochoid-ish: sharp crests, round troughs."""
    return lambda xs: base - amp * (1 - np.abs(np.sin(xs / W * np.pi * k + phase))) ** sharp


# the flat far sea, with strips of reflected sun
s.add(shapes([line_poly(lambda xs: hz + 0 * xs)]), hex2lin("#a7c9c3"), shadow=0.25, off=8, soft=12, rim=0.05)
strips = []
for i, (w, dy) in enumerate([(300, 26), (230, 58), (170, 94), (120, 134), (80, 178)]):
    yy = hz + dy
    strips.append([(sx - w / 2, yy), (sx + w / 2, yy), (sx + w / 2 - 8, yy + 11), (sx - w / 2 + 8, yy + 11)])
s.add(shapes(strips), hex2lin("#f0b596"), shadow=0.18, off=4, soft=5, rim=0.0)

layers = [
    # base, amp, crests across, phase, colour
    (0.585, 22, 7, 0.3, "#8dbab6"),
    (0.640, 30, 6, 1.1, "#72a8a6"),
    (0.705, 40, 5, 2.0, "#56928f"),
    (0.785, 52, 4, 0.6, "#3e7a7c"),
    (0.880, 64, 3.5, 1.7, "#2a6069"),
]
for i, (base, amp, k, ph, col) in enumerate(layers):
    if i == 2:
        # a small sailboat riding between the second and third swell
        bx, by = W * 0.66, H * 0.705 - 44
        hull = [(bx - 70, by), (bx + 78, by), (bx + 52, by + 30), (bx - 50, by + 30)]
        s.add(shapes([hull]), hex2lin("#5b3d33"), shadow=0.3, off=6, soft=6, rim=0.08)
        main = [(bx + 4, by - 8), (bx + 4, by - 190), (bx + 66, by - 8)]
        jib = [(bx - 6, by - 8), (bx - 6, by - 160), (bx - 60, by - 8)]
        s.add(shapes([main, jib]), hex2lin("#fbf6ee"), shadow=0.25, off=6, soft=7, rim=0.0)
    s.add(shapes([line_poly(waves(base * H, amp, k, ph))]), hex2lin(col), shadow=0.40, off=10 + 3 * i, soft=12 + 3 * i, rim=0.10)

s.finish("../wallpapers/16-sea.png", seed=16)
