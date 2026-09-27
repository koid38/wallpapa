"""Pines — a paper-cut forest at night: tiers of firs, drifting mist, a pale moon."""
import numpy as np
from common import *
from paper import *

x, y = grid()
rng = np.random.default_rng(15)

sky = vgradient(y, [(0, "#16232c"), (0.5, "#24383f"), (0.75, "#2f4548"), (1, "#2f4548")])
s = Sheet(sky)

# moon, with two faint paper halos
mx, my = W * 0.68, H * 0.365
s.add(shapes(circles=[(mx, my, 330)]), hex2lin("#3a5057"), shadow=0.18, off=8, soft=18, rim=0.03, alpha=0.55)
s.add(shapes(circles=[(mx, my, 225)]), hex2lin("#4a6066"), shadow=0.18, off=8, soft=18, rim=0.03, alpha=0.55)
s.add(shapes(circles=[(mx, my, 140)]), hex2lin("#efe6d2"), shadow=0.30, off=10, soft=16, rim=0)


def hill(base, amp, seed):
    p = np.random.default_rng(seed).uniform(0, 2 * np.pi, 3)
    return lambda xs: (base * H + amp * np.sin(xs / W * 2 * np.pi * 0.6 + p[0])
                       + amp * 0.5 * np.sin(xs / W * 2 * np.pi * 1.5 + p[1]))


def forest(base, amp, seed, n, hmin, hmax, wr=0.42):
    """A hill with a row of pines standing on it."""
    f = hill(base, amp, seed)
    r = np.random.default_rng(seed + 7)
    xs = np.sort(r.uniform(-60, W + 60, n))
    trees = []
    for tx in xs:
        h = r.uniform(hmin, hmax)
        trees.append(pine(tx, float(f(np.array([tx]))[0]) + 6, h, h * wr * r.uniform(0.85, 1.15), tiers=r.integers(3, 6)))
    return shapes([line_poly(f)] + trees)


layers = [
    # base, amp, seed, trees, hmin, hmax, colour — far layers paler, like air
    (0.580, 20, 1, 70, 60, 120, "#41585c"),
    (0.635, 28, 2, 52, 100, 190, "#304649"),
    (0.705, 34, 3, 36, 170, 300, "#213438"),
    (0.795, 40, 4, 20, 280, 460, "#142226"),
    (0.900, 30, 5, 9, 420, 640, "#0c1518"),
]

for i, (base, amp, seed, n, hmin, hmax, col) in enumerate(layers):
    s.add(forest(base, amp, seed, n, hmin, hmax), hex2lin(col), shadow=0.45, off=10 + 3 * i, soft=12 + 4 * i, rim=0.05)

s.finish("../wallpapers/15-pines.png", seed=15)
