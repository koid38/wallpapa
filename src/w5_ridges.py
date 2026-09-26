"""Ridges — layered sage hills with mist in the valleys and a pale sun."""
import numpy as np
from common import *

x, y = grid()

img = vgradient(y, [(0, "#dde2d8"), (0.45, "#e6e8de"), (1, "#dfe3d8")])

# pale sun with a faint halo
sx, sy, sr = W * 0.64, H * 0.522, 120
sd = np.hypot(x - sx, y - sy)
img = img + hex2lin("#f3e6cc") * (np.exp(-np.maximum(sd - sr, 0) / 240) * 0.12)[..., None]
img = over(img, hex2lin("#f6ead2"), fill(sd - sr, 1.3))

# ridge layers, far -> near
layers = [  # (crest height, amplitude, color, mist amount)
    (0.535, 30, "#c5cdbb", 0.55),
    (0.600, 42, "#a9b49e", 0.50),
    (0.670, 54, "#87957d", 0.45),
    (0.750, 62, "#627260", 0.38),
    (0.835, 68, "#435142", 0.28),
    (0.925, 60, "#2c372d", 0.18),
]
rng = np.random.default_rng(5)
xs = x[0]
for base, amp, col, haze in layers:
    p = rng.uniform(0, 2 * np.pi, 3)
    f = rng.uniform(0.85, 1.25, 3)
    ridge = (base * H
             + amp * np.sin(xs / W * 2 * np.pi * 0.7 * f[0] + p[0])
             + amp * 0.55 * np.sin(xs / W * 2 * np.pi * 1.6 * f[1] + p[1])
             + amp * 0.18 * np.sin(xs / W * 2 * np.pi * 3.9 * f[2] + p[2]))
    ridge = ridge[None, :]
    c = hex2lin(col)
    # darkest at the crest, dissolving into mist toward the valley below
    depth = np.clip((y - ridge) / 260, 0, 1)
    mist = hex2lin("#e6e9e0")
    body = mix(np.broadcast_to(c, img.shape), mist, depth ** 1.4 * haze)
    img = over(img, body, fill(ridge - y, 1.1))

save(img, "../wallpapers/05-ridges.png", grain=0.006, seed=5)
