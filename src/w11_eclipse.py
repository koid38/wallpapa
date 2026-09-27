"""Eclipse — a black moon, a soft streaming corona and one diamond bead. Greyscale."""
import numpy as np
from common import *

x, y = grid()

img = vgradient(y, [(0, "#0b0c0e"), (0.6, "#08090a"), (1, "#050506")])

cx, cy, R = W * 0.5, H * 0.56, 300
dx, dy = x - cx, y - cy
r = np.hypot(dx, dy)
th = np.arctan2(dy, dx)
h = np.maximum(r - R, 0)

# streamers: smooth angular noise, stretched along the radius
rng = np.random.default_rng(11)
ang = np.zeros_like(th)
for k in range(2, 16):
    ang += rng.normal(0, 1) / k ** 0.8 * np.cos(k * th + rng.uniform(0, 2 * np.pi))
ang = ang / ang.std()
# the long equatorial streamers, like a solar-minimum corona
eq = np.exp(-((np.sin(th - 0.25)) / 0.32) ** 2)
stream = np.clip(1 + 0.45 * ang + 1.1 * eq, 0.15, None)

streak = np.clip(stream - 0.9, 0, None)
corona = (np.exp(-h / 20) * 0.9                                   # bright inner ring
          + np.exp(-h / (45 + 70 * stream)) * 0.30 * stream       # body
          + np.exp(-h / (140 + 320 * streak)) * 0.07 * streak)    # long faint streamers
corona = corona * (r > R - 2)
corona = gblur(corona, 4)
white = hex2lin("#e9edf2")
img = img + corona[..., None] * white * 0.5

# the moon, and the thin bright chromosphere hugging its edge
moon = fill(r - R, 1.2)
img = over(img, hex2lin("#030304"), moon)
img = img + (np.exp(-np.abs(r - R - 1.5) / 3.5) * (1 - moon) * 0.35)[..., None] * white

# diamond ring: one bead of photosphere breaking through, upper right
a = np.deg2rad(-38)
bx, by = cx + (R + 2) * np.cos(a), cy + (R + 2) * np.sin(a)
bd = np.hypot(x - bx, y - by)
bead = np.exp(-(bd / 9) ** 2) * 4.0 + np.exp(-bd / 34) * 0.55 + np.exp(-bd / 150) * 0.14
# a faint horizontal flare through the bead
bead += np.exp(-((y - by) / 2.2) ** 2) * np.exp(-np.abs(x - bx) / 240) * 0.10
img = img + (bead * (1 - moon * 0.97))[..., None] * hex2lin("#f6f8fb")

# a few faint stars
n = 70
sx, sy = rng.uniform(0, W, n), rng.uniform(0, H, n)
sb = rng.uniform(0.02, 0.18, n) ** 1.3
for i in range(n):
    if np.hypot(sx[i] - cx, sy[i] - cy) < R * 2.2:
        continue
    y0, y1 = int(max(sy[i] - 6, 0)), int(min(sy[i] + 6, H))
    x0, x1 = int(max(sx[i] - 6, 0)), int(min(sx[i] + 6, W))
    d = np.hypot(x[y0:y1, x0:x1] - sx[i], y[y0:y1, x0:x1] - sy[i])
    img[y0:y1, x0:x1] += (np.exp(-(d / 1.1) ** 2) * sb[i])[..., None]

save(img, "../wallpapers/11-eclipse.png", grain=0.008, seed=11)
