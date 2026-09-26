"""Aura — a defocused cool glow held by one crisp hairline ring, near-black."""
import numpy as np
from common import *

x, y = grid()

bg = vgradient(y, [(0, "#0d1016"), (0.5, "#090b0f"), (1, "#050608")])

cx, cy, R = W * 0.5, H * 0.585, 380

# the light source: a gradient disc, thrown far out of focus
lx, ly, lr = cx + 60, cy + 150, 250
ld = np.hypot(x - lx, y - ly)
t = np.clip(((x - lx) * -0.5 + (y - ly) * 0.87) / (2 * lr) + 0.5, 0, 1)
col = mix(np.broadcast_to(hex2lin("#6d5ce8"), bg.shape), hex2lin("#1fb5a8"), smoothstep(0.15, 0.95, t))
src = col * fill(ld - lr, 1)[..., None]
# warm spark at the core
src = src + hex2lin("#ff8a6a") * (fill(np.hypot(x - lx - 40, y - ly - 10) - 70, 1) * 0.9)[..., None]
glow = gblur(src, 95) * 0.8 + gblur(src, 260) * 0.3

img = bg + glow

# crisp hairline ring: dim where it's in the dark, catching light near the glow
lum = glow.mean(-1)
ring_d = np.hypot(x - cx, y - cy) - R
lit = np.clip(0.10 + lum * 2.2, 0, 0.9)
img = over(img, hex2lin("#eef1f6"), stroke(ring_d, 1.3) * lit)

# a small bright body sitting on the ring, upper-left, like the earlier set
a = np.deg2rad(-128)
px, py = cx + R * np.cos(a), cy + R * np.sin(a)
pd = np.hypot(x - px, y - py)
img = img + hex2lin("#dfe6f2") * (np.exp(-np.maximum(pd - 9, 0) / 16) * 0.06)[..., None]
img = over(img, hex2lin("#f2f4f8"), fill(pd - 9, 1.2))

save(img, "../wallpapers/04-aura.png", grain=0.011, seed=4)
