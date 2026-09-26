"""Fluted — an amber sun half-sunk behind reeded glass, dark."""
import numpy as np
from common import *

x, y = grid()

# --- scene behind the glass -------------------------------------------------
bg = vgradient(y, [(0, "#12161c"), (0.55, "#0c0f13"), (1, "#060709")])

cx, cy, r = W * 0.5, H * 0.585, 300
d = np.hypot(x - cx, y - cy)
# sun body: warm vertical gradient, hot core toward the top edge
t = np.clip((y - (cy - r)) / (2 * r), 0, 1)
sun = mix(np.broadcast_to(hex2lin("#f6c08a"), bg.shape), hex2lin("#c9502a"), t ** 0.9)
halo = np.exp(-np.maximum(d - r, 0) / 170)[..., None] * hex2lin("#b0461f") * 0.26
scene = bg + halo
scene = over(scene, sun, fill(d - r, 1.4))

# --- reeded glass over the lower part -----------------------------------------
yg = cy + 6                      # glass top edge, just under the sun's equator
fw = 64.0                        # flute width
k = np.floor(x / fw)
fc = (k + 0.5) * fw
u = (x - fc) / (fw / 2)          # -1..1 across each flute

frost = gblur(scene, 13)
sx = fc + fw * (1.35 * np.sin(u * np.pi / 2) + 0.35 * u ** 3)
through = sample(frost, sx, y + 18)          # slight downward offset: thickness
# cylindrical shading across each reed
shade = 0.9 + 0.1 * np.cos(u * np.pi / 2) ** 0.6
spec = np.exp(-((u + 0.5) / 0.22) ** 2) * 0.006
through = through * shade[..., None] + spec[..., None]
seam = np.clip(1.2 - np.abs(x - np.round(x / fw) * fw), 0, 1)[..., None]
through = through * (1 - 0.18 * seam)
through += hex2lin("#a9b4c2") * 0.006        # faint cool tint of the glass

glass = smoothstep(yg - 0.8, yg + 0.8, y)
img = over(scene, through, glass)

# top edge of the glass pane catches a little light
edge = stroke(y - yg, 1.2) * (0.25 + 0.75 * np.exp(-((x - cx) / 520) ** 2))
img = over(img, hex2lin("#e8dccb"), edge * 0.55)

save(img, "../wallpapers/01-fluted.png", grain=0.007, seed=1)
