"""Clear — a frosted pane over a cobalt disc, with one round window wiped clean."""
import numpy as np
from common import *

x, y = grid()

# --- scene behind the pane -------------------------------------------------------
bg = vgradient(y, [(0, "#e2e6ea"), (0.6, "#d8dee3"), (1, "#ccd3d9")])

dcx, dcy, dr = W * 0.64, H * 0.60, 430
t = np.clip(((x - dcx) * 0.5 + (y - dcy) * 0.87) / (2 * dr) + 0.5, 0, 1)
disc_col = mix(np.broadcast_to(hex2lin("#4a68b8"), bg.shape), hex2lin("#233a78"), t)
scene = over(bg, disc_col, fill(np.hypot(x - dcx, y - dcy) - dr, 1.2))

# --- frosted pane everywhere ---------------------------------------------------
frost = gblur(scene, 70)
frost = mix(frost, hex2lin("#f4f6f8"), 0.16)
rng = np.random.default_rng(11)
fr = gblur(rng.normal(0, 1, (H, W)).astype(np.float32), 0.7)
frost = frost * (1 + 0.03 * fr)[..., None]

# --- the wiped window ------------------------------------------------------------
wcx, wcy, wr = W * 0.38, H * 0.53, 300
w_sdf = np.hypot(x - wcx, y - wcy) - wr
clear = fill(w_sdf, 1.2)
img = over(frost, scene, clear)

# pane thickness: a faint shadow just inside the cut, light catching the lower edge
inner = np.exp(np.minimum(w_sdf, 0) / 14) * clear
img = img * (1 - 0.10 * inner)[..., None]
ang = np.arctan2(y - wcy, x - wcx)
lit = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(ang + np.pi / 4)) ** 2
img = over(img, hex2lin("#ffffff"), stroke(w_sdf - 0.6, 1.3) * lit * 0.8)

save(img, "../wallpapers/03-clear.png", grain=0.006, seed=3)
