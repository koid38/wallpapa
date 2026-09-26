"""Frost — terracotta disc behind a frosted glass capsule, warm paper."""
import numpy as np
from common import *

x, y = grid()

# --- scene --------------------------------------------------------------------
bg = vgradient(y, [(0, "#ebe6dd"), (0.6, "#e5dfd5"), (1, "#ddd6ca")])

dcx, dcy, dr = W * 0.40, H * 0.50, 330
disc_d = np.hypot(x - dcx, y - dcy)
t = np.clip((y - (dcy - dr)) / (2 * dr), 0, 1)
disc_col = mix(np.broadcast_to(hex2lin("#d27a55"), bg.shape), hex2lin("#b4553a"), t)
scene = over(bg, disc_col, fill(disc_d - dr, 1.2))

# hairline with a dot — the motif from the earlier set; it dissolves in the glass
ly = H * 0.735
ink = hex2lin("#2e3135")
scene = over(scene, ink, stroke(y - ly, 1.4) * 0.85)
scene = over(scene, ink, fill(np.hypot(x - W * 0.20, y - ly) - 17, 1.2))

# --- glass capsule -------------------------------------------------------------
gcx, gcy, ghw, ghh = W * 0.63, H * 0.60, 270, 640
g_sdf = sdf_round_rect(x, y, gcx, gcy, ghw, ghh, ghw)
g_in = fill(g_sdf, 1.2)

# soft contact shadow cast by the pane onto the paper
sh = gblur(fill(sdf_round_rect(x, y, gcx + 10, gcy + 46, ghw - 10, ghh - 10, ghw - 10), 1), 55)
img = scene * (1 - 0.10 * sh * (1 - g_in))[..., None]

# frosted backdrop: heavy blur, lifted toward white, light falling from above
back = gblur(scene, 46)
lift = 0.13 + 0.06 * (1 - (y - (gcy - ghh)) / (2 * ghh)).clip(0, 1)
glass = mix(back, hex2lin("#fbf8f3"), lift)
# inner soft edge glow (thickness of the glass)
inner = np.exp(np.minimum(g_sdf, 0) / 26)
glass = glass + (inner * 0.05)[..., None]
img = over(img, glass, g_in)

# rim: bright on the upper-left, fading around
ang = np.arctan2(y - gcy, x - gcx)
rim_k = 0.35 + 0.65 * (0.5 - 0.5 * np.sin(ang + np.pi / 4)) ** 1.5
img = over(img, hex2lin("#ffffff"), stroke(g_sdf + 1.0, 1.4) * rim_k * 0.85)
img = over(img, hex2lin("#8a7f72"), stroke(g_sdf - 0.6, 1.0) * (1 - rim_k) * 0.18)

# grain inside the glass reads as frosting
rng = np.random.default_rng(7)
fr = gblur(rng.normal(0, 1, (H, W)).astype(np.float32), 0.7)
img = img * (1 + 0.035 * fr * g_in)[..., None]

save(img, "../wallpapers/02-frost.png", grain=0.006, seed=2)
