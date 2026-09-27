"""Albiceleste — Messi's 10 as a frosted glass pane over sky-blue and white stripes."""
import numpy as np
from common import *

x, y = grid()

# --- crisp stripes ----------------------------------------------------------------
celeste, white = hex2lin("#74acdf"), hex2lin("#f4f6f8")
sw = 200.0  # stripe width; the centre stripe is white
k = np.floor((x - W / 2) / sw + 0.5)
stripes = np.where((np.abs(k) % 2 == 1)[..., None], celeste, white)
stripes = gblur(stripes, 1.0)
# light falls from the top, the cloth dims a little toward the bottom
scene = stripes * (1.0 - 0.10 * smoothstep(0.2 * H, H, y))[..., None]

ink = hex2lin("#16213a")
gold = hex2lin("#c9a24c")
cy = H * 0.575

ten = text_mask("10", "BigShoulders-Bold.ttf", 1100, W / 2, cy)

# soft shadow the pane casts on the cloth
shadow = gblur(np.roll(ten, (30, 12), (0, 1)), 30)
img = scene * (1 - 0.30 * shadow * (1 - ten))[..., None]

# --- frosted glass inside the number -----------------------------------------------
glass = gblur(scene, 40)
glass = mix(glass, hex2lin("#f7fafd"), 0.30) * hex2lin("#f2f6fb")
rng = np.random.default_rng(9)
frost = gblur(rng.normal(0, 1, (H, W)).astype(np.float32), 0.7)
glass = glass * (1 + 0.035 * frost)[..., None]
# thickness: a soft glow just inside the edges
inner = np.clip(ten - gblur(ten, 14), 0, 1)
glass = glass + (inner * 0.12)[..., None]
img = over(img, glass, ten)

# edge lighting: light comes from the upper-left
g = gblur(ten, 1.6)
gy, gx = np.gradient(g)
mag = np.hypot(gx, gy) + 1e-6
nx, ny = -gx / mag, -gy / mag            # outward normal
facing = np.clip(nx * -0.6 + ny * -0.8, 0, 1)
edge = np.clip(mag * 3.2, 0, 1) * ten
away = np.clip(-(nx * -0.6 + ny * -0.8), 0, 1)
# the cut edge refracts: a thin cool line all round, deeper on the shaded side
img = over(img, hex2lin("#4f6f98"), edge * (0.22 + 0.25 * away))
# and a bright bevel just inside it on the lit side
bevel = np.clip((ten - gblur(ten, 3.5)) * 2.2, 0, 1)
img = over(img, hex2lin("#ffffff"), bevel * (0.2 + 0.8 * gblur(facing * edge, 3) / (gblur(edge, 3) + 1e-3)) * 0.9)

# --- name and the three stars --------------------------------------------------------
name_y = cy - 450 - 110
img = over(img, ink, text_mask("MESSI", "Outfit-Regular.ttf", 66, W / 2, name_y, tracking=34))
for i in (-1, 0, 1):  # 1978 · 1986 · 2022
    img = over(img, gold, star_mask(W / 2 + i * 74, name_y - 120, 24))

save(img, "../wallpapers/09-albiceleste.png", grain=0.006, seed=9)
