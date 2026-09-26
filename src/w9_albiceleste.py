"""Albiceleste — Messi's 10 on frosted sky-blue and white stripes, three stars."""
import numpy as np
from common import *

x, y = grid()

# --- stripes, seen through frosted glass ------------------------------------------
celeste, white = hex2lin("#74acdf"), hex2lin("#f4f6f8")
sw = 262.0  # stripe width; the centre stripe is white
k = np.floor((x - W / 2) / sw + 0.5)
stripes = np.where((np.abs(k) % 2 == 1)[..., None], celeste, white)
stripes = gblur(stripes, 34)
# light falls from the top, the cloth dims a little toward the bottom
img = stripes * (1.0 - 0.10 * smoothstep(0.2 * H, H, y))[..., None]
rng = np.random.default_rng(9)
img = img * (1 + 0.02 * gblur(rng.normal(0, 1, (H, W)).astype(np.float32), 0.7))[..., None]

ink = hex2lin("#16213a")
gold = hex2lin("#c9a24c")
cy = H * 0.575

# the number, with a soft shadow so it sits on the glass
ten = text_mask("10", "BigShoulders-Bold.ttf", 1100, W / 2, cy)
shadow = gblur(np.roll(ten, (22, 0), (0, 1)), 26)
img = img * (1 - 0.16 * shadow)[..., None]
img = over(img, ink, ten)

# name above, tracked wide
name_y = cy - 450 - 110
img = over(img, ink, text_mask("MESSI", "Outfit-Regular.ttf", 66, W / 2, name_y, tracking=34))

# three stars: 1978 · 1986 · 2022
for i in (-1, 0, 1):
    img = over(img, gold, star_mask(W / 2 + i * 74, name_y - 120, 24))

# the night in Lusail, small, under the number
img = over(img, ink, text_mask("18 · 12 · 2022", "Outfit-Regular.ttf", 40, W / 2, cy + 450 + 110, tracking=14) * 0.6)

save(img, "../wallpapers/09-albiceleste.png", grain=0.006, seed=9)
