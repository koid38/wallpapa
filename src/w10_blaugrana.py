"""Blaugrana — Messi's 10 cut from blue and garnet stripes, glowing on night blue."""
import numpy as np
from common import *

x, y = grid()

blau, grana = hex2lin("#004d98"), hex2lin("#a50044")
gold = hex2lin("#edbb00")

bg = vgradient(y, [(0, "#0c0f1a"), (0.55, "#0a0c15"), (1, "#06070c")])
cy = H * 0.575

# stripes, fixed to the page so the number reads as a cut-out window
sw = 92.0
k = np.floor((x - W / 2) / sw + 0.5)
stripes = np.where((k % 2 == 0)[..., None], blau, grana)
# soft cloth shading across each stripe
u = ((x - W / 2) / sw + 0.5) % 1
stripes = stripes * (0.86 + 0.14 * np.sin(u * np.pi))[..., None]

ten = text_mask("10", "BigShoulders-Bold.ttf", 1100, W / 2, cy)

# the stripes behind the number, thrown far out of focus, as its glow
glow = gblur(stripes * ten[..., None], 110) * 0.6 + gblur(stripes * ten[..., None], 320) * 0.45
img = bg + glow

# the number itself, a touch brighter toward the top
lit = stripes * (1 + 0.25 * smoothstep(cy + 450, cy - 450, y))[..., None]
img = over(img, lit, ten)

name_y = cy - 450 - 110
img = over(img, gold, text_mask("MESSI", "Outfit-Regular.ttf", 66, W / 2, name_y, tracking=34))
img = over(img, hex2lin("#cfd3dc"), text_mask("2004 — 2021", "Outfit-Regular.ttf", 40, W / 2, cy + 450 + 110, tracking=14) * 0.55)

save(img, "../wallpapers/10-blaugrana.png", grain=0.009, seed=10)
