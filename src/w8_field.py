"""Field — two soft colour fields on a deep ground, after Rothko's rust and blue."""
import numpy as np
from common import *

x, y = grid()

ground = vgradient(y, [(0, "#1c2436"), (1, "#141a28")])
mottle = fbm([(180, 1.0), (40, 0.5), (6, 0.25)], seed=8)
img = ground * (1 + 0.05 * mottle)[..., None]


def field(x0, y0, x1, y1, col, feather, seed, wob=14):
    """A rectangle with breathing, hazy edges."""
    n = fbm([(140, 1.0), (45, 0.3)], seed=seed)
    sdf = sdf_round_rect(x, y, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, 40)
    a = smoothstep(feather, -feather, sdf + wob * n)
    body = fbm([(140, 1.0), (30, 0.6), (4, 0.3)], seed=seed + 50)
    glow = smoothstep(0, -220, sdf)  # a touch more luminous toward the middle
    c = hex2lin(col)[None, None, :] * ((1 + 0.06 * body) * (0.93 + 0.1 * glow))[..., None]
    return c, a


m = W * 0.09
c, a = field(m, H * 0.20, W - m, H * 0.54, "#9c4128", 40, 81)
img = over(img, c, a * 0.96)
c, a = field(m, H * 0.585, W - m, H * 0.86, "#2d476c", 46, 82)
img = over(img, c, a * 0.94)

save(img, "../wallpapers/08-field.png", grain=0.012, seed=8)
