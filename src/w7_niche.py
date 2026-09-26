"""Niche — an arched recess in a plaster wall, crossed by a band of late sun."""
import numpy as np
from common import *

x, y = grid()

wall = vgradient(y, [(0, "#e0cdb6"), (0.6, "#d9c4ab"), (1, "#cfb89e")])
plaster = fbm([(60, 1.0), (8, 0.5), (1.5, 0.35)], seed=7)
wall = wall * (1 + 0.018 * plaster)[..., None]

# arch opening: a rectangle capped by a half circle
ncx, hw = W * 0.5, 250
top, bot = H * 0.40, H * 0.80  # spring line of the arch, sill


def arch_sdf(px, py):
    rect = sdf_round_rect(px, py, ncx, (top + bot) / 2, hw, (bot - top) / 2, 0)
    return np.minimum(rect, np.hypot(px - ncx, py - top) - hw)


a_sdf = arch_sdf(x, y)
opening = fill(a_sdf, 1.2)

# sunlight: a slanted band, the light travelling down and to the right
ang = np.deg2rad(62)
dx, dy = np.cos(ang), np.sin(ang)
band_c = x * dy - y * dx  # coordinate across the band
c0 = W * 0.5 * dy - H * 0.56 * dx
band = smoothstep(-262, -250, band_c - c0) * (1 - smoothstep(250, 262, band_c - c0))

# on the back of the niche the band lands shifted by the recess depth,
# and only where the ray made it through the opening
depth = 120
ox, oy = depth * dx / dy * 0.9, depth * 0.9
bx, by = x - ox, y - oy
bc = bx * dy - by * dx - c0
band_back = smoothstep(-266, -248, bc) * (1 - smoothstep(248, 266, bc))
through = fill(arch_sdf(bx, by), 5.0)
lit_back = gblur(band_back * through, 2.0)

sun = hex2lin("#ffe7c7")
front = wall * (0.82 + 0.34 * band[..., None] * sun)

# back of the recess: dimmer, occluded toward its rim
inside_d = -np.minimum(a_sdf, 0)
ao = 1 - 0.22 * np.exp(-inside_d / 40)
back = wall * (0.70 * ao)[..., None] * hex2lin("#f6ece2")
back = back * (1 + 0.40 * lit_back[..., None] * sun)
img = over(front, back, opening)

# thin highlight where the sun catches the arch's lower-right lip
lip = stroke(a_sdf + 1.0, 1.6) * band * smoothstep(-0.2, 0.6, (x - ncx) / hw)
img = over(img, hex2lin("#fff4e6"), lip * 0.5)

save(img, "../wallpapers/07-niche.png", grain=0.006, seed=7)
