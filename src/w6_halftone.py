"""Halftone — a moon printed as a rotated dot screen, cream on charcoal."""
import numpy as np
from common import *

x, y = grid()

img = vgradient(y, [(0, "#131519"), (0.6, "#0f1114"), (1, "#0a0b0d")])

cx, cy, R = W * 0.5, H * 0.575, 440
step = 24.0
s2 = np.sqrt(0.5)

# nearest cell center on a 45° grid
u, v = (x + y) * s2, (x - y) * s2
uc, vc = np.round(u / step) * step, np.round(v / step) * step
px, py = (uc + vc) * s2, (uc - vc) * s2

# lambert shading of the sphere, evaluated at each cell center
nx, ny = (px - cx) / R, (py - cy) / R
r2 = nx * nx + ny * ny
nz = np.sqrt(np.clip(1 - r2, 0, None))
L = np.array([-0.62, -0.38, 0.69])
L = L / np.linalg.norm(L)
shade = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
# maria: large soft dark patches, fixed to the surface
maria = fbm([(55, 1.0), (22, 0.45)], seed=6)
mcell = sample(maria[..., None], np.clip(px, 0, W - 1), np.clip(py, 0, H - 1))[..., 0]
shade = shade * (1 - 0.32 * smoothstep(0.2, 1.4, mcell)) * (r2 < 1)
shade = shade ** 0.85

rad = np.sqrt(shade) * step * 0.5
dot = fill(np.hypot(x - px, y - py) - rad, 1.2) * (rad > 0.6)
img = over(img, hex2lin("#ece4d2"), dot)

# the dark side still shows as a faint disc
img = img + (fill(np.hypot(x - cx, y - cy) - R, 1.5) * 0.0025)[..., None]

save(img, "../wallpapers/06-halftone.png", grain=0.008, seed=6)
