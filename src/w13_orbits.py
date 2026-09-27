"""Orbits — hairline orbits seen from above, planets as frosted glass discs. Blue-grey."""
import numpy as np
from common import *

x, y = grid()

bg = vgradient(y, [(0, "#e4e9ee"), (0.6, "#dde3ea"), (1, "#d3dae2")])
ink = hex2lin("#2f3d52")

cx, cy = W * 0.5, H * 0.60
dx, dy = x - cx, y - cy
r = np.hypot(dx, dy)

# the scene under the glass: orbits and the sun
scene = bg.copy()
orbits = [150, 250, 370, 510, 670, 850]
for i, o in enumerate(orbits):
    scene = over(scene, ink, stroke(r - o, 1.3) * (0.42 - 0.03 * i))
scene = over(scene, ink, fill(r - 34, 1.2))  # the sun, a solid ink dot

# planets: (orbit index, angle in degrees, radius)
planets = [(1, -35, 42), (3, 148, 96), (4, 38, 64), (5, -112, 130)]

img = scene.copy()
rng = np.random.default_rng(13)
frost = gblur(rng.normal(0, 1, (H, W)).astype(np.float32), 0.7)
back = gblur(scene, 1.8)

for oi, deg, pr in planets:
    a = np.deg2rad(deg)
    px, py = cx + orbits[oi] * np.cos(a), cy + orbits[oi] * np.sin(a)
    pd = np.hypot(x - px, y - py)
    # shadow thrown directly away from the sun
    sx, sy = np.cos(a), np.sin(a)
    off = 0.28 * pr
    sh = gblur(fill(np.hypot(x - px - sx * off, y - py - sy * off) - pr, 1), 0.35 * pr)
    disc = fill(pd - pr, 1.2)
    img = img * (1 - 0.16 * sh * (1 - disc))[..., None]
    # frosted body: blurred scene, lifted, lit from the sun's side
    lightness = np.clip(-((x - px) * sx + (y - py) * sy) / pr, -1, 1)
    # a thick disc magnifies a little what lies beneath it
    y0, y1 = int(max(py - pr - 4, 0)), int(min(py + pr + 4, H))
    x0, x1 = int(max(px - pr - 4, 0)), int(min(px + pr + 4, W))
    g = back.copy()
    g[y0:y1, x0:x1] = sample(back, px + (x[y0:y1, x0:x1] - px) * 0.78, py + (y[y0:y1, x0:x1] - py) * 0.78)
    g = mix(g, hex2lin("#f7f9fc"), 0.16 + 0.07 * lightness)
    g = g * (1 + 0.03 * frost)[..., None]
    img = over(img, g, disc)
    # rims: bright on the sun-facing side, a cool refracted line on the far side
    face = np.clip(-((x - px) * sx + (y - py) * sy) / np.maximum(pd, 1), 0, 1)
    img = over(img, hex2lin("#6b7a90"), stroke(pd - pr - 0.8, 1.0) * (0.15 + 0.2 * (1 - face)))
    img = over(img, hex2lin("#ffffff"), stroke(pd - pr + 1.0, 1.4) * (0.2 + 0.8 * face))

save(img, "../wallpapers/13-orbits.png", grain=0.006, seed=13)
