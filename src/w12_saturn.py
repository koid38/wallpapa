"""Saturn — a sand-toned planet wearing rings of frosted glass. Monochrome, light."""
import numpy as np
from common import *

x, y = grid()

bg = vgradient(y, [(0, "#e9e3d9"), (0.6, "#e3dcd0"), (1, "#d9d1c4")])

cx, cy, R = W * 0.5, H * 0.565, 290
dx, dy = x - cx, y - cy
r = np.hypot(dx, dy)

# ring frame: tilted on screen by phi, seen from elev above the ring plane
phi, elev = np.deg2rad(-12), np.deg2rad(24)
u = dx * np.cos(phi) + dy * np.sin(phi)
v = -dx * np.sin(phi) + dy * np.cos(phi)
rho = np.hypot(u, v / np.sin(elev))  # radius within the ring plane
near = v > 0                          # the half that passes in front of the planet

# --- planet --------------------------------------------------------------------------
nz = np.sqrt(np.clip(1 - (r / R) ** 2, 0, None))
nx, ny = dx / R, dy / R
L = np.array([-0.55, -0.42, 0.72])
L /= np.linalg.norm(L)
lam = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
A = np.array([np.cos(elev) * np.sin(phi), -np.cos(elev) * np.cos(phi), np.sin(elev)])
lat = nx * A[0] + ny * A[1] + nz * A[2]
bands = (1 + 0.05 * np.sin(lat * 19) + 0.035 * np.sin(lat * 43 + 1.3)
         - 0.07 * np.exp(-((lat - 0.28) / 0.06) ** 2) - 0.05 * np.exp(-((lat + 0.12) / 0.05) ** 2))
body = hex2lin("#c8b89e") * ((0.2 + 0.9 * lam ** 0.9) * bands)[..., None]
# limb darkening
body = body * (0.72 + 0.28 * nz ** 0.5)[..., None]
planet = fill(r - R, 1.2)

# --- rings, as frosted glass -----------------------------------------------------
edges = [1.28, 1.70, 1.76, 2.10]  # B ring | Cassini gap | A ring
in_b = fill(np.maximum(edges[0] * R - rho, rho - edges[1] * R) * np.sin(elev) * 1.6, 1.0)
in_a = fill(np.maximum(edges[2] * R - rho, rho - edges[3] * R) * np.sin(elev) * 1.6, 1.0)
ring = np.clip(in_b + in_a, 0, 1)
density = in_b * 1.0 + in_a * 0.8
# fine concentric grooves inside the glass
grooves = 1 + 0.02 * np.sin(rho / R * 55)

rng = np.random.default_rng(12)
frost = gblur(rng.normal(0, 1, (H, W)).astype(np.float32), 0.7)


def glass(backdrop, sigma, lift):
    g = gblur(backdrop, sigma)
    g = mix(g, hex2lin("#fdfaf5"), lift * density)
    return g * (grooves * (1 + 0.03 * frost))[..., None]


def rims(img, mask_half):
    for i, e in enumerate(edges):
        d = (rho - e * R) * np.sin(elev) * 1.6
        out = 1 if i % 2 else -1  # which side of the line is outside the glass
        # a cool refracted line on the outside, a bright bevel on the inside
        img = over(img, hex2lin("#8c7f6b"), stroke(d - out * 1.2, 1.0) * mask_half * 0.28)
        img = over(img, hex2lin("#fffdf8"), stroke(d + out * 0.6, 1.2) * mask_half * 0.85)
    return img


# far half of the rings, behind the planet
img = bg.copy()
far = ring * (~near)
img = over(img, glass(bg, 22, 0.24), far)
img = rims(img, (~near).astype(np.float32) * (1 - planet))
# the planet
img = over(img, body, planet)
# near half, in front: the planet shows through, softened
img = over(img, glass(img, 22, 0.24), ring * near)
img = rims(img, near.astype(np.float32))
# where the glass crosses the planet's face, it throws a faint shadow just below
shadow = gblur(ring * near, 10)
shadow = np.roll(shadow, 26, 0) * planet * (1 - ring * near)
img = img * (1 - 0.18 * shadow)[..., None]

save(img, "../wallpapers/12-saturn.png", grain=0.006, seed=12)
