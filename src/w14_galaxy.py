"""Galaxy — a tilted spiral made only of blur, over a field of crisp stars. Deep blue."""
import numpy as np
from common import *

x, y = grid()
rng = np.random.default_rng(14)

cx, cy = W * 0.5, H * 0.565
tilt, squash = np.deg2rad(-28), 0.40
dx, dy = x - cx, y - cy
gu = dx * np.cos(tilt) + dy * np.sin(tilt)
gv = (-dx * np.sin(tilt) + dy * np.cos(tilt)) / squash
r = np.hypot(gu, gv)
th = np.arctan2(gv, gu)

# two logarithmic arms
pitch = np.deg2rad(26)
arms = np.zeros_like(r)
for k in range(2):
    phase = th - k * np.pi - np.log(np.maximum(r, 1) / 60) / np.tan(pitch)
    d = np.angle(np.exp(1j * phase))            # wrapped angular distance to the arm
    width = 0.42 + 0.0004 * r
    arms += np.exp(-(d / width) ** 2)
clumps = fbm([(22, 1.0), (9, 0.5)], seed=14)
arms = arms * (1 + 0.28 * clumps) * np.exp(-r / 290) * smoothstep(50, 170, r) * (1 - smoothstep(620, 900, r))
disk = np.exp(-r / 250) * 0.30
core = np.exp(-(np.hypot(dx, dy) / 20) ** 2) * 0.9 + np.exp(-np.hypot(gu, gv * squash / 0.6) / 70) * 0.55
lum = gblur(arms * 0.8 + disk, 7.0) + core
# dust lane: a dark thread riding the inner edge of each arm
dust = np.zeros_like(r)
for k in range(2):
    phase = th - k * np.pi - 0.28 - np.log(np.maximum(r, 1) / 60) / np.tan(pitch)
    dust += np.exp(-(np.angle(np.exp(1j * phase)) / 0.10) ** 2)
lum = lum * (1 - 0.35 * gblur(dust * smoothstep(80, 200, r) * np.exp(-r / 500), 4.0))
bloom = gblur(lum, 45) * 0.45 + gblur(lum, 160) * 0.25
lum = lum + bloom

# one hue: deep navy -> steel blue -> ice white
deep, mid, ice = hex2lin("#0c1a3a"), hex2lin("#4f78c0"), hex2lin("#eef4ff")
t = np.clip(lum, 0, None)
col = mix(mix(np.broadcast_to(deep, (H, W, 3)), mid, np.clip(t * 1.8, 0, 1)), ice, smoothstep(0.5, 1.6, t))
img = vgradient(y, [(0, "#05070d"), (0.6, "#04060b"), (1, "#030408")])
img = img + col * (1 - np.exp(-t * 1.4))[..., None] * 0.85

# crisp stars, thinning toward the top where the clock sits
n = 380
sx, sy = rng.uniform(0, W, n), rng.uniform(0, H, n)
keep = rng.random(n) < (0.35 + 0.65 * sy / H)
sb = rng.pareto(2.2, n) * 0.08 + 0.04
star = np.zeros((H, W), np.float32)
for i in np.nonzero(keep)[0]:
    y0, y1 = int(max(sy[i] - 8, 0)), int(min(sy[i] + 8, H))
    x0, x1 = int(max(sx[i] - 8, 0)), int(min(sx[i] + 8, W))
    d = np.hypot(x[y0:y1, x0:x1] - sx[i], y[y0:y1, x0:x1] - sy[i])
    star[y0:y1, x0:x1] += np.exp(-(d / (0.9 + 1.5 * min(sb[i], 0.4))) ** 2) * min(sb[i], 1.2)
img = img + star[..., None] * hex2lin("#dfe8ff")

save(img, "../wallpapers/14-galaxy.png", grain=0.008, seed=14)
