"""Lens — a clear glass sphere bending fine ruled lines, cool grey."""
import numpy as np
from common import *

x, y = grid()

# --- paper with ruled lines ------------------------------------------------------
pitch = 26.0
ink = hex2lin("#39414b")
accent = hex2lin("#e2562b")
cx, cy, R = W * 0.5, H * 0.585, 360
acc_y = cy + 0.12 * R  # one line in the stack is orange


def paper(px, py):
    """The flat scene at arbitrary coordinates (so the lens can sample it)."""
    ly = py - (np.round(py / pitch) * pitch)
    # lines fade in below the clock area and out toward the dock
    b = smoothstep(0.26 * H, 0.46 * H, py) * (1 - smoothstep(0.80 * H, 0.97 * H, py))
    a = stroke(ly, 1.1) * 0.22 * b
    base = vgradient(np.clip(py, 0, H - 1), [(0, "#e3e6e9"), (0.6, "#dadee2"), (1, "#cfd4d9")])
    out = over(base, ink, a)
    return over(out, accent, stroke(py - acc_y, 2.4))


# --- ray-trace a glass ball floating a little in front of the paper -------------
gap = 0.9 * R          # distance from the ball's back to the paper
rx, ry = x - cx, y - cy
r2 = rx * rx + ry * ry
inside = fill(np.sqrt(r2) - R, 1.2)
z1 = np.sqrt(np.clip(R * R - r2, 0, None))
N1 = np.stack([rx, ry, z1], -1) / R
cosi = N1[..., 2]  # view ray is (0,0,-1)


def refract(d, n, eta):
    c = -np.sum(n * d, -1, keepdims=True)
    k = 1 - eta * eta * (1 - c * c)
    return eta * d + (eta * c - np.sqrt(np.clip(k, 0, None))) * n


def trace(ior):
    d0 = np.zeros_like(N1)
    d0[..., 2] = -1
    P1 = N1 * R
    t = refract(d0, N1, 1 / ior)
    s = -2 * np.sum(P1 * t, -1, keepdims=True)
    P2 = P1 + s * t
    t2 = refract(t, -P2 / R, ior)
    tz = np.minimum(t2[..., 2], -1e-3)
    s2 = (-R - gap - P2[..., 2]) / tz
    return paper(cx + P2[..., 0] + s2 * t2[..., 0], cy + P2[..., 1] + s2 * t2[..., 1])


lens = np.stack([trace(1.511)[..., 0], trace(1.515)[..., 1], trace(1.519)[..., 2]], -1)
lens = gblur(lens, 0.5) * hex2lin("#f3f6f9")  # faint body tint

# fresnel reflection of a soft studio: bright above, dim below
F = 0.04 + 0.96 * (1 - cosi) ** 5
refl_y = (ry / R) * (1 - 2 * (1 - cosi))  # rough reflected-direction proxy
env = mix(np.broadcast_to(hex2lin("#f7f9fb"), lens.shape), hex2lin("#8f98a3"), smoothstep(-0.8, 0.8, refl_y))
lens = mix(lens, env, np.clip(F, 0, 1) * 0.9)
# thin dark line right at the silhouette, where the glass reflects the room
lens = lens * (1 - 0.35 * smoothstep(0.965, 1.0, np.sqrt(r2) / R))[..., None]

# specular softbox reflection, upper-left
spx, spy = cx - R * 0.40, cy - R * 0.47
e = ((x - spx) * 0.8 + (y - spy) * 0.6) / 64, (-(x - spx) * 0.6 + (y - spy) * 0.8) / 30
spec = smoothstep(1.0, 0.25, np.hypot(*e))
lens = lens + (spec * 0.6 + np.exp(-np.hypot(*e) ** 2 / 6) * 0.08)[..., None]

# --- shadow + caustic on the paper -------------------------------------------------
flat = paper(x, y)
scx, scy = cx + 120, cy + 190
sd = np.hypot((x - scx) / 1.1, (y - scy) / 0.95)
shadow = gblur(fill(sd - R * 0.9, 1), 70)
flat = flat * (1 - 0.24 * shadow)[..., None]
caustic = np.exp(-(((x - scx - 30) / 95) ** 2 + ((y - scy - 40) / 70) ** 2))
flat = flat + (caustic * 0.26)[..., None] * hex2lin("#fff6e8")

img = over(flat, lens, inside)
save(img, "../wallpapers/03-lens.png", grain=0.005, seed=3)
