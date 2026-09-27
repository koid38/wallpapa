"""Aurora — paper-cut snowy peaks under vellum ribbons of northern lights."""
import numpy as np
from common import *
from paper import *

x, y = grid()
rng = np.random.default_rng(18)

sky = vgradient(y, [(0, "#0b141d"), (0.45, "#12212b"), (0.7, "#1a2e36"), (1, "#1a2e36")])
s = Sheet(sky)

# small paper stars and a crescent moon
stars = []
for _ in range(46):
    sx, sy = rng.uniform(30, W - 30), rng.uniform(0.05 * H, 0.55 * H)
    stars.append((sx, sy, rng.choice([3.0, 3.5, 4.5, 6.0], p=[0.4, 0.3, 0.2, 0.1])))
s.add(shapes(circles=stars), hex2lin("#e9e2cf"), shadow=0.3, off=3, soft=3, rim=0)
mx, my, mr = W * 0.24, H * 0.30, 78
moon = shapes(circles=[(mx, my, mr)]) * (1 - shapes(circles=[(mx + 34, my - 20, mr * 0.9)]))
s.add(moon, hex2lin("#efe6d2"), shadow=0.3, off=6, soft=8, rim=0)

# aurora: translucent vellum ribbons, brighter along their lower hem
for i, (y0, amp, ph, hgt, col, a) in enumerate([
        (0.40, 70, 0.4, 330, "#63c7a2", 0.42),
        (0.455, 60, 2.1, 260, "#7fd8b4", 0.40),
        (0.35, 50, 4.0, 220, "#58b7b0", 0.30)]):
    hem = lambda xs, y0=y0, amp=amp, ph=ph: (y0 * H - (xs - W / 2) * 0.22
                                             + amp * np.sin(xs / W * 2 * np.pi * 1.1 + ph)
                                             + amp * 0.4 * np.sin(xs / W * 2 * np.pi * 2.7 + ph * 1.7))
    top = lambda xs, hem=hem, hgt=hgt, ph=ph: hem(xs) - hgt * (0.75 + 0.25 * np.sin(xs / W * 2 * np.pi * 1.9 + ph))
    ribbon = shapes([line_poly(top)]) * (1 - shapes([line_poly(hem)]))
    fade = np.clip((y - top(x)) / np.maximum(hem(x) - top(x), 1), 0, 1) ** 1.6
    s.add(ribbon * (0.25 + 0.75 * fade), hex2lin(col), shadow=0.25, off=8, soft=12, rim=0.0, alpha=a)


def ridge(peaks, seed, rough=18):
    """Jagged polyline through valley/peak points, roughened by midpoint displacement."""
    r = np.random.default_rng(seed)
    pts = [(px * W, py * H) for px, py in peaks]
    for _ in range(4):
        out = []
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            out += [(ax, ay), ((ax + bx) / 2, (ay + by) / 2 + r.normal(0, rough))]
        pts = out + [pts[-1]]
        rough *= 0.55
    return pts


def mountains(peaks, seed, snow_y, lit, shade, snow, snow_shade):
    pts = ridge(peaks, seed)
    body = shapes([[(-10, H + 10)] + pts + [(W + 10, H + 10)]])
    # shade the right-hand faces: from each summit a line runs down, open below
    sh = np.zeros((H, W), np.float32)
    for (ax, ay), (bx, by), (cx_, cy_) in zip(peaks[:-2], peaks[1:-1], peaks[2:]):
        if by < ay and by < cy_:  # a summit
            px, py = bx * W, by * H
            vx, vy = cx_ * W, cy_ * H  # the saddle to the right
            edge = px + (y - py) * 0.18
            stop = vx + np.clip(y - vy, 0, None) * 0.35  # where the next peak's lit face begins
            sh = np.maximum(sh, np.clip(x - edge + 0.5, 0, 1) * np.clip(stop - x + 0.5, 0, 1) * (y > py))
    zig = snow_y * H + 22 * np.abs(((x / 46) % 2) - 1) + 10 * np.sin(x / 90)
    cap = np.clip(zig - y + 0.5, 0, 1)
    col = mix(np.broadcast_to(hex2lin(lit), (H, W, 3)), hex2lin(shade), sh)
    col = mix(col, mix(np.broadcast_to(hex2lin(snow), (H, W, 3)), hex2lin(snow_shade), sh), cap)
    return body, col


layers = [
    ([(-0.05, 0.62), (0.12, 0.50), (0.26, 0.58), (0.44, 0.47), (0.60, 0.57), (0.80, 0.49), (0.98, 0.58), (1.08, 0.55)],
     0.53, "#3a4f5c", "#2f4250", "#b9c6cd", "#94a4ae"),
    ([(-0.08, 0.70), (0.08, 0.60), (0.30, 0.69), (0.55, 0.55), (0.78, 0.68), (0.96, 0.61), (1.1, 0.68)],
     0.605, "#28394a", "#1e2e3c", "#d5dee3", "#a9b7c0"),
    ([(-0.1, 0.80), (0.20, 0.68), (0.42, 0.79), (0.72, 0.66), (1.0, 0.79), (1.12, 0.74)],
     0.71, "#17242f", "#101b24", "#e8eef0", "#b8c5cc"),
]
for i, (peaks, snow_y, lit, shade, snow, snow_shade) in enumerate(layers):
    body, col = mountains(peaks, 40 + i, snow_y, lit, shade, snow, snow_shade)
    s.add(body, col, shadow=0.45, off=10 + 4 * i, soft=14 + 4 * i, rim=0.08)

# a dark band of forest at the foot
f = lambda xs: H * 0.86 + 18 * np.sin(xs / W * 2 * np.pi * 0.8 + 1.0)
r = np.random.default_rng(88)
trees = [pine(tx, float(f(np.array([tx]))[0]) + 6, h, h * 0.4, tiers=int(r.integers(3, 5)))
         for tx, h in zip(np.sort(r.uniform(-40, W + 40, 44)), r.uniform(110, 260, 44))]
s.add(shapes([line_poly(f)] + trees), hex2lin("#0a1218"), shadow=0.5, off=14, soft=18, rim=0.05)

s.finish("../wallpapers/18-aurora.png", seed=18)
