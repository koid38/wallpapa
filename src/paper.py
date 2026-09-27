"""Paper-cut helpers: polygon masks, stacked sheets with drop shadows, paper texture."""
import numpy as np
from PIL import Image, ImageDraw
from common import *

SS = 3  # supersampling for polygon edges


def shift(a, dy, dx=0):
    """Shift a 2D array with zero fill."""
    out = np.zeros_like(a)
    ys = slice(max(dy, 0), H + min(dy, 0))
    yd = slice(max(-dy, 0), H + min(-dy, 0))
    xs = slice(max(dx, 0), W + min(dx, 0))
    xd = slice(max(-dx, 0), W + min(-dx, 0))
    out[ys, xs] = a[yd, xd]
    return out


def shapes(polys=(), circles=()):
    """Anti-aliased mask from polygons [(x, y), ...] and circles (cx, cy, r)."""
    im = Image.new("L", (W * SS, H * SS), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        d.polygon([(px * SS, py * SS) for px, py in p], fill=255)
    for cx, cy, r in circles:
        d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=255)
    return np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255


def line_poly(fy, step=6, bottom=H + 10):
    """Polygon filling everything below the curve y = fy(x)."""
    xs = np.arange(-10, W + 11, step, dtype=np.float64)
    return [(float(a), float(b)) for a, b in zip(xs, fy(xs))] + [(W + 10, bottom), (-10, bottom)]


def pine(x, gy, h, w, tiers=4):
    """A tiered pine: apex at (x, gy - h), trunk down to gy."""
    trunk = h * 0.08
    top = gy - h
    pts_r = [(x, top)]
    for i in range(1, tiers + 1):
        t = i / tiers
        yb = top + (h - trunk) * t
        wi = w * (0.35 + 0.65 * t) / 2
        pts_r.append((x + wi, yb))
        if i < tiers:
            pts_r.append((x + wi * 0.42, yb - (h - trunk) / tiers * 0.12))
    pts_r += [(x + w * 0.05, gy - trunk), (x + w * 0.05, gy + 4)]
    pts_l = [(2 * x - px, py) for px, py in reversed(pts_r[1:])]
    return pts_r + pts_l


class Sheet:
    """A stack of paper layers, composited back to front."""

    def __init__(self, bg):
        self.img = bg.astype(np.float32).copy()

    def add(self, mask, color, shadow=0.35, off=12, soft=14, rim=0.10, alpha=1.0):
        # the new sheet shades whatever lies behind it
        if shadow:
            sh = gblur(shift(mask, off, 2), soft)
            self.img *= (1 - shadow * alpha * sh)[..., None]
        col = color if np.ndim(color) == 3 else np.broadcast_to(color, self.img.shape)
        self.img = over(self.img, col, mask * alpha)
        # the cut edge on top catches the light
        if rim:
            edge = np.clip(mask - shift(mask, 3), 0, 1)
            self.img = over(self.img, hex2lin("#ffffff"), edge * rim * alpha)

    def finish(self, path, seed, grain=0.006):
        fibres = fbm([(0.8, 1.0), (2.5, 0.6), (10, 0.35), (60, 0.25)], seed=seed + 100)
        self.img *= (1 + 0.022 * fibres)[..., None]
        save(self.img, path, grain=grain, seed=seed)


def vgrad_color(y, top, bottom, y0, y1):
    """Per-pixel colour for one sheet: a gentle vertical fade between two hexes."""
    t = np.clip((y - y0) / max(y1 - y0, 1), 0, 1)
    return mix(np.broadcast_to(hex2lin(top), y.shape + (3,)), hex2lin(bottom), t)
