"""Shared helpers: linear-light canvas, soft shapes, gaussian blur, sampling, export."""
import numpy as np
from PIL import Image

W, H = 1440, 3168


def grid():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    return x + 0.5, y + 0.5


def hex2lin(h):
    h = h.lstrip("#")
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    t = np.asarray(t, np.float32)
    if t.ndim == 2:
        t = t[..., None]
    return a + (b - a) * t


def over(base, color, alpha):
    """Composite a flat color (or image) over base with a 2D alpha."""
    return mix(base, color, alpha)


def vgradient(y, stops):
    """stops: list of (pos 0..1, hex). Returns HxWx3 linear image."""
    t = y / H
    out = np.zeros(y.shape + (3,), np.float32)
    pos = [p for p, _ in stops]
    cols = [hex2lin(c) for _, c in stops]
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, [c[ch] for c in cols])
    return out


def sdf_circle(x, y, cx, cy, r):
    return np.hypot(x - cx, y - cy) - r


def sdf_round_rect(x, y, cx, cy, hw, hh, r):
    qx = np.abs(x - cx) - (hw - r)
    qy = np.abs(y - cy) - (hh - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


def fill(sdf, soft=1.0):
    """Anti-aliased coverage of the region sdf < 0."""
    return np.clip(0.5 - sdf / soft, 0, 1).astype(np.float32)


def stroke(sdf, width):
    return np.clip(width / 2 + 0.5 - np.abs(sdf), 0, 1).astype(np.float32)


def gblur(img, sigma):
    """Gaussian blur via FFT with edge padding. Works on HxW or HxWxC."""
    if sigma <= 0:
        return img
    pad = int(3 * sigma) + 2
    squeeze = img.ndim == 2
    a = img[..., None] if squeeze else img
    a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    h, w = a.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    k = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (fx ** 2 + fy ** 2)).astype(np.float32)
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        out[..., c] = np.fft.irfft2(np.fft.rfft2(a[..., c]) * k, s=(h, w))
    out = out[pad:-pad, pad:-pad]
    return out[..., 0] if squeeze else out


def sample(img, sx, sy):
    """Bilinear sample img at float coords (pixel centers at +0.5)."""
    sx = np.clip(sx - 0.5, 0, img.shape[1] - 1.001)
    sy = np.clip(sy - 0.5, 0, img.shape[0] - 1.001)
    x0 = sx.astype(np.int32)
    y0 = sy.astype(np.int32)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def save(img, path, grain=0.006, seed=0):
    """Linear -> sRGB, add fine monochrome grain + dither to kill banding."""
    rng = np.random.default_rng(seed)
    s = lin2srgb(img)
    n = rng.normal(0, 1, s.shape[:2]).astype(np.float32)[..., None]
    s = s + n * grain + (rng.random(s.shape, dtype=np.float32) - 0.5) / 255
    out = (np.clip(s, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(out).save(path, optimize=True)
    return out
