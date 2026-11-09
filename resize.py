"""Lanczos-3 resizing with PIL's coefficients, applied on the GPU as two matrix products, in float.

PIL widens the kernel by the scale when shrinking, so downscaling is antialiased (cv2's INTER_LANCZOS4 is not).
Unlike PIL, nothing is rounded to 8 bits between the horizontal and vertical passes.
"""
import math


def lanczos_taps(n_in, n_out, a=3):
    """Per output pixel: (first input index, weights). The coefficients of PIL's Lanczos (Resample.c
    precompute_coeffs)."""
    def kernel(x):
        if x == 0:
            return 1.0
        if -a < x < a:
            px = math.pi * x
            return a * math.sin(px) * math.sin(px / a) / (px * px)
        return 0.0

    scale = n_in / n_out
    filterscale = max(scale, 1.0)
    support = a * filterscale
    taps = []
    for i in range(n_out):
        center = (i + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), n_in)
        weights = [kernel((j - center + 0.5) / filterscale) for j in range(lo, hi)]
        total = sum(weights)
        taps.append((lo, [w / total for w in weights] if total else weights))
    return taps


def lanczos_matrix(n_in, n_out, device=None):
    """Dense [n_out, n_in] resampling matrix."""
    import torch
    m = torch.zeros(n_out, n_in, dtype=torch.float32)
    for i, (lo, weights) in enumerate(lanczos_taps(n_in, n_out)):
        m[i, lo:lo + len(weights)] = torch.tensor(weights)
    return m.to(device) if device is not None else m


class Resizer:
    """Resize [1, C, H, W] float frames to (width, height); matrices are built once per input size."""

    def __init__(self, width, height, device=None):
        self.size = (width, height)
        self.device = device
        self._for = None

    def __call__(self, x):
        h, w = x.shape[2], x.shape[3]
        if (w, h) != self._for:
            self._for = (w, h)
            self.rows = lanczos_matrix(h, self.size[1], x.device)
            self.cols_t = lanczos_matrix(w, self.size[0], x.device).T.contiguous()
        return (self.rows @ (x[0].float() @ self.cols_t)).clamp(0, 1).unsqueeze(0)
