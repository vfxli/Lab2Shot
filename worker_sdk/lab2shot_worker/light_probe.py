"""The light-probe family (DiffusionLight, LuxDiT): one lat-long convention, the probe frame, the result files and
statistics, shared by every worker that estimates an HDRI from a plate. Node side: lab2shot/nodes/results.py
light_probe(). The raw contract (save_probe):

    raw/envmap.exr    float32 RGB lat-long HDR, scene-linear Rec.709, width = 2 x height, camera-relative (ORIENTATION)
    raw/preview.png   what the node shows next to it (sRGB): the method's own picture of the light
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lab2shot_shared.light_probe import (ENVMAP, LUMA, ORIENTATION, PREVIEW, REC709_CHROMATICITIES,  # noqa: F401
                                         USD_DOMELIGHT, latlong_directions)

from . import Job, fail, files

def probe_frame(job: Job) -> tuple[int, Path]:
    """(frame number, plate) to light from: the node chose it and sends only that frame (probe_plate)."""
    if len(job.frames) != 1:
        fail("E-PROBE-ONEFRAME", count=len(job.frames))
    return job.frames[0]


def save_probe(raw: Path, frame: int, hdr: np.ndarray, preview) -> dict:
    """raw/envmap.exr (float32 R,G,B, ZIP, the standard chromaticities attribute: Rec.709 / D65) and raw/preview.png
    (`preview`: a PIL image). Returns the result.json fields naming them."""
    raw.mkdir(parents=True, exist_ok=True)
    attrs = {"frame": frame, "projection": "latlong", "orientation": "camera-relative; u=0.5 = view direction; top = camera up"}
    files.write_exr(raw / ENVMAP, hdr, "RGB", attrs, REC709_CHROMATICITIES)
    preview.save(raw / PREVIEW)
    return {"envmap": ENVMAP, "envmap_size": [hdr.shape[1], hdr.shape[0]], "preview": PREVIEW}


def hdr_stats(hdr: np.ndarray) -> dict:
    lum = hdr @ LUMA
    h, w = lum.shape
    # Brightest region: centroid of the top 0.1% pixels (solid-angle weighted), as a direction.
    weight = np.sin((np.arange(h) + 0.5) / h * np.pi)[:, None] * np.ones((1, w))
    thresh = np.quantile(lum, 0.999)
    sel = lum >= thresh
    d = latlong_directions(h, w)
    c = (d[sel] * (lum[sel] * weight[sel])[:, None]).sum(0)
    c /= max(np.linalg.norm(c), 1e-12)
    azimuth = math.degrees(math.atan2(c[0], -c[2]))  # 0 = forward, +90 = camera right
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, c[1]))))
    theta = math.atan2(-c[0], c[2]) % (2 * math.pi)
    y, x = np.unravel_index(int(np.argmax(lum)), lum.shape)
    return {
        "max_luminance": float(lum.max()),
        "median_luminance": float(np.median(lum)),
        "p99_luminance": float(np.quantile(lum, 0.99)),
        "dynamic_range_max_over_median": float(lum.max() / max(np.median(lum), 1e-10)),
        "max_pixel": [int(x), int(y)],
        "brightest_direction": {
            "azimuth_deg": round(azimuth, 1),
            "elevation_deg": round(elevation, 1),
            "uv": [round(theta / (2 * math.pi), 4), round(math.acos(max(-1.0, min(1.0, c[1]))) / math.pi, 4)],
            "meaning": "centroid of the brightest 0.1% (solid-angle weighted). azimuth 0 = camera forward, +90 = camera right, +-180 = behind; elevation +90 = camera up",
        },
    }
