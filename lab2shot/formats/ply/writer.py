"""Binary little-endian PLY: 3D gaussians in Inria's layout (active scales/opacity become log/logit), point clouds as
x y z + red green blue (uchar, the display colour)."""
from __future__ import annotations

import numpy as np


def write(path, sample):
    from ...data.gaussian import validate

    p, s, r, a, sh = validate(*(sample[k] for k in ("points", "scales", "rotations", "opacity", "sh")))
    rest = sh[:, 1:].transpose(0, 2, 1).reshape(len(p), (sh.shape[1] - 1) * 3)
    names = [*"xyz", "nx", "ny", "nz", *(f"f_dc_{i}" for i in range(3)),
             *(f"f_rest_{i}" for i in range(rest.shape[1])), "opacity", *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4))]
    clipped = np.clip(a, 1e-7, 1 - 1e-7)
    columns = np.concatenate((p, np.zeros_like(p), sh[:, 0], rest, (np.log(clipped) - np.log1p(-clipped))[:, None], np.log(s), r), axis=1)
    data = np.empty(len(p), dtype=[(name, "<f4") for name in names])
    for i, name in enumerate(names):
        data[name] = columns[:, i]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex " + str(len(p)) + "\n" + "".join("property float " + k + "\n" for k in names) + "end_header\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(data.tobytes())
    return path


def write_points(path, points, colors):
    """A point cloud: float x y z and uchar red green blue (the display colours 0..1, as USD's displayColor holds them,
    rounded to 0..255) — the vertex layout every point-cloud tool reads (MeshLab, CloudCompare, Open3D, Houdini)."""
    p = np.asarray(points, np.float32).reshape(-1, 3)
    c = np.clip(np.rint(np.asarray(colors, np.float64).reshape(-1, 3) * 255.0), 0, 255).astype(np.uint8)
    data = np.empty(len(p), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, name in enumerate("xyz"):
        data[name] = p[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        data[name] = c[:, i]
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex " + str(len(p)) + "\n"
              + "".join("property float " + k + "\n" for k in "xyz")
              + "".join("property uchar " + k + "\n" for k in ("red", "green", "blue")) + "end_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(data.tobytes())
    return path
