"""3D gaussian arithmetic shared by the core (data/gaussian.py) and the reconstruction workers: one implementation of
the covariance, the real spherical-harmonics basis and the rigid / similarity bake of a splat set.

Conventions (Inria 3DGS): rotations are unit quaternions (w, x, y, z), scales are linear standard deviations, opacity
is linear, SH coefficients are [N, C, 3] in the Inria real-SH order with the DC term first (C = 1, 4, 9 or 16).
"""

from __future__ import annotations

import numpy as np

C0 = 0.28209479177387814  # the DC basis value: colour = C0 * sh[:, 0] + 0.5


def _covariance(scales, rotations) -> np.ndarray:
    """The full 3x3 covariance per splat in float64: a thin splat's variances span many decades (an axis ratio of 1e4
    is 1e8 in variance), beyond float32 once it is turned and decomposed again."""
    w, x, y, z = np.asarray(rotations, np.float64).T
    r = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), axis=1).reshape(-1, 3, 3)
    return (r * np.asarray(scales, np.float64)[:, None, :] ** 2) @ r.transpose(0, 2, 1)


def covariance(scales, rotations) -> np.ndarray:
    """Six symmetric covariance entries xx, xy, xz, yy, yz, zz per splat, in the splat's local space."""
    return _covariance(scales, rotations)[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].astype(np.float32)


def sh_basis(directions, coefficients: int) -> np.ndarray:
    """The real SH basis in Inria order at unit `directions` [N,3] -> [N, coefficients]; the viewer's shader
    (webui/src/view/gaussian3d.tsx) evaluates the same polynomial."""
    x, y, z = np.asarray(directions, np.float64).T
    values = [np.full_like(x, C0)]
    if coefficients > 1:
        values += [-.4886025119029199 * y, .4886025119029199 * z, -.4886025119029199 * x]
    if coefficients > 4:
        values += [1.0925484305920792 * x * y, -1.0925484305920792 * y * z,
                   .31539156525252005 * (2 * z * z - x * x - y * y), -1.0925484305920792 * x * z,
                   .5462742152960396 * (x * x - y * y)]
    if coefficients > 9:
        values += [-.5900435899266435 * y * (3 * x * x - y * y), 2.890611442640554 * x * y * z,
                   -.4570457994644658 * y * (4 * z * z - x * x - y * y),
                   .3731763325901154 * z * (2 * z * z - 3 * x * x - 3 * y * y),
                   -.4570457994644658 * x * (4 * z * z - x * x - y * y), 1.445305721320277 * z * (x * x - y * y),
                   -.5900435899266435 * x * (x * x - 3 * y * y)]
    return np.stack(values, axis=-1)


def uniform_part(linear) -> tuple[np.ndarray, bool]:
    """(the rotation or reflection of a 3x3 linear map, whether the map is that times a uniform scale). The SH of a
    splat only follow a map whose directions change rigidly; a non-uniform scale bends directions nonlinearly."""
    u, s, vt = np.linalg.svd(np.asarray(linear, np.float64))
    return u @ vt, bool(np.allclose(s, s[0], rtol=1e-5, atol=1e-7))


def rotate_sh(sh, rotation) -> np.ndarray:
    """Turn real SH coefficients [N,C,3] by a rotation (or reflection) 3x3. The SH bands are closed under rotation, so
    the map is linear: it is solved exactly from the basis at a fixed spread of directions (a Fibonacci sphere with
    more points than coefficients), the same for every splat."""
    h = np.asarray(sh)
    if h.shape[1] == 1:
        return h
    n = 96
    i = np.arange(n)
    z = 1 - 2 * (i + .5) / n
    theta = i * np.pi * (3 - np.sqrt(5))
    radial = np.sqrt(1 - z * z)
    dirs = np.stack((radial * np.cos(theta), radial * np.sin(theta), z), axis=-1)
    basis = sh_basis(dirs, h.shape[1])
    turned = sh_basis(dirs @ np.asarray(rotation, np.float64), h.shape[1])
    transform = np.linalg.lstsq(basis, turned, rcond=None)[0]
    return np.einsum("ij,njc->nic", transform, h).astype(np.float32)


def _matrix_to_quat(r: np.ndarray) -> np.ndarray:
    from .motion import matrix_to_quat

    return matrix_to_quat(r)


def transform(values: dict, matrix) -> dict:
    """Bake a 4x4 column-vector transform into one splat set: positions moved, the covariance (scales + rotations)
    turned and scaled, the SH turned with the map's rotation. A non-uniform scale is carried by the covariance, but SH
    above degree 0 cannot follow it: ValueError (the caller says so in its own words). `values`: points [N,3], scales
    [N,3], rotations [N,4] wxyz, opacity [N], sh [N,C,3]."""
    m = np.asarray(matrix, np.float64)
    out = dict(values)
    out["points"] = (np.asarray(values["points"], np.float64) @ m[:3, :3].T + m[:3, 3]).astype(np.float32)
    full = _covariance(values["scales"], values["rotations"])
    eigen, basis = np.linalg.eigh(m[:3, :3] @ full @ m[:3, :3].T)
    basis[:, :, 0] *= np.linalg.det(basis)[:, None]  # a proper rotation, whatever order eigh returned
    out["scales"] = np.sqrt(np.maximum(eigen, 1e-20)).astype(np.float32)
    out["rotations"] = _matrix_to_quat(basis).astype(np.float32)
    rotation, uniform = uniform_part(m[:3, :3])
    if np.asarray(values["sh"]).shape[1] > 1 and not uniform:
        raise ValueError("non-uniform scale on view-dependent SH")
    out["sh"] = rotate_sh(values["sh"], rotation)
    return out
