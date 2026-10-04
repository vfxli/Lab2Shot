"""Reading a PLY file (ASCII, binary little- or big-endian) into scene arrays: the vertex element as one point cloud
(x, y, z; red, green, blue when present, 0-255 or 0-1) and the face element, if any, as one model (the vertex_indices
lists; per-vertex nx, ny, nz as face-varying normals). Items are named after the file stem. PLY carries no time, so its
items have no frames and apply to every frame of a shot. Depends only on numpy."""

from __future__ import annotations

from ... import i18n

from pathlib import Path

import numpy as np
from lab2shot_shared.scene_arrays import SceneArrays

from ...data.units import DEFAULT_FPS
from ...errors import Invalid
from ...messages import Msg

TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2", "ushort": "u2",
         "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4",
         "double": "f8", "float64": "f8"}
STILL: list[int] = []  # no frames: the items apply to every frame (a still scene packet)


def _refuse(path: Path, detail: str) -> Invalid:
    return Invalid(Msg("E-PLY-UNREADABLE", file=path.name, detail=detail))


def _header(path: Path, data: bytes) -> tuple[str, list[tuple[str, int, list[tuple]]], int]:
    """(format, elements [(name, count, properties)], body offset). A property is (name, type), or
    (name, count type, item type) for a list."""
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise _refuse(path, i18n.t("reader.ply.no_header"))
    body = data.index(b"\n", end) + 1
    fmt, elements = "", []
    for line in data[:end].decode("ascii", "replace").splitlines()[1:]:
        words = line.split()
        if not words or words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "format":
            fmt = words[1]
        elif words[0] == "element":
            elements.append((words[1], int(words[2]), []))
        elif words[0] == "property" and elements:
            prop = (words[4], TYPES[words[2]], TYPES[words[3]]) if words[1] == "list" else (words[2], TYPES[words[1]])
            elements[-1][2].append(prop)
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise _refuse(path, i18n.t("reader.ply.unknown_format", format=fmt or i18n.t("reader.ply.unset")))
    return fmt, elements, body


def _read_elements(path: Path, data: bytes) -> dict[str, dict[str, object]]:
    fmt, elements, at = _header(path, data)
    out: dict[str, dict[str, object]] = {}
    if fmt == "ascii":
        words = data[at:].split()
        i = 0
        for name, count, props in elements:
            if all(len(p) == 2 for p in props):  # scalar properties only: read as one block
                n = count * len(props)
                if i + n > len(words):
                    raise _refuse(path, i18n.t("reader.ply.cut", name=name, count=count))
                block = np.array(words[i:i + n], np.float64).reshape(count, len(props))
                out[name] = {p[0]: block[:, k].astype(p[1]) for k, p in enumerate(props)}
                i += n
                continue
            values: dict[str, list] = {p[0]: [] for p in props}
            for _ in range(count):
                for p in props:
                    if i >= len(words):
                        raise _refuse(path, i18n.t("reader.ply.cut", name=name, count=count))
                    if len(p) == 3:
                        k = int(words[i])
                        values[p[0]].append([int(float(w)) for w in words[i + 1:i + 1 + k]])
                        i += 1 + k
                    else:
                        values[p[0]].append(float(words[i]))
                        i += 1
            out[name] = values
        return out
    order = "<" if fmt == "binary_little_endian" else ">"
    for name, count, props in elements:
        if all(len(p) == 2 for p in props):
            dtype = np.dtype([(p[0], order + p[1]) for p in props])
            if at + dtype.itemsize * count > len(data):
                raise _refuse(path, i18n.t("reader.ply.cut", name=name, count=count))
            rows = np.frombuffer(data, dtype, count, at)
            out[name] = {p[0]: rows[p[0]].astype(p[1]) for p in props}
            at += dtype.itemsize * count
            continue
        values = {p[0]: [] for p in props}
        for _ in range(count):
            for p in props:
                if len(p) == 3:
                    head = np.dtype(order + p[1])
                    if at + head.itemsize > len(data):
                        raise _refuse(path, i18n.t("reader.ply.cut", name=name, count=count))
                    k = int(np.frombuffer(data, head, 1, at)[0])
                    item = np.dtype(order + p[2])
                    if at + head.itemsize + item.itemsize * k > len(data):
                        raise _refuse(path, i18n.t("reader.ply.cut", name=name, count=count))
                    values[p[0]].append(np.frombuffer(data, item, k, at + head.itemsize).tolist())
                    at += head.itemsize + item.itemsize * k
                else:
                    one = np.dtype(order + p[1])
                    if at + one.itemsize > len(data):
                        raise _refuse(path, i18n.t("reader.ply.cut", name=name, count=count))
                    values[p[0]].append(np.frombuffer(data, one, 1, at)[0])
                    at += one.itemsize
        out[name] = values
    return out


def arrays(path: Path) -> dict:
    """The file's scene arrays (SceneArrays.arrays()): vertices as a point cloud, faces as a model."""
    elements = _read_elements(path, path.read_bytes())
    vertex = elements.get("vertex")
    if not vertex or not all(k in vertex for k in "xyz"):
        raise _refuse(path, i18n.t("reader.ply.no_xyz"))
    points = np.stack([np.asarray(vertex[k], np.float64) for k in "xyz"], axis=-1)
    name, where = path.stem, f"/{path.stem}"
    out = SceneArrays(DEFAULT_FPS)
    fields = [*(f"f_dc_{i}" for i in range(3)), "opacity", *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4))]
    if all(k in vertex for k in fields):
        from ...data.gaussian import validate

        rest = sorted((k for k in vertex if k.startswith("f_rest_")), key=lambda k: int(k[7:]))
        if rest != [f"f_rest_{i}" for i in range(len(rest))] or len(rest) not in (0, 9, 24, 45):
            raise _refuse(path, i18n.t("reader.ply.sh_cut"))
        dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=-1)[:, None, :]
        h = np.stack([vertex[k] for k in rest], axis=-1).reshape(len(points), 3, len(rest) // 3).transpose(0, 2, 1) if rest else np.empty((len(points), 0, 3))
        raw = np.asarray(vertex["opacity"], np.float64)
        opacity = np.exp(-np.logaddexp(0, -raw))
        scales = np.exp(np.stack([vertex[f"scale_{i}"] for i in range(3)], axis=-1))
        rotations = np.stack([vertex[f"rot_{i}"] for i in range(4)], axis=-1)
        try:
            p, s, r, a, h = validate(points, scales, rotations, opacity, np.concatenate((dc, h), axis=1))
        except Invalid as exc:  # say which file it was
            raise _refuse(path, str(exc)) from None
        out.add("gaussian", name, where, STILL, np.eye(4)[None], counts=np.array([len(p)]),
                points=p, scales=s, rotations=r, opacity=a, sh=h)
        return out.arrays()
    colours = None
    if all(k in vertex for k in ("red", "green", "blue")):
        raw = np.stack([np.asarray(vertex[k]) for k in ("red", "green", "blue")], axis=-1)
        colours = raw.astype(np.float64) / 255.0 if raw.dtype.kind in "iu" else raw.astype(np.float64)
    out.add("points", name, where, STILL, np.eye(4)[None], counts=np.array([len(points)], np.int64), points=points,
            **({"colors": colours} if colours is not None else {}))
    faces = elements.get("face", {})
    lists = faces.get("vertex_indices", faces.get("vertex_index"))
    if lists:
        counts = np.array([len(f) for f in lists], np.int32)
        indices = np.array([i for f in lists for i in f], np.int32)
        if len(indices) and (indices.min() < 0 or indices.max() >= len(points)):
            raise _refuse(path, i18n.t("reader.ply.bad_face"))
        normals = None
        if all(k in vertex for k in ("nx", "ny", "nz")):
            normals = np.stack([np.asarray(vertex[k], np.float32) for k in ("nx", "ny", "nz")], axis=-1)[indices]
        out.add("model", name, where, STILL, np.eye(4)[None], counts=counts, indices=indices, points=points[None],
                **({"normals": normals} if normals is not None else {}))
    return out.arrays()
