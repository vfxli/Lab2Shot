"""Reading a Nuke script (.nk, as Nuke and 3DEqualizer's Nuke export write it): its top-level nodes and their knobs,
a knob's constant or keyed curve at frames, 3DE's LD_3DE lens distortion nodes as a distortion, a Camera node as a
tracked camera.

This is the single Nuke parser in the program. The hand-over block (lab2shot/datasets/blocks/nuke_script.py), the
benchmark ground truths and this package's reader (formats/nuke/reader.py, which adds the script's wiring) all use it.

LD_3DE knobs are matched to the model's parameter names by comparing lowercase letters and digits only (3DE writes
spaces and punctuation in names as underscores); a knob absent from the script takes its default.
"""


from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ...data import units
from ...errors import Invalid
from ...io.sequence import format_frame_range
from ...messages import Msg
from ...data.lens_models import MODELS, in_limit

SCENE_UNITS = ("cm", "m", "mm")  # length units a tracking or compositing scene may declare
CM_TO_MM = units.UNITS["cm"].per_base / units.UNITS["mm"].per_base  # 3DE's built-in lens knobs are in centimetres


def text_of(source) -> str:
    """The text of a hand-over source: the string itself, or the contents of the file."""
    return Path(source).read_text(encoding="utf-8", errors="replace") if isinstance(source, Path) else str(source)


def scene_cm(unit: str) -> float:
    if unit not in SCENE_UNITS:
        raise Invalid(Msg("E-NUKE-UNIT", unit=str(unit)[:10], units=list(SCENE_UNITS)))
    return units.UNITS[unit].per_base


def nodes(text: str) -> list[tuple[str, dict[str, str]]]:
    """The top-level nodes of a Nuke script as (class, {knob: raw value}); braces in knob values are preserved."""
    out, i, n = [], 0, len(text)
    while i < n:
        m = re.compile(r"(?m)^\s*([A-Za-z_][\w.]*)\s*\{").search(text, i)
        if not m:
            break
        start, depth, j = m.end(), 1, m.end()
        while j < n and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        out.append((m.group(1), _knobs(text[start:j - 1])))
        i = j
    return out


def _knobs(body: str) -> dict[str, str]:
    knobs, i, n = {}, 0, len(body)
    while i < n:
        while i < n and body[i].isspace():
            i += 1
        m = re.compile(r"[A-Za-z_][\w.]*").match(body, i)
        if not m:
            i += 1
            continue
        name, i = m.group(0), m.end()
        while i < n and body[i] in " \t":
            i += 1
        if i < n and body[i] == "{":
            depth, j = 1, i + 1
            while j < n and depth:
                depth += {"{": 1, "}": -1}.get(body[j], 0)
                j += 1
            knobs[name], i = body[i:j], j
        else:
            m = re.compile(r"[^\s]*").match(body, i)
            knobs[name], i = m.group(0), m.end()
    return knobs


def _groups(value: str) -> list[str]:
    """The top-level brace groups inside a knob value `{a} {b}` / `{{curve ...} {curve ...}}`, or its plain words."""
    v = value.strip()
    if v.startswith("{") and v.endswith("}"):
        v = v[1:-1].strip()
    parts, i, n = [], 0, len(v)
    while i < n:
        if v[i].isspace():
            i += 1
        elif v[i] == "{":
            depth, j = 1, i + 1
            while j < n and depth:
                depth += {"{": 1, "}": -1}.get(v[j], 0)
                j += 1
            parts.append(v[i + 1:j - 1].strip())
            i = j
        else:
            m = re.compile(r"[^\s{}]+").match(v, i)
            parts.append(m.group(0))
            i = m.end()
    return parts


def curve(text: str) -> dict[int, float] | float:
    """A Nuke curve ("curve x1001 1.2 1.3 x1010 2" with interpolation letters and tangents) as its keys, or a constant."""
    words = text.split()
    if not words or words[0] != "curve":
        return float(words[0]) if words else 0.0
    keys, frame = {}, None
    for w in words[1:]:
        if re.fullmatch(r"x-?\d+(\.\d+)?", w):
            frame = int(round(float(w[1:])))
        elif re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", w):
            if frame is None:
                frame = 1
            keys[frame] = float(w)
            frame += 1
        # Interpolation letters (K L C R Z s t u v ...) and tangents (s0.5) are not keys.
    return keys


def _channel(value: str, count: int, frames: list[int], defaults: tuple[float, ...]) -> np.ndarray:
    """A knob's `count` channels at `frames`: linear between keys, held beyond the ends; constants repeated."""
    if count == 1:  # focal 35, focal {{curve x1 35 36}}
        v = value.strip()
        while v.startswith("{") and v.endswith("}"):
            v = v[1:-1].strip()
        groups = [v] if v else []
    else:  # translate {0 1.5 3}, translate {{curve ...} {curve ...} {curve ...}}
        groups = _groups(value) if value else []
    out = np.zeros((len(frames), count))
    for c in range(count):
        raw = groups[c] if c < len(groups) else None
        got = curve(raw) if raw is not None else defaults[c]
        if isinstance(got, dict):
            xs = sorted(got)
            out[:, c] = np.interp(frames, xs, [got[x] for x in xs]) if xs else defaults[c]
        else:
            out[:, c] = got
    return out


def euler(deg: np.ndarray, order: str) -> np.ndarray:
    """Rotations [F,3,3] from Euler angles (degrees, x y z channels) applied in `order` (Nuke's rot_order: ZXY rotates
    about Z first)."""
    r = np.radians(deg)
    mats = {}
    c, s = np.cos(r), np.sin(r)
    one, zero = np.ones(len(r)), np.zeros(len(r))
    mats["X"] = np.stack([np.stack([one, zero, zero], -1), np.stack([zero, c[:, 0], -s[:, 0]], -1), np.stack([zero, s[:, 0], c[:, 0]], -1)], 1)
    mats["Y"] = np.stack([np.stack([c[:, 1], zero, s[:, 1]], -1), np.stack([zero, one, zero], -1), np.stack([-s[:, 1], zero, c[:, 1]], -1)], 1)
    mats["Z"] = np.stack([np.stack([c[:, 2], -s[:, 2], zero], -1), np.stack([s[:, 2], c[:, 2], zero], -1), np.stack([zero, zero, one], -1)], 1)
    out = np.repeat(np.eye(3)[None], len(r), 0)
    for axis in order.upper():
        out = mats[axis] @ out  # the first axis in the order is applied first, so it remains rightmost
    return out


# ---- LD_3DE lens distortion nodes


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def ld_model(cls: str) -> str:
    """The distortion model id (lab2shot/data/lens_models.py MODELS) of an LD_3DE node class; "" for an unsupported model."""
    n = _norm(cls)
    if "rescaled" in n:
        return ""
    if "anamorphicstandarddegree4" in n:
        return "3de4_anamorphic_std_deg4"
    if "radialstandarddegree4" in n:
        return "3de4_radial_std_deg4"
    if "classic" in n:
        return "3de4_classic"
    return ""


def _ld(text: str) -> list[tuple[str, str, dict[str, str]]]:
    return [(k.get("name", f"{cls}{i + 1}"), cls, k) for i, (cls, k) in enumerate(nodes(text)) if _norm(cls).startswith("ld3de")]


def lens_nodes(text: str) -> list[dict]:
    return [{"name": name, "class": cls, "model": ld_model(cls), "label": MODELS[ld_model(cls)].label if ld_model(cls) else cls,
             "animated": any("curve" in v for v in k.values())} for name, cls, k in _ld(text)]


def _knob_value(node: str, knob: str, raw: str):
    """A knob's value: a number, or its keys as {frame: value} (a curve with a single key yields a number)."""
    v = raw.strip()
    while v.startswith("{") and v.endswith("}"):
        v = v[1:-1].strip()
    try:
        got = curve(v)
    except ValueError:
        raise Invalid(Msg("E-NUKE-LDVALUE", name=node, knob=knob, value=raw[:40])) from None
    if isinstance(got, dict):
        if not got:
            raise Invalid(Msg("E-NUKE-LDVALUE", name=node, knob=knob, value=raw[:40]))
        return next(iter(got.values())) if len(got) == 1 else got
    return got


def _scaled(value, factor: float):
    if isinstance(value, dict):
        keys = sorted(value.items())
        return {"frames": [f for f, _ in keys], "values": [v * factor for _, v in keys]}
    return value * factor


def lens(text: str, name: str, width: int, height: int) -> dict:
    found = _ld(text)
    hit = next(((cls, k) for n, cls, k in found if n == name), None)
    if hit is None:
        raise Invalid(Msg("E-NUKE-LDNODE", name=name, have=[n for n, _, _ in found] or ["没有"]))
    cls, knobs = hit
    model = ld_model(cls)
    if not model:
        raise Invalid(Msg("E-NUKE-LDMODEL", name=name, cls=cls, models=[MODELS[m].label for m in MODELS if m.startswith("3de4")]))
    by_norm = {_norm(k): k for k in knobs}
    params: dict = {}
    for p in MODELS[model].params:
        knob = by_norm.get(_norm(p.name))
        value = p.default if knob is None else _knob_value(name, knob, knobs[knob])
        for x in value.values() if isinstance(value, dict) else [value]:
            in_limit(float(x), p)
        params[p.name] = _scaled(value, 1.0)
    out: dict = {"model": model, "params": params, "direction": knobs.get("direction", "").strip("{} "), "node": name, "raster": [width, height]}

    def cm(knob: str):
        k = by_norm.get(_norm(knob))
        return None if k is None else _knob_value(name, k, knobs[k])

    focal = cm("tde4_focal_length_cm")
    if focal is not None:
        out["focal_mm"] = _scaled(focal, CM_TO_MM)
    w, h = cm("tde4_filmback_width_cm"), cm("tde4_filmback_height_cm")
    if isinstance(w, float) and isinstance(h, float):
        out["filmback_mm"] = [w * CM_TO_MM, h * CM_TO_MM]
    x, y = cm("tde4_lens_center_offset_x_cm"), cm("tde4_lens_center_offset_y_cm")
    if isinstance(x, float) or isinstance(y, float):
        out["center_mm"] = [float(x or 0.0) * CM_TO_MM, float(y or 0.0) * CM_TO_MM]
    aspect, focus = cm("tde4_pixel_aspect"), cm("tde4_custom_focus_distance_cm")
    if isinstance(aspect, float):
        out["pixel_aspect"] = aspect
    if isinstance(focus, float):
        out["focus_cm"] = focus
    return out


# ---- Camera nodes


def _cameras(text: str) -> list[tuple[str, dict[str, str]]]:
    return [(k.get("name", f"{cls}{i + 1}"), k) for i, (cls, k) in enumerate((c, k) for c, k in nodes(text) if c.startswith("Camera"))]


def keyed_frames(knobs: dict[str, str]) -> list[int]:
    """The frames on which a camera has keys; empty for an unkeyed camera, which is constant over any range."""
    frames = set()
    for key in ("translate", "rotate", "focal", "haperture"):
        for g in _groups(knobs.get(key, "")) or []:
            got = curve(g)
            if isinstance(got, dict):
                frames |= set(got)
    return sorted(frames)


def camera(text: str, name: str, frames: list[int], unit: str) -> dict:
    scale = scene_cm(unit)
    found = _cameras(text)
    knobs = next((k for n, k in found if n == name), None)
    if knobs is None:
        raise Invalid(Msg("E-NUKE-NOCAMERAINFILE", name=name, have=[n for n, _ in found] or ["没有"]))
    order = knobs.get("rot_order", "ZXY").strip("{}").strip() or "ZXY"
    if sorted(order.upper()) != ["X", "Y", "Z"] or knobs.get("xform_order", "SRT").strip() not in ("SRT", ""):
        raise Invalid(Msg("E-NUKE-ORDER", name=name, rot=order, xform=knobs.get("xform_order", "SRT")))
    t = _channel(knobs.get("translate", ""), 3, frames, (0.0, 0.0, 0.0)) * scale
    rot = _channel(knobs.get("rotate", ""), 3, frames, (0.0, 0.0, 0.0))
    focal = _channel(knobs.get("focal", ""), 1, frames, (50.0,))[:, 0]
    h_ap = _channel(knobs.get("haperture", ""), 1, frames, (24.576,))[:, 0]
    m = np.repeat(np.eye(4)[None], len(frames), 0)
    m[:, :3, :3] = euler(rot, order)
    m[:, :3, 3] = t
    keyed = sorted({f for key in ("translate", "rotate", "focal") for g in (_groups(knobs.get(key, "")) or [])
                    for f in (curve(g).keys() if isinstance(curve(g), dict) else [])})
    return {"c2w": m, "focal_mm": focal, "h_aperture_mm": h_ap, "keyed": keyed_frames(knobs),
            "source": {"format": "nk", "camera": name, "unit": unit, "rot_order": order, "keys": format_frame_range(keyed) if keyed else "常数"}}
