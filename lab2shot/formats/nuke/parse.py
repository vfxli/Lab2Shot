"""Reading a Nuke script (.nk, as Nuke and 3DEqualizer's Nuke export write it): its top-level nodes and their knobs,
a knob's constant or keyed curve, and 3DE's LD_3DE lens distortion nodes as a distortion.

This is the single Nuke parser in the program: 「LensDistortion」 reads a pasted lens with it
(nodes/core/lens_distortion.py read_pasted).

LD_3DE knobs are matched to the model's parameter names by comparing lowercase letters and digits only (3DE writes
spaces and punctuation in names as underscores); a knob absent from the script takes its default.
"""


from __future__ import annotations

import re


from ...data import units
from ...errors import Invalid
from ...messages import Msg
from ...data.lens_models import MODELS, in_limit

CM_TO_MM = units.UNITS["cm"].per_base / units.UNITS["mm"].per_base  # 3DE's built-in lens knobs are in centimetres


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
