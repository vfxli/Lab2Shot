"""What a delivered OpenEXR file holds, read from its header with the standard library only: its channels and the
`lab2shot:layers` attribute the server writes (each layer's meaning: an ST map's direction, a depth's scale, the
colour space of its pictures). Nothing is decoded; only the header is read."""

from __future__ import annotations

import json
import struct

MAGIC = 20000630
HEADER_MOST = 1 << 20


def header(path: str) -> dict:
    """{"channels": [names], "layers": {layer: meta}} of the first part; {} when the file is not an OpenEXR."""
    with open(path, "rb") as f:
        data = f.read(HEADER_MOST)
    if len(data) < 8 or struct.unpack("<i", data[:4])[0] != MAGIC:
        return {}
    pos, out = 8, {"channels": [], "layers": {}}
    while pos < len(data):
        end = data.index(b"\0", pos)
        name = data[pos:end].decode("latin-1")
        pos = end + 1
        if not name:
            break
        end = data.index(b"\0", pos)
        kind = data[pos:end].decode("latin-1")
        pos = end + 1
        size = struct.unpack("<i", data[pos:pos + 4])[0]
        pos += 4
        value = data[pos:pos + size]
        pos += size
        if kind == "chlist":
            i = 0
            while i < len(value) and value[i:i + 1] != b"\0":
                end = value.index(b"\0", i)
                out["channels"].append(value[i:end].decode("latin-1"))
                i = end + 1 + 16  # pixel type, pLinear + reserved, xSampling, ySampling
        elif name == "lab2shot:layers" and kind == "string":
            try:
                got = json.loads(value.decode("utf-8"))
                out["layers"] = got if isinstance(got, dict) else {}
            except ValueError:
                pass
    return out


def layer_names(channels: list[str]) -> list[str]:
    """The layers of the channels as Nuke names them (R, G, B, A alone: rgba)."""
    out = []
    for c in channels:
        layer = c.rsplit(".", 1)[0] if "." in c else "rgba"
        if layer not in out:
            out.append(layer)
    return out
