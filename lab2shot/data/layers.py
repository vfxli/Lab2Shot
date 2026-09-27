"""Layers of an EXR (2D data is per-frame named layers, one to one with the layers of an EXR): a file's
channels grouped the way Nuke groups them, what a layer is taken as, read in its type's contract, and typed 2D
results written back as the layers of one EXR, segmentations as Cryptomatte.

Grouping: channel "diffuse.R" is R of layer "diffuse" (everything before the last dot names the layer); the bare R,
G, B, A (and Y, a grey picture) are "rgba"; a bare Z is "depth" (Nuke writes depth.Z, many renderers a bare Z); any
other bare channel is a layer of its own. Cryptomatte's CryptoObject00, CryptoObject01 ... (named in the file's
header) are one layer, CryptoObject. Nuke's motion vectors, forward.u/v and backward.u/v, are one layer, motion (and
<name>_forward / <name>_backward one layer <name>, as 「多层 EXR 输出」 writes a second set).

Cryptomatte (the Psyop specification 1.2): a name's id is MurmurHash3_32 of its UTF-8 bytes (seed 0), stored as the
float32 with those bits (exponent kept off 0 and 255, so it is never denormal, infinite or NaN); the layer <name>00
holds per pixel the id and coverage of the two objects covering it most (R, G and B, A); the header lists the names:
cryptomatte/<key>/name, hash, conversion and manifest ({name: 8-digit hex id}), <key> the first 7 hex digits of the
layer name's hash.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..io import images
from ..errors import Invalid
from ..messages import Msg
from ..data.payloads import CHANNEL_LETTERS, VALIDITY

RGBA = "rgba"
MOTION = "motion"  # Nuke's layer of motion vectors: forward.u, forward.v, backward.u, backward.v
HEADER = "layers"  # lab2shot:layers in an EXR written by this project: per layer its type and description (scale, space, classes)

_ALIASES = {"red": "R", "green": "G", "blue": "B", "alpha": "A", "r": "R", "g": "G", "b": "B", "a": "A",
            "x": "X", "y": "Y", "z": "Z", "u": "U", "v": "V", "w": "W"}
_ORDER = "RGBXYZUVW"  # channel order inside a layer; then A, then the rest by name
# common ST-map layer names (preferred by stmap_layout when choosing a channel pair); no meaning is guessed from other names
STMAP_NAMES = ("stmap", "st", "distort", "undistort", "distortion")


def role(channel: str) -> str:
    """What a channel is inside its layer: "depth.Z" -> Z, "diffuse.red" -> R."""
    short = channel.rsplit(".", 1)[-1]
    return _ALIASES.get(short.lower(), short)


def _order(channel: str) -> tuple[int, str]:
    r = role(channel)
    return (_ORDER.index(r) if r in _ORDER else len(_ORDER) + (r != "A"), r)


def group(names: list[str], crypto: list[str] = ()) -> dict[str, list[str]]:
    """Channel names -> {layer: its channels}: rgba first, depth next, the rest as they come; channels in R G B / X Y Z
    / U V order, then A. `crypto`: the Cryptomatte layers the header names (their preview channels are left out)."""
    prefixed = {n.rsplit(".", 1)[0] for n in names if "." in n}
    layers: dict[str, list[str]] = {}
    for n in names:
        if "." in n:
            layer = n.rsplit(".", 1)[0]
        elif role(n) in ("R", "G", "B", "A", "Y"):
            layer = RGBA
        elif role(n) == "Z" and "depth" not in prefixed:
            layer = "depth"
        else:
            layer = n
        for c in crypto:
            if re.fullmatch(re.escape(c) + r"\d\d", layer):
                layer = c
            elif layer == c:  # the preview picture next to a Cryptomatte's ranks
                layer = ""
        if layer:
            layers.setdefault(layer, []).append(n)
    layers = {x: sorted(chs, key=None if x in crypto else _order) for x, chs in layers.items()}
    layers = _motion_layers(layers)
    first = [x for x in (RGBA, "depth") if x in layers]
    return {x: layers[x] for x in first + [x for x in layers if x not in first]}


def _motion_layers(layers: dict[str, list[str]]) -> dict[str, list[str]]:
    """Nuke's forward and backward layers (u, v each) as one layer of motion vectors, where the forward one was:
    forward.u/v + backward.u/v -> motion; <name>_forward + <name>_backward -> <name>."""
    def uv(name: str) -> bool:
        return sorted(role(c) for c in layers.get(name, ())) == ["U", "V"]

    pairs = {name.removesuffix("forward") for name in layers
             if name.endswith("forward") and uv(name) and uv(name.removesuffix("forward") + "backward")}
    out: dict[str, list[str]] = {}
    for name, channels in layers.items():
        if name.endswith("forward") and name.removesuffix("forward") in pairs:
            prefix = name.removesuffix("forward")
            out[prefix.removesuffix("_") or MOTION] = channels + layers[prefix + "backward"]
        elif not (name.endswith("backward") and name.removesuffix("backward") in pairs):
            out[name] = channels
    return out


def is_motion(channels: list[str]) -> bool:
    """A layer _motion_layers made: forward u, v then backward u, v."""
    parts = [c.rsplit(".", 1)[0] for c in channels]
    return len(channels) == 4 and all(p.endswith("forward") for p in parts[:2]) and all(p.endswith("backward") for p in parts[2:])


STMAP_DIRECTIONS = ("undistort", "distort")  # what an ST-map does


def stmap_direction_of_name(name: str) -> str:
    """The direction a file or layer name says, as 3DE and Nuke name them (lens_undistort.exr, redistort): "undistort",
    "distort", or "" when it says neither."""
    n = name.lower()
    if "undistort" in n:
        return "undistort"
    return "distort" if "distort" in n else ""


def stmap_layout(channels: list[str], direction: str = "") -> tuple[list[str], str]:
    """The one rule for which channels of a file hold an ST-map: the (u, v) channels
    and the direction the channels themselves say ("" when they say none).

    - NukeX LensDistortion's motion layer: forward.u / forward.v undistort, backward.u / backward.v distort (the channel
      names decide; with both, `direction` picks, else forward);
    - otherwise one layer's pair (the channels grouped as group() groups them): R and G (3DE's ST-maps, Nuke's STMap,
      this project's) or u and v (stmap.u, stmap.v); a layer named for an ST-map (stmap, st, undistort ...) first, then a u v
      pair, then the picture's own R G. Several pairs left alike, or none, are refused (E-LAYER-STMAPCHANNELS): an
      ST-map is never taken from whichever two channels come first."""
    def uv(side: str) -> list[str]:
        got = {role(c): c for c in channels if c.rsplit(".", 1)[0].lower().endswith(side) and role(c) in ("U", "V")}
        return [got["U"], got["V"]] if len(got) == 2 else []

    forward, backward = uv("forward"), uv("backward")
    if forward or backward:
        if backward and (not forward or direction == "distort"):
            return backward, "distort"
        return forward, "undistort"
    pairs = []  # (rank, the pair): lower is preferred
    for layer, members in group(list(channels)).items():
        roles = {role(c): c for c in members}
        named = re.sub(r"[^a-z0-9]", "", layer.rsplit(".", 1)[-1].lower()) in STMAP_NAMES
        for rank, (a, b) in ((1, ("U", "V")), (2, ("R", "G"))):
            if a in roles and b in roles:
                pairs.append((0 if named else rank, [roles[a], roles[b]]))
                break
    best = sorted(pairs, key=lambda p: p[0])
    if best and (len(best) == 1 or best[0][0] < best[1][0]):
        return best[0][1], ""
    raise Invalid(Msg("E-LAYER-STMAPCHANNELS", channels=list(channels)))


def channels_named(channels: list[str], picked: list[str]) -> list[str]:
    """Resolve the channel names the user picked (`R`, `Z`, `u`: the file's short names) to the full names in this layer
    (`depth.Z`, `forward.u`), in the order given. A name the layer lacks is refused (E-LAYER-NOCHANNEL)."""
    have: dict[str, str] = {}
    for c in channels:
        have.setdefault(role(c), c)
        have.setdefault(c, c)
    out = []
    for name in picked:
        got = have.get(name) or have.get(role(name))
        if got is None:
            raise Invalid(Msg("E-LAYER-NOCHANNEL", channel=name, channels=" ".join(role(c) for c in channels)))
        out.append(got)
    return out


def read(path, channels: list[str], crypto: dict | None = None, view: str | None = None,
         box: tuple | None = None, says_valid: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """One frame of a layer: (values [H,W,C], validity or None). The named channels are read as they are, without
    conversion by meaning: a channel's meaning is defined by its use, so no conversion to centimetres, normalisation or
    clipping to 0..1 is done here; that is the consuming node's responsibility.

    Non-finite values count as having no value. The `valid` channel in EXRs written by this project marks where values
    exist and is not a data channel; in multi-layer EXRs delivered to Nuke it is `<layer>.A` (Nuke's convention), and
    with `says_valid` the last channel is read as validity too.
    `crypto` {"manifest", "classes"}: the file's metadata marks the layer as Cryptomatte; it is decoded to a label map.
    `box`: the region of the file to read (the packet's data window, data/windows.py Window); None for the format."""
    if crypto is not None:
        return crypto_labels(images.read_named(path, list(crypto_rank(channels)), view, box=box), crypto)[..., None], None
    said = next((c for c in channels if role(c) == VALIDITY), None)
    if said is None and says_valid and len(channels) > 1:
        said = channels[-1]  # a multi-layer EXR delivered by this project: the last channel A marks validity, recorded in the header (HEADER validity)
    data_ch = [c for c in channels if c != said]
    named = images.read_named(path, sorted(set(channels)), view, box=box)
    data = np.stack([named[c] for c in data_ch], axis=-1)
    finite = np.isfinite(data).all(axis=-1)
    data = np.where(finite[..., None], data, 0.0).astype(np.float32)
    if said is None:
        return data, None if bool(finite.all()) else finite.astype(np.float32)
    return data, np.where(finite, np.clip(named[said], 0, 1), 0.0).astype(np.float32)


def describe_file(path, view: str | None = None) -> dict[str, dict]:
    """Every layer of the file: {layer: {"channels", plus what the file itself states}}.

    No meaning is guessed for a layer; channels are channels. EXRs written by this project record scale, coordinate
    system, projection and direction in the header (lab2shot:layers), which are copied as stated; Cryptomatte is
    likewise identified from the file's own metadata, not guessed."""
    attrs = images.header_attributes(path)
    manifests = crypto_manifests(attrs, Path(path).parent)
    sidecars = crypto_sidecars(attrs)
    told = json.loads(attrs.get(f"lab2shot:{HEADER}", "{}"))
    out = {}
    for layer, channels in group(images.channel_names(path, view), list(manifests)).items():  # one view of a stereo file
        info = {"channels": channels, **{k: v for k, v in told.get(layer, {}).items() if k != "type"}}
        if layer in manifests:
            # a manifest kept beside the file (manif_file) and not uploaded with it, or none: the objects by the ids in
            # the picture, unnamed (the node says which file would name them: missing_manifest)
            manifest = manifests[layer] or crypto_seen(path, channels, view)
            if not manifests[layer] and layer in sidecars:
                info["missing_manifest"] = sidecars[layer]
            info["manifest"] = manifest
            info.setdefault("classes", [{"index": k + 1, "name": n} for k, n in enumerate(sorted(manifest))])
        out[layer] = info
    return out


# ------------------------------------------------------------------ Cryptomatte


def murmur3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3 x86 32-bit (Austin Appleby), as Cryptomatte names its ids."""
    c1, c2, mask = 0xCC9E2D51, 0x1B873593, 0xFFFFFFFF
    h = seed & mask
    n = len(data) // 4 * 4
    for i in range(0, n, 4):
        k = int.from_bytes(data[i:i + 4], "little")
        k = (k * c1) & mask
        k = ((k << 15) | (k >> 17)) & mask
        k = (k * c2) & mask
        h ^= k
        h = ((h << 13) | (h >> 19)) & mask
        h = (h * 5 + 0xE6546B64) & mask
    tail = data[n:]
    if tail:
        k = int.from_bytes(tail, "little")
        k = (k * c1) & mask
        k = ((k << 15) | (k >> 17)) & mask
        k = (k * c2) & mask
        h ^= k
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & mask
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & mask
    h ^= h >> 16
    return h


def crypto_hash(name: str) -> int:
    return murmur3_32(name.encode("utf-8"))


def crypto_bits(h: int) -> int:
    """A name's hash -> the bits of the float32 id (uint32_to_float32: exponent kept within 1..254)."""
    exponent = min(max((h >> 23) & 0xFF, 1), 254)
    return (h & 0x80000000) | (exponent << 23) | (h & 0x7FFFFF)


def crypto_id(name: str) -> np.float32:
    return np.array([crypto_bits(crypto_hash(name))], np.uint32).view(np.float32)[0]


def crypto_key(layer: str) -> str:
    return f"{crypto_hash(layer):08x}"[:7]


def crypto_channels(layer: str) -> list[str]:
    """One rank pair: each pixel belongs to one object of a segmentation (R id, G coverage; B, A the second rank)."""
    return [f"{layer}00.{c}" for c in "RGBA"]


def crypto_header(layer: str, names: list[str]) -> dict[str, str]:
    key = f"cryptomatte/{crypto_key(layer)}"
    return {f"{key}/name": layer, f"{key}/hash": "MurmurHash3_32", f"{key}/conversion": "uint32_to_float32",
            f"{key}/manifest": json.dumps({n: f"{crypto_hash(n):08x}" for n in names}, ensure_ascii=False)}


def class_names(classes: list[dict]) -> dict[int, str]:
    """Label -> a name unique within the table (Cryptomatte tells objects apart by their names)."""
    out: dict[int, str] = {}
    for c in classes:
        name = str(c.get("name") or f"id {c['index']}")
        while name in out.values():
            name = f"{name} {c['index']}"
        out[int(c["index"])] = name
    return out


def crypto_encode(labels: np.ndarray, classes: list[dict]) -> np.ndarray:
    """A segmentation frame (labels [H,W], 0 = nothing) -> the <layer>00 RGBA of Cryptomatte."""
    names = class_names(classes)
    labels = np.rint(labels).astype(np.int64)
    ids = np.zeros(labels.shape, np.float32)
    for label in np.unique(labels):
        if label > 0:
            ids[labels == label] = crypto_id(names.get(int(label), f"id {label}"))
    out = np.zeros((*labels.shape, 4), np.float32)
    out[..., 0], out[..., 1] = ids, (labels > 0)
    return out


def crypto_manifests(attrs: dict[str, str], folder: Path | None = None) -> dict[str, dict[str, str]]:
    """The Cryptomatte layers a header describes: {layer: {name: hex id}}. <key> is any short string (Nuke's
    Cryptomatte reads xc7bccc as well as hex digits). A manifest kept beside the file (manif_file) is read from
    `folder` only when it was uploaded with the file, never looked for elsewhere; one not there,
    or none at all, is empty."""
    out = {}
    sidecars = crypto_sidecars(attrs)
    for key, value in attrs.items():
        m = re.fullmatch(r"cryptomatte/([^/]+)/name", key)
        if m:
            out[value] = json.loads(attrs.get(f"cryptomatte/{m.group(1)}/manifest", "{}"))
            beside = folder / sidecars[value] if folder is not None and value in sidecars else None
            if not out[value] and beside is not None and beside.is_file() and beside.parent == folder:
                out[value] = json.loads(beside.read_text(encoding="utf-8"))
    return out


def crypto_sidecars(attrs: dict[str, str]) -> dict[str, str]:
    """{layer: the file name its manifest is kept in}, for the layers whose header names one (manif_file)."""
    names = {m.group(1): v for k, v in attrs.items() if (m := re.fullmatch(r"cryptomatte/([^/]+)/name", k))}
    return {names[m.group(1)]: Path(v).name for k, v in attrs.items()
            if (m := re.fullmatch(r"cryptomatte/([^/]+)/manif_file", k)) and m.group(1) in names}


def crypto_rank(channels) -> tuple[str, str]:
    """The id and coverage channels of a Cryptomatte's first rank (<layer>00.R / .G, or .red / .green as most renderers
    name them)."""
    rank = [c for c in channels if c.rsplit(".", 1)[0].endswith("00")]
    ident, cover = next((c for c in rank if role(c) == "R"), None), next((c for c in rank if role(c) == "G"), None)
    if ident is None or cover is None:
        raise Invalid(Msg("E-LAYER-CHANNELS", count=2, have=len(rank), channels=list(channels)))
    return ident, cover


def crypto_seen(path, channels, view: str | None = None) -> dict[str, str]:
    """A Cryptomatte without its manifest in the header: {"id <hex>": hex} for every id the picture's first rank
    covers, so its objects still stay apart (Nuke's Cryptomatte picks them by id alike)."""
    ident, cover = crypto_rank(channels)
    named = images.read_named(path, [ident, cover], view)
    ids = np.ascontiguousarray(named[ident], np.float32).view(np.uint32)[named[cover] > 0]
    return {f"id {v:08x}": f"{v:08x}" for v in np.unique(ids) if v}


def crypto_labels(named: dict[str, np.ndarray], crypto: dict) -> np.ndarray:
    """A Cryptomatte frame -> labels [H,W]: per pixel the class of the object covering it most (rank 0), 0 where
    nothing does or the id is not in the manifest. `crypto`: {"manifest": {name: hex}, "classes": [{index, name}]}."""
    ident, cover = crypto_rank(list(named))
    rank_id, coverage = named[ident], named[cover]
    index = {name: c for c, name in class_names(crypto["classes"]).items()}
    bits = {crypto_bits(int(h, 16)): index[name] for name, h in crypto["manifest"].items() if name in index}
    ids = np.ascontiguousarray(rank_id, np.float32).view(np.uint32)
    labels = np.zeros(ids.shape, np.float32)
    for b, label in bits.items():
        labels[(ids == b) & (coverage > 0)] = label
    return labels


# ------------------------------------------------------------------ writing several results into one EXR
#
# What a layer is matters only when delivering to other software: Nuke recognises names such as depth.Z. Packets in
# this project have no such notion (2D data has only a channel count), so this table exists only here, keyed by output
# port name rather than data type.

# Output port name -> the conventional EXR layer name for delivery. Ports not listed use their own name as the layer
# name (stmap, confidence, roughness, disparity, ... are already industry terms).
# The optical flow port is `flow` (`motion` is reserved for the 「线性蒙皮变形」 animation), but its layer is Nuke's
# `motion`, the name Nuke recognises (a different name would produce `<name>_forward.u` and lose Nuke's motion layer).
LAYER_FOR_PORT = {"image": "rgba", "alpha": "mask", "mask": "mask", "normal": "N", "position": "P", "flow": MOTION}
# Layer name -> its channel names. rgba / depth / forward / backward / mask / disparityL / disparityR are Nuke's
# built-in layers (Python Developer's Guide, "Working with Channels and Layers"; case as documented: depth uses upper-case
# Z, forward / backward lower-case u v). motion is the forward / backward pair written when delivering 「运动矢量」.
NUKE_CHANNELS = {
    RGBA: ("R", "G", "B", "A"), "depth": ("Z",), "forward": ("u", "v"), "backward": ("u", "v"), "mask": ("a",),
    "disparityL": ("x", "y"), "disparityR": ("x", "y"),
    MOTION: ("forward.u", "forward.v", "backward.u", "backward.v"),
}


def layer_for_port(port: str) -> str:
    """The EXR layer an output port is delivered as (the port's own name when there is no conventional name)."""
    return LAYER_FOR_PORT.get(port, port)


# fallback layer name by channel count when a row names none (the page normally fills it in from the source port)
BY_COUNT = {1: "float", 2: "uv", 3: "rgb", 4: RGBA}


def layer_names(given: list[str], counts: list[int]) -> list[str]:
    """A layer name per result: the one the row gives (the page fills it in from the wired port: rgba, depth, N, P,
    mask, stmap ...), else one by channel count; a name taken already gets a number (mask, mask2)."""
    out: list[str] = []
    for i, count in enumerate(counts):
        base = (given[i].strip() if i < len(given) else "") or BY_COUNT.get(count, "layer")
        name, n = base, 2
        while name in out:
            name, n = f"{base}{n}", n + 1
        out.append(name)
    return out


# what a row of 「多层 EXR 输出设置」's 图层 table may name its channels (a Nuke channel name: letters, digits and _)
CHANNEL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def row_channels(layer: str, count: int, labels: bool = False) -> list[str]:
    """The default channels a layer is written with: a layer with a class table as Cryptomatte (channels fixed by the
    specification); a Nuke built-in layer name with its own channels (depth Z, forward u v, ..., when the count
    matches); otherwise the first channels of R G B A by count."""
    if labels:
        return [role(c) for c in crypto_channels("x")]  # Cryptomatte's own rank channels
    want = NUKE_CHANNELS.get(layer)
    return list(want) if want and len(want) == count else list(CHANNEL_LETTERS[:count])


def checked_channels(layer: str, count: int, labels: bool, given: str | list[str] | None) -> list[str]:
    """The channel names a row writes: the ones it names (spaces or commas between them), checked, or the layer's own
    when it names none. As many as the data has channels, each a Nuke channel name, none twice; a Cryptomatte's and
    Nuke's motion channels are that software's standard and are refused if changed."""
    names = [n for n in re.split(r"[\s,]+", given) if n] if isinstance(given, str) else list(given or ())
    standard = row_channels(layer, count, labels)
    if not names or names == standard:  # nothing said, or the layer's own spelled out (the editor fills a new row in)
        return standard
    if labels or layer == MOTION:
        raise Invalid(Msg("E-LAYER-FIXEDCHANNELS", kind=layer, channels=" ".join(standard)))
    if len(names) != len(standard) or len(set(names)) != len(names) or not all(CHANNEL_NAME.fullmatch(n) for n in names):
        raise Invalid(Msg("E-LAYER-CHANNELNAMES", kind=layer, count=len(standard), given=" ".join(names)))
    return names


def layer_channels(layer: str, names: list[str], said: bool = False, labels: bool = False) -> list[str]:
    """The full channel names written to the EXR: the row's channel names (`names`) under the layer name (depth.Z,
    N.R, ...), plus a channel A marking valid pixels when `said` (a picture's alpha uses the same channel).
    The rgba layer's channels are bare (R G B A); segmentations use Cryptomatte's own set;
    motion vectors use Nuke's own (forward.u ... backward.v, or <name>_forward.u ... under another layer name)."""
    if labels:
        return crypto_channels(layer)
    if set(names) == set(NUKE_CHANNELS[MOTION]):
        return [c if layer == MOTION else f"{layer}_{c}" for c in names]
    letters = list(names)
    if said:
        letters.append("A")
    return letters if layer == RGBA else [f"{layer}.{c}" for c in letters]
