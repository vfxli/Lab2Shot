"""Names in 3D files: the one rule for turning a name (a joint, a blend shape, a subset, a mesh, a group, a curve
attribute — whatever the artist or the source file called it) into an identifier where a format needs one (USD prim and
property names, Alembic objects), and for telling apart siblings that come out the same.

identifier(): a name that already is an identifier stays as it is. Otherwise ASCII letters and digits stay, every other
ASCII character becomes "_" (readable in a DCC: Mixamo's mixamorig:Hips shows as mixamorig_Hips), every non-ASCII
character is spelled "_u" + four hex digits ("_U" + eight above U+FFFF), so 左眨眼 and 右眨眼 stay two names, and a
leading digit gets a "_" in front (USD names cannot start with one).

The identifier is never decoded: where it differs from the name, the writer keeps the name itself beside it (a
skeleton's jointNames, blend-shape tokens, customData ORIGINAL on a prim or a property — lab2shot/io/usd.py
keep_original / name_of), and readers take that or the identifier as it is. So a name from another tool that happens to
look like a spelling (Bone_u0041) reads as itself.

unique(): a name already taken among its siblings gets "_2", "_3"… — the one tie-break wherever names must differ
(USD prims, Alembic objects, a skeleton's jointNames: sibling_unique, so its joints read alike in every format). A name
is kept as given, spaces included. FBX takes any text and repeats, so its writer puts the original name itself on the
node, never an identifier or a tie-break.

join_path() / split_path(): a place in a hierarchy as one string of names (the scene arrays' `path` from a reader, their
`shown` for a writer), a "/" or "\\" inside a name escaped with "\\": a model named a/b stays one name, not two groups.

Pure standard library: the core (lab2shot/io/usd.py) and the format workers (adapters/fbx, adapters/alembic) import it."""

from __future__ import annotations

import re
from collections.abc import Container

ORIGINAL = "lab2shot:name"  # customData key holding a name where its identifier differs from it

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _spelled(c: str) -> str:
    return f"_u{ord(c):04x}" if ord(c) <= 0xFFFF else f"_U{ord(c):08x}"


def identifier(name: str) -> str:
    """`name` as an identifier ([A-Za-z_][A-Za-z0-9_]*), by the rule in the module docstring. "" gives "_"."""
    if _IDENTIFIER.fullmatch(name):
        return name
    out = "".join(c if c.isascii() and c.isalnum() else "_" if c.isascii() else _spelled(c) for c in name)
    return f"_{out}" if out[:1].isdigit() else out or "_"


def unique(base: str, taken: Container[str]) -> str:
    """`base`, or the first of base_2, base_3… that is not in `taken` (the caller records the one it uses)."""
    out, n = base, 1
    while out in taken:
        n += 1
        out = f"{base}_{n}"
    return out


def sibling_unique(names: list[str], parents) -> list[str]:
    """Each joint's name made unique among the joints of its parent (unique), in order: USD's jointNames, FBX's and
    Alembic's nodes come out alike."""
    taken: dict[int, set[str]] = {}
    out = []
    for name, parent in zip(names, parents):
        seen = taken.setdefault(int(parent), set())
        out.append(unique(str(name), seen))
        seen.add(out[-1])
    return out


def join_path(parts) -> str:
    """Names from the top down as one path, "/" and "\\" inside a name escaped."""
    return "/" + "/".join(str(p).replace("\\", "\\\\").replace("/", "\\/") for p in parts)


def split_path(path: str) -> list[str]:
    """join_path back: the names, top down (empty parts dropped)."""
    parts, cur, escaped = [], "", False
    for c in path:
        if escaped:
            cur, escaped = cur + c, False
        elif c == "\\":
            escaped = True
        elif c == "/":
            parts.append(cur)
            cur = ""
        else:
            cur += c
    parts.append(cur)
    return [p for p in parts if p]


def decode(data: bytes) -> tuple[str, bool]:
    """Text a file keeps names in, by the one policy for every format: UTF-8 (a BOM allowed); bytes that are not are
    replaced by U+FFFD and said (the bool: something was replaced), never guessed as another encoding — a guess
    turns Shift-JIS into Chinese without a word. The FBX binding does the same (fbxio.cpp text(), W-FBX-NAMEENCODING);
    PyAlembic cannot hand the bytes over at all, so an Alembic with such a name is refused (E-ALEMBIC-NAMEENCODING)."""
    try:
        return data.decode("utf-8-sig"), False
    except UnicodeDecodeError:
        return data.decode("utf-8-sig", errors="replace"), True  # a BOM goes either way, never into the text
