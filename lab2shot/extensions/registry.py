"""Find adapters: every adapters/<name>/extension.py defining EXTENSION (loaded and checked in lab2shot/adapters.py)."""

from __future__ import annotations

from functools import cache

from ..adapters import adapters
from ..errors import NotFound
from ..messages import Msg
from .spec import Extension


@cache
def extensions() -> dict[str, Extension]:
    """Every extension that loads. A broken one (e.g. half-edited) is reported in broken_extensions() and left out,
    so it cannot take the others down."""
    return dict(adapters().extensions)


def broken_extensions() -> dict[str, str]:
    """Adapter folder -> why its extension, or its nodes, did not load."""
    return dict(adapters().broken)


def get_extension(name: str) -> Extension:
    exts = extensions()
    if name not in exts:
        raise NotFound(Msg("E-EXT-UNKNOWN", name=name, known=list(exts)) if exts else Msg("E-EXT-NONE", name=name))
    return exts[name]
