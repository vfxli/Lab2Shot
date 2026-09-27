"""SHA-256 digests for fingerprinted data; the single digest implementation in the codebase. Parts are hashed
sequentially in the given order: bytes as-is, text as UTF-8, and files by their bytes read in chunks (never loaded
whole into memory). A digest over a name followed by a file therefore equals hashing the two in sequence.

    sha256(*parts)   hex digest of the parts in order
    key(value, n)    short deterministic name for a value: the first n hex characters of the sha256 of its canonical
                     JSON (sorted keys; values JSON cannot represent are serialized as text). Used for cache folders,
                     staged layers and record file names; equal values always yield equal keys.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHUNK = 1 << 22  # bytes read from a file per chunk


def sha256(*parts: bytes | str | Path) -> str:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, Path):
            with open(part, "rb") as f:
                for chunk in iter(lambda: f.read(CHUNK), b""):
                    h.update(chunk)
        else:
            h.update(part.encode("utf-8") if isinstance(part, str) else part)
    return h.hexdigest()


def key(value: object, n: int) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))[:n]


def files(folder: Path) -> list[tuple[str, str]]:
    """Return every file under `folder` as (path relative to the folder, sha256), sorted by path."""
    folder = Path(folder)
    return [(p.relative_to(folder).as_posix(), sha256(p)) for p in sorted(folder.rglob("*")) if p.is_file()]


def tree(path: Path) -> str:
    """Return a file's sha256; for a folder, the sha256 over each file's relative path and sha256 in path order."""
    path = Path(path)
    if not path.is_dir():
        return sha256(path)
    return sha256(*(part for rel, digest in files(path) for part in (rel, digest)))
