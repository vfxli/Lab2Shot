"""Names and paths the plugin makes: cleaned, bounded, never clashing (安全守则「名字和路径」).

Every name the plugin creates — a scene object, a namespace, a group, a file, a folder — is cleaned first: ASCII
letters, digits and underscores only, starting with a letter; a node name at most NODE_MOST characters, a file name
at most FILE_MOST, a whole path within PATH_MOST (Windows' limit, with room). A name that is taken gets _002, _003 …
until one is free; nothing is ever overwritten. The original text (a Chinese name from the server, a character's own
name) is kept by the caller beside the object (a string attribute), never used as the name itself.

The user's own names are never cleaned: what the user's scene holds is read and written as it is.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import unicodedata

NODE_MOST = 60
FILE_MOST = 80
PATH_MOST = 240
_NOT_WORD = re.compile(r"[^A-Za-z0-9_]+")
_RESERVED = re.compile(r"(?i)^(con|prn|aux|nul|com[0-9]|lpt[0-9])$")


def _ascii(text: str) -> str:
    """Accented letters lose their accents (é → e); anything else not ASCII goes."""
    return unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")


def clean(text, most: int = NODE_MOST, fallback: str = "item") -> str:
    """A safe name: [A-Za-z][A-Za-z0-9_]*, at most `most` characters. A text with nothing usable (all Chinese, emoji)
    becomes `fallback` with a short hash of the text, so two different texts never clean to the same name."""
    raw = str(text or "")
    word = _NOT_WORD.sub("_", _ascii(raw)).strip("_")
    word = re.sub(r"_+", "_", word)
    if not word or len(word) < len(raw.strip()) / 3:  # mostly lost: keep it apart from other texts by a hash
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
        word = f"{word}_{digest}" if word else f"{fallback}_{digest}"
    if not (word[0].isascii() and word[0].isalpha()):
        word = f"{fallback[:1] or 'n'}_{word}"
    if _RESERVED.match(word):
        word += "_"
    if len(word) > most:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
        word = word[: most - 7].rstrip("_") + "_" + digest
    return word


def clean_file(name: str, most: int = FILE_MOST) -> str:
    """A safe file name: its stem cleaned (clean), its extension kept when it is a plain one."""
    stem, ext = os.path.splitext(str(name or ""))
    ext = ext if re.match(r"^\.[A-Za-z0-9]{1,8}$", ext or "") else ""
    # a sequence's frame number stays as it is: plate.1001.png → plate.1001.png
    frame = re.search(r"\.(\d+)$", stem)
    tail = f".{frame.group(1)}" if frame else ""
    stem = stem[: frame.start()] if frame else stem
    return clean(stem, max(8, most - len(ext) - len(tail)), "file") + tail + ext


def numbered(base: str, taken, most: int = NODE_MOST, first_plain: bool = True) -> str:
    """`base` itself when free, else base_002, base_003 … — the first one `taken(name)` says is free. Always within
    `most` characters (the base is shortened to make room for the number)."""
    if first_plain and not taken(base[:most]):
        return base[:most]
    for i in range(2, 100000):
        suffix = f"_{i:03d}"
        name = base[: most - len(suffix)] + suffix
        if not taken(name):
            return name
    raise RuntimeError("no free name")


def free_path(folder: str, name: str) -> str:
    """A path in `folder` for `name` (cleaned) that does not exist yet (numbered when taken)."""
    stem, ext = os.path.splitext(clean_file(name))
    chosen = numbered(stem, lambda s: os.path.exists(os.path.join(folder, s + ext)), FILE_MOST - len(ext))
    return os.path.join(folder, chosen + ext)


def path_fits(path: str) -> bool:
    return len(os.path.abspath(path)) <= PATH_MOST


def writable(folder: str) -> str:
    """"" when the plugin can write into `folder` (made if missing); else why not (in words)."""
    try:
        os.makedirs(folder, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".l2s_probe_", dir=folder)
        os.close(fd)
        os.remove(probe)
        return ""
    except OSError as exc:
        from .paths import text

        return text("dcc.safety.not_writable", folder=folder, error=exc.strerror or exc)


def free_bytes(folder: str) -> int:
    probe = folder
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or ".").free
    except OSError:
        return 0


def under(folder: str, rel: str) -> str | None:
    """A relative path from the server, cleaned part by part, inside `folder` (None when it would lead out)."""
    parts = [p for p in str(rel or "").replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    clean_parts = [clean(p, FILE_MOST, "dir") for p in parts[:-1]] + [clean_file(parts[-1])]
    target = os.path.abspath(os.path.join(folder, *clean_parts))
    root = os.path.abspath(folder)
    return target if target.startswith(root + os.sep) else None
