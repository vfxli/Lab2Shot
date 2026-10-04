"""A node type's words: its name, its description and everything else it shows, looked up by key in the language now.

The words are in the language catalogues (lab2shot/i18n), never in code:

    lab2shot/i18n/<lang>/nodes.toml       the core's nodes (node.<type>.subtitle / .description / .param.<p>.… / .port.<n>.…)
                                          and the keys every node may fall back to (param.*, port.*, option.*, group.*,
                                          stage.*, button.*)
    adapters/<name>/i18n/<lang>.toml      an extension's nodes, and the shared keys for its own nodes only (`scope`)

`word(cls, …)` is the one lookup: node.<type>.<path…> with the shared fallbacks (i18n.node_text), the extension's own
first. A node with no name in the language now shows its type id; a word with no entry is None and the caller decides.

An account that manages node categories edits a node's name or description in the node menu (server/categories.py
menu_text): `save` writes it into the catalogue file of the language now that holds the node's words (the core's
nodes.toml, or the extension's <lang>.toml), in place: the table [node."<type>"] gets its subtitle / description lines,
the rest of the file is left as it is. A catalogue edited outside the server is read again when its modification time
changes (`refresh`, called before the catalogue route answers and before a save); `stamp` is what the answers worked
out from the words are cached under. Writes keep a .bak copy first and use atomic replacement (io/atomic.py).
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import tomllib
from pathlib import Path
from typing import Any

from .. import i18n
from ..errors import Invalid, NotFound
from ..io.atomic import write_text
from ..messages import Msg

# a name fits on one line of the node menu, a description is shown in full there: per language (English words are
# longer than Chinese characters); lab2shot check holds every catalogue to them (limit_problems)
SUBTITLE_CHARS = {"zh": 30, "en": 48}
TEXT_CHARS = {"zh": 520, "en": 1400}


def scope_of(cls) -> str | None:
    """The extension whose shared keys a node type's words fall back to first (None: a core node)."""
    runtime = getattr(cls, "runtime", "core")
    return None if runtime == "core" else runtime


def word(cls, *path: str, **params: Any) -> str | None:
    """One of a node type's words in the language now: node.<type>.<path…>, else the shared key it falls back to
    (param.*, port.*, option.*, stage.*, button.*, fact.*), the extension's own first; None when there is none."""
    type_id = getattr(cls, "id", "")
    if not type_id:
        return None
    return i18n.node_text(type_id, *path, in_scope=scope_of(cls), **params)


def param_label(cls, name: str) -> str:
    """How a message names one of its parameters: the label of its table (nodes/params.py dress), kept by its key
    (i18n.Word) so the message reads in whoever's language follows it; its name when it has no such parameter."""
    return next((p["label"] for p in cls.param_specs() if p["name"] == name), name)


def subtitle(cls) -> str:
    """Its subtitle in the language now (shown under its type name); its type id when it has none."""
    return word(cls, "subtitle") or getattr(cls, "id", "")


def description(cls) -> str:
    return word(cls, "description") or ""


def file_for(cls, lang: str | None = None) -> Path:
    """The catalogue file of language `lang` a node type's words are in: its extension's, or the core's nodes.toml."""
    lang = lang or i18n.current()
    scope = scope_of(cls)
    if scope:
        folder = i18n.extension_folders().get(scope)
        if folder is not None:
            return i18n.extension_file(folder, lang)
    return i18n.lang_dir(lang) / "nodes.toml"


def files() -> list[Path]:
    """Every catalogue file of every language (what a node's words may be in), those that are there."""
    out: list[Path] = []
    for lang in i18n.LANGS:
        out += i18n.word_files(lang)
        out += [p for f in i18n.extension_folders().values() if (p := i18n.extension_file(f, lang)).is_file()]
    return out


def entries(path: Path) -> dict[str, dict[str, str]]:
    """The nodes' names and descriptions one catalogue file holds: {type: {"subtitle": …, "description": …}}."""
    out: dict[str, dict[str, str]] = {}
    try:
        flat = i18n.read(path)
    except i18n.TextError:
        return {}
    for key, value in flat.items():
        parts = i18n.node_key_parts(key)
        if parts is not None and parts[1] in (["subtitle"], ["description"]):
            out.setdefault(parts[0], {})[parts[1][0]] = value
    return out


def stamp() -> tuple:
    """Every catalogue file with its modification time: what `refresh` compares and the answers worked out from the
    words are cached under."""
    out = []
    for f in files():
        try:
            out.append((str(f), f.stat().st_mtime_ns))
        except OSError:
            out.append((str(f), 0))
    return tuple(out)


_read_at: tuple | None = None  # stamp() of the catalogues as last read
_lock = threading.RLock()  # one writer of the catalogues at a time (requests run in a thread pool)


def refresh(types=None) -> tuple:
    """Read the catalogues again when one changed since (one edited by hand, restored, or saved here): the words
    looked up and the node tables dressed with them are forgotten. Returns the stamp the words now match (the
    caller's cache key). `types` is not needed (kept for the callers that hand the node types over)."""
    global _read_at
    with _lock:
        at = stamp()
        if at != _read_at:
            if _read_at is not None:
                i18n.clear()
                from .params import forget_words

                forget_words()
            _read_at = at
        return at


def problems() -> list[str]:
    """The catalogue files that do not read (not TOML, a key twice, a value that is not text)."""
    out = []
    for lang in i18n.LANGS:
        try:
            i18n.words(lang)
        except i18n.TextError as exc:
            out.append(str(exc))
    return out


def limit_problems() -> list[str]:
    """Every node's name and description, in every language, against the limits an edit is held to (_check)."""
    from .registry import node_types

    out = []
    for type_id, cls in sorted(node_types().items()):
        for lang in i18n.LANGS:
            with i18n.using(lang):
                said = word(cls, "subtitle")
                if said is None:
                    continue
                try:
                    _check(said, description(cls))
                except Invalid as exc:
                    out.append(f"{lang} {type_id}: {exc}")
    return out


def _check(subtitle: str, description: str) -> tuple[str, str]:
    """Within the limits of the language now."""
    lang = i18n.current()
    subtitle, description = subtitle.strip(), description.strip()
    if not subtitle or len(subtitle) > SUBTITLE_CHARS[lang]:
        raise Invalid(Msg("E-NODETEXT-SUBTITLE", most=SUBTITLE_CHARS[lang]))
    if len(description) > TEXT_CHARS[lang]:
        raise Invalid(Msg("E-NODETEXT-TEXT", most=TEXT_CHARS[lang]))
    return subtitle, description


def _string(text: str) -> str:
    """Text as a TOML basic string (JSON's escapes are TOML's)."""
    return json.dumps(text, ensure_ascii=False)


def with_words(source: str, type_id: str, subtitle: str, description: str) -> str:
    """A catalogue's text with the table [node."<type>"] holding this subtitle and description: its lines replaced where
    they are, added under the table's header when not, the table added at the end when there is none."""
    header = re.compile(r'^\[node\.(?:"' + re.escape(type_id) + r'"|' + re.escape(type_id) + r')\]\s*$')
    lines = source.splitlines()
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    new = {"subtitle": f"subtitle = {_string(subtitle)}", "description": f"description = {_string(description)}"}
    if start is None:
        return source.rstrip("\n") + f'\n\n[node.{_string(type_id)}]\n{new["subtitle"]}\n{new["description"]}\n'
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    for key in ("subtitle", "description"):
        at = next((i for i in range(start + 1, end) if re.match(rf"^{key}\s*=", lines[i])), None)
        if at is None:
            lines.insert(start + 1, new[key])
            end += 1
        else:
            lines[at] = new[key]
    return "\n".join(lines) + "\n"


def save(type_id: str, subtitle: str, description: str) -> tuple[dict, bool]:
    """A node's subtitle and description in the language now, written to its catalogue file and used at once. Returns the
    entry and whether anything changed."""
    from .registry import node_types

    types = node_types()
    cls = types.get(type_id)
    if cls is None:
        raise NotFound(Msg("E-NODE-NOSUCH", type=type_id))
    subtitle, description = _check(subtitle, description)
    lang = i18n.current()
    path = file_for(cls, lang)
    entry = {"subtitle": subtitle, "description": description}
    with _lock:  # read, change, write and read again as one step: two administrators never overwrite each other's words
        refresh()
        if (word(cls, "subtitle"), word(cls, "description") or "") == (subtitle, description):
            return entry, False
        try:
            source = path.read_text(encoding="utf-8") if path.is_file() else ""
        except OSError as exc:
            raise Invalid(Msg("E-NODETEXT-UNWRITABLE", file=str(path), why=str(exc)[:80])) from exc
        text = with_words(source, type_id, subtitle, description)
        try:
            got = i18n.flatten(tomllib.loads(text), str(path))
        except (tomllib.TOMLDecodeError, i18n.TextError) as exc:  # the file holds the node's words some other way
            raise Invalid(Msg("E-NODETEXT-UNWRITABLE", file=str(path), why=str(exc)[:80])) from exc
        if (got.get(f"node.{type_id}.subtitle"), got.get(f"node.{type_id}.description")) != (subtitle, description):
            raise Invalid(Msg("E-NODETEXT-UNWRITABLE", file=str(path), why=f"node.{type_id}"))
        try:
            if path.is_file():
                shutil.copyfile(path, path.with_name(path.name + ".bak"))
            write_text(path, text)
        except OSError as exc:  # the folder is not writable here (a packaged install): said, not a 500
            raise Invalid(Msg("E-NODETEXT-UNWRITABLE", file=str(path), why=str(exc)[:80])) from exc
        refresh()
        return entry, True

