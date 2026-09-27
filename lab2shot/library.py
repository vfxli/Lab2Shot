"""Templates are files: three locations, one format, one loader.

    adapters/<package>/templates/*.json     templates shipped with an adapter; cannot be deleted; enabled
                                            state and category are written to the file itself              owner = "adapter"
    templates/*.json                        project presets: created, copied, deleted and categorized by
                                            the administrator                                              owner = "admin"
    work/users/<username>/templates/*.json  a user's own templates ("我的模板"); deleted ones move to _bin/ owner = <username>
    templates/_categories.json              the template panel's category tree (read and written by lab2shot/categories.py)

Each file is a node graph (lab2shot.graph/1) with an additional `owner` in `meta`. The file's location is authoritative:
`owner` is written from it and the location wins on read. A card's category is stored in its own file
(`meta.deliverable`: a category or subcategory id; absent means "未分类"), and dragging a card in the template dialog
changes only this field, for adapter templates as well. The file name without .json is the card's stem; the card id is
source plus stem: `admin~<stem>`, `adapter~<package>~<stem>`, `user~<username>~<stem>` (separator SEP). The database
holds no template data.

The words `admin` and `adapter` are reserved and cannot be registered as usernames (lab2shot/accounts.py
RESERVED_USERNAMES reads them from here).

This is product-layer storage and must not import `lab2shot.server` (layering rule). The routes are in
server/library.py (users) and server/templates.py (administrator) and handle only the HTTP layer.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from functools import lru_cache
from pathlib import Path

from .config import TEMPLATES_DIR, settings
from .errors import Invalid, NotFound, TooLarge
from .io.atomic import write_text
from .messages import Msg

OWNER_ADMIN = "admin"
OWNER_ADAPTER = "adapter"
RESERVED_USERNAMES = frozenset({OWNER_ADMIN, OWNER_ADAPTER})

MAX_BYTES = 4 << 20  # maximum size of one graph (well above the largest built-in template); larger input is data
MAX_PER_ACCOUNT = 200  # maximum templates per account, including those in the bin
NAME_CHARS = 40  # card title fits on one line without wrapping or truncation (longest existing: 35)
TEXT_CHARS = 240  # card description, about ten lines at three-column card width (longest existing: 223)
BIN = "_bin"  # subfolder for templates deleted by the user; the administrator can restore or purge them
_STEM = re.compile(r"[^\w一-鿿-]+")  # what a file name keeps of a template's name


# ------------------------------------------------------------------ files


def _json_files(folder: Path) -> list[Path]:
    """The template files of one folder (a name starting with _ is not a template: _categories.json, _bin/)."""
    return sorted(f for f in folder.glob("*.json") if not f.name.startswith("_")) if folder.is_dir() else []


def _text(value: object, most: int, what: str) -> str:
    s = str(value or "").strip()
    if len(s) > most:
        raise Invalid(Msg("E-LIBRARY-TOOLONG", what=what, most=most))
    return s


def _stem_for(name: str, folder: Path) -> str:
    """A file name for a template called `name`, unused in `folder`."""
    base = _STEM.sub("_", name).strip("_") or "template"
    stem, n = base, 1
    while (folder / f"{stem}.json").exists() or (folder / BIN / f"{stem}.json").exists():
        n += 1
        stem = f"{base}_{n}"
    return stem


def _write(path: Path, graph: dict, meta: dict) -> None:
    """A graph file with these meta words on it (its own meta otherwise kept: exposed, author, created …)."""
    from .engine.graph import SCHEMA

    data = {**graph, "schema": SCHEMA, "meta": {**(graph.get("meta") or {}), **meta}}
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise TooLarge(Msg("E-LIBRARY-TOOBIG", mb=len(text.encode("utf-8")) / 2**20, most=MAX_BYTES / 2**20))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text(path, text)
    except OSError as exc:  # the folder is not writable here (a packaged install): said, not a 500
        raise Invalid(Msg("E-TEMPLATES-UNWRITABLE", template=path.name, why=str(exc)[:80])) from exc


def _load(path: Path) -> dict:
    from .engine.templates import load_graph

    return load_graph(path)


# ------------------------------------------------------------------ ids


SEP = "~"  # between the parts of a card id: safe in a file name (a delivery is named after its template: no colon on Windows) and in a URL


def card_id(owner: str, stem: str, adapter: str = "") -> str:
    if owner == OWNER_ADAPTER:
        return f"{OWNER_ADAPTER}{SEP}{adapter}{SEP}{stem}"
    if owner == OWNER_ADMIN:
        return f"{OWNER_ADMIN}{SEP}{stem}"
    return f"user{SEP}{owner}{SEP}{stem}"


def parse_id(card: str) -> tuple[str, str, str]:
    """(kind, who, stem): kind is admin / adapter / user; who the adapter's or the user's name ("" for admin)."""
    parts = str(card).split(SEP, 2)
    if parts[0] == OWNER_ADMIN and len(parts) == 2:
        return OWNER_ADMIN, "", parts[1]
    if parts[0] in (OWNER_ADAPTER, "user") and len(parts) == 3 and parts[1] and parts[2]:
        return parts[0], parts[1], parts[2]
    raise NotFound(Msg("E-TEMPLATES-NOSUCH", template=card))


# ------------------------------------------------------------------ the presets: adapters' and the project's


def _folders() -> list[tuple[Path, str, str]]:
    """Where the preset files are: (folder, owner, adapter name), the project's first, then every adapter's."""
    from .extensions import extensions

    out = [(TEMPLATES_DIR, OWNER_ADMIN, "")]
    for name, ext in sorted(extensions().items()):
        out.append((ext.adapter_dir / "templates", OWNER_ADAPTER, name))
    return out


def _stamp() -> tuple:
    """Everything the preset list depends on: every file's mtime, the folders' (a deleted file changes its folder) and the
    category tree's. The one key `presets` is cached by."""
    parts: list[tuple] = []
    for folder, owner, adapter in _folders():
        try:
            parts.append((str(folder), folder.stat().st_mtime_ns if folder.is_dir() else 0))
        except OSError:
            parts.append((str(folder), 0))
        for f in _json_files(folder):
            try:
                parts.append((str(f), f.stat().st_mtime_ns))
            except OSError:
                pass
    from .categories import templates as tree

    parts.append(("_categories", int(tree.changed_at() * 1e9)))
    return tuple(parts)


def changed_at() -> tuple:
    """Every preset file, folder and the tree with its modification time: the templates route's cache key (per
    file: a file restored with an older time still counts as changed)."""
    return _stamp()


def presets() -> list[dict]:
    """Every preset card (the adapters' and the project's), in the catalogue's order (engine/templates.py order): what
    engine.templates.templates() hands out. Each: id, owner, adapter, path, name, intro, deliverable (where its card
    sits: a subcategory or first-level category id, "" for 未分类), category (the first-level one, "" for 未分类), project,
    follows, graph, runtimes, licence, enabled, author, created, updated, bytes.
    Read again only when a file changes; treat the result as read-only."""
    return _presets(_stamp())


@lru_cache(maxsize=1)
def _presets(stamp: tuple) -> list[dict]:
    from .engine.templates import order
    from .logs import get as log

    out = []
    for folder, owner, adapter in _folders():
        for f in _json_files(folder):
            try:
                out.append(_card(f, owner, adapter))
            except Exception:  # noqa: BLE001 - one unreadable file must not affect the others
                log("library").warning("模板 %s 读不出卡片，跳过", f, exc_info=True)
    return order(out)


def _card(path: Path, owner: str, adapter: str) -> dict:
    """One file as a card. Where it sits is its own meta.deliverable, read against the templates tree
    (lab2shot/categories.py): a place the tree no longer has is 未分类. Nothing is derived from the graph."""
    from .categories import templates as tree
    from .engine.templates import core_project
    from .nodes import node_types
    from .nodes.tags import graph_tags

    data = _load(path)
    meta = data.get("meta") or {}
    types = node_types()
    cid = card_id(owner, path.stem, adapter)
    deliverable = str(meta.get("deliverable") or "")
    runtimes = list(dict.fromkeys(types[n["type"]].runtime for n in data.get("nodes", []) if n["type"] in types))
    st = path.stat()
    return {"id": cid, "owner": owner, "adapter": adapter, "path": str(path),
            "name": str(meta.get("name") or path.stem), "intro": str(meta.get("intro") or ""),
            "deliverable": deliverable, "category": tree.category_of(deliverable),
            "project": core_project(data, types),
            # the card whose 逐项 (per-item) run this card is ("" when it leads): `order`'s one input
            "follows": str(meta.get("follows", "")),
            "graph": data,
            "runtimes": [r for r in runtimes if r != "core"],
            "licence": sorted(graph_tags(data)),  # nodes/tags.py: its chips, and who may use it
            "enabled": meta.get("enabled") is not False,
            "author": str(meta.get("author") or ""), "created": str(meta.get("created") or ""),
            "updated": st.st_mtime, "bytes": st.st_size}


def preset(card: str) -> dict:
    found = next((t for t in presets() if t["id"] == card), None)
    if found is None:
        raise NotFound(Msg("E-TEMPLATES-NOSUCH", template=card))
    return found


def _own(card: str) -> tuple[dict, Path]:
    """A project preset (the administrator's file), or E-TEMPLATES-READONLY for an adapter's."""
    found = preset(card)
    if found["owner"] != OWNER_ADMIN:
        raise Invalid(Msg("E-TEMPLATES-READONLY", template=found["name"]))
    return found, Path(found["path"])


def _edit_meta(path: Path, **values) -> None:
    data = _load(path)
    _write(path, data, {**(data.get("meta") or {}), **values, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})


# ---- the administrator's operations (server/templates.py)


def save_preset(graph: dict, name: str, intro: str = "", deliverable: str = "", author: str = "") -> dict:
    """"保存为预设模板" (save as preset): a new file under templates/. `deliverable`: where its card sits ("" for 未分类);
    `author`: who saved it."""
    name = _text(name, NAME_CHARS, "名字")
    if not name:
        raise Invalid(Msg("E-LIBRARY-NONAME"))
    stem = _stem_for(name, TEMPLATES_DIR)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(TEMPLATES_DIR / f"{stem}.json", graph,
           {"name": name, "intro": _text(intro, TEXT_CHARS, "简介"), "owner": OWNER_ADMIN, "deliverable": str(deliverable or ""),
            "author": str(author or ""), "created": now, "updated": now})
    return preset(card_id(OWNER_ADMIN, stem))


def edit_preset(card: str, name: str, intro: str) -> dict:
    """A card's name and intro, written into its file's meta (an adapter's file too: the two words)."""
    name = _text(name, NAME_CHARS, "名字")
    if not name:
        raise Invalid(Msg("E-LIBRARY-NONAME"))
    found = preset(card)
    _edit_meta(Path(found["path"]), name=name, intro=_text(intro, TEXT_CHARS, "简介"))
    return preset(card)


def copy_preset(card: str, name: str = "", author: str = "") -> dict:
    """A card's graph as a new project preset ("<name> 副本" unless named), placed where the original is."""
    found = preset(card)
    copy_name = name
    if not copy_name:  # "<name> 副本", within the limit and distinct from existing card names (a copy of a copy)
        taken = {t["name"] for t in presets()}
        base = found["name"][:NAME_CHARS - 3].rstrip()
        copy_name, n = f"{base} 副本", 2
        while copy_name in taken:
            tail = f" 副本{n}"
            copy_name, n = f"{found['name'][:NAME_CHARS - len(tail)].rstrip()}{tail}", n + 1
    return save_preset(found["graph"], copy_name, found["intro"], found["deliverable"], author)


def delete_preset(card: str) -> dict:
    """A project preset's file is removed (an adapter's cannot be: E-TEMPLATES-READONLY; disable it instead)."""
    found, path = _own(card)
    path.unlink(missing_ok=True)
    return found


def set_enabled(card: str, on: bool) -> dict:
    """A card on or off: written into its file's meta (an adapter's file too: the one field)."""
    found = preset(card)
    _edit_meta(Path(found["path"]), enabled=bool(on))
    return preset(card)


def place(card: str, where: str) -> dict:
    """A card under `where` (a subcategory or first-level category id; "" for 未分类): its file's meta.deliverable."""
    found = preset(card)
    _edit_meta(Path(found["path"]), deliverable=str(where or ""))
    return preset(card)


def unplace(gone: set[str]) -> list[str]:
    """The places in `gone` are no longer in the tree (lab2shot/categories.py remove): every card there becomes 未分类
    and is never deleted with its category. Returns the names of the cards whose file could not be rewritten (they still
    name the old place; the route says so)."""
    from .logs import get as log

    stuck = []
    for folder, _owner, _adapter in _folders():  # every file, not only the cards that read: an unreadable file is stuck too
        for f in _json_files(folder):
            name = f.stem
            try:
                data = _load(f)
                name = str((data.get("meta") or {}).get("name") or f.stem)
                if str((data.get("meta") or {}).get("deliverable") or "") in gone:
                    _edit_meta(f, deliverable="")
            except Exception:  # noqa: BLE001 - one file that cannot be read or written must not stop the rest (it shows as 未分类 anyway)
                log("library").warning("模板 %s 的分类没能清掉", f, exc_info=True)
                stuck.append(name)
    return stuck


def disabled() -> frozenset[str]:
    return frozenset(t["id"] for t in presets() if not t["enabled"])


# ------------------------------------------------------------------ "我的模板" (user templates): one folder per account


def user_dir(username: str) -> Path:
    return settings().work_dir / "users" / username / "templates"


def _user_row(path: Path, username: str, bin_: bool) -> dict:
    """One of the user's files as the page lists it (the graph itself is fetched when opened, never in a listing)."""
    try:
        meta = json.loads(path.read_text(encoding="utf-8")).get("meta") or {}
    except (OSError, ValueError):
        meta = {}
    st = path.stat()
    return {"id": card_id(username, path.stem), "stem": path.stem, "name": str(meta.get("name") or path.stem),
            "intro": str(meta.get("intro") or ""), "bytes": st.st_size, "updated": st.st_mtime,
            "deleted": float(meta.get("deleted") or st.st_mtime) if bin_ else None,
            "deleted_by": str(meta.get("deleted_by") or "") if bin_ else ""}


def user_cards(username: str, bin_: bool = False) -> list[dict]:
    """The account's templates (`bin_`: the ones it deleted, kept for the administrator to restore), newest first."""
    folder = user_dir(username) / BIN if bin_ else user_dir(username)
    return sorted((_user_row(f, username, bin_) for f in _json_files(folder)), key=lambda r: -r["updated"])


def _user_file(username: str, stem: str) -> tuple[Path, bool]:
    """The file of one of the user's templates, and whether it is in the bin; E-LIBRARY-NOSUCH when there is none."""
    if not stem or "/" in stem or stem.startswith("_") or stem.startswith("."):
        raise NotFound(Msg("E-LIBRARY-NOSUCH"))
    live, binned = user_dir(username) / f"{stem}.json", user_dir(username) / BIN / f"{stem}.json"
    if live.is_file():
        return live, False
    if binned.is_file():
        return binned, True
    raise NotFound(Msg("E-LIBRARY-NOSUCH"))


def user_get(username: str, stem: str) -> dict:
    path, binned = _user_file(username, stem)
    return {**_user_row(path, username, binned), "graph": _load(path)}


def user_save(username: str, graph: dict, name: str, intro: str = "", stem: str = "") -> dict:
    """Save a graph as one of the account's templates (`stem`: overwrite that one of its own). The quota is the caller's
    to check first (server/quota.py)."""
    name = _text(name, NAME_CHARS, "名字")
    if not name:
        raise Invalid(Msg("E-LIBRARY-NONAME"))
    folder = user_dir(username)
    if stem:
        path, binned = _user_file(username, stem)
        if binned:
            path.rename(folder / path.name)  # saving over a deleted one brings it back
            path = folder / path.name
    else:
        if len(user_cards(username)) + len(user_cards(username, bin_=True)) >= MAX_PER_ACCOUNT:
            raise Invalid(Msg("E-LIBRARY-TOOMANY", most=MAX_PER_ACCOUNT))
        stem = _stem_for(name, folder)
        path = folder / f"{stem}.json"
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    kept = (json.loads(path.read_text(encoding="utf-8")).get("meta") or {}) if path.is_file() else {}
    _write(path, graph, {"name": name, "intro": _text(intro, TEXT_CHARS, "简介"), "owner": username,
                         "created": kept.get("created") or now, "updated": now, "deleted": None, "deleted_by": ""})
    return _user_row(path, username, False)


def user_bin(username: str, stem: str, by: str) -> dict:
    """Into the bin (`by`: user / admin). Nothing is lost; the administrator (or the user, when they binned it) restores it."""
    path, binned = _user_file(username, stem)
    if binned:
        return _user_row(path, username, True)
    target = user_dir(username) / BIN / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _load(path)
    _write(target, data, {**(data.get("meta") or {}), "deleted": time.time(), "deleted_by": by})
    path.unlink()
    return _user_row(target, username, True)


def user_restore(username: str, stem: str, by: str = "admin") -> dict:
    """Out of the bin. One the administrator binned is not the user's to take back (E-LIBRARY-ADMINBINNED)."""
    path, binned = _user_file(username, stem)
    if not binned:
        return _user_row(path, username, False)
    row = _user_row(path, username, True)
    if by == "user" and row["deleted_by"] == "admin":
        raise Invalid(Msg("E-LIBRARY-ADMINBINNED", name=row["name"]))
    target = user_dir(username) / path.name
    data = _load(path)
    _write(target, data, {**(data.get("meta") or {}), "deleted": None, "deleted_by": ""})
    path.unlink()
    return _user_row(target, username, False)


def user_purge(username: str, stem: str) -> dict:
    path, binned = _user_file(username, stem)
    row = _user_row(path, username, binned)
    path.unlink()
    return row


def user_bytes(username: str) -> int:
    """What the account's templates take (the bin does not count against it: it is kept for the administrator)."""
    return sum(r["bytes"] for r in user_cards(username))


def user_graphs(username: str) -> list[dict]:
    """Every graph of the account, the binned ones too (server/quota.py: the uploads they name stay)."""
    out = []
    for bin_ in (False, True):
        for r in user_cards(username, bin_):
            try:
                out.append(user_get(username, r["stem"])["graph"])
            except Exception:  # noqa: BLE001
                pass
    return out


def remove_user(username: str) -> int:
    """An account is purged (lab2shot/accounts.py purge): its folder goes with it, including its templates and the graph
    files of its jobs (jobs/, lab2shot/farm/queue.py graph_file). Returns how many templates it held."""
    n = len(user_cards(username)) + len(user_cards(username, bin_=True))
    shutil.rmtree(settings().work_dir / "users" / username, ignore_errors=True)
    return n


def rows_for_account(username: str) -> list[dict]:
    """The account's templates for the administrator's registry (lab2shot/resources.py): name, state, size, when; plus
    the registry's two general fields (dimmed, its actions)."""
    found = []
    for bin_ in (False, True):
        for r in user_cards(username, bin_):
            found.append({"id": r["id"], "name": r["name"],
                          "state": "用户已删除" if bin_ and r["deleted_by"] == "user" else "已删除" if bin_ else "我的模板",
                          "bytes": r["bytes"], "updated": r["updated"],
                          "__dim": bin_, "__acts": ["restore", "purge"] if bin_ else ["bin"]})
    return found
