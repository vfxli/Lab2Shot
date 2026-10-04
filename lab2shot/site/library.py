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
RESERVED_USERNAMES holds them; tests/test_accounts.py holds the two to agree).

This is product-layer storage and must not import `lab2shot.server` (layering rule). The routes are in
server/library.py (users) and server/templates.py (administrator) and handle only the HTTP layer.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

from .. import i18n
from ..config import TEMPLATES_DIR, settings
from ..errors import Invalid, NotFound, TooLarge
from ..io.atomic import write_text
from ..messages import Msg

OWNER_ADMIN = "admin"
OWNER_ADAPTER = "adapter"

MAX_BYTES = 4 << 20  # maximum size of one graph (well above the largest built-in template); larger input is data
MAX_PER_ACCOUNT = 200  # maximum templates per account, including those in the bin
NAME_CHARS = 60  # card title: room for a full English name (camera_da3long: "Depth and Camera · Depth Anything 3 Streaming")
# card description, about ten lines at three-column card width, by display width: a full-width character (Chinese)
# counts 2, a half-width one (English) 1, so 240 Chinese characters or 480 English letters fill the same lines
TEXT_WIDTH = 480
BIN = "_bin"  # subfolder for templates deleted by the user; the administrator can restore or purge them
_STEM = re.compile(rf"[^\w{chr(0x4E00)}-{chr(0x9FFF)}-]+")  # what a file name keeps of a template's name (Han by code point)


# ------------------------------------------------------------------ files


def _json_files(folder: Path) -> list[Path]:
    """The template files of one folder (a name starting with _ is not a template: _categories.json, _bin/)."""
    return sorted(f for f in folder.glob("*.json") if not f.name.startswith("_")) if folder.is_dir() else []


def width(text: str) -> int:
    """How wide a text shows: a full-width (East Asian wide) character 2, any other 1 (webui/src/ui/NameSheet.tsx
    textWidth, the same rule)."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _text(value: object, most: int, what: str) -> str:
    s = str(value or "").strip()
    if len(s) > most:
        raise Invalid(Msg("E-LIBRARY-TOOLONG", what=what, most=most))
    return s


def _intro(value: object, what: str) -> str | dict:
    """A card's intro as it is kept: a user's own words (a string), or a built-in card's both languages ({zh, en}),
    each held to TEXT_WIDTH (display width: E-LIBRARY-TOOWIDE)."""
    if isinstance(value, dict):
        return {lang: _intro(text, what) for lang, text in value.items() if lang in i18n.LANGS}
    s = str(value or "").strip()
    if width(s) > TEXT_WIDTH:
        raise Invalid(Msg("E-LIBRARY-TOOWIDE", what=what, most=TEXT_WIDTH // 2, half=TEXT_WIDTH))
    return s


_HAN = re.compile(rf"[{chr(0x3400)}-{chr(0x9FFF)}]")


def language_of(text: str) -> str:
    """The language a plain (one-language) display text is taken to be in: Chinese when it has a Chinese character,
    else English (webui/src/state/cookInputs.ts languageOf, the same rule)."""
    return "zh" if _HAN.search(text) else "en"


def _in_language(old: object, new: str) -> str | dict:
    """A card's name or intro edited on the page, which shows one language: written into the language now, the other
    kept as it is ({zh, en}, read with i18n.pick). An old plain string is first put where its language says
    (language_of); nothing there before: {language now: text} (the page's edited(), the same rule). Unchanged from what
    the page showed (i18n.pick: possibly the other language, when the language now has none): kept as it is, so a
    field left alone is not copied into the other language."""
    if old and new == i18n.pick(old):
        return old
    if isinstance(old, dict):
        base = {k: v for k, v in old.items() if k in i18n.LANGS and isinstance(v, str)}
    elif isinstance(old, str) and old.strip():
        base = {language_of(old): old}
    else:
        base = {}
    return {**base, i18n.current(): new}


def _new_words(text: str | dict) -> str | dict:
    """A name or intro given for a new card: {language now: text} (nothing typed: "")."""
    return {i18n.current(): text} if isinstance(text, str) and text else text


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
    from ..engine.graph import SCHEMA

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
    from ..engine.templates import load_graph

    return load_graph(path)


# ------------------------------------------------------------------ ids


SEP = "~"  # between the parts of a card id: safe in a file name (a delivery is named after its template: no colon on Windows) and in a URL


def card_id(owner: str, stem: str, adapter: str = "") -> str:
    """A preset's card id (owner admin / adapter). A user's template has its own, `user_card_id`: the owner word cannot
    tell them apart, because the built-in administrator's username is also `admin` (accounts.ADMIN_NAME)."""
    if owner == OWNER_ADAPTER:
        return f"{OWNER_ADAPTER}{SEP}{adapter}{SEP}{stem}"
    if owner == OWNER_ADMIN:
        return f"{OWNER_ADMIN}{SEP}{stem}"
    return user_card_id(owner, stem)


def user_card_id(username: str, stem: str) -> str:
    """A user's template ("我的模板"): always `user~<username>~<stem>`, whatever the username. The built-in administrator
    is called `admin` (accounts.ADMIN_NAME); through `card_id` its templates got `admin~<stem>`, which `parse_id` reads as
    a project preset, so they could be neither opened nor deleted (E-LIBRARY-NOSUCH)."""
    return f"user{SEP}{username}{SEP}{stem}"


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
    from ..extensions import extensions

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
    from ..categories import templates as tree

    parts.append(("_categories", int(tree.changed_at() * 1e9)))
    return tuple(parts)


def changed_at() -> tuple:
    """Every preset file, folder and the tree with its modification time: the templates route's cache key (per
    file: a file restored with an older time still counts as changed)."""
    return _stamp()


def presets() -> list[dict]:
    """Every preset card (the adapters' and the project's), in the catalogue's order (engine/templates.py order): the
    one list of cards (engine/templates.py only parses them). Each: id, owner, adapter, path, name, intro, deliverable (where its card
    sits: a subcategory or first-level category id, "" for 未分类), category (the first-level one, "" for 未分类), project,
    follows, graph, runtimes, licence, enabled, author, created, updated, bytes.
    Read again only when a file changes; treat the result as read-only. A card's name and intro are in the language
    now (a built-in card holds both: {"zh": …, "en": …}, i18n.pick), so each language has its own list."""
    with _PRESETS_LOCK:  # the warm-up and a first request: the list is worked out once, the other waits for it
        return _presets(_stamp(), i18n.current())


_PRESETS_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _presets(stamp: tuple, _lang: str) -> list[dict]:
    from ..engine.templates import order
    from ..logs import get as log

    out = []
    for folder, owner, adapter in _folders():
        for f in _json_files(folder):
            try:
                out.append(_card(f, owner, adapter))
            except Exception:  # noqa: BLE001 - one unreadable file must not affect the others
                log("library").warning("template %s: its card does not read, skipped", f, exc_info=True)
    return order(out)


def _card(path: Path, owner: str, adapter: str) -> dict:
    """One file as a card. Where it sits is its own meta.deliverable, read against the templates tree
    (lab2shot/categories.py): a place the tree no longer has is 未分类. Nothing is derived from the graph."""
    from ..categories import templates as tree
    from ..nodes import node_types

    data = _load(path)
    meta = data.get("meta") or {}
    types = node_types()
    cid = card_id(owner, path.stem, adapter)
    deliverable = str(meta.get("deliverable") or "")
    runtimes = list(dict.fromkeys(types[n["type"]].runtime for n in data.get("nodes", []) if n["type"] in types))
    st = path.stat()
    return {"id": cid, "owner": owner, "adapter": adapter, "path": str(path),
            "name": i18n.pick(meta.get("name")) or path.stem, "intro": i18n.pick(meta.get("intro")),
            "deliverable": deliverable, "category": tree.category_of(deliverable),
            # the card whose 逐项 (per-item) run this card is ("" when it leads): `order`'s one input
            "follows": str(meta.get("follows", "")),
            "graph": data,
            "runtimes": [r for r in runtimes if r != "core"],
            **_licences(str(path), st.st_mtime_ns, st.st_size, registry_print()),
            "enabled": meta.get("enabled") is not False,
            # who saved it: an account id (author_of names it when asked), or, in a file saved before ids, a name
            "author_id": meta.get("author_id"), "author": str(meta.get("author") or ""), "created": str(meta.get("created") or ""),
            "updated": st.st_mtime, "bytes": st.st_size}


LICENCES_FILE = "licences.json"  # work/: what the cards' licences were worked out to, kept across restarts
# the modules that judge a card's licences: a change to them is a change to every answer
_JUDGES = ("engine/templates.py", "nodes/tags.py", "nodes/applies.py")
_kept_lock = threading.Lock()


@lru_cache(maxsize=1)
def _registry_print_of(_types_id: int) -> str:
    from ..io.digest import key, sha256
    from ..nodes import node_types
    from ..nodes.applies import option_traits, resolve_params
    from ..nodes.params import param_defaults

    here = Path(__file__).parent.parent  # lab2shot/, where _JUDGES are
    rows: list = [[sha256(here / f) for f in _JUDGES]]
    for tid, t in sorted(node_types().items()):
        tags = sorted(resolve_params(t, param_defaults(t.Params)).licence.tags)
        options = [(p["name"], [str(o) for o in p["options"]]) for p in t.param_specs() if p.get("options")]
        rows.append([tid, getattr(t, "version", None), t.runtime, tags, option_traits(t), options])
    return key(rows, 24)


def registry_print() -> str:
    """What a card's licences are worked out from besides its own file: every node type's id, version, runtime,
    licence with its defaults, choices and the licences its choices switch to, and the code that judges them. A
    licence kept on disk for another one is worked out again (an extension upgraded, a node's licence changed)."""
    from ..nodes import node_types

    return _registry_print_of(id(node_types()))


def _kept_path() -> Path:
    return settings().work_dir / LICENCES_FILE


@lru_cache(maxsize=1)
def _kept() -> dict:
    """The licences kept on disk: {"registry": registry_print, "cards": {file sha256: {licence, route_licences}}}."""
    try:
        data = json.loads(_kept_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("cards"), dict) else {}
    except (OSError, ValueError):
        return {}


def _keep(registry: str, digest: str, found: dict) -> None:
    """Keep a card's licences on disk (best effort: an unwritable work folder only means working them out again)."""
    with _kept_lock:
        kept = _kept()
        if kept.get("registry") != registry:
            kept.clear()
            kept.update({"registry": registry, "cards": {}})
        kept["cards"][digest] = found
        try:
            write_text(_kept_path(), json.dumps(kept, ensure_ascii=False))
        except OSError:
            pass


@lru_cache(maxsize=1024)
def _licences(path: str, _mtime_ns: int, _size: int, registry: str) -> dict:
    """A card file's licence chip and its routes' licences: the same in every language, so worked out once per file
    as it stands (judging every route takes the most of reading the presets), not again for each language, and kept
    on disk (LICENCES_FILE) by the file's contents and the node registry (registry_print), so a server that starts
    again does not judge every card again."""
    from ..engine.templates import card_tags, route_tags
    from ..io.digest import sha256

    file = Path(path)
    digest = sha256(file)
    kept = _kept()
    if kept.get("registry") == registry and isinstance(found := kept["cards"].get(digest), dict):
        return found
    data = _load(file)
    found = {"licence": sorted(card_tags(data)),  # its chip: what its deliveries are made from with its defaults
             # every route a user can take through it (its menus' values), each's tags: its best one is said beside
             # the chip (who sees the card is decided by its nodes alone: server/access.py templates_for)
             "route_licences": [sorted(r) for r in route_tags(data)]}
    _keep(registry, digest, found)
    return found


def author_of(card: dict) -> str:
    """Who saved a card, as a page shows it: its account's label (DELETED once that account is gone), or the name a
    file saved before ids holds."""
    from ..accounts import deleted_label, get
    from ..errors import NotFound

    if card.get("author_id") is None:
        return card["author"]
    try:
        return get(int(card["author_id"])).label
    except (NotFound, ValueError, TypeError):
        return deleted_label()


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


def save_preset(graph: dict, name: str, intro: str = "", deliverable: str = "", author: int | None = None) -> dict:
    """"保存为预设模板" (save as preset): a new file under templates/. `deliverable`: where its card sits ("" for 未分类);
    `author`: the account that saved it, by id: the file never holds a person's name, which permanent deletion of the
    account could not take back out of a file of the repository (author_of names it while the account is there)."""
    name = _text(name, NAME_CHARS, i18n.t("library.name"))
    if not name:
        raise Invalid(Msg("E-LIBRARY-NONAME"))
    stem = _stem_for(name, TEMPLATES_DIR)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(TEMPLATES_DIR / f"{stem}.json", graph,
           {"name": _new_words(name), "intro": _new_words(_intro(intro, i18n.t("library.intro"))), "owner": OWNER_ADMIN, "deliverable": str(deliverable or ""),
            "author_id": author, "created": now, "updated": now})
    return preset(card_id(OWNER_ADMIN, stem))


def same_name_preset(name: str) -> dict | None:
    """「保存为预设模板」同名时要覆盖的那张：名字（去掉首尾空白）相同、归管理员的项目预设（owner admin）。有几张同名时
    取最近改过的（节点图不记来自哪张模板：从模板存出来的就是一张普通的图）。只有接入层自带的同名时 E-TEMPLATES-SAMENAMEREADONLY
    （它的文件不能改，沿用 E-TEMPLATES-READONLY 的口径）；没有同名的返回 None（新建）。"""
    name = str(name or "").strip()
    same = [t for t in presets() if t["name"].strip() == name]
    if not same:
        return None
    own = [t for t in same if t["owner"] == OWNER_ADMIN]
    if not own:
        raise Invalid(Msg("E-TEMPLATES-SAMENAMEREADONLY", template=name))
    return max(own, key=lambda t: t["updated"])


def replace_preset(card: str, graph: dict, name: str, intro: str = "", deliverable: str = "") -> dict:
    """覆盖一张项目预设（「保存为预设模板」同名、管理员确认之后）：节点图换成 `graph`，名字、简介、分类按表单；文件名
    （卡片 id）不变，`created`、`author`、文件自己的 `id`、开关和 `follows`（排在谁后面）保留原值，
    `updated` 更新。
    接入层自带的不能覆盖（E-TEMPLATES-READONLY）。"""
    name = _text(name, NAME_CHARS, i18n.t("library.name"))
    if not name:
        raise Invalid(Msg("E-LIBRARY-NONAME"))
    _found, path = _own(card)
    old = _load(path).get("meta") or {}
    kept = {k: old[k] for k in ("id", "enabled", "follows") if k in old}
    _write(path, graph,
           {"name": _in_language(old.get("name"), name), "intro": _in_language(old.get("intro"), _intro(intro, i18n.t("library.intro"))),
            "owner": OWNER_ADMIN, "deliverable": str(deliverable or ""),
            **{k: old[k] for k in ("author", "author_id") if k in old}, "created": str(old.get("created") or ""), **kept,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return preset(card)


def edit_preset(card: str, name: str, intro: str) -> dict:
    """A card's name and intro, written into its file's meta (an adapter's file too: the two words)."""
    name = _text(name, NAME_CHARS, i18n.t("library.name"))
    if not name:
        raise Invalid(Msg("E-LIBRARY-NONAME"))
    found = preset(card)
    meta = _load(Path(found["path"])).get("meta") or {}
    _edit_meta(Path(found["path"]), name=_in_language(meta.get("name"), name),
               intro=_in_language(meta.get("intro"), _intro(intro, i18n.t("library.intro"))))
    return preset(card)


def copy_preset(card: str, name: str = "", author: int | None = None) -> dict:
    """A card's graph as a new project preset ("<name> 副本" unless named), placed where the original is."""
    found = preset(card)
    copy_name = name
    if not copy_name:  # "<name> 副本", within the limit and distinct from existing card names (a copy of a copy)
        taken = {t["name"] for t in presets()}
        base = found["name"][:NAME_CHARS - 3].rstrip()
        copy_name, n = i18n.t("library.copy", name=base), 2
        while copy_name in taken:
            tail = i18n.t("library.copy_tail", n=n)
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
    from ..logs import get as log

    stuck = []
    for folder, _owner, _adapter in _folders():  # every file, not only the cards that read: an unreadable file is stuck too
        for f in _json_files(folder):
            name = f.stem
            try:
                data = _load(f)
                name = i18n.pick((data.get("meta") or {}).get("name")) or f.stem
                if str((data.get("meta") or {}).get("deliverable") or "") in gone:
                    _edit_meta(f, deliverable="")
            except Exception:  # noqa: BLE001 - one file that cannot be read or written must not stop the rest (it shows as 未分类 anyway)
                log("library").warning("template %s: its category could not be cleared", f, exc_info=True)
                stuck.append(name)
    return stuck


def disabled() -> frozenset[str]:
    return frozenset(t["id"] for t in presets() if not t["enabled"])


# ------------------------------------------------------------------ "我的模板" (user templates): one folder per account


def user_dir(username: str) -> Path:
    """The account's own templates folder. `username` must be a username by the accounts' rule (accounts.USERNAME):
    anything else (「..」, a path) names no folder, whoever calls."""
    from ..accounts import USERNAME

    if not USERNAME.match(username):
        raise NotFound(Msg("E-LIBRARY-NOSUCH"))
    return settings().work_dir / "users" / username / "templates"


def _user_row(path: Path, username: str, bin_: bool) -> dict:
    """One of the user's files as the page lists it (the graph itself is fetched when opened, never in a listing)."""
    try:
        meta = json.loads(path.read_text(encoding="utf-8")).get("meta") or {}
    except (OSError, ValueError):
        meta = {}
    st = path.stat()
    return {"id": user_card_id(username, path.stem), "stem": path.stem, "name": i18n.pick(meta.get("name")) or path.stem,
            "intro": i18n.pick(meta.get("intro")), "bytes": st.st_size, "updated": st.st_mtime,
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


def user_same_name(username: str, name: str) -> str:
    """「保存到我的模板」同名时要覆盖的那张的文件名（stem）：这个账号自己的、不在回收站里的、名字相同的一张；有几张同名时
    取最近改过的。没有同名的返回 ""（新建）。"""
    name = str(name or "").strip()
    same = [r for r in user_cards(username) if r["name"].strip() == name]  # newest first
    if not same:
        return ""
    return same[0]["stem"]


def user_get(username: str, stem: str) -> dict:
    path, binned = _user_file(username, stem)
    return {**_user_row(path, username, binned), "graph": _load(path)}


def user_save(username: str, graph: dict, name: str, intro: str = "", stem: str = "") -> dict:
    """Save a graph as one of the account's templates (`stem`: overwrite that one of its own). The quota is the caller's
    to check first (server/quota.py)."""
    name = _text(name, NAME_CHARS, i18n.t("library.name"))
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
    text = _intro(intro, i18n.t("library.intro"))
    words = ({"name": _in_language(kept.get("name"), name), "intro": _in_language(kept.get("intro"), text)} if kept
             else {"name": _new_words(name), "intro": _new_words(text)})
    _write(path, graph, {**words, "owner": username,
                         "created": kept.get("created") or now, "updated": now, "deleted": None, "deleted_by": ""})
    return _user_row(path, username, False)


def user_bin(username: str, stem: str, by: str) -> dict:
    """Into the bin (`by`: user / admin). Nothing is lost: only an administrator restores it (user_restore), whoever binned
    it, since the user has no bin of their own to take it from."""
    path, binned = _user_file(username, stem)
    if binned:
        return _user_row(path, username, True)
    target = user_dir(username) / BIN / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _load(path)
    _write(target, data, {**(data.get("meta") or {}), "deleted": time.time(), "deleted_by": by})
    path.unlink()
    return _user_row(target, username, True)


def user_restore(username: str, stem: str) -> dict:
    """Out of the bin, by the administrator (templates.restore): the user has no bin of their own to take it from."""
    path, binned = _user_file(username, stem)
    if not binned:
        return _user_row(path, username, False)
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
    """What the account's templates take (the bin does not count against it: it is kept for the administrator): their
    files' sizes, without reading them."""
    return sum(f.stat().st_size for f in _json_files(user_dir(username)))


def remove_user(username: str) -> int:
    """An account is purged (lab2shot/accounts.py purge): its folder goes with it, and its templates with it. Returns how
    many templates it held."""
    n = len(user_cards(username)) + len(user_cards(username, bin_=True))
    shutil.rmtree(settings().work_dir / "users" / username, ignore_errors=True)
    return n


# ---- the administrator's registry of an account's templates (lab2shot/site/resources.py)


def rows_for_account(username: str) -> list[dict]:
    """The account's templates for the administrator's registry (lab2shot/site/resources.py): name, state, size, when; plus
    the registry's two general fields (dimmed, its actions)."""
    found = []
    for bin_ in (False, True):
        for r in user_cards(username, bin_):
            found.append({"id": r["id"], "name": r["name"],
                          "state": i18n.t("library.state.user_deleted" if bin_ and r["deleted_by"] == "user" else "library.state.deleted" if bin_ else "library.state.mine"),
                          "bytes": r["bytes"], "updated": r["updated"],
                          "__dim": bin_, "__acts": ["restore", "purge"] if bin_ else ["bin"]})
    return found
