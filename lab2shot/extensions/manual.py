"""Files people download by hand: one registry of what they are, one inbox they all go into.

Some things the installer cannot fetch: their page wants a registration (SMPL, SMPL-X, MANO, FLAME) or a person to
accept a licence (the Autodesk FBX SDK). The user downloads them and drops each file as it came (not unpacked, not
renamed) into the inbox, INBOX = <Lab2Shot>/downloads/. check() recognises every file there by what is inside it and
installs what installs by itself (the admin page, the installer, the setup menu); state() says where an item stands
without installing anything (the extensions' install status, nodes, the command line); view() is what the admin page's
「扩展包」 手动下载 shows.

items() is the registry, assembled from the shared body-model table and from what the loaded extensions declare
(Extension.manual_items), never hand-written here. An item says what it is, where it is downloaded and what that page asks of the user, how its
file is recognised (`markers`: names inside the archive or folder, whatever the file itself is called; a file on its
own is known by its name) and how it is installed (an Install: unpacked into its folder, or a vendor installer that
runs only after the user accepted its licence on the page, see accept()). Extensions ask for an item with
manual_weight(item) / body_model_weight(key); which extensions need an item is read from them.

The originals of what was installed move to INBOX/installed/ (kept, never replaced: they may be deleted). What stays
in the inbox still wants something: a licence to accept, or a file nobody recognised (listed with why).
"""

from __future__ import annotations

import os

import fnmatch
import json
import shutil
import tarfile
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lab2shot_shared import body_models as bodies

from ..config import ROOT, THIRD_PARTY_DIR, settings
from ..io import files
from ..io.digest import sha256, tree
from ..errors import Invalid, MessageError
from ..messages import Msg
from .spec import Weight

INBOX = ROOT / "downloads"
DONE = "installed"  # INBOX/installed/: the originals of what was installed
# INBOX/datasets/: a folder of datasets, not a hand-downloaded package; it is never walked or listed here: it holds
# datasets of tens of thousands of files (P3M-10k, AMASS), and walking them would make every page that asks for the
# extensions' state take seconds
DATASETS = "datasets"
README = "README.txt"
README_TEXT = """\
这里放所有要你自己下载的文件：人体模型（SMPL、SMPL-X、MANO、FLAME）、Autodesk FBX SDK……

下载好的文件原样放进来：不用解压，不用改名，也不用分类。
Lab2Shot 按文件里面的内容认出每一个，自动装到它该去的地方；
装好的原文件移到 installed 文件夹里（可以删掉，也可以留着备份）。

要下载什么、去哪下载、现在还缺什么：管理员打开 Lab2Shot 的后台管理页，看「扩展包」里的「手动下载」。
认不出的文件留在这里，页面上写着为什么。
"""

PARTIAL_SUFFIXES = (".crdownload", ".part", ".partial", ".download", ".tmp")


# ------------------------------------------------------------------ how an item is installed


class Install:
    """How a recognised file becomes a usable item. `consent`: installing it means accepting a licence, which only
    the user's own click on the page does (accept()); otherwise check() installs it right away (put())."""

    consent = False

    def ready(self) -> bool:
        raise NotImplementedError

    def path(self) -> Path | None:
        """Where the installed item is (what an extension builds against or loads); None while it is not."""
        raise NotImplementedError

    def version(self, names: list[str]) -> str:
        """The version a recognised file holds, from its member names ("" when the item has none)."""
        return ""

    def has(self, version: str) -> bool:
        """This version is installed already (a second copy of it just moves to installed/)."""
        return self.ready()

    def put(self, file: Path, names: list[str]) -> None:
        """Install a recognised file (items without consent)."""
        raise NotImplementedError

    def licence(self, file: Path, names: list[str], scratch: Path) -> str:
        """The licence the user is asked to accept, taken from the file without accepting it (consent items)."""
        raise NotImplementedError

    def accept(self, file: Path, names: list[str], scratch: Path) -> None:
        """Install with the licence accepted: only after the user's click (consent items)."""
        raise NotImplementedError


@dataclass(frozen=True)
class BodyModelFiles(Install):
    """A body model: unpacked into third_party/_body_models/<model>/ (a file on its own is copied there), where the
    workers find it (lab2shot_shared.body_models find)."""

    model: str

    def ready(self) -> bool:
        return self.path() is not None

    def path(self) -> Path | None:
        return bodies.find(self.model)

    def put(self, file: Path, names: list[str]) -> None:
        if file.is_dir():
            # a dropped folder: only the model file and the files beside it are taken (identify already refused deeper
            # nesting); the folder is not copied whole into _body_models/, otherwise a project folder that happens to
            # contain SMPLX_NEUTRAL.npz would be copied entirely and then moved into installed/
            model = bodies.MODELS[self.model]
            _copy_needed(file, bodies.ROOT / self.model, lambda n: _matches(n, (*model.markers, *model.files)))
            return
        files.unpack(file, bodies.ROOT / self.model)


@dataclass(frozen=True)
class ExtensionWeights(Install):
    """A model file the installer cannot fetch (its author put it on a file-sharing site that needs a browser):
    copied into third_party/<extension>/weights/, where that extension's worker finds it exactly like a weight the
    installer downloaded. `files` are all the files the item is made of; one of them is enough to install and to run
    the settings that use it, and a worker asking for one that is not there says which one and where it comes from."""

    extension: str
    files: tuple[str, ...]

    @property
    def folder(self) -> Path:
        return THIRD_PARTY_DIR / self.extension / "weights"

    def ready(self) -> bool:
        return any((self.folder / name).is_file() for name in self.files)

    def path(self) -> Path | None:
        return self.folder if self.ready() else None

    def put(self, file: Path, names: list[str]) -> None:
        if file.is_dir():  # unpack's `keep` does not apply to a folder (copytree): only the item's files come along
            _copy_needed(file, self.folder, lambda n: n in self.files, siblings=False)
        else:
            files.unpack(file, self.folder, keep=lambda n: n.rsplit("/", 1)[-1] in self.files)
        # an archive that wraps its files in a folder of its own (StableMotion's save/stablemotion/): the files
        # themselves are what this item is, so they end up in weights/, whatever the archive nested them under
        for found in sorted(self.folder.rglob("*")):
            if found.is_file() and found.parent != self.folder and found.name in self.files:
                found.replace(self.folder / found.name)
        for empty in sorted(self.folder.rglob("*"), reverse=True):
            if empty.is_dir() and not any(empty.iterdir()):
                empty.rmdir()


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name.lower(), p.lower()) for p in patterns)


def _copy_needed(src: Path, dest: Path, wanted, siblings: bool = True) -> None:
    """A dropped folder into `dest`: only the files `wanted(name)` takes (with, when `siblings`, the other files
    lying next to one: an archive's folder holds the model and its readme together), at their path inside the folder.
    Never the whole tree (io.files.unpack copies a folder entire)."""
    found = [p for p in sorted(src.rglob("*")) if p.is_file() and not p.is_symlink() and wanted(p.name)]
    if siblings:
        found = sorted({q for p in found for q in p.parent.iterdir() if q.is_file() and not q.is_symlink()})
    for p in found:
        target = dest / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def unwrap_licence(raw: bytes) -> str:
    """The licence text an installer printed, as it was written. Autodesk's installers break their text into lines
    of 80 bytes, cutting through words: every 81st byte of the text is an added line break. Those breaks are found
    (the longest run of line breaks 81 bytes apart that reaches the end of the text) and taken out; the first break
    of the run is the heading's own and stays. Their text is Windows-1252."""
    body = raw.rstrip()
    first, length = 0, 0  # the run: its first break and how many
    for start in range(81):
        run_first, run = start, 0
        for p in range(start, len(body), 81):
            if body[p] != 10:
                run_first, run = p + 81, 0
            else:
                run += 1
        if run > length:
            first, length = run_first, run
    text = bytearray(body)
    if length > 3:
        for p in reversed(range(first + 81, first + 81 * length, 81)):
            del text[p]
    return text.decode("cp1252", "replace").strip("\n").replace("\r\n", "\n")


# ------------------------------------------------------------------ the registry


@dataclass(frozen=True)
class ManualItem:
    key: str  # what extensions name: manual_weight(key)
    title: str  # the official name
    what: str  # what it is, for people
    page: str  # where it is downloaded
    download: str  # which download on that page, as the page names it
    filename: str  # the name the downloaded file has (the contents are what is recognised: it may be renamed)
    note: str  # what the page asks of the user: registration, licence
    markers: tuple[str, ...]  # names of files or folders inside the archive or folder that identify it (patterns)
    install: Install
    alone: tuple[str, ...] = ()  # the name a file needs when it comes on its own, not in its archive
    looks_like: tuple[str, ...] = ()  # file names meant to be this item (patterns): said so when it is not recognised
    nested: bool = True  # a marker may match at any depth inside a dropped archive or folder (a vendor archive with
    # the model buried in it). False for an item downloaded as a bare file: then only the dropped file itself or a
    # direct child of a dropped folder counts, so a large unrelated folder that happens to hold a copy of it deep
    # inside (a prefetch folder with one of every weight) is not taken for this item and moved away
    hint: Msg | None = None  # what to do instead when a file looks like this item but is not recognised
    noncommercial: bool = False
    registration: bool = False  # each user must register on its page for it (nodes/tags.py 需注册)

    def recognises(self, names: list[str]) -> bool:
        """Any file or folder name inside matches a marker (case-insensitive); `nested` False: only the top level."""
        wanted = names if self.nested else [n for n in names if "/" not in n]
        parts = {part.lower() for n in wanted for part in n.split("/") if part}
        return any(fnmatch.fnmatch(part, marker.lower()) for marker in self.markers for part in parts)


def _body(key: str) -> ManualItem:
    """A body model item, generated from the lab2shot_shared.body_models table (download page, file names and
    recognition markers are all in that table)."""
    model = bodies.MODELS[key]
    return ManualItem(
        key=key, title=model.title, what=model.what, page=model.page, download=model.download, filename=model.filename,
        note="要先在官网注册登录，仅限非商用科研，禁止再分发", markers=model.markers, install=BodyModelFiles(key),
        alone=model.files, looks_like=model.looks_like, noncommercial=True, registration=True,
        hint=Msg("W-MANUAL-NOTTHEFILE", title=model.title, file=model.files[0], download=model.download))


BODY_KEYS = ("smplx", "smpl", "mano", "flame")  # SMPL-X before SMPL: its archives' names contain "SMPL"


def items() -> dict[str, ManualItem]:
    """The registry, assembled, never hand-written here: the body models from lab2shot_shared.body_models (each shared
    by several extensions), then every item a loaded extension declares for itself (`Extension.manual_items`: the FBX
    SDK is the FBX extension's, StableMotion's checkpoint is StableMotion's). A third-party developer who needs the user
    to download something by hand declares it in their own extension.py; the core never learns the item's name."""
    from .registry import extensions

    out = {key: _body(key) for key in BODY_KEYS}
    for ext in extensions().values():
        for item in ext.manual_items:
            out.setdefault(item.key, item)
    return out


def item_of(ext, key: str) -> ManualItem:
    """The item one extension's manual weight names, without the registry: a body model from the shared table, else
    the extension's own declaration (`manual_items`). The installer asks this while the extensions are still being
    loaded (env_fingerprint runs on load), when items() would re-enter the loader."""
    if key in BODY_KEYS:
        return _body(key)
    for item in ext.manual_items:
        if item.key == key:
            return item
    raise KeyError(f"{ext.name} declares manual weight {key!r} but no manual_items entry for it")


def manual_weight(item: ManualItem) -> Weight:
    """What an extension declares for an item it declares itself (in `manual_items`) and needs the user to download by hand."""
    return Weight(key=item.key, kind="manual", source=item.key, dest="", note=f"{item.title}：{item.what}，{item.note}")


def body_model_weight(key: str) -> Weight:
    """What an extension declares for a body / hand / face model it needs (smplx, smpl, mano, flame): the item is the
    shared table's, so several extensions declare the same key and the inbox lists them all under it."""
    return manual_weight(_body(key))


def needed_by() -> dict[str, list[dict]]:
    """Item key -> the extensions that need it ({name, title})."""
    from .registry import extensions

    out: dict[str, list[dict]] = {key: [] for key in items()}
    for ext in extensions().values():
        for w in ext.weights:
            if w.kind == "manual":
                out[w.source].append({"name": ext.name, "title": ext.title})
    return out


# ------------------------------------------------------------------ files: what they carry, unpacking, moving


class ManualError(MessageError, RuntimeError):
    """What went wrong with a hand-downloaded file, for the user."""


def members(path: Path) -> list[str]:
    """The names a hand-downloaded file carries (io/files.py members); a file named like an archive that is none is
    refused (E-MANUAL-BADARCHIVE)."""
    if not path.is_dir() and not zipfile.is_zipfile(path) and not tarfile.is_tarfile(path) and files.is_archive_name(path.name):
        raise ManualError(Msg("E-MANUAL-BADARCHIVE"))
    return files.members(path)


def move_keeping(src: Path, folder: Path) -> list[Path]:
    """Move a file, or a folder with everything in it, into `folder` without replacing anything: a taken name gets
    " (2)", " (3)" ...; a folder already there is merged into, file by file. Returns where every file went."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / src.name
    if src.is_dir() and not src.is_symlink() and target.is_dir():
        moved = [p for child in sorted(src.iterdir()) for p in move_keeping(child, target)]
        src.rmdir()
        return moved
    target = _free_name(target)
    shutil.move(str(src), target)
    return sorted(p for p in target.rglob("*") if p.is_file()) if target.is_dir() else [target]


def _free_name(target: Path) -> Path:
    name = target.name
    compound = next((s for s in (".tar.gz", ".tar.bz2", ".tar.xz") if name.lower().endswith(s)), None)
    suffix = name[-len(compound):] if compound else (target.suffix if not target.is_dir() else "")
    stem = name[: len(name) - len(suffix)]
    n = 1
    while target.exists() or target.is_symlink():
        n += 1
        target = target.with_name(f"{stem} ({n}){suffix}")
    return target


# ------------------------------------------------------------------ the inbox


def ensure_inbox() -> Path:
    """The inbox with its README (made on start and on every check)."""
    INBOX.mkdir(parents=True, exist_ok=True)
    readme = INBOX / README
    if not readme.is_file() or readme.read_text(encoding="utf-8") != README_TEXT:
        readme.write_text(README_TEXT, encoding="utf-8")
    return INBOX


@dataclass(frozen=True)
class Match:
    item: ManualItem | None
    names: list[str]
    why: Msg | None  # None when recognised; else why not, for the user
    unrelated: bool = False  # a folder, or a file, plainly not meant for the inbox at all (view() groups these apart,
    # calmly, instead of listing them as "认不出的文件": test material dropped into a sub-folder of the inbox by
    # mistake is not an error: the inbox only looks at what items() names, and everything else is left alone)


_seen: dict[tuple[str, int, int], Match] = {}  # (path, size, mtime): big archives are read once


def identify(path: Path) -> Match:
    """Which item a dropped file (or folder) is, from what is inside it; a file on its own from its name. Anything
    else in the inbox that plainly is not one of items() (a folder, or a file whose name resembles nothing listed
    either) is `unrelated`, not an error: the inbox only ever uses the files items() names, everything else just sits
    there unused (see view())."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if not path.is_dir() and key in _seen:
        return _seen[key]
    names: list[str] = []
    item, why, unrelated = None, None, False
    if path.name.lower().endswith(PARTIAL_SUFFIXES):
        why = Msg("N-MANUAL-DOWNLOADING")
    else:
        try:
            names = members(path)
            item = next((item for item in items().values() if item.recognises(names)), None)
        except (ManualError, OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
            why = exc.message if isinstance(exc, ManualError) else Msg("W-MANUAL-UNREADABLE", detail=str(exc))
        except (RuntimeError, NotImplementedError) as exc:  # zipfile: encrypted / an unsupported compression method
            why = Msg("W-MANUAL-UNSUPPORTEDARCHIVE", detail=str(exc)[:200])
        if item is not None and path.is_dir() and _depth(item, names) > 1:
            # a project folder that happens to contain the model file deep inside is not recognised (otherwise it would be
            # moved whole into installed/); the message states what to drop in
            why = Msg("W-MANUAL-FOLDERNESTED", title=item.title, file="、".join(item.alone[:3] or item.markers[:3]))
    if item is None:  # not recognised: whose it looks like, by its name, to say so there
        low = path.name.lower()
        item = next((i for i in items().values() if any(fnmatch.fnmatch(low, p) for p in i.looks_like)), None)
        if why:
            pass  # already explained (still downloading, unreadable): that reason stands even if the name also matches
        elif item is not None:
            why = item.hint
            if not path.is_dir() and files.is_archive_name(path.name):  # a folder can't be "put in as an archive"
                why = Msg("W-MANUAL-REPACKED", hint=why)
        else:
            why, unrelated = Msg("N-MANUAL-UNRELATEDFOLDER" if path.is_dir() else "N-MANUAL-UNRELATEDFILE"), True
    match = Match(item, names, why, unrelated)
    if not path.is_dir():
        _seen[key] = match
    return match


def _depth(item: ManualItem, names: list[str]) -> int:
    """How deep inside a dropped folder the item's marker files lie: 0 directly in it, 1 in a folder of it ... (the
    least over every match; -1 when nothing matches)."""
    depths = [i for n in names for i, part in enumerate(n.split("/")) if part and _matches(part, item.markers)]
    return min(depths) if depths else -1


def _ours(path: Path) -> bool:
    """What the inbox keeps for itself, or holds for another part of Lab2Shot: never a dropped file."""
    return path.name in (README, DONE, DATASETS) or path.name.startswith(".")


def _zone_mark(path: Path) -> bool:
    """Windows' "downloaded from the internet" mark, which becomes a file of its own when a download is copied into
    WSL (name:Zone.Identifier). It is not the user's: check() deletes it."""
    if not path.name.endswith(":Zone.Identifier") or not path.is_file() or path.stat().st_size > 4096:
        return False
    return path.read_bytes().lstrip(b"\xef\xbb\xbf").startswith(b"[ZoneTransfer]")


def _dropped() -> list[tuple[Path, Match]]:
    """What the user dropped into the inbox, each with what it was recognised as."""
    if not INBOX.is_dir():
        return []
    # p.exists(): a link whose target is gone is not a file anyone dropped (identify() would fail on it)
    return [(p, identify(p)) for p in sorted(INBOX.iterdir()) if p.exists() and not _ours(p) and not _zone_mark(p)]


def _waits(m: Match) -> bool:
    """A recognised file that waits for the user: its licence is to be accepted (and this version is not installed)."""
    return m.item is not None and not m.why and m.item.install.consent and not m.item.install.has(m.item.install.version(m.names))


_lock = threading.RLock()


def check() -> dict:
    """Sort the inbox: install every recognised file that installs by itself (its original moves to installed/, as
    does a second copy of what is installed already) and delete Windows' download marks. What waits for the user and
    what is not recognised stays. Returns view()."""
    with _lock:
        ensure_inbox()
        for path in INBOX.iterdir():
            if _zone_mark(path):
                path.unlink()
        for path, m in _dropped():
            if m.item is None or m.why or _waits(m):
                continue
            try:
                if not m.item.install.consent:
                    m.item.install.put(path, m.names)
            except (OSError, zipfile.BadZipFile, tarfile.TarError, shutil.Error, RuntimeError, NotImplementedError) as exc:
                # RuntimeError / NotImplementedError: raised by zipfile for encrypted archives or unsupported compression
                # methods. They must be caught: the install preflight calls check() first, and one such file would make
                # it fail, turning every extension's preflight into a 500
                stat = path.stat()  # said from now on, until the file changes
                why = Msg("W-MANUAL-UNSUPPORTEDARCHIVE", detail=str(exc)[:200]) if isinstance(exc, (RuntimeError, NotImplementedError)) \
                    else Msg("E-MANUAL-INSTALLFAILED", detail=str(exc))
                _seen[(str(path), stat.st_size, stat.st_mtime_ns)] = Match(None, m.names, why)
                continue
            move_keeping(path, INBOX / DONE)
        return view()


def state(key: str, dropped: list[tuple[Path, Match]] | None = None) -> str:
    """Where an item stands: "ready"; "consent" (its file is in the inbox, waiting for the user to accept the
    licence); "unrecognised" (a file that looks like it is in the inbox, but not what is needed); "missing".
    A pure look: nothing is installed (that is check())."""
    item = items()[key]
    if item.install.ready():
        return "ready"
    mine = [m for _, m in (_dropped() if dropped is None else dropped) if m.item is item]
    if any(_waits(m) for m in mine):
        return "consent"
    return "unrecognised" if any(m.why for m in mine) else "missing"


def view() -> dict:
    """What the page's 手动下载 shows: the inbox (as Linux and Windows write it), every item with its state, the
    extensions that need it, where to download it, the file expected, the files of it in the inbox; the files nobody
    recognised that still need something from the user (a licence, or the right download, each with the reason); and,
    apart from those (calmly, not as a warning; see identify()), what else sits in the inbox that the inbox simply
    never looks at (a folder, an unrelated file)."""
    dropped, needs = _dropped(), needed_by()
    rows = []
    for key, item in items().items():
        files = [{"name": p.name, "state": "unrecognised", "why": m.why.text, "code": m.why.code} if m.why
                 else {"name": p.name, "state": "consent", "version": item.install.version(m.names)}
                 for p, m in dropped if m.item is item and (m.why or _waits(m))]
        installed = item.install.path()
        rows.append({
            "key": key, "title": item.title, "what": item.what, "page": item.page, "download": item.download,
            "filename": item.filename, "alone": list(item.alone), "note": item.note, "noncommercial": item.noncommercial,
            "consent": item.install.consent, "state": state(key, dropped), "files": files, "needed_by": needs[key],
            "installed": _shown(installed) if installed else "",
        })
    unknown = [{"name": p.name, "why": m.why.text, "code": m.why.code} for p, m in dropped if m.item is None and not m.unrelated]
    unrelated = [p.name + ("/" if p.is_dir() else "") for p, m in dropped if m.item is None and m.unrelated]
    return {"inbox": {"path": _shown(INBOX), "open": _openable(INBOX)}, "items": rows, "unknown": unknown,
            "unrelated": {"count": len(unrelated), "sample": unrelated[:5]}}


def explain(key: str, state: str) -> Msg:
    """One message for the user: what an item that is not ready still needs."""
    item = items()[key]
    if state == "consent":
        return Msg("N-MANUAL-CONSENT", title=item.title)
    return Msg("E-MANUAL-MISSING", title=item.title, what=item.what, page=item.page, download=item.download)


def _shown(path: Path) -> str:
    """How a path inside the project reads on the page: 「./Lab2Shot/downloads」, not a bare 「downloads」 nobody can
    place. The project's own name, not the checkout's folder name (a worktree is called something else). Outside the
    project it is the whole path."""
    try:
        return f"./Lab2Shot/{path.relative_to(ROOT).as_posix()}"
    except ValueError:
        return str(path)


def _openable(path: Path) -> str:
    r"""The path to put in a file manager's address bar. When the service runs in WSL and the browser on Windows, a
    Linux path opens nothing; there the UNC form does (\wsl.localhost\<distro>\home\...). Elsewhere the absolute
    path is already openable."""
    distro = os.environ.get("WSL_DISTRO_NAME")
    return "\\\\wsl.localhost\\" + distro + str(path).replace("/", "\\") if distro else str(path)


# ------------------------------------------------------------------ licences: shown, and accepted only by the user


def _scratch() -> Path:
    folder = settings().work_dir / "manual" / "scratch"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _consent_file(name: str) -> tuple[ManualItem, Path, Match]:
    """A file in the inbox recognised as an item whose install needs the user's consent. Where the name may lead is
    decided in the project's one place for it (io.files.inside: the name is resolved first, so a link planted in the
    inbox leads nowhere either), and it must be directly in the inbox, not in a folder of it."""
    try:
        path = files.inside(INBOX, name)
    except Invalid:
        raise ManualError(Msg("E-MANUAL-NOTININBOX", name=name)) from None
    if path.parent != INBOX.resolve() or not path.exists():
        raise ManualError(Msg("E-MANUAL-NOTININBOX", name=name))
    m = identify(path)
    if m.item is None or m.why or not m.item.install.consent:
        raise ManualError(Msg("E-MANUAL-NOCONSENT", name=name))
    return m.item, path, m


def _digest(path: Path) -> str:
    """A file's sha256; a folder's over each file's path and digest in order (lab2shot/io/digest.py tree)."""
    return tree(path)


def licence(name: str) -> dict:
    """The licence of an inbox file, taken from it without accepting it: {item, title, file, version, text, sha256}.
    Kept in work/manual/licences/ by the file's sha256 (the installer runs once per file)."""
    item, path, m = _consent_file(name)
    file_sha = _digest(path)
    cache = settings().work_dir / "manual" / "licences" / f"{file_sha}.json"
    try:
        text = json.loads(cache.read_text(encoding="utf-8"))["text"]
    except (OSError, ValueError, KeyError):
        with tempfile.TemporaryDirectory(dir=_scratch()) as scratch:
            text = item.install.licence(path, m.names, Path(scratch))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    return {"item": item.key, "title": item.title, "file": name, "version": item.install.version(m.names),
            "text": text, "sha256": sha256(text), "file_sha256": file_sha}


def accept(name: str, shown_sha256: str, who: dict) -> dict:
    """The user accepted the licence they were shown (its sha256) for an inbox file: record who, when, what and
    which version in the database (consents), then install it. Only the user's own acceptance calls this: the page's
    同意并安装, or the licence shown in full and agreed to at the command line (cli/setup.py _consent_pending)."""
    from ..database import db, json_text

    with _lock:
        shown = licence(name)
        if shown["sha256"] != shown_sha256:
            raise ManualError(Msg("E-MANUAL-LICENCECHANGED"))
        item, path, m = _consent_file(name)
        record = {"item": item.key, "title": item.title, "version": shown["version"], "file": name,
                  "file_sha256": shown["file_sha256"], "licence_sha256": shown["sha256"],
                  "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "who": who}
        with db().write() as c:
            c.execute("INSERT INTO consents (record) VALUES (?)", (json_text(record),))
        with tempfile.TemporaryDirectory(dir=_scratch()) as scratch:
            item.install.accept(path, m.names, Path(scratch))
        move_keeping(path, INBOX / DONE)
        return check()


