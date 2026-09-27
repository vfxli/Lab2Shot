"""What an answer made from slowly changing state depends on, as a key.

A route declaring `keyed=` (server/routes.py Access) is answered from its key: the key is its ETag, so a page that has
the answer gets 304 and the handler never runs, and another request with the same key gets the answer made once. A key
names everything the answer reads, each part cheap to look at (names, sizes and times of files, a count in the database),
never the answer itself:

    code(request)      this run of the server and its work folder, the node types it loaded, the template files (engine/templates.py
                       reads them again when they change)
    account(request)   what of the answer is this account's: its licence tags (nodes/tags.py), its role, whether it
                       sees the server's folders (access.py scrub), the answer fields it gets (available.py FIELDS)
    installs(request)  what the extensions' and hand downloads' state is read from: each extension's environment and
                       install record, the inbox (downloads/), what the hand downloads installed
                       (extensions/manual.py), the accepted licences, and the install tasks running now (server/installs.py
                       revision: the answer carries them)

catalog and templates put them together for their routes."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from starlette.requests import Request

from ..io.digest import sha256


def _digest(parts: tuple) -> str:
    return sha256(repr(parts))[:24]


def _stat(path: Path) -> tuple:
    try:
        s = path.stat()
        return (s.st_mtime_ns, s.st_size)
    except OSError:
        return ()


def _listing(folder: Path, depth: int = 2, skip: frozenset[str] = frozenset()) -> tuple:
    """Names, sizes and times of what is in `folder`, `depth` levels down (a file added, removed or replaced shows)."""
    try:
        entries = sorted(os.scandir(folder), key=lambda e: e.name)
    except OSError:
        return ()
    out = []
    for e in entries:
        if e.name in skip:
            continue
        try:
            s = e.stat(follow_symlinks=True)
        except OSError:
            continue
        inner = _listing(Path(e.path), depth - 1) if depth > 1 and e.is_dir(follow_symlinks=True) else ()
        out.append((e.name, s.st_mtime_ns, s.st_size, inner))
    return tuple(out)


def code(request: Request) -> tuple:
    from ..config import TEMPLATES_DIR, settings
    from ..nodes import node_types
    from . import restart

    return (restart.BOOT, str(settings().work_dir), tuple(sorted(node_types())),
            tuple((f.name, _stat(f)) for f in sorted(TEMPLATES_DIR.glob("*.json"))))


def account(request: Request) -> tuple:
    from ..availability import resolve
    from ..roles import SessionFacts
    from . import auth, available

    s = auth.session(request)
    if s is None:
        return (None,)
    allowed = None if s.user.allowed is None else tuple(sorted(s.user.allowed))
    return (allowed, s.user.role, s.can("logs.view"), tuple(resolve(available.FIELDS, SessionFacts.of(s)).available))


def installs(request: Request) -> tuple:
    from lab2shot_shared import body_models as bodies

    from ..config import THIRD_PARTY_DIR, settings
    from ..extensions import extensions, manual

    from .installs import revision

    exts = tuple((name, _stat(ext.paths.python), _stat(ext.paths.state_file)) for name, ext in sorted(extensions().items()))
    # third_party/ two levels down: every environment, and what a hand download installs there; the install tasks
    # running now, because the answer carries them (a card follows one and shows why it stopped)
    return (exts, _listing(manual.INBOX, 2, frozenset({manual.DATASETS})), _listing(bodies.ROOT), _listing(THIRD_PARTY_DIR),
            _listing(settings().work_dir / "manual"), revision())


def of(*parts: Callable[[Request], tuple]) -> Callable[[Request], str]:
    """The key of an answer made of these parts."""
    return lambda request: _digest(tuple(part(request) for part in parts))


def kept(request: Request) -> tuple:
    """What the administrator keeps in files and every account is handed: the template files and their tree, the two
    category trees and the node placements (lab2shot/categories.py), and every node's name and description
    (nodes/text.py)."""
    from .. import categories, library
    from ..nodes import text

    try:
        return (library.changed_at(), categories.changed_at(), text.stamp())
    except Exception:  # no work folder yet (the page before a login)
        return ()


catalog = of(code, account, installs, kept)  # the category tree rides in the catalogue
templates = of(code, account, kept)
