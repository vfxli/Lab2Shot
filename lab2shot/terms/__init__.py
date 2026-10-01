"""用户协议与隐私政策: the two texts everyone who uses this server agrees to, and who agreed to which version.

The texts are data, not code: agreement.md and privacy.md beside this file are the ones that come with the program;
an administrator's edited copy (后台「注册设置」→「用户协议与隐私政策」, the right terms.edit) is kept under the work
folder (EDITED/<id>.md, both texts together) and, while it is there, is the one in effect. 恢复默认 removes it.

One version number covers both texts. The texts in effect are compared with the newest version the database knows
(terms_versions, database/schema.py) by their digest whenever they may have changed (current()): different texts,
whoever changed them and however (an update of the program, the administrator's save), are the next version, kept
there word for word, so what anyone agreed to can always be read again.

Who must agree (owed): every account but the owner's (the one who runs this server offers the texts; it does not
accept them), to the current version, before it uses anything: registering agrees in the same transaction that makes
the account (lab2shot/registration.py), and every other account (one an administrator made, or anyone after the
texts changed) is asked when it next opens a page, and refused everything but logging in and out until it agrees
(server/access.py Guard). The machine's own token is the owner's, so the command line is never asked.

A text may name a few values of the settings (fills(): {任务保留天数} ...), filled in when it is shown, so it never says a
number the server no longer keeps to; the version is of the text as written, so changing such a setting is not a new
version."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from ..accounts import ADMIN_ID, LOGIN_LOG_KEPT_S, SYSTEM, Actor
from ..config import settings
from ..database import db
from ..errors import Invalid
from ..io.atomic import write_text
from ..messages import Msg
from ..text import plain_lines

HERE = Path(__file__).parent
DOCS: dict[str, str] = {"agreement": "用户协议", "privacy": "隐私政策"}  # id (the file's name) -> its title
EDITED = "terms"  # the administrator's copy: <work folder>/terms/<id>.md
MOST = 20_000  # characters of one text an administrator may save (each is about a page)
DAY = 86400


def fills() -> dict[str, str]:
    """Placeholder -> what it says now."""
    s = settings()
    return {"{任务保留天数}": str(s["tasks.keep_days"]), "{登录记录保留天数}": str(LOGIN_LOG_KEPT_S // DAY),
            "{数据库备份份数}": str(s["database.backups"])}


@dataclass(frozen=True)
class Terms:
    version: int
    at: float  # when this version came into effect
    by: str  # who saved it; "" for the program's own text
    edited: bool  # the administrator's copy is in effect
    texts: dict[str, str]  # id -> the text as written

    def shown(self) -> list[dict]:
        """The texts as a person reads them: the placeholders filled in."""
        said = fills()
        out = []
        for doc, title in DOCS.items():
            text = self.texts[doc]
            for name, value in said.items():
                text = text.replace(name, value)
            out.append({"id": doc, "title": title, "text": text})
        return out


def _edited_dir() -> Path:
    return settings().work_dir / EDITED


def _sources() -> dict[str, Path]:
    """Where each text is read from now: the administrator's copy when it is there (both texts), else the program's."""
    edited = {doc: _edited_dir() / f"{doc}.md" for doc in DOCS}
    return edited if all(p.is_file() for p in edited.values()) else {doc: HERE / f"{doc}.md" for doc in DOCS}


def _digest(texts: dict[str, str]) -> str:
    return hashlib.sha256("\0".join(texts[doc] for doc in DOCS).encode("utf-8")).hexdigest()


_known: tuple[tuple, Terms] | None = None  # (the sources' stamp, the version they are) of the last look


def _stamp(sources: dict[str, Path]) -> tuple:
    return (str(db().path), *((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sources.values()))


def _look(by: Actor = SYSTEM) -> Terms:
    """The texts in effect as a version: the newest one the database knows when they are the same, else a new one
    kept now (`by`: who saved them). Looked at again only when a source file changed. The database's write lock is
    the one lock: whatever changes the texts or a version holds it (edit, reset), so no two versions are made of one
    change, and a caller already inside a transaction (registering) never waits on a lock of its own here."""
    global _known
    known = _known
    if known is not None and known[0] == _stamp(_sources()):
        return known[1]
    with db().write() as c:
        sources = _sources()
        stamp = _stamp(sources)
        edited = next(iter(sources.values())).parent != HERE
        texts = {doc: p.read_text(encoding="utf-8") for doc, p in sources.items()}
        digest = _digest(texts)
        r = c.execute("SELECT * FROM terms_versions ORDER BY version DESC LIMIT 1").fetchone()
        if r is None or r["digest"] != digest:
            version = (r["version"] + 1) if r is not None else 1
            c.execute("INSERT INTO terms_versions (version, at, by, by_id, digest, agreement, privacy) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (version, time.time(), by.label, by.id, digest, texts["agreement"], texts["privacy"]))
            r = c.execute("SELECT * FROM terms_versions WHERE version = ?", (version,)).fetchone()
        found = Terms(version=r["version"], at=r["at"], by=r["by"], edited=edited, texts={doc: r[doc] for doc in DOCS})
        _known = (stamp, found)
        return found


def current() -> Terms:
    """The version in effect now."""
    return _look()


def owed(user) -> int | None:
    """The version this account has to agree to before it goes on (None: nothing: it agreed to the current one, or
    it is the owner's)."""
    if user.owner:
        return None
    version = current().version
    found = db().row("SELECT 1 FROM terms_agreed WHERE user_id = ? AND version = ?", (user.id, version))
    return None if found else version


def record(c, user_id: int, version: int, ip: str) -> None:
    """That the account agreed to `version` now (inside the caller's transaction `c`), if it is the current one:
    otherwise the texts changed since the page showed them, and they must be read again (E-TERMS-CHANGED)."""
    if version != current().version:
        raise Invalid(Msg("E-TERMS-CHANGED"))
    c.execute("INSERT OR IGNORE INTO terms_agreed (user_id, version, at, ip) VALUES (?, ?, ?, ?)",
              (user_id, version, time.time(), ip[:100]))


def agree(user_id: int, version: int, ip: str) -> None:
    with db().write() as c:
        record(c, user_id, version, ip)


def agreed_count(version: int) -> dict:
    """How many of the accounts that must agree (usable ones, the owner's aside) agreed to `version`."""
    r = db().row("SELECT COUNT(*) AS n, COUNT(a.user_id) AS agreed FROM users u "
                 "LEFT JOIN terms_agreed a ON a.user_id = u.id AND a.version = ? "
                 "WHERE u.deleted IS NULL AND u.enabled = 1 AND u.id != ?", (version, ADMIN_ID))
    return {"accounts": r["n"], "agreed": r["agreed"]}


def edit(texts: dict[str, object], by: Actor) -> Terms:
    """The administrator's copy of both texts, in effect from now: each made plain lines of text (text.py
    plain_lines), neither empty nor longer than MOST. The same texts as now change nothing; different ones are the
    next version, which everyone is asked to agree to."""
    clean = {doc: plain_lines(texts.get(doc), MOST + 1) for doc in DOCS}
    for doc, text in clean.items():
        if not text:
            raise Invalid(Msg("E-TERMS-EMPTY", title=DOCS[doc]))
        if len(text) > MOST:
            raise Invalid(Msg("E-TERMS-TOOLONG", title=DOCS[doc], most=MOST))
    kept = {doc: text + "\n" for doc, text in clean.items()}  # a text file ends with a line break, as the program's do
    with db().write():
        if _digest(kept) == _digest(current().texts):
            return current()
        for doc, text in kept.items():
            write_text(_edited_dir() / f"{doc}.md", text)
        return _look(by)


def reset(by: Actor) -> Terms:
    """Back to the program's own texts: the administrator's copy goes (a new version, unless it said the same)."""
    with db().write():
        for doc in DOCS:
            (_edited_dir() / f"{doc}.md").unlink(missing_ok=True)
        return _look(by)
