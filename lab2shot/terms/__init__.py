"""The Terms of Service and the Privacy Policy: the two texts everyone who uses this server agrees to, in every interface
language, and who agreed to which version in which language.

The texts are data, not code: <lang>/agreement.md and <lang>/privacy.md beside this file are the ones that come with the
program (one folder per language of lab2shot/i18n LANGS); an administrator's edited copy (the admin page's Registration
Settings → Terms of Service and Privacy Policy, the right terms.edit) is kept under the work folder
(EDITED/<lang>/<id>.md, every text of every language together) and, while it is there, is the one in effect. Restoring
the defaults removes it.

One version number covers both texts in every language. The texts in effect are compared with the newest version the
database knows (terms_versions, database/schema.py: one row per version and language) by their digest whenever they may
have changed (current()): different texts in any language, whoever changed them and however (an update of the program,
the administrator's save), are the next version, kept there word for word, so what anyone agreed to can always be read
again. The English texts are a translation; each says that the Chinese one prevails.

Who must agree (owed): every account but the owner's (the one who runs this server offers the texts; it does not
accept them), to the current version, in whichever language it read them, before it uses anything: registering agrees
in the same transaction that makes the account (lab2shot/site/registration.py), and every other account (one an
administrator made, or anyone after the texts changed) is asked when it next opens a page, and refused everything but
logging in and out until it agrees (server/access.py Guard). The machine's own token is the owner's, so the command
line is never asked. The language agreed in is the request's (terms_agreed.lang).

A text may name a few values of the settings (fills(): {task_keep_days} ...), filled in when it is shown, so it never
says a number the server no longer keeps to; the version is of the text as written, so changing such a setting is not
a new version."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .. import i18n
from ..accounts import ADMIN_ID, LOGIN_LOG_KEPT_S, SYSTEM, Actor
from ..config import settings
from ..database import db
from ..errors import Invalid
from ..io.atomic import write_text
from ..messages import Msg
from ..text import plain_lines
from ..io.digest import sha256

HERE = Path(__file__).parent
DOCS: tuple[str, ...] = ("agreement", "privacy")  # ids (the files' names); their titles: terms.title.<id>
EDITED = "terms"  # the administrator's copy: <work folder>/terms/<lang>/<id>.md
MOST = 20_000  # characters of one text an administrator may save (each is about a page)
DAY = 86400


def fills() -> dict[str, str]:
    """Placeholder -> what it says now."""
    s = settings()
    return {"{task_keep_days}": str(s["tasks.keep_days"]), "{login_log_days}": str(LOGIN_LOG_KEPT_S // DAY),
            "{database_backups}": str(s["database.backups"])}


def title(doc: str) -> str:
    """A text's title in the language now."""
    return i18n.t(f"terms.title.{doc}")


def lang_of(lang: str | None = None) -> str:
    """One of LANGS: `lang`, else the language now."""
    return i18n.normal(lang) or i18n.current()


@dataclass(frozen=True)
class Terms:
    version: int
    at: float  # when this version came into effect
    by: str  # who saved it; "" for the program's own text
    edited: bool  # the administrator's copy is in effect
    texts: dict[str, dict[str, str]]  # lang -> id -> the text as written

    def shown(self, lang: str | None = None) -> list[dict]:
        """The texts as a person reads them, in `lang` (else the language now): the placeholders filled in."""
        said = fills()
        lang = lang_of(lang)
        out = []
        for doc in DOCS:
            text = self.texts[lang][doc]
            for name, value in said.items():
                text = text.replace(name, value)
            out.append({"id": doc, "title": title(doc), "text": text, "lang": lang})
        return out


def _edited_dir() -> Path:
    return settings().work_dir / EDITED


def _sources() -> dict[tuple[str, str], Path]:
    """Where each text of each language is read from now: the administrator's copy when it is there (every text of
    every language), else the program's."""
    edited = {(lang, doc): _edited_dir() / lang / f"{doc}.md" for lang in i18n.LANGS for doc in DOCS}
    if all(p.is_file() for p in edited.values()):
        return edited
    return {(lang, doc): HERE / lang / f"{doc}.md" for lang in i18n.LANGS for doc in DOCS}


def _digest(texts: dict[str, dict[str, str]]) -> str:
    return sha256("\0".join(texts[lang][doc] for lang in i18n.LANGS for doc in DOCS))


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
        edited = next(iter(sources.values())).parent.parent != HERE
        texts: dict[str, dict[str, str]] = {lang: {} for lang in i18n.LANGS}
        for (lang, doc), p in sources.items():
            texts[lang][doc] = p.read_text(encoding="utf-8")
        digest = _digest(texts)
        r = c.execute("SELECT * FROM terms_versions ORDER BY version DESC LIMIT 1").fetchone()
        if r is None or r["digest"] != digest:
            version = (r["version"] + 1) if r is not None else 1
            at = time.time()
            for lang in i18n.LANGS:
                c.execute("INSERT INTO terms_versions (version, lang, at, by, by_id, digest, agreement, privacy) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (version, lang, at, by.label, by.id, digest, texts[lang]["agreement"], texts[lang]["privacy"]))
            r = c.execute("SELECT * FROM terms_versions WHERE version = ? LIMIT 1", (version,)).fetchone()
        rows = c.execute("SELECT * FROM terms_versions WHERE version = ?", (r["version"],)).fetchall()
        kept = {row["lang"]: {doc: row[doc] for doc in DOCS} for row in rows}
        found = Terms(version=r["version"], at=r["at"], by=r["by"], edited=edited, texts=kept)
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


def record(c, user_id: int, version: int, ip: str, lang: str | None = None) -> None:
    """That the account agreed to `version` now, read in `lang` (else the language now), inside the caller's
    transaction `c`, if it is the current one: otherwise the texts changed since the page showed them, and they must be
    read again (E-TERMS-CHANGED)."""
    now = current()
    if version != now.version:
        raise Invalid(Msg("E-TERMS-CHANGED"))
    lang = lang_of(lang)
    if lang not in now.texts:  # a version kept before this language was there: what was read is the one there is
        lang = next(iter(now.texts))
    c.execute("INSERT OR IGNORE INTO terms_agreed (user_id, version, lang, at, ip) VALUES (?, ?, ?, ?, ?)",
              (user_id, version, lang, time.time(), ip[:100]))


def agree(user_id: int, version: int, ip: str, lang: str | None = None) -> None:
    with db().write() as c:
        record(c, user_id, version, ip, lang)


def agreed_count(version: int) -> dict:
    """How many of the accounts that must agree (usable ones, the owner's aside) agreed to `version`."""
    r = db().row("SELECT COUNT(*) AS n, COUNT(a.user_id) AS agreed FROM users u "
                 "LEFT JOIN terms_agreed a ON a.user_id = u.id AND a.version = ? "
                 "WHERE u.deleted IS NULL AND u.enabled = 1 AND u.id != ?", (version, ADMIN_ID))
    return {"accounts": r["n"], "agreed": r["agreed"]}


def edit(texts: dict[str, dict[str, object]], by: Actor) -> Terms:
    """The administrator's copy of both texts in every language ({id: {lang: text}}), in effect from now: each made
    plain lines of text (text.py plain_lines), none empty nor longer than MOST. The same texts as now change nothing;
    different ones are the next version, which everyone is asked to agree to."""
    kept: dict[str, dict[str, str]] = {lang: {} for lang in i18n.LANGS}
    for doc in DOCS:
        given = texts.get(doc) if isinstance(texts.get(doc), dict) else {}
        for lang in i18n.LANGS:
            text = plain_lines(given.get(lang), MOST + 1)
            where = f"{title(doc)} ({i18n.t(f'lang.{lang}')})"
            if not text:
                raise Invalid(Msg("E-TERMS-EMPTY", title=where))
            if len(text) > MOST:
                raise Invalid(Msg("E-TERMS-TOOLONG", title=where, most=MOST))
            kept[lang][doc] = text + "\n"  # a text file ends with a line break, as the program's do
    with db().write():
        if _digest(kept) == _digest(current().texts):
            return current()
        for lang, docs in kept.items():
            for doc, text in docs.items():
                write_text(_edited_dir() / lang / f"{doc}.md", text)
        return _look(by)


def reset(by: Actor) -> Terms:
    """Back to the program's own texts: the administrator's copy goes (a new version, unless it said the same)."""
    with db().write():
        for lang in i18n.LANGS:
            for doc in DOCS:
                (_edited_dir() / lang / f"{doc}.md").unlink(missing_ok=True)
        return _look(by)
