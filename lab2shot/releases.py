"""Release notes: what changed in each version, shown by the top bar's 「更新说明」 dialog. They live in one file at the
root of the repository, CHANGELOG.toml, so a one-click update brings the notes with the code and publishing a version
only means adding a table at its top; no code changes. The file:

    project = "https://github.com/..."     the project's address, shown first in the dialog
    [[release]]                             one table per version, the newest first
    name = {zh = "…", en = "…"}             its name
    date                                    a TOML date
    about = {zh = "…", en = "…"}            one sentence
    changes = {zh = [...], en = [...]}      a few short lines for everyone (may be empty), the same count in both
    admin = {zh = [...], en = [...]}        optional: lines only administrators are shown (the back office)

Every text is in both languages (i18n.LANGS), as a built-in template's words: the route sends both and the page shows
the one it speaks (webui/src/i18n/t.ts pick), so switching the page's language needs no new request.

The server reads it once when it starts (current()); `lab2shot check releases` runs the same reading and reports every
problem (problems), so a broken file is caught before it ships. The text is data only: the page shows it as plain text.

This module is product-layer data and does not import `lab2shot.server` (the layering rule); the route is in
server/releases.py.
"""

from __future__ import annotations

import datetime
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

from . import i18n
from .config import ROOT

FILE = ROOT / "CHANGELOG.toml"
FIELDS = ("name", "date", "about", "changes")  # every version has these
OPTIONAL = ("admin",)  # lines for administrators only: served to them alone (server/releases.py)


@dataclass(frozen=True)
class Notes:
    """What the file says: the project's address and the versions, newest first (plain JSON-ready values; the date as
    YYYY-MM-DD). `problems`: why it cannot be shown, in the words the check prints; when there is any, the rest is
    empty rather than half read."""

    project: str = ""
    releases: tuple[dict, ...] = ()
    problems: tuple[str, ...] = ()


def read(file: Path = FILE) -> Notes:
    """Read and check the release notes file: every problem it has, or the notes."""
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Notes(problems=(i18n.t("release.no_file"),))
    except tomllib.TOMLDecodeError as exc:
        return Notes(problems=(i18n.t("release.not_toml", why=exc),))
    except (OSError, UnicodeDecodeError) as exc:
        return Notes(problems=(i18n.t("release.unreadable", why=exc),))
    bad: list[str] = []
    project = data.get("project")
    if not _https(project):
        bad.append(i18n.t("release.project", now=repr(project)))
    extra = sorted(set(data) - {"project", "release"})
    if extra:
        bad.append(i18n.t("release.top_unknown", keys=i18n.separator().join(extra)))
    rows = data.get("release")
    if not isinstance(rows, list) or not rows:
        bad.append(i18n.t("release.none"))
        rows = []
    releases, names = [], set()
    for i, row in enumerate(rows, 1):
        where = i18n.t("release.nth", n=i)
        if not isinstance(row, dict):
            bad.append(i18n.t("release.not_table", where=where))
            continue
        missing = [k for k in FIELDS if k not in row]
        unknown = sorted(set(row) - set(FIELDS) - set(OPTIONAL))
        if missing:
            bad.append(i18n.t("release.missing", where=where, keys=i18n.separator().join(missing)))
        if unknown:
            bad.append(i18n.t("release.unknown", where=where, keys=i18n.separator().join(unknown)))
        name, date, about, changes = (row.get(k) for k in FIELDS)
        if "name" in row and not i18n.is_both(name):
            bad.append(i18n.t("release.name", where=where))
        elif i18n.is_both(name):
            where = i18n.t("release.named", where=where, name=i18n.pick(name))
            if any((lang, name[lang]) in names for lang in i18n.LANGS):
                bad.append(i18n.t("release.name_again", where=where))
            names.update((lang, name[lang]) for lang in i18n.LANGS)
        # a TOML date without quotes; a date-time (with a time of day) is a datetime, which is a date too: refused
        if "date" in row and not (isinstance(date, datetime.date) and not isinstance(date, datetime.datetime)):
            bad.append(i18n.t("release.date", where=where, now=repr(date)))
            date = None
        if "about" in row and not i18n.is_both(about):
            bad.append(i18n.t("release.about", where=where))
        if "changes" in row:
            bad += _lines(changes, "release.changes", where)
        admin = row.get("admin", {lang: [] for lang in i18n.LANGS})
        bad += _lines(admin, "release.admin", where)
        if releases and isinstance(date, datetime.date) and releases[-1]["date"] and date.isoformat() > releases[-1]["date"]:
            bad.append(i18n.t("release.order", where=where, date=date, above=releases[-1]["date"]))
        releases.append({"name": name, "date": date.isoformat() if isinstance(date, datetime.date) else "",
                         "about": about, "changes": changes, "admin": admin})
    if bad:
        return Notes(problems=tuple(bad))
    return Notes(project=project, releases=tuple(releases))


@cache
def current() -> Notes:
    """The notes as the server shows them: the file read once, on first use (the server does it when it starts)."""
    return read()


def _https(value: object) -> bool:
    if not isinstance(value, str) or value != value.strip() or any(c.isspace() for c in value):
        return False
    parts = urlsplit(value)
    return parts.scheme == "https" and bool(parts.hostname)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _lines(value: object, key: str, where: str) -> list[str]:
    """What is wrong with a list of lines in both languages ({zh: [...], en: [...]}: non-empty text, one line each, as
    many in every language): `key` the field's word (release.changes / release.admin)."""
    if not (isinstance(value, dict) and set(value) == set(i18n.LANGS)
            and all(isinstance(v, list) and all(_text(c) for c in v) for v in value.values())):
        return [i18n.t(key, where=where)]
    counts = {lang: len(value[lang]) for lang in i18n.LANGS}
    if len(set(counts.values())) > 1:
        return [i18n.t("release.count", where=where, field=key.rpartition(".")[2], **counts)]
    return []
