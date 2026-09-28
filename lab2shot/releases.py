"""Release notes: what changed in each version, shown by the top bar's 「更新说明」 dialog. They live in one file at the
root of the repository, CHANGELOG.toml, so a one-click update brings the notes with the code and publishing a version
only means adding a table at its top; no code changes. The file:

    project = "https://github.com/..."     the project's address, shown first in the dialog
    [[release]]                             one table per version, the newest first
    name, date (a TOML date), about         名称, 日期, 说明
    changes = ["...", ...]                  更新: a few short lines for everyone (may be empty)
    admin = ["...", ...]                    optional: lines only administrators are shown (the back office)

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
        return Notes(problems=("文件不存在",))
    except tomllib.TOMLDecodeError as exc:
        return Notes(problems=(f"不是正确的 TOML：{exc}",))
    except (OSError, UnicodeDecodeError) as exc:
        return Notes(problems=(f"文件读不出来：{exc}",))
    bad: list[str] = []
    project = data.get("project")
    if not _https(project):
        bad.append(f"project 应是 https:// 开头的网址，现在是 {project!r}")
    extra = sorted(set(data) - {"project", "release"})
    if extra:
        bad.append(f"文件顶层有不认识的项：{'、'.join(extra)}")
    rows = data.get("release")
    if not isinstance(rows, list) or not rows:
        bad.append("一个版本也没有：至少要有一段 [[release]]")
        rows = []
    releases, names = [], set()
    for i, row in enumerate(rows, 1):
        where = f"第 {i} 个版本"
        if not isinstance(row, dict):
            bad.append(f"{where}不是一段 [[release]] 表")
            continue
        missing = [k for k in FIELDS if k not in row]
        unknown = sorted(set(row) - set(FIELDS) - set(OPTIONAL))
        if missing:
            bad.append(f"{where}缺少 {'、'.join(missing)}")
        if unknown:
            bad.append(f"{where}有不认识的项：{'、'.join(unknown)}")
        name, date, about, changes = (row.get(k) for k in FIELDS)
        if "name" in row and not _text(name):
            bad.append(f"{where}的 name 应是一段不空的文字")
        elif _text(name):
            where = f"{where}（{name}）"
            if name in names:
                bad.append(f"{where}的名称和前面的版本重复了")
            names.add(name)
        # a TOML date without quotes; a date-time (with a time of day) is a datetime, which is a date too: refused
        if "date" in row and not (isinstance(date, datetime.date) and not isinstance(date, datetime.datetime)):
            bad.append(f"{where}的 date 应是不加引号的日期 YYYY-MM-DD，现在是 {date!r}")
            date = None
        if "about" in row and not _text(about):
            bad.append(f"{where}的 about 应是一段不空的文字")
        if "changes" in row and not (isinstance(changes, list) and all(_text(c) for c in changes)):
            bad.append(f"{where}的 changes 应是一列不空的文字，每条一行")
        admin = row.get("admin", [])
        if not (isinstance(admin, list) and all(_text(c) for c in admin)):
            bad.append(f"{where}的 admin 应是一列不空的文字，每条一行")
        if releases and isinstance(date, datetime.date) and releases[-1]["date"] and date.isoformat() > releases[-1]["date"]:
            bad.append(f"{where}的日期 {date} 比上面一个版本（{releases[-1]['date']}）新：最新的版本要写在最上面")
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
