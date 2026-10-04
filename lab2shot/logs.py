"""The server's log: work/logs/lab2shot.log, one line per event, so problems can be traced afterwards.

What goes in: requests that failed and every request that changes something, jobs (submitted, started, finished,
failed, stopped), uploads, the administrator's actions, errors with their traceback, and what users send from
their browser's log window. The file is rotated at the size the administrator set (logs.max_mb), keeping logs.files
old files; with logs.debug on it also takes the finer records (DEBUG). Both apply at once (/admin 设置).
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

from . import i18n
from .config import settings
from .messages import Msg, localized

FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"

_lock = threading.Lock()


def log_file() -> Path:
    return settings().work_dir / "logs" / "lab2shot.log"


class WorkLogHandler(logging.Handler):
    """Appends to the current work folder's log (resolved per record: each server uses its own), by the
    settings as they are now."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.INFO or bool(settings()["logs.debug"])

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            path = log_file()
            s = settings()
            keep = int(s["logs.files"])
            with _lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.stat().st_size > int(s["logs.max_mb"]) << 20:
                    for old in path.parent.glob(f"{path.name}.*"):  # the oldest, and any beyond a smaller count
                        if old.suffix[1:].isdigit() and int(old.suffix[1:]) >= keep:
                            old.unlink()
                    for i in range(keep - 1, 0, -1):
                        older = path.with_name(f"{path.name}.{i}")
                        if older.exists():
                            older.replace(path.with_name(f"{path.name}.{i + 1}"))
                    path.replace(path.with_name(f"{path.name}.1"))
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            self.handleError(record)


def get(area: str) -> logging.Logger:
    """The logger for one part of Lab2Shot (farm, http, uploads, admin, client ...)."""
    root = logging.getLogger("lab2shot")
    if not any(isinstance(h, WorkLogHandler) for h in root.handlers):
        handler = WorkLogHandler()
        handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)  # the handler lets DEBUG through when logs.debug is on
    return root.getChild(area)


CONTINUED = "    | "  # how each line after a record's first begins (say)


def say(logger: logging.Logger, message, detail: str = "", *, about: str = "", debug: bool = False) -> None:
    """One line of the server's log, by its message: the only way anything of Lab2Shot writes a line (no logger call
    outside this function). "[CODE] about text", at the level its letter says
    (E error, W warning, anything else info; `debug`: the finer records, kept only with logs.debug on), its text in
    English (i18n.LOG_LANG; a sentence not yet translated reads in Chinese until its area is moved). `message` is a Msg,
    or a message as it travelled ({code, text}: a job's event); `about` what it is about when the message does not say
    (a job and its node); `detail` (a traceback, a worker's line) on the lines after it."""
    with i18n.using(i18n.LOG_LANG):  # the log is the back end's: always English, whoever's work it was
        code, text, letter = ((message.code, message.text, message.level) if isinstance(message, Msg)
                              else (message["code"], localized(message)["text"], message["code"][0]))
    level = logging.DEBUG if debug else {"E": logging.ERROR, "W": logging.WARNING}.get(letter, logging.INFO)
    # every line after a record's first is marked as its continuation: what came from outside (a page's log window, a
    # name) can never write a line that reads as a record of its own
    body = f"{about} " if about else ""
    body += text + (f"\n{detail}" if detail else "")
    logger.log(level, "[%s] %s", code, body.replace("\r", "").replace("\n", "\n" + CONTINUED))


def error_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc)).rstrip()


def tail(lines: int = 500) -> list[str]:
    """The last lines of the log (the current file, then the previous one when that is not enough)."""
    out: list[str] = []
    for path in (log_file(), log_file().with_name(log_file().name + ".1")):
        if len(out) >= lines or not path.exists():
            break
        with _lock:
            text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        out = text[-(lines - len(out)):] + out
    return out
