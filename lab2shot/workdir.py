"""Which copy of Lab2Shot a work folder belongs to.

A work folder (work/: the database, the cache, uploads, task folders) is used by one checkout of Lab2Shot: the code
that migrates its database and writes its files. It says which one in work/owner.json ({"root": <the checkout>}).
Whatever opens the folder's records (database.Database, the one door) calls `own()` first:

    owner.json names this checkout            it opens
    owner.json names another checkout         refused (E-WORK-OTHERCHECKOUT), nothing touched
    no owner.json, no database yet            a new folder: this checkout claims it
    no owner.json, a database already there   claimed only when it is this checkout's own work/ (ROOT/work);
                                              anywhere else refused (E-WORK-UNCLAIMED)
                                              until `lab2shot db claim` is run from the checkout that should own it

Why: a process of another checkout (a test run in a worktree whose LAB2SHOT_WORK_DIR points here, say) that opens the
database migrates it to a schema version this checkout's code does not know, and this server no longer starts. So a
process of another checkout can't open it at all, whatever its environment says.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import ROOT
from .errors import MessageError
from .messages import Msg

OWNER = "owner.json"


class WorkDirError(MessageError, RuntimeError):
    """This process may not use this work folder; the message says whose it is and what to do."""


def owner(work_dir: Path) -> str | None:
    """The checkout the work folder belongs to (None: it does not say)."""
    try:
        return str(json.loads((Path(work_dir) / OWNER).read_text(encoding="utf-8"))["root"])
    except FileNotFoundError:
        return None


def claim(work_dir: Path) -> None:
    """This checkout owns the work folder from now on (`lab2shot db claim`, or a new folder's first use)."""
    from .io.atomic import write_text

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    write_text(work / OWNER, json.dumps({"root": str(ROOT), "claimed": time.time()}, ensure_ascii=False) + "\n")


def own(work_dir: Path) -> None:
    """Raises WorkDirError when this checkout may not open the work folder's records; claims a new one."""
    work = Path(work_dir).expanduser().resolve()
    found = owner(work)
    if found is not None:
        if found != str(ROOT):
            raise WorkDirError(Msg("E-WORK-OTHERCHECKOUT", work=str(work), owner=found, root=str(ROOT)))
        return
    if (work / "db" / "lab2shot.db").exists() and work != (ROOT / "work").resolve():
        raise WorkDirError(Msg("E-WORK-UNCLAIMED", work=str(work), root=str(ROOT)))
    claim(work)


def main_checkout() -> Path:
    """The main checkout this one belongs to (a git worktree's own common folder), itself when it is the main one."""
    import subprocess

    try:
        common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ROOT
    return Path(common).resolve().parent


