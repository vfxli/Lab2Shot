"""Files that travel between the user's machine and this server. Nodes never read or write the user's paths: the
user's files come in as uploads, and what output-settings nodes write goes back as outputs of a task.

uploads     plates, videos and camera files the user picked, kept by content (the same bytes once)
tasks       a task's folder: its graph, its footage, its outputs and its logs
outputs     what 「输出」 collected and packed in a task: a folder DCC plugins fetch files from, and its zip
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from ..errors import Invalid
from ..messages import Msg

# bytes one HTTP request may carry (server/routes.py MAX_BODY reads it from here; server/access.py enforces it): a graph, a
# form — far below this. It lives in this layer because uploads.py sizes the declared head by it (HEAD_MAX), and transfer
# may not import server.
BODY_MAX = 32 << 20


NAME_MAX = 255  # bytes one part of a name may take: what every file system this runs on allows (Linux NAME_MAX, NTFS 255)
PATH_MAX = 1024  # bytes the whole name may take, sub-folders included


def relative_name(name: str) -> PurePosixPath:
    """A file name inside an upload or an output: sub-folders allowed, never a drive, an absolute path, "." or "..",
    never a control character, and never longer than a file system takes — a name the disk would refuse is refused
    here, said in words, instead of failing as an internal error while it is being written."""
    p = PurePosixPath(name.replace("\\", "/"))
    if not name or p.is_absolute() or not p.parts or set(p.parts) & {"..", "."} \
            or re.match(r"^[A-Za-z]:", name) or re.search(r"[\x00-\x1f]", name):
        raise Invalid(Msg("E-UPLOAD-BADNAME", name=repr(name[:200])))
    if len(str(p).encode()) > PATH_MAX or any(len(part.encode()) > NAME_MAX for part in p.parts):
        raise Invalid(Msg("E-UPLOAD-NAMETOOLONG", name=repr(name[:80]), most=NAME_MAX, whole=PATH_MAX))
    return p
