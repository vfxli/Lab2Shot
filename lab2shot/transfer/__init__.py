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
from ..io.files import NAME_MAX, PATH_MAX  # what a name on the disk may be: one rule (io/files.py)
from ..messages import Msg

# bytes a request of the upload routes that carry many names or a file's head may take (server/routes.py BIG_BODY, which
# those routes declare; server/access.py enforces it). It lives in this layer because uploads.py sizes the declared head
# by it (HEAD_MAX), and transfer may not import server.
BODY_MAX = 32 << 20


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
