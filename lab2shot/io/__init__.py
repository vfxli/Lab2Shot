"""Standard file I/O shared by every adapter: sequences, video, color, USD."""

from ..errors import MessageError


class FileProblem(MessageError, OSError):
    """A file that cannot be opened, read or written; the message names the file and the cause. Subclasses OSError."""

    status = 422


class NotThere(MessageError, FileNotFoundError):
    """A missing file or folder; the message names the path. Subclasses FileNotFoundError."""

    status = 404
