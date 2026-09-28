"""A user's words made safe to keep, show and put in the name of a file the user's machine gets. The lowest layer (it
imports nothing of Lab2Shot): whatever takes text a person typed or picked (registration, transfer, the server's
downloads) cleans it here, so the rules exist once."""

from __future__ import annotations

import re
import unicodedata

# What a user typed or picked (a file's name, a graph's name, a group's name) is data only: made plain text in one place
# before it is kept, shown or put in a file name the user's machine gets. Never part of a path on the server.
_BREAKS = "\t\r\n\v\f\x85\u2028\u2029"  # line breaks, the line and paragraph separators among them: each becomes a space
_BIDI = {*range(0x202A, 0x202F), *range(0x2066, 0x206A)}  # direction overrides and isolates: they turn the text around them
_UNSAFE = re.compile(r'[/\\:*?"<>|]+')  # path separators and what Windows refuses in a file name


def plain_text(text: object, most: int) -> str:
    """A user's words as one line of plain text: Unicode NFC, without control characters, surrogates, private-use
    characters and direction overrides, line breaks and runs of whitespace as one space, trimmed, at most `most`
    characters. What a page shows and the database keeps (a group's name)."""
    s = unicodedata.normalize("NFC", str(text or ""))
    s = "".join(" " if ch in _BREAKS else ch for ch in s
                if ch in _BREAKS or (unicodedata.category(ch) not in ("Cc", "Cs", "Co") and ord(ch) not in _BIDI))
    return " ".join(s.split())[:most].strip()


def plain_lines(text: object, most: int) -> str:
    """Words written as lines (a document the administrator edits): each line as plain_text makes one (a tab, a run
    of spaces as one space), line breaks kept as \\n, more than one blank line in a row as one, trimmed, at most
    `most` characters."""
    s = unicodedata.normalize("NFC", str(text or "")).replace("\r\n", "\n")
    lines = [plain_text(line, len(line)) for line in re.split(r"[\n\r\v\f\x85  ]", s)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()[:most].strip()


def file_part(text: object, most: int) -> str:
    """A user's words made fit for part of a file name the user's machine gets (a download's name, a folder inside it;
    never a path on the server): plain_text, without any format character either, path separators and what Windows
    refuses as `_`, spaces as `_`, no dots or `_` at either end, at most `most` characters. "" when nothing is left."""
    s = "".join(ch for ch in plain_text(text, 4 * most + 64) if unicodedata.category(ch) != "Cf")
    return _UNSAFE.sub("_", s).replace(" ", "_")[:most].strip("._ ")
