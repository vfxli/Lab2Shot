"""A word of the catalogue (lab2shot/i18n, a key of lab2shot/i18n/<lang>/server.toml) said when it is shown, not when it
is made: a parameter of a message (Msg renders a parameter that is not a message with str()), a kind on the admin
page's 安全 list. It is said in the language now each time (a request's, the log's English), so one made while serving
a request in Chinese still reads in English in the log line written from it; kept in a message (an audit row) it
travels as its key (messages.wire) and reads in its reader's language. The class is i18n.Word."""

from __future__ import annotations

from typing import Any

from ..i18n import Word  # noqa: F401 (server.words.Word is i18n.Word)


def said(value: Any) -> Any:
    """`value` with a Word in it said (str) now; anything else as it is."""
    return str(value) if isinstance(value, Word) else value
