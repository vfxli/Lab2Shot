"""Errors that mean something to whoever asked, raised anywhere and turned into an answer in one place (the server's
exception handlers, lab2shot/server/app.py).

Every one a user sees carries a message (lab2shot/messages: a code and its parameters, its Chinese in the catalogue):
`exc.message` is the Msg, `str(exc)` its text."""

from __future__ import annotations

from pathlib import Path

from .messages import Msg


class MessageError(Exception):
    """An error a user reads: its message is a catalogue code with parameters."""

    def __init__(self, message: Msg):
        if not isinstance(message, Msg):
            raise TypeError(f"{type(self).__name__} takes a Msg (lab2shot/messages), not {type(message).__name__}")
        self.message = message
        super().__init__(message.code)

    def __str__(self) -> str:
        return self.message.text

    @property
    def code(self) -> str:
        return self.message.code


class GraphError(MessageError, ValueError):
    """A node graph does not check out (a node type that is not there, a wire that can't be, a cycle)."""


class Refused(GraphError):
    """A usage check the instance cannot get past (a B- code: two things of one name, too few frames). `check` is the
    whole check as engine/lint.py produced it (code, level, text, params, the input concerned and the one-click fix),
    so the error and the panel's message list carry identical content and the same fix."""

    def __init__(self, check: dict):
        super().__init__(Msg(check["code"], **check["params"]))
        self.check = check


class Invalid(MessageError, ValueError):
    """A value, a parameter or an input a user gave that can't be taken (the message says why and what fits)."""


class CookError(MessageError, RuntimeError):
    """A node's cook() failed to produce its result; `node_id` says which, `log` where its worker's own log (if any)
    lives, `param` the parameter it is about (the message's anchor: the panel goes there), "" none."""

    def __init__(self, node_id: str, message: Msg, log: Path | None = None, param: str = ""):
        super().__init__(message)
        self.node_id = node_id
        self.log = log
        self.param = param


def message_of(exc: BaseException) -> Msg:
    """What an error says to a user: its own message, or (an error from a library, the file system) its words under
    E-COOK-PROBLEM."""
    if isinstance(exc, MessageError):
        return exc.message
    code, params = getattr(exc, "code", None), getattr(exc, "params", None)
    if isinstance(code, str) and isinstance(params, dict):  # the worker SDK's Failure(code, **params), raised in the core
        return Msg(code, **params)
    return Msg("E-COOK-PROBLEM", detail=str(exc) or type(exc).__name__)


class NothingToCook(MessageError):
    """A node found nothing to give this time (no face in the whole shot, no point to track): not an error. Raised by a
    node's cook (or by run_worker for a worker's nothing()), the engine gives every output of the node an empty
    packet and attaches the message, a notice (N-) stating what was not found and what to try (engine/cook.py).
    Downstream nodes decide whether they can proceed without it (Port.takes_empty, optional)."""


class CookCancelled(Exception):
    """The cook was stopped (by the user who started it, or by the administrator)."""


class NotFound(MessageError, LookupError, ValueError):
    """What was asked for is not there (a job, an upload, a delivery, a template): HTTP 404. It is also a
    ValueError, so code that turns bad input into a node's error message treats it like one."""


class Unavailable(MessageError, RuntimeError):
    """The server can't do it just now, but will again soon (it is restarting): HTTP 503."""


class TooMany(MessageError, RuntimeError):
    """One client asks too often, or the pool it asks for is full (light cooks, farm/queue.py): HTTP 429."""


class NotSignedIn(MessageError, PermissionError):
    """Only the administrator may do this, and whoever asked has not logged in (or the password they gave is wrong):
    HTTP 401."""


class Forbidden(MessageError, PermissionError):
    """Whoever asked may not do this (not theirs, not their licence): HTTP 403."""


class TooManyTries(MessageError, RuntimeError):
    """Too many attempts at once from one client or one account (wrong passwords, or open event streams); the request
    is refused for now, and the message states how long to wait or what to close: HTTP 429."""


class Conflict(MessageError, RuntimeError):
    """Not now: what was asked would get in the way of what the server is doing (a restart onto a port that is taken,
    an upload the server is moving away, a clean while jobs run): HTTP 409."""


class TooLarge(MessageError, ValueError):
    """More than the server takes in one ask (an upload past its size limit, a disk too full to hold it, too many
    files asked about at once): HTTP 413."""


class Unviewable(MessageError, RuntimeError):
    """A result's view data could not be produced (out of memory, time limit exceeded, the worker died). The failure
    is reported explicitly and never replaced by a silently degraded view (所见即所得): HTTP 422."""


class Failed(MessageError, RuntimeError):
    """Work the server itself does failed: HTTP 500."""
