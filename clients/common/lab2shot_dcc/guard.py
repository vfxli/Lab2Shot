"""Every way into the plugin (a menu item, a button, a Qt slot, a deferred call, a scene callback) goes through
`guarded`: whatever it raises is written to the plugin's log and told to whoever listens (the panel's status line),
never passed on to the DCC. A DCC that sees no exception from the plugin cannot be brought down by one."""

from __future__ import annotations

import functools
import threading
import traceback

from . import log

_listeners: list = []
_lock = threading.Lock()


def on_error(listener) -> None:
    """`listener(text)` hears every error a guarded call caught (the panel shows the latest)."""
    with _lock:
        _listeners.append(listener)


def off_error(listener) -> None:
    with _lock:
        if listener in _listeners:
            _listeners.remove(listener)


def report(text: str) -> None:
    with _lock:
        listeners = list(_listeners)
    for listener in listeners:
        try:
            listener(text)
        except Exception:  # noqa: BLE001 - a listener that fails is no reason to fail
            log.get().warning("an error listener failed: %s", traceback.format_exc())


def said(exc: BaseException) -> str:
    """The sentence an exception is shown as: the server's own words when it came from the server or the client."""
    text = str(exc).strip()
    return text or type(exc).__name__


def guarded(fn=None, *, what: str = "", default=None):
    """Decorator: call `fn`, and on any exception log it with its traceback, report it, and return `default`."""

    def wrap(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - nothing ever reaches the DCC
                log.get().error("%s failed: %s\n%s", what or f.__qualname__, exc, traceback.format_exc())
                report(said(exc))
                return default
        return inner

    return wrap(fn) if fn is not None else wrap


def call(fn, *args, **kwargs):
    """Call once, guarded (for lambdas and deferred calls)."""
    return guarded(fn)(*args, **kwargs)
