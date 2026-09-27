"""A picture for the viewer is made once: however many pages ask for the same frame at the same moment, one of them
makes it and the others wait for that one; once it is there nobody makes it again. The editor's

frames and data maps (view/frames.py), video frames and light textures (server/view.py) and the proxies (view/proxy.py)
are all written through here.

    once(target, make, made)   run make() unless made() (default: target exists), one caller at a time per target
    write_once(target, write)  write(part) to a file beside it, moved in place when whole; returns target"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

from ..errors import Unviewable
from ..messages import Msg

MAKE_WAIT_S = 300.0  # 等别人做这一张最多等这么久：做的那个卡死了（挂掉的 worker、锁住的盘），等的人不该跟着永远挂住


class _Making:
    """The pictures being made now (target -> the event its maker sets): only while they are made, never kept."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.now: dict[Path, threading.Event] = {}


_MAKING = _Making()


def _has_content(target: Path) -> bool:
    """The file is there and not empty (an empty one is what a system crash leaves of a picture not yet on the disk)."""
    try:
        return target.stat().st_size > 0
    except OSError:
        return False


def once(target: Path, make: Callable[[], object], made: Callable[[], bool] | None = None) -> None:
    """`make()` for `target` unless it is made already (made(), default: the file is there), one caller at a time: a
    caller that finds another making it waits for that one and looks again (a maker that failed leaves it to the
    next)."""
    done = made or (lambda: _has_content(target))
    while not done():
        with _MAKING.lock:
            if done():
                return
            busy = _MAKING.now.get(target)
            if busy is None:
                mine = _MAKING.now[target] = threading.Event()
        if busy is not None:
            if not busy.wait(MAKE_WAIT_S) and not done():
                raise Unviewable(Msg("E-VIEW-MAKEWAIT", seconds=MAKE_WAIT_S))
            continue
        try:
            make()
        finally:
            with _MAKING.lock:
                _MAKING.now.pop(target, None)
            mine.set()
        return


def write_once(target: Path, write: Callable[[Path], None]) -> Path:
    """`target` written by write(part) once (once): into a file beside it (the same suffix: a writer goes by it), moved
    in place when whole, so nobody ever reads half a picture."""

    def make() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(f".{target.stem}.{os.getpid()}.{threading.get_ident()}{target.suffix}")
        try:
            write(part)
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)

    once(target, make)
    return target
