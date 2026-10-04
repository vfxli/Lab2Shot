"""The plugin's log: one file per host in the user's plugin folder (%USERPROFILE%/lab2shot/<host>.log), rotated, never
the DCC's script editor. Everything that goes wrong is written here with its traceback; the panel shows one sentence."""

from __future__ import annotations

import logging
import logging.handlers
import os

from . import paths

_NAME = "lab2shot_dcc"
_set_up: dict[str, str] = {}


def setup(host_name: str = "plugin") -> str:
    """Write the log to <user folder>/<host_name>.log (once per host name); returns the file."""
    if host_name in _set_up:
        return _set_up[host_name]
    path = os.path.join(paths.user_dir(), f"{host_name}.log")
    logger = logging.getLogger(_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # never into the DCC's own handlers (its script editor)
    try:
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=2 << 20, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
        logger.addHandler(handler)
    except OSError:
        logger.addHandler(logging.NullHandler())
    _set_up[host_name] = path
    return path


def get() -> logging.Logger:
    logger = logging.getLogger(_NAME)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
    return logger


def file() -> str:
    return next(iter(_set_up.values()), "")
