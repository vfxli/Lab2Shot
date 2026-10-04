"""What upstream reads from hard-coded paths in its checkout (EnvSpec.places): where each declared file is, which are
missing, and putting them there again on the live environment.

The install puts them there itself (installer/run.py step_place, on every install, after the weights). This module
also serves `lab2shot ext place <name>`: weights deleted by mistake, or a checkout cleaned by hand, are placed again
without installing anything. The core knows no project's files: the paths are the adapter's declaration and the
placing is the adapter's build script, run with the single argument "place".
"""

from __future__ import annotations


from ..extensions.build_state import located, missing  # noqa: F401 (read-only: the extensions' status reads them too)
from ..extensions.spec import Extension, InstallError
from ..messages import Msg
from .events import Sink


def place(ext: Extension, sink: Sink, *, live=None) -> None:
    """The build script's placing (`<build> place`) on the live environment and checkout, then every place checked."""
    from .run import Context, Live, locked, step_place
    from .sources import Policy

    if not ext.env.places:
        raise InstallError(Msg("E-PLACE-NOTHING", title=ext.title))
    with locked(ext):
        paths = ext.paths  # the live environment and checkout (active_env.json)
        if not paths.python.exists():
            raise InstallError(Msg("E-ADOPT-NOENV", path=str(paths.python)))
        step_place(Context(ext, paths, sink, Policy.from_settings(), live or Live()))


__all__ = ["located", "missing", "place"]
