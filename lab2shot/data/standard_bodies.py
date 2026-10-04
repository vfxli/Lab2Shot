"""The bodies the core node 「标准人」 (nodes/core/scene.py StandardHuman) offers under 「骨架」: a registry.

The core brings one, SMPL-X 22 joints (data/skeleton.py neutral_body: the SMPL-X neutral body each administrator
downloads after registering). Every other body is declared by the extension it comes with (Extension.standard_bodies,
e.g. Kimodo's SOMA 30 and G1 34, whose skins ship with its pinned repository) and registered here when the extension
loads (lab2shot/adapters.py load). The core names none of them: the node's options, their names, their licence and
whether they can be picked now are all read from this table, the way 「镜头内参组」 reads Extension.lens_groups.

This module is data only: it imports nothing above the data layer. What it registers is plain declarations (a
callable that makes the body, the files it reads), so the node layer reads it without reaching up to the extensions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..errors import Invalid
from ..messages import Msg

CORE_BODY = "smplx"  # the core's own body (data/skeleton.py STANDARD_BODY), and the node's default


@dataclass(frozen=True)
class StandardBody:
    """One body 「标准人」 can make.

    id            the 「骨架」 option's value, kept in graphs and templates (never renamed once used)
    model         the body's own name, said in messages ("SOMA", "G1", "SMPL-X"): a proper name, not translated
    make          height_cm (None: the model's own) -> (the skinned character, what the node reports of it)
    word          the catalogue key of its name in the 「骨架」 list (an extension's own: extension.<name>.body.<id>)
    files         the data files it reads, in lab2shot_shared.body_models.ROOT / <id> (an extension copies them there
                  when it installs: Extension.post_install); one missing greys the option and refuses the cook
    licence       the licence class picking it switches the node to (nodes/tags.py NONCOMMERCIAL / RESEARCH), "" none
    registration  it needs a model each user registers for (the 需注册 tag)
    owner         the extension that declared it ("" the core), set when it is registered
    owner_title   that extension's name as people know it ("Kimodo"), for the message that says to install it
    """

    id: str
    model: str
    make: Callable[[float | None], tuple[Any, dict]]
    word: str = ""
    files: tuple[str, ...] = ()
    licence: str = ""
    registration: bool = False
    owner: str = ""
    owner_title: str = ""

    def folder(self) -> Path:
        from lab2shot_shared import body_models

        return body_models.ROOT / self.id

    def missing_file(self) -> str | None:
        """The first of its data files that is not there, None when all are."""
        return next((f for f in self.files if not (self.folder() / f).is_file()), None)

    def lacking(self) -> Msg | None:
        """Why it cannot be picked now (its extension's data not installed), None when it can: the option's reason."""
        if self.missing_file() is None:
            return None
        return Msg("I-BODY-NOTINSTALLED", model=self.model, extension=self.owner_title or self.owner)

    def build(self, height_cm: float | None = None):
        """The body (make), after its data files are found: a missing one is refused saying which and from where."""
        missing = self.missing_file()
        if missing is not None:
            raise Invalid(Msg("E-BODY-NOEXTDATA", model=self.model, file=missing, folder=str(self.folder()),
                              extension=self.owner_title or self.owner))
        return self.make(height_cm)


def _core_body(height_cm: float | None = None):
    from .skeleton import neutral_body

    return neutral_body(height_cm, body_only=True)


# SMPL-X: its model file is checked by neutral_body itself (lab2shot_shared.body_models.find knows its names), so
# `files` is left empty: the core's option is never greyed, the cook says what to download (E-BODY-NOMODEL).
_CORE = StandardBody(CORE_BODY, "SMPL-X", _core_body, word="node.standard_human.param.skeleton.option.smplx",
                     licence="noncommercial", registration=True)
_REGISTERED: dict[str, StandardBody] = {}


def register(body: StandardBody, owner: str = "", owner_title: str = "") -> None:
    """Add an extension's body (lab2shot/adapters.py, as the extension loads). The first one of an id stays: an id is
    one body, and the core's cannot be taken."""
    if body.id == CORE_BODY or body.id in _REGISTERED:
        return
    _REGISTERED[body.id] = replace(body, owner=owner or body.owner, owner_title=owner_title or body.owner_title)


def forget(owner: str) -> None:
    """Drop what one extension registered (a test, or an extension loaded again)."""
    for k in [k for k, b in _REGISTERED.items() if b.owner == owner]:
        del _REGISTERED[k]


def standard_bodies() -> Mapping[str, StandardBody]:
    """Every body there is: the core's first, then the extensions' in the order they registered."""
    return MappingProxyType({CORE_BODY: _CORE, **_REGISTERED})


def standard_body(skeleton: str, height_cm: float | None = None):
    """「标准人」's body for the chosen 「骨架」: (character, what it reports). One no extension declares (its extension
    removed since the graph was made) is refused, naming the ones there are."""
    from .. import i18n

    body = standard_bodies().get(str(skeleton))
    if body is None:
        raise Invalid(Msg("E-BODY-UNKNOWN", skeleton=str(skeleton),
                          choices=i18n.Both.of(lambda: i18n.separator().join(b.model for b in standard_bodies().values()))))
    return body.build(height_cm)
