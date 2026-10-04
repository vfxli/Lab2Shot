"""Node names (Houdini's): a node in a graph is called by its id, `retarget1`, `char_fbx`; its type, `retarget` or
`fbx.import`, is said beside it. The one place both the rule for a name and how a message points at a node live
(webui/src/graph/naming.ts is the page's twin: the same rule, the same default numbering).

- a name: `[a-z][a-z0-9_]*`, at most NAME_MOST characters, unique in its graph. A graph file whose ids break the rule
  still opens (engine/graph.py check_shape only refuses what would break a key); renaming asks for a good one, and the
  templates must follow it (`lab2shot check naming`).
- a new node is called after its type with the dot made an underscore and the least number not taken:
  `fbx.import` -> `fbx_import1`, `retarget` -> `retarget1`.
- a message pointing at a node says `name（type）`: 「fbx_import1（fbx.import）」. A comment never goes into one."""

from __future__ import annotations

import re

from .. import i18n

NAME = re.compile(r"[a-z][a-z0-9_]*")
NAME_MOST = 32


def node_ref(node_id: str, type_id: str) -> i18n.Word:
    """How a message points at a node: its name, then its type (the brackets are the language's: engine.node_ref)."""
    # kept by its key (i18n.Word): a message naming it reads right in whoever's language follows it
    return i18n.Word("engine.node_ref", name=node_id, type=type_id)


def name_stem(type_id: str) -> str:
    """The stem of a type's default node names: the type name with its dot an underscore."""
    return type_id.replace(".", "_")


def is_default_name(node_id: str, type_id: str) -> bool:
    """Whether the node still has a default name of its type (never renamed by anyone)."""
    return bool(re.fullmatch(re.escape(name_stem(type_id)) + r"[0-9]+", node_id))


# The formats are known from what is registered, never listed here: a format is a module whose nodes import a file
# (nodes/formats.py ImportNode) or set how one is written (nodes/output.py OutputSettings), named `<format>.<task>`
# (`usd.output`, `tracks.output`, `fbx.import`). A new format module is a new prefix by itself.


def format_extensions() -> frozenset[str]:
    """The format modules under adapters/ (Extension.format_module: a format whose library needs an environment of its
    own, FBX, Alembic), the only extensions named after a format."""
    from ..extensions import extensions

    return frozenset(name for name, ext in extensions().items() if ext.format_module)


def core_format_prefixes() -> frozenset[str]:
    """The prefixes of the core's format nodes (`usd`, `exr`, `tracks`): reserved, no third-party library under
    adapters/ may take one as its name (`lab2shot check naming`), or `adapters.why_missing`, which reads the extension
    off the head of a type name, would blame the wrong one."""
    from ..nodes.formats import ImportNode
    from ..nodes.output import OutputSettings
    from ..nodes.registry import node_types

    return frozenset(tid.partition(".")[0] for tid, t in node_types().items()
                     if t.runtime == "core" and "." in tid and issubclass(t, (ImportNode, OutputSettings)))


def format_prefixes() -> frozenset[str]:
    """Every format's prefix: the core's and the format modules'."""
    return core_format_prefixes() | format_extensions()


def delivery_names() -> tuple[str, ...]:
    """The outside names a card that delivers several files names each file by (`exr_name`, `usd_name`, …: one per
    format, templates/_conventions.md), in order."""
    return tuple(f"{p}_name" for p in sorted(format_prefixes()))
