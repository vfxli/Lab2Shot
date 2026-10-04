"""All node types: core nodes plus the nodes of every extension that loaded, as the top layer hands them over
(nodes/services.py Services.extensions: lab2shot/adapters.py loads them, lab2shot/site/catalog.py installs it)."""

from __future__ import annotations

from functools import cache, lru_cache

from ..messages import Msg
from .base import NodeDef
from .core import CORE_NODES
from .services import services


def core_nodes() -> tuple[type[NodeDef], ...]:
    """The node types of the core's own environment: the node kit's (nodes/core) and the core's format modules'
    (lab2shot/formats, the same layer as the nodes: cli/check_arch.py LAYERS)."""
    from ..formats import NODES as FORMAT_NODES

    return (*CORE_NODES, *FORMAT_NODES)


@cache
def node_types() -> dict[str, type[NodeDef]]:
    # their words are looked up in the language now, never stored on them (nodes/text.py)
    return {**{n.id: n for n in core_nodes()}, **services().extensions().nodes}


def registry_key() -> tuple:
    """What an answer worked out from the node types depends on, as a cache key: the types loaded and the catalogues
    their words are in (a label edited on the admin page, a file restored by hand: nodes/text.py refresh, read again
    here first), in the language now (i18n.current). Cheap: file times, no reading."""
    from . import text

    from .. import i18n

    types = node_types()
    return (tuple(types), text.refresh(types), i18n.current())  # their words are said in the language now


def converter(data_type: str, port_type: str) -> str:
    """The node type that turns data of `data_type` into what an input of `port_type` takes (NodeDef.converts: it
    takes the data and gives such data), "" none: the one click offered on such a wire (Graph.fix_for)."""
    from ..data.types import accepts

    return next((t.id for t in node_types().values() if t.converts and accepts(t.converts[0], data_type)
                 and accepts(port_type, t.converts[1])), "")


def type_tables() -> dict:
    """_type_tables of the node types there are (worked out once per set of them)."""
    return _type_tables(tuple(sorted(node_types())))


@lru_cache(maxsize=4)
def _type_tables(_ids: tuple[str, ...]) -> dict:
    """The lookup tables the editor reads before a status reply can tell (a wire still being drawn, a node just added):
    over every type any port of any node type is declared with (alternatives like "scene|scene[]" as they are written) and
    every data type, "accepts": port type -> the types it takes (data/types.py accepts), and "converters": data type ->
    port type -> the node type put in between ("" none are left out). The page looks these up; it never works out
    whether a type fits (webui/src/graph/rules.ts)."""
    from ..data.types import DATA_TYPES, accepts, is_list, list_of

    types = set(DATA_TYPES)
    for t in node_types().values():
        ports = [*t.inputs, *t.outputs, *(t.param_port(p["name"]) for p in t.param_specs() if p["wire"])]
        types |= {p.type for p in ports} | ({t.ports_from_type} if t.ports_from_type else set())
    # A port that lists alternatives ("scene|scene[]", 「拆成列表」's every list type) also puts each alternative in the
    # table, and every data type its list form: a block (「拆成列表」/「逐项结束」) makes a list of whatever went in, so
    # 人物框[] and 相机[] really travel down wires. The page only looks this table up — a row it does not have is a
    # wire it cannot judge.
    types |= {a for t in list(types) for a in t.split("|")}
    types |= {list_of(t) for t in list(types) if "|" not in t and not is_list(t)}
    ordered = sorted(types)
    conversions = [(t.id, *t.converts) for t in node_types().values() if t.converts]
    converters: dict[str, dict[str, str]] = {}
    for d in ordered:
        for p in ordered:
            via = next((i for i, a, b in conversions if accepts(a, d) and accepts(p, b)), "")
            if via:
                converters.setdefault(d, {})[p] = via
    return {"accepts": {p: [d for d in ordered if accepts(p, d)] for p in ordered}, "converters": converters}


def why_missing(type_id: str) -> Msg:
    """Why there is no node type `type_id`: its extension (the id's first part) is not there, or its nodes did not
    load, or it has no such node (the extension loader knows: Adapters.why_missing)."""
    return services().extensions().why_missing(type_id)
