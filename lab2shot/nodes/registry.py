"""All node types: core nodes plus the nodes of every extension that loaded, as the top layer hands them over
(nodes/services.py Services.extensions: lab2shot/adapters.py loads them, lab2shot/catalog.py installs it)."""

from __future__ import annotations

from functools import cache, lru_cache

from ..messages import Msg
from .base import NodeDef
from .core import CORE_NODES
from .services import services


@cache
def node_types() -> dict[str, type[NodeDef]]:
    from . import text

    types = {**{n.id: n for n in CORE_NODES}, **services().extensions().nodes}
    text.apply(types)  # every node's name and description come from its folder's nodes.json (nodes/text.py)
    return types


def converter(data_type: str, port_type: str) -> str:
    """The node type that turns data of `data_type` into what an input of `port_type` takes (NodeDef.converts: it
    takes the data and gives such data), "" none: the one click offered on such a wire (Graph.fix_for)."""
    from ..data.types import accepts

    return next((t.id for t in node_types().values() if t.converts and accepts(t.converts[0], data_type)
                 and accepts(port_type, t.converts[1])), "")


def type_tables() -> dict:
    """type_tables_of the node types there are now (worked out once per set of them)."""
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
