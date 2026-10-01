"""Scopes and node instances. This module is the single place where the rules of a 逐项处理 block are derived, from
the graph alone; they are never stored in the graph file and never recomputed by a page.

A scope is one block: `Scope(kind, name, begin, ends, members, parent)`. `kind` is "each", the one kind in
SCOPE_KINDS. The rules (`rules(graph)`, stored on the Graph as `graph.scopes`):
1. Pairing: a node declaring a scope begin pairs with the nodes declaring a scope end by kind and block name (a
   parameter); exactly one begin and at least one end. Any other configuration is B-EACH-UNPAIRED on that node.
2. Members: the begin and every node reachable downstream from its outputs, stopping at the scope's ends. The ends
   are not members: they sit one level outside and gather the members' results for every item.
3. The only exit is an end: a wire from a member into a node that is also downstream of one of the scope's ends is
   B-EACH-LEAK on that wire (fix: insert an end of that kind). A wire into an end from a node that is not a member of
   its scope is B-EACH-OUTSIDE.
4. Nesting: two scopes that share nodes must be nested (the inner begin, members and ends all lie within the outer
   members); otherwise they cross, reported as B-EACH-CROSS on the wires where one leaves the other.
A node's chain is the list of scopes it is a member of, outermost first (a begin's chain ends with its own scope; an
end's chain is the chain its scope sits in).

Node instances. An instance is a node together with its item path, `Inst(node, path)`: the path holds one item key per
scope of the node's chain, outermost first; a node outside every block has a single instance with the empty path.
Every table kept by the evaluation, every cook step, every event and every status entry is per instance
(engine/evaluation.py, engine/cook.py). An item is `Item(name, packet)`: its name (unique within its list) and the
fingerprint of its packet (a list never copies data; the item is that packet). Its key is the hash of both, so two
items holding the same data under different names are distinct instances.

Declarations on node types (implemented by the nodes in nodes/core/flow.py). The engine reads these class attributes
and class methods; nodes/base.py does not reference them:
- a block's begin: `scope_role = "begin"`, `scope_kind`, `scope_name(params)`, `item_input` (the input port supplying the
  items), `item_output` (the output port that yields the item itself; its packet is the item's packet and is never
  rewritten), `scope_items(params, packet)` (the items of the packet wired into `item_input`, in order),
  `item_outputs(params, item, list_packet)` (the remaining outputs: port -> (packet fingerprint, value)). A begin
  instance is cooked like any other node with `CookContext.item` set (an ItemAt) and writes those outputs. Each output
  is addressed by what it depends on (data/items.py `port_fp`): 名字 = (the item's key, i.e. its name and packet, the
  port), 序号 = (the item's packet, the port, the index), 总数 = (the list's packet, the port). Adding an item therefore never re-cooks
  existing items, while moving an item gives its 序号 a different packet. Outputs not taken by any wire are not written
  (Evaluation.needed_outputs, CookContext.wanted).
- a block's end: `scope_role = "end"`, `scope_kind`, `scope_name(params)`. An end instance receives, on each input, the
  packets of the items that have a result, item by item (wire by wire within an item), and `CookContext.items`
  (ItemAt, in the same order). What it gathers when there is nothing to gather is one rule (gathers_empty): an item
  whose result on any input failed or was skipped is omitted and reported (W-EACH-FAILED), and when every item is
  omitted so, the end is skipped (the error to fix is up there); an item that gave nothing (an empty packet) is omitted
  but is no failure, and a list with no items has none to give: when nothing remains for either reason the end gives
  the empty list (N-EACH-NOTHING when items gave nothing), never an empty packet, whatever reads it after.
- a switch: `condition_input` (the input port whose value makes the choice) and
  `chosen_inputs(params, condition_packet) -> the input ports it needs`. It is called only once the condition is
  known; until then the instance is pending on it. Unchosen inputs are treated as unwired: nothing upstream of them is
  planned, cooked, failed or skipped on this instance's behalf, and a node needed only by them is "unused". A choice
  that names none of its inputs (「切换」's 「走哪一路」 past the ways it has) is refused before anything is planned
  (Graph.check_inputs: B-SWITCH-RANGE, B-SWITCH-WIREDRANGE).

Pending (待定): information an instance cannot know until an upstream node is cooked is a `Pending(kind, on, port)`:
"value" (a parameter driven by a wire), "items" (the number and names of items, for an end), "condition" (which inputs
a switch needs). All of them are determined by the single function Evaluation.lookup (engine/presence.py); the engine cooks `on`
first, then re-plans what follows (Evaluation.waits, Evaluation.order).

The status reply (Evaluation.status), as read by the editor:
- "scopes": one entry per block: {"kind", "name", "begin", "ends": [...], "members": [...], "parent": begin of the
  enclosing block or null, "lists": [{"path": [...parent item keys], "items": [{"key", "name"}...],
  "summary": {"total", followed by each STATE with a nonzero count}} or {"path": [...], "pending": true}]}. The summary
  is the block's own 「3 条 · 2/3 已算」: an item's state is the combined state of its members (node_state), so a block
  item counts as cached only when every node in the block has that item's result. The page displays this summary and
  never aggregates members itself.
- "nodes": per node (fingerprint, cached, outputs, error, outcome, skipped, messages, values ... for the instance with
  the empty path), where "cached" means "every output taken by a wire exists" (an output taken by no wire is never
  written: Evaluation.needed_outputs, CookContext.wanted) and "present" lists the existing outputs, plus
  - "state": one of STATES: "cached", "todo" (not cached), "pending" (等上游: waiting for an upstream result),
    "failed", "skipped", "unused" (needed only by a switch input that is not chosen), "error" (cannot be planned);
  - for a node inside a block (non-empty chain), the same fields for the item currently viewed (the viewed item of
    each block is passed in the request, Evaluation.status(view={begin: item key}); defaults to the first item), plus
    "item": {"path": [keys], "names": [item names, outermost first]} and "summary": {"total": number of instances,
    followed by each STATE with a nonzero count: {state: count}}, e.g. 「4/5 条已算 · 1 条失败」. Its "state" is the
    state of the whole node: failed if any instance failed, else skipped, error, pending (including an item list not
    yet known), todo, unused (only if all instances are), cached. The status reply never contains per-item entries;
    those are served by a separate call, Evaluation.items(node, offset, limit), with its own route.
Cook events carry "path" (a list, empty outside every block) alongside "node".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

from ..messages import Msg

if TYPE_CHECKING:
    from .graph import Graph

SCOPE_KINDS = ("each",)  # available scope kinds (counted by core_health); subnets are intended as the next kind
EACH = SCOPE_KINDS[0]

BEGIN, END = "begin", "end"

CACHED, TODO, PENDING, FAILED, SKIPPED, UNUSED, ERROR = "cached", "todo", "pending", "failed", "skipped", "unused", "error"
STATES = (CACHED, TODO, PENDING, FAILED, SKIPPED, UNUSED, ERROR)

ItemPath = tuple[str, ...]


class Inst(NamedTuple):
    """A node instance: the node and its item path (one item key per scope of its chain, outer first)."""

    node: str
    path: ItemPath = ()


@dataclass(frozen=True)
class ItemAt:
    """An item with its position in its list, as provided by a begin instance and gathered by an end instance."""

    name: str
    key: str
    packet: str
    index: int
    count: int


@dataclass(frozen=True)
class Pending:
    """What an instance is waiting for: `kind` is "value" | "items" | "condition"; `on` is the instance whose output
    `port` is not yet known."""

    kind: str
    on: Inst
    port: str


# ------------------------------------------------------------------ node type declarations


def role(node_type: Any) -> str:
    """Return "begin", "end", or "" for a node that is not part of a scope."""
    return getattr(node_type, "scope_role", "")


def chooses(node_type: Any) -> bool:
    return hasattr(node_type, "condition_input") and hasattr(node_type, "chosen_inputs")


# ------------------------------------------------------------------ the rules


@dataclass(frozen=True)
class Scope:
    kind: str
    name: str
    begin: str
    ends: tuple[str, ...]
    members: frozenset[str]
    parent: str | None = None

    def describe(self) -> dict:
        return {"kind": self.kind, "name": self.name, "begin": self.begin, "ends": list(self.ends),
                "members": sorted(self.members), "parent": self.parent}


Wire = tuple[str, str, str, str]  # (source node, source port, destination node, destination port)


@dataclass
class Scopes:
    scopes: dict[str, Scope] = field(default_factory=dict)  # by begin
    chains: dict[str, tuple[str, ...]] = field(default_factory=dict)  # node -> begins of its scopes, outer first
    ended: dict[str, str] = field(default_factory=dict)  # end -> the begin of its scope
    problems: dict[Wire, tuple[Msg, str]] = field(default_factory=dict)  # refused wire -> (reason, node type that fixes it)
    unpaired: dict[str, tuple[Msg, str]] = field(default_factory=dict)  # node -> (B-EACH-UNPAIRED, node type that completes the pair)

    def chain(self, node_id: str) -> tuple[str, ...]:
        return self.chains.get(node_id, ())

    def depth(self, node_id: str) -> int:
        return len(self.chains.get(node_id, ()))

    def source_path(self, src: str, node_id: str, path: ItemPath) -> ItemPath | None:
        """The path of the instance of `src` that an instance of `node_id` at `path` takes from: its own path cut to
        `src`'s depth, when `src` sits in the blocks around it (the same chain, no deeper than the path); None when the
        path does not reach it. The one rule every wire into an instance follows (Evaluation.wires, the values read
        before planning, provenance, stand-ins); a block's end is the one exception, one wire per item (wires)."""
        depth = self.depth(src)
        if depth > len(path) or self.chain(src) != self.chain(node_id)[:depth]:
            return None
        return path[:depth]


def _kind(node_type: Any) -> str:
    """The kind of block a begin or end belongs to (NodeDef.scope_kind; 逐项处理 when it says none)."""
    return getattr(node_type, "scope_kind", EACH)


def _scope_type(kind: str, want: str) -> str:
    """Return the node type that begins or ends a scope of `kind` ("" if none is registered). Used as the one-click
    fix for a leaking wire and for a block missing one of its ends."""
    from ..nodes import node_types

    return next((t.id for t in node_types().values() if role(t) == want and _kind(t) == kind), "")


def rules(graph: Graph) -> Scopes:
    """Compute every scope of the graph, the chain of every node, and the violations of the rules (see the module
    docstring)."""
    nodes = graph.nodes
    out = Scopes()
    begins = [n for n, g in nodes.items() if role(g.type) == BEGIN]
    ends = [n for n, g in nodes.items() if role(g.type) == END]
    if not begins and not ends:
        return out

    def key_of(nid: str) -> tuple[str, str]:
        t = nodes[nid].type
        return _kind(t), str(t.scope_name(nodes[nid].params))

    count = Counter(key_of(b) for b in begins)
    for b in begins:
        kind, name = key_of(b)
        mine = tuple(e for e in ends if key_of(e) == (kind, name))
        if count[(kind, name)] > 1:  # two begins of one name: which block is which can't be told; rename one
            out.unpaired[b] = (Msg("B-EACH-TWICE", node=nodes[b].label, block=name, count=count[(kind, name)]), "")
            continue
        if not mine:  # no end: the one click inserts one
            out.unpaired[b] = (Msg("B-EACH-UNPAIRED", node=nodes[b].label, block=name), _scope_type(kind, END))
            continue
        out.scopes[b] = Scope(kind, name, b, mine, frozenset())
    paired = {e for s in out.scopes.values() for e in s.ends}
    for e in ends:
        if e in paired:
            continue
        kind, name = key_of(e)
        if count[(kind, name)] > 1:  # its begin is there, twice: renaming one begin is the fix, not a third begin
            out.unpaired[e] = (Msg("B-EACH-TWICE", node=nodes[e].label, block=name, count=count[(kind, name)]), "")
        else:
            out.unpaired[e] = (Msg("B-EACH-UNPAIRED", node=nodes[e].label, block=name), _scope_type(kind, BEGIN))

    def reach(starts, stop: set[str]) -> set[str]:
        """What is downstream of `starts` through one wire or more, not going past `stop`."""
        from .graph import walk

        onward = lambda n: [d for d in graph.targets_of(n) if d not in stop]  # noqa: E731
        return set(walk([d for s in starts for d in onward(s)], onward))

    def wires_into(targets: set[str], sources: set[str]):
        for (dst, dport), wires in graph.inputs.items():
            if dst in targets:
                for src, sport in wires:
                    if src in sources:
                        yield (src, sport, dst, dport)

    def label(nid: str, port: str) -> str:
        p = graph.input_port(nid, port)
        return p.label if p else port

    # members, and wires that leak out of a block
    members: dict[str, set[str]] = {}
    for b, s in out.scopes.items():
        inside = {b} | reach([b], set(s.ends))
        after = reach(s.ends, set()) - {b}
        leaking = inside & after
        fix = _scope_type(s.kind, END)
        for w in wires_into(leaking, inside - leaking):
            src, _, dst, dport = w
            out.problems[w] = (Msg("B-EACH-LEAK", node=nodes[dst].label, input=label(dst, dport), source=nodes[src].label,
                                   block=s.name), fix)
        members[b] = inside - leaking
        for e in s.ends:  # an end gathers the members' per-item results; a parameter takes a single value, not one per item
            for (dst, dport), wires in graph.inputs.items():
                port = graph.input_port(e, dport) if dst == e else None
                for src, sport in wires if port is not None else ():
                    if port.param and src in members[b]:
                        said = Msg("B-EACH-LEAK", node=nodes[e].label, input=port.label, source=nodes[src].label, block=s.name)
                    elif not port.param and src not in members[b]:
                        said = Msg("B-EACH-OUTSIDE", node=nodes[e].label, input=port.label, source=nodes[src].label, block=s.name)
                    else:
                        continue
                    out.problems.setdefault((src, sport, e, dport), (said, ""))
    # nesting: two blocks sharing nodes must be nested; otherwise they cross. A node that breaks it belongs to neither
    # block's chain (astray): it stands with its wire's error at the level the blocks are at, never planned as a
    # member of a block it is not properly in (an end waiting forever for the other block's items)
    order, astray = list(out.scopes), set()
    for i, b in enumerate(order):
        for c in order[i + 1:]:
            s, t = out.scopes[b], out.scopes[c]
            a, z = members[b] | set(s.ends), members[c] | set(t.ends)
            both = a & z
            if not both:
                continue
            if b in members[c] and a <= members[c] or c in members[b] and z <= members[b]:
                continue
            (outer, os, inner, ins, mine, theirs) = (c, t, b, s, a, z) if b in members[c] else (b, s, c, t, z, a)
            if outer in members[inner] or inner in members[outer]:  # nested, but something inside wired past the inner end
                wrong, sources = mine - members[outer], mine & members[outer]
            else:
                wrong, sources = both, (a | z) - both
            astray |= wrong
            for w in wires_into(wrong, sources):
                src, _, dst, _ = w
                if dst in os.ends and inner in members[outer]:  # a node of the inner block wired straight into the
                    # outer block's end: it has to go through the inner block's end first (the one click inserts one)
                    said = (Msg("B-EACH-SKIPEND", node=nodes[dst].label, source=nodes[src].label, inner=ins.name, outer=os.name),
                            _scope_type(ins.kind, END))
                else:
                    said = (Msg("B-EACH-CROSS", node=nodes[dst].label, block=s.name, other=t.name), "")
                out.problems.setdefault(w, said)
    for b in members:
        members[b] -= astray
    # chains, outermost first: a scope encloses another when the other's begin is among its members
    for b, s in out.scopes.items():
        parent = [c for c in out.scopes if c != b and b in members[c]]
        parent.sort(key=lambda c: sum(c in members[x] for x in out.scopes if x != c))
        out.scopes[b] = Scope(s.kind, s.name, b, s.ends, frozenset(members[b]), parent[-1] if parent else None)
    depth = {b: sum(b in members[c] for c in out.scopes) for b in out.scopes}  # includes the node's own scope
    for nid in nodes:
        mine = [b for b in out.scopes if nid in members[b]]
        if mine:
            out.chains[nid] = tuple(sorted(mine, key=lambda b: depth[b]))
    for b, s in out.scopes.items():
        for e in s.ends:
            out.ended[e] = b
    return out


def summary(states) -> dict[str, int]:
    """Return {state: instance count} in STATES order."""
    c = Counter(states)
    return {s: c[s] for s in STATES if c[s]}


def gathers_empty(node_type) -> bool:
    """A block's end: nothing to gather is its empty list, not「nothing given」(the rule in this module's docstring):
    the engine's rule that a required input which came empty leaves the node nothing to cook (Engine._context) does not
    apply to it."""
    return role(node_type) == END


def node_state(states: list[str], pending: bool) -> str:
    """Derive the state of a node inside a block from its instances' states and whether an upstream item list is still
    unknown. No instance and nothing pending (its block's list is empty): nothing of it is used (never 「已算」, which
    would say its results are there while the node's `cached` says there are none)."""
    if not states and not pending:
        return UNUSED
    for s in (FAILED, SKIPPED, ERROR):
        if s in states:
            return s
    if pending or PENDING in states:
        return PENDING
    if TODO in states:
        return TODO
    if all(s == UNUSED for s in states):
        return UNUSED
    return CACHED
