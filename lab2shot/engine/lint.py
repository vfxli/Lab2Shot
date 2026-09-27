"""Usage checks over a graph (质检): each input's expectations (nodes/expects.py) are checked against what is known
of the data wired into it, and failures are reported as warnings.

Known data: before the upstream node is cooked, what the graph itself provides (Evaluation.info: frames, frame rate,
still or not); after it is cooked, the packet description (meta), read through the evaluation's manifest cache rather
than read separately. Consequently, a check that depends on data content appears only after the upstream node has
been cooked (by the queue, or by the editor cooking light nodes itself), while a check that depends only on the graph
appears immediately. Graph status (Evaluation.status) attaches the warnings to each node; a cook reports a node's
warnings when it starts that node, at which point all upstream data is known.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data.contracts import shot_meta
from ..errors import CookError
from ..nodes import node_types
from ..nodes.expects import Checked, Seen

if TYPE_CHECKING:
    from .evaluation import Evaluation


def warnings(ev: Evaluation, node_id: str, path: tuple[str, ...] = ()) -> list[dict]:
    """Return the instance's failed expectations as messages (lab2shot/messages Msg.json(): code, level, text, params),
    each with the input concerned ("port") and, when inserting a single node between the port and its source resolves
    it, that node ("fix": {"insert": node type, "label"}). A node with invalid wires reports only the one-click fixes
    for its refused wires ("refused": True)."""
    g = ev.graph
    node = g.nodes[node_id]
    if node_id in g.scopes.unpaired:  # a block with only one of its two ends: the fix inserts the missing end
        said, via = g.scopes.unpaired[node_id]
        fix = {"fix": {"insert": via, "label": f"插入「{node_types()[via].label}」"}} if via else {}
        return [{**said.json(), "refused": True, **fix}]
    if node_id in g.wiring:  # a wire from a missing or incompatible output is the only reported problem; a refused
        # wire that an inserted node would fix is offered as a one-click fix (Graph.fixes)
        return [{**said.json(), "port": port, "refused": True, "fix": {"insert": via, "label": label}}
                for port, via, said, label in g.fixes.get(node_id, [])]
    taken = ev.taken(node_id, path)  # wires actually taken: excludes wires whose source failed (W-INPUT-UNUSED) and switch
    # inputs not selected; one per item into a block's end
    inputs = {port.name: tuple(s for src, sport, p in taken[port.name] if (s := _seen(ev, src, sport, p)))
              for port in g.input_ports(node_id)}
    checked = Checked(node.label, ev.params(node_id, path), inputs)
    out = []
    for port in g.input_ports(node_id):
        for got in inputs[port.name]:
            for expect in node.type.expectations(port):
                said = expect.check(got, checked)
                if said:
                    fix = {"fix": {"insert": expect.fix, "label": f"插入「{node_types()[expect.fix].label}」"}} if expect.fix else {}
                    out.append({**said.json(), "port": port.name, **fix})
    return out


def _seen(ev: Evaluation, src: str, port: str, path: tuple[str, ...] = ()) -> Seen | None:
    """Return what is known of output `port` of instance `src`, or None when it provides nothing (an empty packet is
    treated as not connected)."""
    g = ev.graph
    try:
        plan = ev.plan(src, path)
        info = ev.info(src, path)
    except (CookError, OSError, ValueError):  # not yet plannable (GraphError is a ValueError); reported on that node
        return Seen(src, g.nodes[src].label, g.output_type(src, port))
    m = ev.manifest(plan.outputs[port]) if port in plan.outputs else None
    meta = m["meta"] if m else None
    if meta and meta.get("empty"):
        return None
    if meta is not None:
        frames, still = tuple(meta.get("frames") or ()), bool(meta.get("still"))
    else:
        frames, still = info.frames, info.still
    # shot description: from the packet once cooked, otherwise from the graph (Evaluation.shot), so checks on the
    # plate's lens do not have to wait for a cook
    shot = shot_meta(meta) if meta is not None else ev.shot(src, port, path)
    return Seen(src, g.nodes[src].label, g.output_type(src, port), frames, still, meta, shot)
