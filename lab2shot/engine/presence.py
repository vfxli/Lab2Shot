"""Whether a packet, or what an instance gives, is there: THE one place an evaluation finds out (engine/evaluation.py
Evaluation is made of this and the other parts: routing, demand, status). A packet's manifest is read at most once
(`manifest`, `packet`); what an instance's output gives is known once its packet is on disk, or from the node's own
parameters (`known`); what something waits for before an instance can be planned is read through `lookup`, the only
door (a wired parameter's value, a parameter an input sets, a block's items, a switch's condition), which answers one
of three: the packet, what it waits for (Pending), or EMPTY (it gives nothing: taken everywhere as not connected).
Whether a packet is on disk is `present`, the one formula (data/packet.py valid: complete, its files there and
unchanged, the packets it references present).

Two other tests of a packet stand beside it, each for its own question, not another answer to this one:
- NodePlan.has (engine/records.py): `present` over the ports a reader wants (Demand.cached; a read never writes the
  cache); what a cook still has to write is asked of `present` itself (Demand.missing);
- Packet.exists (data/packet.py: marked complete, its manifest readable): only whether a manifest can be read, where
  the files were vouched for already or are read right after — `manifest` here (what a packet says, never whether it
  may be used), a list's items (nodes/core/flow.py _item_packets: the list was `valid`, which covers the packets it
  references), a packet handed to 「输出」 (transfer/outputs.py collect: the cook took it as an input, checked then)."""

from __future__ import annotations

from ..data.packet import MANIFEST, Packet, packet_dir, valid
from .records import KEPT, PLAN_ERRORS, Memo, _hash
from .scopes import Inst, ItemPath, Pending


class Empty:
    """What an input gives when its packet is empty (data/packet.py empty_packet): nothing, as if not connected."""

    def __repr__(self) -> str:
        return "EMPTY"


EMPTY = Empty()


def generation(fp: str) -> tuple | None:
    """Which generation of a packet is on disk (its manifest file's identity: one stat), None when none is: a forced
    recook writes the same fingerprint anew (fresh_dir), so a fingerprint alone does not say which (Presence.manifest,
    Evaluation.still_true)."""
    try:
        st = (packet_dir(fp) / MANIFEST).stat()
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns


def present(fp: str) -> bool:
    """THE formula for 「this packet is on disk」: complete, its files there and unchanged (data/packet.py valid). What
    an evaluation knows (known, on_disk), what the status reply lists (present, gens) and what a cook still has to
    write (Engine._to_give, Outputs) all ask it."""
    return valid(packet_dir(fp))


class Presence:
    """The presence part of Evaluation (a mixin: every method reads the evaluation's own graph and tables)."""

    def _open_presence(self) -> None:
        self._manifests: Memo[tuple, dict | None] = self._memo(KEPT)  # (packet fingerprint, generation) -> its manifest, once
        self._theres: Memo[tuple, bool] = self._memo(KEPT)  # (packet fingerprint, generation) -> present, once
        # (node, port, path, kind) -> what wired_input answered, Pending too: it changes only when what the wire comes
        # from is cooked or fails, and that forgets the reader (Demand.forget)
        self._read_inputs: Memo[tuple, Any] = self._memo(lambda k: Inst(k[0], k[2]))

    def manifest(self, fp: str) -> dict | None:
        """A packet's manifest (type, meta, node), read from disk once per generation of it in this evaluation's
        lifetime. A fingerprint is not a generation: a forced recook writes the same fingerprint anew (fresh_dir: a new
        folder, a new manifest file), so what is remembered is keyed by (fingerprint, the manifest file's identity: one
        stat), and a reader never pairs an old manifest with the new files. None: not committed (never remembered: a
        cook of this evaluation's own graph may commit it in a moment), and removed while it was read (a clean, a
        recook elsewhere): not there, the one place that race is answered (no caller catches it itself)."""
        if (gen := generation(fp)) is None:
            return None
        key = (fp, gen)
        d = packet_dir(fp)
        if key in self._manifests:
            return self._manifests[key]
        try:
            if not Packet.exists(d):
                return None
            p = Packet.load(d)
        except (OSError, ValueError):  # removed (or rewritten) between the look and the read: not there
            return None
        return self._manifests.put(key, {"type": p.type, "meta": p.meta, "node": p.node, "messages": p.messages,
                                         "created": p.created})

    def there(self, fp: str) -> bool:
        """`present(fp)` as the reading side asks it (the status, what is known before cooking): once per generation
        of the packet in this evaluation (one stat to tell the generation, not a re-read of the manifest and what it
        names on every poll). The cook asks `present` itself, fresh, for what it writes."""
        if (gen := generation(fp)) is None:
            return False
        key = (fp, gen)
        if key in self._theres:
            return self._theres[key]
        return self._theres.put(key, present(fp))

    def packet(self, fp: str) -> Packet | None:
        """The packet a fingerprint names, from `manifest` (no second read of manifest.json)."""
        m = self.manifest(fp)
        return None if m is None else Packet(packet_dir(fp), m["type"], m["meta"], m["node"], m["messages"])

    def lookup(self, kind: str, on: Inst, port: str) -> Packet | Pending | Empty:
        """THE one place something an instance needs before it can be planned is read (engine/scopes.py「Pending」): a
        wired parameter's value or a parameter an input sets ("value"), a block's items ("items"), the input a switch
        chooses ("condition"). What `on`'s `port` gives when known (Evaluation.known); what waits for it while not;
        EMPTY when it gives nothing (an empty packet), which every caller takes as not connected: a parameter keeps its
        own value, a block has no items, a switch goes by its own 「走哪一路」."""
        packet = self.known(on.node, port, on.path)
        if packet is None:
            return Pending(kind, on, port)
        return EMPTY if packet.meta.get("empty") else packet

    def wired_input(self, node_id: str, port: str, path: ItemPath, kind: str) -> tuple[str, str, Packet] | Pending | None:
        """THE one reader of what a wire brings before the instance can be planned: a switch's condition ("condition"),
        a wired parameter or a parameter an input sets ("value"), a block's items ("items"). The same checks for all,
        in this order: the first wire into `port` from the instance its source is at (Scopes.source_path); a wire the
        graph refuses (Graph.wire_problem) gives nothing, its node says why; a "value" whose source failed is gone
        (dropped: the parameter keeps its own value) — a condition's or items' source failing skips the node instead;
        then `lookup`. (source, its port, the packet) once known; Pending while not; None for no wire, a refused or
        dropped one, or an empty packet (nothing given counts as not connected). Remembered, Pending included: a chain
        of switches each waiting on the one before asks it once per instance, not once per way of asking."""
        key = (node_id, port, path, kind)
        if key in self._read_inputs:
            return self._read_inputs[key]
        return self._read_inputs.put(key, self._wired_input(node_id, port, path, kind))

    def _wired_input(self, node_id: str, port: str, path: ItemPath, kind: str) -> tuple[str, str, Packet] | Pending | None:
        g = self.graph
        wire = next(iter(g.inputs.get((node_id, port), [])), None)
        if wire is None:
            return None
        src, sport = wire
        at = g.scopes.source_path(src, node_id, path)
        if at is None or g.wire_problem(src, sport, node_id, port):
            return None
        if kind == "value" and (port, src, sport, at) in self.dropped(node_id, path):
            return None
        got = self.lookup(kind, Inst(src, at), sport)
        if got is EMPTY:
            return None
        return got if isinstance(got, Pending) else (src, sport, got)

    def known(self, node_id: str, port: str, path: ItemPath = ()) -> Packet | None:
        """What an instance's output gives, when that is known before cooking downstream: its packet once cooked, or
        what the node gives from its parameters alone (NodeDef.known_outputs: a constant value node, nodes/core/values.py); None not yet,
        and for a path that is not one instance of the node (a node in a block asked about outside it), and while the
        instance itself waits (Evaluation.waits: a wired value, a condition, its items not known yet): its parameters
        are not what they will be, so what it gives is not known either — the wait passes on to whatever reads it
        (a value node whose value a wire still to be cooked sets does not answer with the value typed into it)."""
        if len(path) != self.graph.scopes.depth(node_id) or self.waits(node_id, path):
            return None
        try:
            plan = self.plan(node_id, path)
        except PLAN_ERRORS:
            plan = None  # it cannot be cooked yet (no picture wired into 「AnyCalib 镜头标定」): a value that comes out
            # of its own parameters is known all the same (the graph tells it without a cook)
        if plan is not None and port in plan.outputs and self.there(plan.outputs[port]):  # each output on its own (Engine: wanted)
            if (found := self.packet(plan.outputs[port])) is not None:  # (removed meanwhile: not known, manifest)
                return found
        node = self.graph.nodes[node_id]
        try:
            meta = node.type.known_outputs(self.params(node_id, path)).get(port)
        except PLAN_ERRORS:
            return None
        if meta is None:
            return None
        # a value read out of its own meta needs no folder; it gets the cache folder it will land in once there is one,
        # and without a plan one named by the instance's output that is never written (a read there finds nothing, never
        # the process's own folder)
        where = packet_dir(plan.outputs[port] if plan is not None and port in plan.outputs
                           else _hash(["known", node_id, port, list(path)]))
        return Packet(where, self.graph.output_type(node_id, port), meta)

    def on_disk(self, node_id: str, port: str, path: ItemPath = ()) -> str | None:
        """The fingerprint of the instance's output `port` when that packet is on disk (cooked), else None."""
        try:
            fp = self.plan(node_id, path).outputs.get(port)
        except PLAN_ERRORS:
            return None
        return fp if fp and self.there(fp) else None
