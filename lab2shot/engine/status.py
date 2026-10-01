"""The graph's state as the editor polls it (`status`, `items`): each node's and each instance's state, plan, outcome,
messages and values, every block's items, worked out here from the other parts so the page never derives a rule
itself. One of the parts Evaluation is made of (engine/evaluation.py; the others: presence, routing, demand)."""

from __future__ import annotations

from ..errors import GraphError, message_of
from ..messages import Msg
from . import scopes as sc
from .records import PLAN_ERRORS, WHOLE, Memo
from .scopes import Inst, ItemAt, ItemPath, Pending


class Status:
    """The status part of Evaluation (a mixin: every method reads the evaluation's own graph and tables)."""

    def _open_status(self) -> None:
        self._provisional: Memo[Inst, bool] = self._memo()  # it, or an instance it takes from, still waits (provisional)
        self._useds: Memo[tuple, tuple[frozenset[Inst], frozenset[str]] | None] = self._memo(WHOLE)  # _used

    def state(self, node_id: str, path: ItemPath, used: frozenset[Inst] | None, shown: frozenset[str] = frozenset()) -> str:
        """The instance's state as the editor shows it (engine/scopes.py STATES); `shown`: the outputs the viewer shows
        of it (Demand.cached)."""
        if used is not None and Inst(node_id, path) not in used:  # no result takes it: unused, unless it has its result
            return sc.CACHED if self.cached(node_id, path, shown) else sc.UNUSED
        if (o := self.outcome(node_id, path)) is not None:  # the one answer to 「it has no result because of an error」
            return o.state
        try:
            self.plan(node_id, path)
        except PLAN_ERRORS:  # (it waits, and can't be planned with what is known yet)
            return sc.ERROR
        if self.provisional(node_id, path):
            return sc.PENDING
        return sc.CACHED if self.cached(node_id, path, shown) else sc.TODO

    def provisional(self, node_id: str, path: ItemPath = ()) -> bool:
        """Whether what the instance is planned with is not final yet: it waits for something (`waits`: a block's items,
        a switch's condition, a wired value), or an instance it takes from does. A node after a block's end whose items
        are not known (「列表合并」 after 「逐项结束」) is planned on an empty list for now: pending, like the end, not
        「待算」."""
        if Inst(node_id, path) not in self._provisional:
            self._sources_first(node_id, path, self._provisional.__contains__, self.provisional)
        return self._provisional_at(Inst(node_id, path), set())

    def _provisional_at(self, key: Inst, visiting: set[Inst]) -> bool:
        if key in self._provisional:
            return self._provisional[key]
        if key in visiting:  # a cycle (the rules refuse one anyway): stops here
            return False
        visiting.add(key)
        try:
            found = bool(self.waits(*key)) or any(self._provisional_at(Inst(src, p), visiting)
                                                  for ws in self.wires(*key).values() for src, _, p in ws)
        except PLAN_ERRORS:
            found = False
        return self._provisional.put(key, found)

    def _used(self) -> tuple[frozenset[Inst], frozenset[str]] | None:
        """The instances some result of the graph needs (every node no wire leaves, through the inputs switches
        choose) and the nodes behind what is pending; None when that can't be told (a node's plan error on the way:
        its own status says it). Worked out once per what a cook takes (WHOLE, with `_orders`)."""
        if () in self._useds:
            return self._useds[()]
        sinks = [n for n in self.graph.nodes if not self.graph.outputs_by_node.get(n)]
        if getattr(self._warming, "used", False):  # asked while it is being worked out (planning a node on the way
            return None  # asks what its readers need): not known yet
        self._warming.used = True
        try:
            order, behind_at = self.order(sinks)
            found = frozenset(order), frozenset(i.node for i in behind_at)
        except PLAN_ERRORS:
            found = None
        finally:
            self._warming.used = False
        return self._useds.put((), found)

    def _value_outputs(self, nid: str, path: ItemPath, ports: list[str]) -> dict[str, str]:
        """The instance's value outputs as the panel says them (「38.6 mm」), for those already known: cooked, or known
        from the node's own parameters (NodeDef.known_outputs). One place, used whether or not it can be planned."""
        from ..data.values import describe, is_value

        return {port: describe(pk) for port in ports
                if is_value(self.graph.output_type(nid, port))
                and (pk := self.known(nid, port, path)) is not None and not pk.meta.get("empty")}

    def _instance_status(self, nid: str, path: ItemPath, state: dict, used: frozenset[Inst] | None,
                         shown: frozenset[str] = frozenset()) -> dict:
        """One instance's entry: its plan, outcome, messages and value outputs (the same fields as a node outside every
        block)."""
        g, node = self.graph, self.graph.nodes[nid]
        # every check is listed as it is, B- ones included (the page draws it by its level letter, 「提交前拦下」):
        # a B- one is also the instance's error below, the same way a refused wire is both
        entry = {"messages": [*state["messages"], *(m for m in self.checks(nid, path) if m not in state["messages"])]}
        entry["messages"] += [{**said.json(), **({"port": port} if port else {})} for said, port in self.unused_inputs(nid, path)]
        try:
            p = self.plan(nid, path)
        except PLAN_ERRORS as exc:
            entry.update({"fingerprint": None, "cached": False, "state": self.state(nid, path, used, shown)})
            # an instance with no plan at all (its file is not there, a precondition it can't get past) still says
            # whose error it is: its own when it failed here, the one above it when it is only skipped
            if (o := self.outcome(nid, path)) is not None and o.state != sc.ERROR:
                entry["outcome"] = {"state": o.state, "root": o.root}
                entry["error" if o.state == sc.FAILED else "skipped"] = o.message
            else:  # its own planning error (an ERROR outcome says the same)
                entry["error"] = message_of(exc).json()
            # an instance that cannot cook may still give values: outputs determined by its own parameters
            # (NodeDef.known_outputs, e.g. 「镜头模型」 of 「AnyCalib 镜头标定」) are sent as usual, so downstream panels
            # can show 「OpenCV 鱼眼 · 来自 AnyCalib」 before a picture is wired
            if values := self._value_outputs(nid, path, [port.name for port in g.outputs(nid)]):
                entry["values"] = values
            return entry
        entry.update({"fingerprint": p.fingerprint, "cached": self.cached(nid, path, shown), "outputs": p.outputs,
                      "present": sorted(port for port, fp in p.outputs.items() if self.there(fp)),
                      # the generation of each port on disk (the packet's commit time, data/packet.py created): the
                      # page includes it in its cache key, so a recook of the same fingerprint gives new keys, and the
                      # page drops stale packet descriptions when it changes
                      "gens": {port: (self.manifest(fp) or {}).get("created", "") for port, fp in p.outputs.items() if self.there(fp)},
                      # the ports of this node that are needed (`needed_outputs`: those wired into an input its reader
                      # takes on the route it takes, or all when none). Sent so the page can decide which channels to upload this time: only wired channels
                      # are uploaded. The decision is made only in `needed_outputs`; the page follows this list.
                      "needed": sorted(self.needed_outputs(nid, path))})
        # the channels to upload (`NodeDef.upload_channels`, answerable only by file-reading nodes): `{"take", "write"}`;
        # absent means the whole file is uploaded (PNG / JPG, all channels needed, or nodes that read no file). The
        # mapping from ports to channel names lives only in the node; the page decodes per this list in a worker and
        # uploads with the chunked protocol (`webui/src/transfer/planes.ts`)
        if (planes := self._upload_channels(nid, path)) is not None:
            entry["channels"] = planes
        if (o := self.outcome(nid, path)) is not None:  # no result because of an error: its own, or one above it
            entry["outcome"] = {"state": o.state, "root": o.root}
            entry["error" if o.state in (sc.FAILED, sc.ERROR) else "skipped"] = o.message
        if self.cached(nid, path):  # what it said while it was cooked, kept with its result (Packet.commit)
            said = next((m["messages"] for o in p.outputs.values() if (m := self.manifest(o)) and m["messages"]), [])
            entry["messages"] = [*entry["messages"], *(m for m in said if m not in entry["messages"])]
        if values := self._value_outputs(nid, path, list(p.outputs)):
            entry["values"] = values
        if strip := self.strip_values(nid, path):
            entry["strip"] = strip
        entry["state"] = self.state(nid, path, used, shown)
        return entry

    def strip_values(self, node_id: str, path: ItemPath = ()) -> list[dict]:
        """Current values of the parameters shown in the strip under the view (NodeDef.strip), one {label, text} each: a
        wired value as on the wire (the same number `sources` states), an entered value as entered, otherwise empty
        (the page draws 「—」). These are not output ports (see nodes/base.py strip)."""
        from ..data.values import describe_value, option_label, read, say, unit_problem

        node = self.graph.nodes[node_id]
        if not node.type.strip:
            return []
        params = self.params(node_id, path)
        specs = {s["name"]: s for s in node.type.param_specs()}
        out = []
        for name, label in node.type.strip.items():
            spec = specs[name]
            text = ""
            wire = self._wired_packet(node_id, name, path)
            if wire is not None and not isinstance(wire, Pending):
                v = read(wire[2])
                v = v.in_unit(spec["unit"]) if spec["unit"] and not unit_problem(v.unit, spec["unit"]) else v
                text = (option_label(spec, v.value) if not v.per_frame else "") or describe_value(v)
            elif wire is None and params.get(name) is not None and params.get(name) != "":
                text = say(spec, params[name])
            out.append({"label": label, "text": text})
        return out

    def _upload_channels(self, nid: str, path: ItemPath = ()) -> dict | None:
        """The channels of this node's source to upload for the current cook (NodeDef.upload_channels; None: upload the
        whole file, or the node reads no file). A header not read yet, parameters not filled in or an upload already
        cleaned only mean there is no answer: the whole file is uploaded, and the graph status must not fail."""
        node = self.graph.nodes[nid]
        try:
            return node.type.upload_channels(self.params(nid, path), self.needed_outputs(nid, path))
        except Exception:  # noqa: BLE001 no answer means the whole file (reading reports its own errors)
            return None

    def view_path(self, node_id: str, view: dict[str, str] | None = None) -> ItemPath | None:
        """The instance of the node the view is on: the item each block around it is showing (`view`: begin -> item
        key, from the request), the first item of it otherwise; None while an item list is not known yet."""
        path: ItemPath = ()
        for n, begin in enumerate(self.graph.scopes.chain(node_id)):
            items = self.item_list(begin, path[:n])
            if isinstance(items, Pending) or not items:
                return None
            want = (view or {}).get(begin)
            path += (next((i.key for i in items if i.key == want), items[0].key),)
        return path

    def items(self, node_id: str, offset: int = 0, limit: int = 50) -> dict:
        """Item by item for one node inside a block (what the status reply does not carry, asked for on
        its own): {"total", "offset", "items": [one entry per instance, as the status has it for the view item, with
        "item": {"path", "names"}]}."""
        paths, _pending = self.instances(node_id)
        used = (self._used() or (None, frozenset()))[0]
        window = paths[offset:offset + limit] if limit else paths[offset:]
        return {"total": len(paths), "offset": offset,
                "items": [{**self._instance_status(node_id, p, {"messages": []}, used),
                           "item": {"path": list(p), "names": self._names(node_id, p)}} for p in window]}

    def status(self, view: dict[str, str] | None = None, shown: dict[str, frozenset[str]] | None = None) -> dict:
        """The graph as the editor reads it, everything worked out here so the
        page never derives a rule itself (webui/src/graph/rules.ts only reads it); the shape of the block parts
        ("scopes", a node's "state", "items", "summary") is written in engine/scopes.py:
        - "nodes": per node its fingerprint, whether its result is cached, its outputs' packets, the parameters its
          connections and settings make inactive (with why), its cost and licence with its parameters, its usage
          warnings (engine/lint.py), where its parameters get their values (sources) and, once cooked, what its value
          outputs hold ("values": port -> "38.6 mm"); its ports as they are in this graph (Graph.ports), the viewer
          handles that apply (Graph.handles), what a click on 计算 cooks ("policy") and its state; a switch whose
          route is known, the inputs it takes ("taken"); for the inputs its parameters' options come from, the packets
          that stand for them ("stand_ins", Evaluation.stand_ins). A wired value it can't take (once known) is its error.
        - "wires": every wire's type and state (Graph.wire_states).
        - "scopes": every 逐项处理 block, its members and its item lists.
        - "deliver": what 交付 cooks (every 「输出」 together), null without one.
        A node inside a block answers for the item the view is on (`view`: begin -> item key; the first item by
        default) and says how all its items stand ("summary"); item by item is `items()`. `shown`: the node the viewer
        shows -> the outputs of it shown, as 「计算」 submits them: its state and policy are of that cook (the same
        readiness its look gives), not of the whole node.
        The server adds the graph's key, the page's cook-inputs version and the shown node's plan (server/packets.py)."""
        g = self.graph
        used_behind = self.guarded("used", self._used, None)
        used = None if used_behind is None else used_behind[0]
        behind = frozenset() if used_behind is None else used_behind[1]
        out = {}
        for nid in g.nodes:
            try:
                out[nid] = self._node_status(nid, used, behind, view, frozenset((shown or {}).get(nid) or ()))
            except Exception as exc:  # noqa: BLE001 one node's fault (a bug) never takes the whole graph's status down
                out[nid] = self._broken_status(nid, exc)
        # what a click on 「计算」 cooks, for each node: the node and what it needs (worked out together: each_case)
        policies = self.guarded("policy", lambda: self.each_case(list(g.nodes)), {})
        for nid, ports in (shown or {}).items():  # the shown node: what its 「计算」 with those outputs cooks
            if nid in g.nodes and ports:
                policies[nid] = self.guarded("policy", lambda nid=nid, ports=ports: self._case([nid], frozenset(ports)), policies.get(nid))
        for nid in g.nodes:
            out[nid]["policy"] = policies.get(nid) or {"targets": [nid], "computes": [], "uses": [], "delivers": False}
        deliveries = g.deliveries()
        # the whole graph's parts, each on its own: a fault in one (a bug) never takes the node entries down
        return {"nodes": out, "wires": g.wire_states(self.guarded("wires", self.unchosen_wires, frozenset())),
                "scopes": self.guarded("scopes", lambda: self._scopes_status(used), []),
                "deliver": self.guarded("deliver", lambda: self._case(deliveries), None) if deliveries else None}

    def guarded(self, what: str, fn, fallback):
        """`fn()`, or `fallback` when it fails for a fault of the program (said in the server's log with its trace; a
        callable `fallback` is given the E-FARM-INTERNAL message): the one guard every answer about a graph goes through
        (the status reply's parts, the look before a cook, a block node's items), so one node's bug never makes a 500."""
        from .. import logs

        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 one part's fault never takes the whole reply down
            said = Msg("E-FARM-INTERNAL", detail=f"{type(exc).__name__}: {exc}")
            logs.say(logs.get("farm"), said, logs.error_text(exc), about=what)
            return fallback(said) if callable(fallback) else fallback

    def _node_status(self, nid: str, used: frozenset[Inst] | None, behind: frozenset[str], view: dict[str, str] | None,
                     ports: frozenset[str] = frozenset()) -> dict:
        """One node's entry of `status`."""
        g = self.graph
        node = g.nodes[nid]
        inside = bool(g.scopes.depth(nid))
        r = self.resolved(nid) if not inside else g.resolved(nid)
        # what its declarations say here (nodes/applies.py): the parameters that do nothing and why, those a cook
        # still has to tell, what it costs and whose licence it is with these parameters
        state = {"applies": r.params.json(),
                 "cost": r.cost.describe(), "licence": r.licence.describe(),
                 "messages": [], "sources": self.sources(nid) if not inside else {},
                 "ports": g.ports(nid), "handles": g.handles(nid),
                 # a switch whose route is known: the inputs it takes (taken_ports, the one answer; the page reads
                 # it rather than working out 「走哪一路」 itself)
                 **({"taken": sorted(t)} if not inside and (t := self.taken_ports(nid)) is not None else {}),
                 # where a placing node puts what it gives (nodes/handles.py Places), for the viewer's preview
                 **({"places": node.type.places.placement()} if node.type.places else {})}
        if not inside:
            entry = self._instance_status(nid, (), state, used if nid not in behind else None, ports)
            if stand := self.stand_ins(nid):  # what its option editors read before it can cook
                entry["stand_ins"] = stand
            # a B- check is its error (plan) and is listed among its messages with its one click, like a refused wire
            return {**state, **entry}
        # inside a block: the item the view is on, and how the items stand together (item by item
        # is `items()`, asked for on its own)
        paths, pending = self.instances(nid)
        states = [self.state(nid, p, used) for p in paths]
        summary = {"total": len(paths), **sc.summary(states)}
        shown = self.view_path(nid, view)
        up = next((o for w in pending if (o := self.outcome(*w.on)) is not None), None)
        entry = ({**self._instance_status(nid, shown, state, used), "item": {"path": list(shown), "names": self._names(nid, shown)},
                  "sources": self.sources(nid, shown)} if shown is not None
                 else {"fingerprint": None, "cached": False, "outputs": {}, "messages": [],
                       "item": {"path": [], "names": []}, "state": sc.PENDING})
        entry["summary"] = summary
        if stand := self.stand_ins(nid, shown or ()):  # what its option editors read before it can cook
            entry["stand_ins"] = stand
        entry["cached"] = bool(paths) and all(s == sc.CACHED for s in states)
        wiring = None
        if shown is None and up is None:  # its items not known yet: what holds for every one of them (its wires) is
            try:  # said now, not only once an item is planned
                g.check_inputs(nid, every_item=True)
            except GraphError as exc:
                wiring = exc.message.json()
                entry["error"] = wiring
        # its state, once, by precedence: a wiring error of the node itself; on a route no result takes (a switch's
        # other way, _used: unused like the nodes outside the block on it, even while its items are not known);
        # skipped behind what it waits on; else how its instances stand together (node_state)
        unused = used is not None and nid not in behind and not any(Inst(nid, p) in used for p in paths)
        entry["state"] = (sc.ERROR if wiring is not None else sc.UNUSED if unused
                          else sc.SKIPPED if up is not None and not paths else sc.node_state(states, bool(pending)))
        if up is not None and not paths:
            o = self._skipped(node, g.nodes[up.root].label, up)
            entry.update({"outcome": {"state": o.state, "root": o.root}, "skipped": o.message})
        return {**state, **entry}

    def _broken_status(self, nid: str, exc: Exception) -> dict:
        """The entry of a node whose status could not be worked out (a fault of the program): an error that says so,
        written to the server's log with its trace, so the rest of the graph still shows."""
        from .. import logs

        said = Msg("E-FARM-INTERNAL", detail=f"{type(exc).__name__}: {exc}")
        logs.say(logs.get("farm"), said, logs.error_text(exc), about=nid)
        return {"fingerprint": None, "cached": False, "outputs": {}, "messages": [], "state": sc.ERROR, "error": said.json(),
                "applies": {}, "sources": {}, "ports": {"inputs": [], "outputs": []}, "handles": []}

    def item_names(self, node_id: str, path: ItemPath) -> list[str]:
        """The item names of an instance's path, outer block first ([] outside every block)."""
        return self._names(node_id, path) if path else []

    def _names(self, node_id: str, path: ItemPath) -> list[str]:
        """The item names of an instance's path, outer first."""
        names = []
        for n, begin in enumerate(self.graph.scopes.chain(node_id)):
            items = self.item_list(begin, path[:n])
            names.append(next((i.name for i in items if i.key == path[n]), "") if not isinstance(items, Pending) else "")
        return names

    def _scopes_status(self, used: frozenset[Inst] | None) -> list[dict]:
        """Every 逐项处理 block: its members, its items and how they stand together. 「3 条 · 2/3 已算」 is worked out
        here, like every other rule, so the page only shows it."""
        out = []
        for begin, s in self.graph.scopes.scopes.items():
            lists = []
            for p in self._parent_paths(begin):
                items = self.item_list(begin, p)
                lists.append({"path": list(p), "pending": True} if isinstance(items, Pending)
                             else {"path": list(p), "items": [{"key": i.key, "name": i.name} for i in items],
                                   "summary": self._scope_summary(s, p, items, used)})
            out.append({**s.describe(), "lists": lists})
        return out

    def _scope_summary(self, s: sc.Scope, path: ItemPath, items: list[ItemAt], used: frozenset[Inst] | None) -> dict:
        """How the items of one block stand: {"total": how many, then the STATES that have any}. An item counts as
        what its members make of it together, the same rule a node's own instances follow (engine/scopes.py
        node_state), so a block is 「已算」 only once every node in it has that item's result."""
        by_item: dict[str, list[str]] = {i.key: [] for i in items}
        depth = len(path)
        for nid in s.members:
            paths, _pending = self.instances(nid)
            for ip in paths:
                if len(ip) > depth and ip[:depth] == path and ip[depth] in by_item:
                    by_item[ip[depth]].append(self.state(nid, ip, used))
        states = [sc.node_state(v, not v) for v in by_item.values()]
        return {"total": len(items), **sc.summary(states)}

    def _parent_paths(self, begin: str) -> list[ItemPath]:
        """The paths the block `begin` starts sits at, as far as the item lists of the blocks around it are known."""
        scopes = self.graph.scopes
        chain = scopes.chain(begin)[:-1]
        paths: list[ItemPath] = [()]
        for b in chain:
            deeper = []
            for p in paths:
                items = self.item_list(b, p[:scopes.depth(b) - 1])
                if not isinstance(items, Pending):
                    deeper += [p + (i.key,) for i in items]
            paths = deeper
        return paths
