"""What applies: one declaration for when a parameter does something, when an output is there, what running a node costs
and under which licence, and the one place that resolves them.

Like Houdini's disable_when, a node declares it next to what it is about, and nothing else decides it:

    min_confidence: float = P(0.5, label="置信度门槛", applies=Wired("confidence"))
    Port("camera", "scene.camera", "相机", when=Param("camera").set())
    cost = Cost(gpu=True, vram_gb=5.9, seconds_per_frame=0.07)
    traits = (OptionTrait(Param("sift_gpu").one_of(True), gpu=True, licence=NONCOMMERCIAL),)

A condition here is a leaf of the one availability mechanism (lab2shot/availability.py: its kinds, All / AnyOf / Not
and the resolver) over what is known of one node (NodeFacts: its parameters, what is wired into it and the facts it
knows), three-valued: True, False, or None for not known (a fact only a cook tells: how many chunks a shot was cut into:
the `cook` kind, shown as applying and said as pending). Once the node is cooked and says the fact (CookContext.fact),
a parameter it turns out not to have used is said on the node (engine/cook.py). A node's conditions never ask a
capability: a node does not know who looks at it (check_declarations).

`resolve(node_type, facts)` gives everything the server says about a node from these declarations: its outputs, the
ones that wait for a selection, what 3D data they carry, its parameters' availability (availability.Availability: the
ones that do nothing and why, those not known yet), its cost and its licence. The graph, the evaluation, the queue,
the tags and the catalogue read it; the web editor reads what the server resolved (the
status reply's `applies`, through webui/src/api/applies.ts) and the lookup tables the catalogue gives (option_traits),
never the rules themselves.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ..availability import AVAILABLE, CAPABILITY, COOK, Availability, Cond, level
from ..availability import resolve as resolve_subjects
from ..data.types import DATA_TYPES, accepts
from ..messages import Both, Msg

if TYPE_CHECKING:
    from .base import NodeDef, Port

CLIPBOARD = "clipboard"  # the subject id of 「复制到 Nuke」 on a node (Pasteable.clipboard_when)
MEASURED_ON = "RTX 4090 24 GB"  # the card the nodes' VRAM and time were measured on (Cost.measured_on; card information)


class _Wired:
    """A parameter's value while a wire drives it (Param conditions see this instead of the typed value): it is set,
    and what it is, is not known before the wire's source is cooked."""

    def __repr__(self) -> str:
        return "WIRED"


WIRED = _Wired()


@dataclass(frozen=True)
class Fact:
    """Something a node knows about itself beyond its parameters (NodeDef.facts before cooking, CookContext.fact while
    cooking): its value (None: not known) and how a message names it (NodeDef.fact_labels)."""

    value: Any
    label: str = ""


@dataclass(frozen=True)
class NodeFacts:
    """What is known of one node when its declarations are resolved: its parameters (a parameter a wire drives is
    WIRED), each input port that has wires -> the data types they carry ("" not known, or open between alternatives),
    its own facts, and the facts of what is wired into each input."""

    params: Mapping[str, Any]
    wired: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    own: Mapping[str, Fact] = field(default_factory=dict)
    # each input port -> the facts of what is wired into it, as the node above declared them (NodeDef.facts of the
    # source). A port with no wire, or a source that cannot tell yet, is simply absent: not known (Incoming)
    incoming: Mapping[str, Mapping[str, Fact]] = field(default_factory=dict)
    # the node's own outputs that have a wire out of them (None: not known here, as for a catalogue card or a standalone
    # node): a parameter that only matters for one output (a point cloud's spacing) says so with WiredOut
    wired_out: frozenset[str] | None = None
    # each input port that has wires -> for each wire, whether it brings values (True), a picture (False) or that is
    # not known (None): engine/graph.py output_data, which WiredPicture reads
    wired_data: Mapping[str, tuple[bool | None, ...]] = field(default_factory=dict)


# ------------------------------------------------------------------ conditions: the node's leaves (availability.Cond)


def _param_label(t: type[NodeDef], name: str) -> str:
    # kept in both languages (messages.Both): a message said with it reads right in whoever's language follows the cook
    return Both.of(lambda: next((p["label"] for p in t.param_specs() if p["name"] == name), name))


def _port_label(t: type[NodeDef], name: str) -> str:
    if name == t.ports_from and t.ports_from_side == "inputs":  # every row of a table that makes inputs: the table's label
        return _param_label(t, name)
    return t.port_label(name)  # in both languages (nodes/base.py NodeDef.port_label)


@dataclass(frozen=True)
class Param:
    """A parameter to state a condition on: Param("gravity").set(), Param("mode").one_of("video")."""

    name: str

    def set(self) -> ParamSet:
        return ParamSet(self.name)

    def one_of(self, *values: Any) -> ParamIn:
        return ParamIn(self.name, tuple(values))

    def suffix(self, *suffixes: str) -> ParamSuffix:
        return ParamSuffix(self.name, tuple(suffixes))

    def gt(self, x: float) -> ParamCmp:
        return ParamCmp(self.name, x)

    def wired(self) -> "ParamWired":
        """The parameter is driven by a wire (its own promoted input port is connected)."""
        return ParamWired(self.name)


def _is_set(value: Any) -> bool:
    if value is WIRED:
        return True
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None and value != [] and value != ()


@dataclass(frozen=True)
class _OneName(Cond):
    """Shared part of the two "is it wired" leaves: the message when it holds, the message when it does not, and which
    slot of `names()` the name belongs to. `holds` differs between the two and is defined by each.
    `_param` True: the leaf refers to a parameter's promoted input port; False: to a declared input port."""

    _yes: ClassVar[str] = ""
    _no: ClassVar[str] = ""
    _key: ClassVar[str] = "name"   # keyword in the message template ("name" / "input")
    _param: ClassVar[bool] = True

    @property
    def _subject(self) -> str:
        return self.name if self._param else self.port  # type: ignore[attr-defined]

    def _said(self, t: type[NodeDef]) -> str:
        return _param_label(t, self._subject) if self._param else _port_label(t, self._subject)

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg(self._yes, **{self._key: self._said(t)})

    def why_not(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg(self._no, **{self._key: self._said(t)})

    def names(self):
        one = frozenset({self._subject})
        return (one, frozenset(), frozenset()) if self._param else (frozenset(), one, frozenset())


@dataclass(frozen=True)
class ParamWired(_OneName):
    """The parameter's promoted input port has a wire (`Param("camera_rotate").wired()`).

    This differs from `Wired("camera")` (a declared input port has a wire): a parameter's port belongs to the
    parameter (`NodeDef.wired_ports`, `P(wired=True)`). When a node has no declared camera input and the wire goes to
    the parameter, `Not(Wired("camera"))` is always false and the parameter is never greyed; this condition is the
    one to use.

    `ParamSet` cannot replace it: a wired parameter still has a default value in the panel, so `.set()` always holds."""

    name: str
    _yes, _no, _key, _param = "I-APPLIES-PARAMWIRED", "I-APPLIES-PARAMUNWIRED", "name", True

    def holds(self, f: NodeFacts) -> bool:
        return f.params.get(self.name) is WIRED


@dataclass(frozen=True)
class ParamSet(Cond):
    """The parameter has a value: not None, not empty text, not an empty list (a wire driving it counts)."""

    name: str

    def holds(self, f: NodeFacts) -> bool:
        return _is_set(f.params.get(self.name))

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg("I-APPLIES-SET", name=_param_label(t, self.name))

    def why_not(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg("I-APPLIES-UNSET", name=_param_label(t, self.name))

    def names(self):
        return frozenset({self.name}), frozenset(), frozenset()


@dataclass(frozen=True)
class ParamIn(Cond):
    """The parameter is one of these values."""

    name: str
    values: tuple[Any, ...]

    def holds(self, f: NodeFacts) -> bool | None:
        value = f.params.get(self.name)
        return None if value is WIRED else value in self.values

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        param = _param_label(t, self.name)
        if len(self.values) == 1 and isinstance(self.values[0], bool):
            return Msg("I-APPLIES-ON" if self.values[0] else "I-APPLIES-OFF", name=param)
        return Msg("I-APPLIES-CHOICE", name=param,
                   choices=[Msg("I-APPLIES-OPTION", label=t.option_label(self.name, v)) for v in self.values])

    def names(self):
        return frozenset({self.name}), frozenset(), frozenset()


@dataclass(frozen=True)
class ParamSuffix(Cond):
    """A file parameter's name ends with one of these suffixes."""

    name: str
    suffixes: tuple[str, ...]

    def holds(self, f: NodeFacts) -> bool | None:
        value = f.params.get(self.name)
        if value is WIRED:
            return None
        return str(value or "").lower().endswith(tuple(s.lower() for s in self.suffixes))

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg("I-APPLIES-SUFFIX", name=_param_label(t, self.name), suffixes=list(self.suffixes))

    def names(self):
        return frozenset({self.name}), frozenset(), frozenset()




@dataclass(frozen=True)
class ParamCmp(Cond):
    """A number parameter greater than a value (an empty one does not compare)."""

    name: str
    value: float

    def holds(self, f: NodeFacts) -> bool | None:
        v = f.params.get(self.name)
        if v is WIRED:
            return None
        return v is not None and v > self.value

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg("I-APPLIES-GT", name=_param_label(t, self.name), value=self.value)

    def names(self):
        return frozenset({self.name}), frozenset(), frozenset()


@dataclass(frozen=True)
class Wired(_OneName):
    """An input port has a wire."""

    port: str
    _yes, _no, _key, _param = "I-APPLIES-WIRED", "I-APPLIES-UNWIRED", "input", False

    def holds(self, f: NodeFacts) -> bool:
        return bool(f.wired.get(self.port))


@dataclass(frozen=True)
class WiredOut(Cond):
    """Something is wired to this output of the node: a parameter that only shapes that output does nothing until then
    (a point cloud's spacing on a depth node). Not known where the graph is not known (a catalogue card)."""

    port: str

    def holds(self, f: NodeFacts) -> bool | None:
        return None if f.wired_out is None else self.port in f.wired_out

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg("I-APPLIES-WIREDOUT", output=_out_label(t, self.port))

    def why_not(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg("I-APPLIES-UNWIREDOUT", output=_out_label(t, self.port))

    def names(self):
        return frozenset(), frozenset(), frozenset()


def _out_label(t: type[NodeDef], name: str) -> str:
    return Both.of(lambda: next((p.label for p in all_outputs(t) if p.name == name), name))


def _type_words(types: tuple[str, ...] | list[str]) -> list[str]:
    return [Both.of(lambda x=x: DATA_TYPES[x].label) if x in DATA_TYPES else x for x in types]


@dataclass(frozen=True)
class WiredType(Cond):
    """What is wired into an input is of one of these types (a wire of a subtype counts). Several wires: any one is;
    nothing wired, or a type not known yet, is not known."""

    port: str
    types: tuple[str, ...]

    def __init__(self, port: str, *types: str):
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "types", tuple(types))

    def holds(self, f: NodeFacts) -> bool | None:
        got = f.wired.get(self.port) or ()
        known = [t for t in got if t and "|" not in t]
        if any(accepts("|".join(self.types), t) for t in known):
            return True
        return None if not got or len(known) < len(got) else False

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        got = sorted(set(f.wired.get(self.port) or ()))
        return Msg("I-APPLIES-WIREDTYPE", input=_port_label(t, self.port), want=_type_words(self.types), got=_type_words(got))

    def names(self):
        return frozenset(), frozenset({self.port}), frozenset()


@dataclass(frozen=True)
class WiredPicture(Cond):
    """What is wired into an input is a picture, not values (engine/graph.py output_data: the source port's
    declaration, Port.data). Several wires: any one is; nothing wired, or not known yet, is not known. For what only a
    picture has: a colour space, a half-float bit depth (「多层 EXR 输出设置」)."""

    port: str

    def __init__(self, port: str):
        object.__setattr__(self, "port", port)

    def holds(self, f: NodeFacts) -> bool | None:
        got = f.wired_data.get(self.port) or ()
        if any(d is False for d in got):
            return True
        return None if not got or None in got else False

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg("I-APPLIES-WIREDPICTURE", input=_port_label(t, self.port))

    def names(self):
        return frozenset(), frozenset({self.port}), frozenset()


@dataclass(frozen=True)
class Incoming(Cond):
    """A fact of the data wired into an input, compared: incoming("tracks", "points").eq(4).

    Where `fact(...)` asks what this node knows about itself, this asks what the node above it declared about what it
    gives (NodeDef.facts / fact_labels on the source; Graph.facts collects them per port). It is how one choice of a
    parameter states what data it is for: 「2D 跟踪点输出设置」's CornerPin takes a plane's four corners and nothing
    else, so on 400 grid points that one option is greyed with why, instead of the whole cook failing at the end.

    Nothing wired, or a source that does not declare the fact, is not known: a data condition, so the subject
    stays usable (the cook says what it really is). It never asks the node above to cook first."""

    port: str
    name: str
    op: str
    value: Any

    def _now(self, f: NodeFacts) -> Fact | None:
        got = (f.incoming or {}).get(self.port) or {}
        known = got.get(self.name)
        return known if known is not None and known.value is not None else None

    def holds(self, f: NodeFacts) -> bool | None:
        known = self._now(f)
        if known is None:
            return None
        v = known.value
        return v > self.value if self.op == "gt" else v == self.value

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        known = self._now(f)
        return Msg("I-APPLIES-INGT" if self.op == "gt" else "I-APPLIES-INEQ", input=_port_label(t, self.port),
                   fact=(Both.of(lambda: known.label if known and known.label else self.name)), value=self.value,
                   now=(known.value if known else None))

    def names(self):
        # the fact belongs to whatever is wired in, not to this node: only the port is checked here
        return frozenset(), frozenset({self.port}), frozenset()


@dataclass(frozen=True)
class IncomingName:
    """A fact of what is wired into an input, to state a condition on: incoming("tracks", "points").eq(4)."""

    port: str
    name: str

    def eq(self, x: Any) -> Incoming:
        return Incoming(self.port, self.name, "eq", x)

    def gt(self, x: float) -> Incoming:
        return Incoming(self.port, self.name, "gt", x)


def incoming(port: str, name: str) -> IncomingName:
    """A condition on a fact of the data wired into `port` (the node above it names it in its fact_labels)."""
    return IncomingName(port, name)


@dataclass(frozen=True)
class FactName:
    """A fact to state a condition on: fact("segments").gt(2)."""

    name: str

    def gt(self, x: float) -> FactCmp:
        return FactCmp(self.name, "gt", x)

    def eq(self, x: Any) -> FactCmp:
        return FactCmp(self.name, "eq", x)

    def true(self) -> FactCmp:
        return FactCmp(self.name, "true", True)


def fact(name: str) -> FactName:
    """A condition on a fact of the node (NodeDef.fact_labels names it)."""
    return FactName(name)


@dataclass(frozen=True)
class FactCmp(Cond):
    """A fact of the node compared: known from its parameters, or only once it cooks (the cook kind: not known yet, it
    applies and is said as pending)."""

    name: str
    op: str
    value: Any
    kind = COOK

    def until(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        return Msg("I-APPLIES-PENDING", fact=[t.fact_label(self.name)])

    def holds(self, f: NodeFacts) -> bool | None:
        known = f.own.get(self.name)
        if known is None or known.value is None:
            return None
        v = known.value
        return v > self.value if self.op == "gt" else bool(v) if self.op == "true" else v == self.value

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg:
        label = t.fact_label(self.name)
        now = f.own[self.name].value if self.name in f.own else None
        if self.op == "true":
            return Msg("I-APPLIES-FACTTRUE", fact=label)
        return Msg("I-APPLIES-FACTGT" if self.op == "gt" else "I-APPLIES-FACTEQ", fact=label, value=self.value, now=now)

    def names(self):
        return frozenset(), frozenset(), frozenset({self.name})


# ------------------------------------------------------------------ cost and licence


@dataclass(frozen=True)
class Cost:
    """What running a node costs: `gpu` it always runs on one (else it takes a CPU slot: farm/scheduler), the peak VRAM
    (GB) and seconds per frame measured on an RTX 4090 at its default parameters, the system memory its worker takes at
    its peak when that is a lot (the queue starts it only with that much free), and a note beside those numbers on the
    administrator's card page (farm/cards.py).
    `whole`: why it has no seconds per frame (one picture, a pair, a whole clip at once); a node that works frame by
    frame gives the number instead. `vram_measured`: False when its VRAM is an estimate, not a measurement on a card
    (this flag says so, never the note's words). vram_gb is the whole card's peak over its idle baseline (as nvidia-smi
    samples it, the card to itself), never torch's own max_memory_reserved / max_memory_allocated: those count one
    process's allocator, leave out the CUDA context and anything not allocated through torch, and so come out lower;
    underestimating is the dangerous error, since the scheduler starts a node on a card only when this much is free
    there. A worker's own note may quote torch's reserved peak as a reference; it is not this number. `measured_on`:
    the card the numbers were measured on. The VRAM figures and the card are card information: no text a node shows
    carries them; the resolved cost gives them structurally for the server to gate.
    `vram_full_gb`: a node that steps down on a smaller card (fewer frames per chunk, a smaller size: its worker picks
    by the card's memory) declares the peak of its full tier here, `vram_gb` being what its stepped-down run takes
    on a 24 GB card. On a machine with a card authorized for jobs that holds the full tier, it asks for that much (it
    runs on such a card and nowhere else, so it always runs full there); otherwise it asks for `vram_gb` and runs
    stepped down. Which of the two is in its cache key (engine/evaluation.py Evaluation.full_tier): a stepped-down
    result is never handed out for a full one, nor the other way round. Its worker gets the VRAM its tier asked for
    (job parameter `vram_budget_gb`, engine/external.py job_params) and picks its step by that alone, never by what
    the card happens to have free, so the result is the tier its key says. A full tier that grows with a parameter
    is worked out by the node (NodeDef.vram_full). 0: it never steps down."""

    gpu: bool = False
    vram_gb: float = 0.0
    vram_full_gb: float = 0.0
    seconds_per_frame: float | None = None
    ram_gb: float = 0.0
    note: bool = False  # True: it has a note (node.<type>.cost.note)
    whole: bool = False  # True: it works on the whole at once (node.<type>.cost.whole says why)
    vram_measured: bool = True
    measured_on: str = MEASURED_ON

    def said(self, t) -> str:
        """What the card page says beside the numbers for node type `t`, in the language now: why there is no seconds
        per frame (node.<type>.cost.whole), then the note (node.<type>.cost.note)."""
        from . import text
        from .. import i18n

        parts = [text.word(t, "cost", k) or "" for k, v in (("whole", self.whole), ("note", self.note)) if v]
        return i18n.t("cost.sep").join(x for x in parts if x)


@dataclass(frozen=True)
class Licence:
    """A node's licence class when it differs from its extension's (nodes/tags.py), and why it is what it is: `note`
    True, its words node.<type>.licence.note."""

    tag: str = ""
    note: bool = False
    # it needs a model each user registers for and downloads (SMPL-X, MANO, FLAME): 需注册, the tag an extension gets
    # from the body models among its manual weights (adapters.py), for a node that declares its own (the core's)
    registration: bool = False
    # the data and body models under their own licence it runs on (nodes/tags.py DATA_LICENCES), as LicenseInfo.uses:
    # a core node's class is exactly theirs, nothing else of it being licensed
    uses: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptionTrait:
    """What a parameter value changes about a node: on a GPU, non-commercial parts, measured VRAM / seconds per frame
    (only ever dearer than the default), system memory. What changes the GPU or the licence is a choice
    (Param(...).one_of): the catalogue lists it per option (option_traits)."""

    when: Cond
    gpu: bool | None = None
    # the licence class this choice switches the node to (nodes/tags.py: NONCOMMERCIAL, or RESEARCH — only academic
    # research, stricter); "" none. The same classes as an extension's whole licence, so accounts and cards judge both
    # one way (tags.may)
    licence: str = ""
    # the choice needs a model each user must register for (SMPL-X, MANO, FLAME): the 需注册 tag, as Licence.registration
    # is for a whole node, but for one choice only (the core's 「标准人」: only its SMPL-X skeleton does)
    registration: bool = False
    vram_gb: float | None = None
    seconds_per_frame: float | None = None
    ram_gb: float | None = None

    def __post_init__(self) -> None:
        from . import tags

        if self.licence not in ("", tags.NONCOMMERCIAL, tags.RESEARCH):
            raise TypeError(f"an option's licence is {tags.NONCOMMERCIAL!r} or {tags.RESEARCH!r}, not {self.licence!r}")
        if (self.gpu or self.licence or self.registration) and not isinstance(self.when, ParamIn):
            raise TypeError(f"an option trait changing the GPU, the licence or the registration is a choice "
                            f"(Param(...).one_of), not {self.when!r}")


def licence_traits(options: Mapping[str, Mapping[Any, str]]) -> tuple[OptionTrait, ...]:
    """The node's licensed choices from its extension's one table (extension.py OPTION_LICENCES: parameter -> value ->
    the licence class it switches to, nodes/tags.py NONCOMMERCIAL or RESEARCH): the extension says which of its
    downloads are restricted and how, the node never restates it (lab2shot check numbers: option_licences_declared)."""
    by: dict[tuple[str, str], list] = {}
    for name, values in options.items():
        for value, level in values.items():
            by.setdefault((name, level), []).append(value)
    return tuple(OptionTrait(Param(name).one_of(*values), licence=level) for (name, level), values in by.items())


@dataclass(frozen=True)
class ResolvedCost:
    gpu: bool  # it runs on a GPU (the scheduler gives it a card), else on a CPU slot
    vram_gb: float  # the best measured peak with these parameters (0 off a GPU): where the scheduler places it
    seconds_per_frame: float | None
    ram_gb: float
    rating: dict | None  # {"tier"} 低/中/高/超高 (nodes/compute.py); None off a GPU
    vram_measured: bool = True  # False: vram_gb is an estimate (Cost.vram_measured)
    measured_on: str = ""  # the card vram_gb and seconds_per_frame were measured on ("" off a GPU)
    vram_full_gb: float = 0.0  # Cost.vram_full_gb: its full tier's peak when it steps down on a smaller card; 0 never

    def describe(self) -> dict:
        """vram_gb, vram_full_gb, vram_measured and measured_on are card information: the server shows them only
        behind farm.cards."""
        return {"gpu": self.gpu, "vram_gb": self.vram_gb, "seconds_per_frame": self.seconds_per_frame,
                "ram_gb": self.ram_gb, "rating": self.rating, "vram_measured": self.vram_measured, "measured_on": self.measured_on,
                "vram_full_gb": self.vram_full_gb}


@dataclass(frozen=True)
class ResolvedLicence:
    tags: frozenset[str]
    commercial: bool
    note: str

    def describe(self) -> dict:
        # `word`: the one licence word this node shows (tags.strictest: 「仅限研究」 over 「非商用」, ...). The page
        # never picks among the tags itself, and never hard-codes one of them (a 仅限研究 node would be drawn 「非商用」).
        from .tags import strictest

        return {"tags": sorted(self.tags), "commercial": self.commercial, "note": self.note,
                "word": strictest(self.tags)}


@dataclass(frozen=True)
class Resolved:
    outputs: tuple[Port, ...]  # the outputs with these parameters (a `when` port only while it holds; ports_from's)
    waiting: tuple[Port, ...]  # declared outputs absent only because their `when` does not hold yet
    kinds: Mapping[str, frozenset[str] | None]  # output -> the 3D data it carries when the node knows (Port.kinds_from)
    # its parameters that declare a condition (a table row's field: "layers[2].scale"): available, inactive with why,
    # pending a fact only a cook tells (availability.Availability; a parameter without a condition is not in it)
    params: Availability
    # Outputs that are currently inactive -> the reason (Port.applies; the port stays in place, greyed, with the reason
    # on hover, and cannot be wired). Same conditions and resolver as inputs and parameters (available / inactive,
    # never hidden). Resolved here with the rest of the node's declarations; engine/graph.py ports() reads it.
    out_inactive: Mapping[str, Msg]
    cost: ResolvedCost
    licence: ResolvedLicence


# ------------------------------------------------------------------ resolving


def all_outputs(t: type[NodeDef]) -> tuple[Port, ...]:
    """Every output the node type can have, before parameters decide: `outputs` and the 置信度 a node that declares a
    confidence gives (WorkerNode.confidence), in the type order (data/types.py PORT_ORDER)."""
    from ..data.types import in_port_order

    confidence = getattr(t, "confidence", None)
    return in_port_order(t.outputs + ((confidence.port(t),) if confidence is not None else ()))


def output_ports(t: type[NodeDef], params: Mapping[str, Any]) -> tuple[Port, ...]:
    """The outputs with these parameters: the declared ones whose `when` holds, then one per entry of the ports_from
    parameter. Only parameters: the graph asks it while it checks wires, before anything else is known."""

    f = NodeFacts(params)
    declared = tuple(p for p in all_outputs(t) if level(p.when, f) is AVAILABLE)
    return declared + (t.made_ports(params) if t.ports_from_side == "outputs" else ())


def waiting_port(t: type[NodeDef], port: str, params: Mapping[str, Any]) -> Port | None:
    """The declared output `port` when it is absent only because its `when` does not hold yet (an import node's kind
    with nothing of it selected): a wire from it waits instead of being wrong. None: the port is there, or not one
    that comes back by choosing."""
    f = NodeFacts(params)
    return next((p for p in all_outputs(t) if p.name == port and level(p.when, f) is not AVAILABLE), None)


def model_conditions(model) -> dict[str, Cond]:
    """Every field of a parameters model that declares when it applies (P(applies=)) -> its condition, in field order."""
    out = {}
    for name, f in model.model_fields.items():
        cond = getattr(f.json_schema_extra, "applies", None)
        if cond is not None:
            out[name] = cond
    return out


def supplying_port(cond: Cond | None) -> str:
    """The input port that supplies this parameter's value when the parameter is greyed for that reason, else "".

    Only one form is recognized: `Not(Wired("<port>"))`, optionally wrapped in `Because`, meaning that once the port
    is wired the parameter is unused and its value comes from the port. The node's bottom row can then state the source
    (e.g. "Focal Length · 来自相机（ViPE 相机解算）") instead of showing only a greyed parameter. Other conditions
    (options, facts) do not describe a value supplied by a port and return ""."""
    from ..availability import Because, Not

    while isinstance(cond, Because):
        cond = cond.cond
    if isinstance(cond, Not) and isinstance(cond.cond, Wired):
        return cond.cond.port
    return ""


def param_conditions(t: type[NodeDef]) -> dict[str, Cond]:
    """Every parameter of the node type that declares when it applies -> its condition, in field order."""
    return model_conditions(t.Params)


def lost_output_params(t: type[NodeDef]) -> list[str]:
    """The parameters that only shape an output (applies=WiredOut(port)) which this node's family declares but this
    node does not have: the family's 「点云」 made from depth + camera stays only on a node with a native point cloud
    (nodes/base.py __init_subclass__), and its spacing and point size, inherited with the family's parameters, could
    then never do anything. The node class drops them (nodes/base.py) so they are not shown, sent or saved.

    Only an output some class in its ancestry declares: a WiredOut naming an output no class declares is a typo, which
    check_declarations refuses rather than dropping the parameter silently. Only a condition that is that one output
    (with or without an authored reason): one that combines it with anything else is refused there as well."""
    from ..availability import Because

    have = {p.name for p in all_outputs(t)}
    declared = {p.name for k in t.__mro__ for p in vars(k).get("outputs", ())}
    out = []
    for name, cond in param_conditions(t).items():
        while isinstance(cond, Because):
            cond = cond.cond
        if isinstance(cond, WiredOut) and cond.port not in have and cond.port in declared:
            out.append(name)
    return out


def option_conditions(t: type[NodeDef]) -> dict[str, Cond]:
    """Every choice of a parameter that declares when it can be picked (P(option_applies={value: cond}), and a
    registry's choice that says so: P(options_from=) Choice.applies) -> its condition, keyed "<parameter>=<value>".
    Same mechanism as a parameter's own `applies`, one level down: the server resolves it, the page greys that one
    option and writes why beside it."""
    from .params import registered_options

    out = {}
    for name, f in t.Params.model_fields.items():
        for value, cond in (getattr(f.json_schema_extra, "option_applies", None) or {}).items():
            out[f"{name}={value}"] = cond
    for name, choices in registered_options(t.Params).items():
        for c in choices:
            if c.applies is not None:
                out[f"{name}={c.value}"] = c.applies
    return out


def traits_of(t: type[NodeDef]) -> tuple[OptionTrait, ...]:
    """What the node's choices change about it: its own traits, and the licence each registry choice switches it to
    (P(options_from=) Choice.licence / registration, the same as an OptionTrait on Param(x).one_of(value))."""
    from .params import registered_options

    more = tuple(OptionTrait(Param(name).one_of(c.value), licence=c.licence, registration=c.registration)
                 for name, choices in registered_options(t.Params).items() for c in choices
                 if c.licence or c.registration)
    return (*t.traits, *more) if more else t.traits


@dataclass(frozen=True)
class Provided(Cond):
    """Something this server has or lacks, not the node's own data (a body's files an extension installs:
    data/standard_bodies.py): holds while `lacking()` says nothing, greyed with what it says otherwise. Not known
    before a cook and never a capability: the `data` kind, so it is shown greyed with the reason."""

    lacking: Callable[[], Msg | None]

    def holds(self, f: NodeFacts) -> bool:
        return self.lacking() is None

    def why(self, f: NodeFacts, t: type[NodeDef]) -> Msg | None:
        return self.lacking()  # asked only when it does not hold, i.e. when lacking() says what is missing


def table_conditions(t: type[NodeDef]) -> dict[str, dict[str, Cond]]:
    """A table parameter (a list of entries, P over list[SomeModel]) whose entries' fields declare when they apply ->
    those conditions; each row is resolved on its own fields (读取序列's layers: 尺度 only for a layer taken as depth)."""
    from .params import _entry_model

    out = {}
    for name, f in t.Params.model_fields.items():
        entry = _entry_model(f.annotation)
        if entry is not None and (conds := model_conditions(entry)):
            out[name] = conds
    return out


def conditions_of(t: type[NodeDef]) -> list[tuple[str, Cond]]:
    """Every condition a node type declares, as (where it is written, the condition), for check_declarations.

    This is the single list of the conditions a node carries: a new kind of declaration (a new `*_when`, a new
    condition field) is added here as one line and is then covered by the check automatically. The per-row conditions
    of table parameters are not listed here: they may refer only to their own row's fields and are checked separately
    by check_declarations through `table_conditions`."""
    out: list[tuple[str, Cond]] = [(f"parameter {n}", c) for n, c in param_conditions(t).items()]
    out += [(f"option {k}", c) for k, c in option_conditions(t).items()]
    out += [(f"input {p.name}", p.applies) for p in t.inputs if p.applies is not None]
    out += [(f"output {p.name}", p.when) for p in t.outputs if p.when is not None]
    out += [(f"output {p.name}'s applies", p.applies) for p in t.outputs if p.applies is not None]
    out += [("a trait", tr.when) for tr in t.traits]
    out += [(f"handle {h.kind}", h.when) for h in t.handles if h.when is not None]
    if getattr(t, "pinhole_when", None) is not None:
        out.append(("pinhole_when", t.pinhole_when))
    if getattr(t, "clipboard_when", None) is not None:
        out.append(("clipboard_when", t.clipboard_when))
    return out


class _Row:
    """One row of a table as the conditions' messages see it: the fields' labels and choices."""

    inputs: tuple = ()
    fact_labels: dict = {}

    def __init__(self, specs: list[dict]):
        self._specs = specs

    def param_specs(self) -> list[dict]:
        return self._specs


def _rating(vram: float, seconds: float | None) -> dict:
    from .compute import compute_tier

    return {"tier": compute_tier(vram, seconds)}


def resolve_cost(t: type[NodeDef], f: NodeFacts) -> ResolvedCost:
    from .compute import TIERS

    c = t.cost
    held = [tr for tr in t.traits if may_hold(tr, f)]
    gpu = c.gpu or any(tr.gpu for tr in held)
    ram = max([c.ram_gb, *(tr.ram_gb for tr in held if tr.ram_gb is not None)])
    if not gpu:
        return ResolvedCost(False, 0.0, None, ram, None)
    points = [(tr.vram_gb if tr.vram_gb is not None else c.vram_gb, tr.seconds_per_frame if tr.seconds_per_frame is not None else c.seconds_per_frame)
              for tr in held if tr.vram_gb is not None or tr.seconds_per_frame is not None]
    worst = max(points or [(c.vram_gb, c.seconds_per_frame)], key=lambda p: TIERS.index(_rating(*p)["tier"]))
    vram = max([c.vram_gb, *(tr.vram_gb for tr in held if tr.vram_gb is not None), *setting_vram(t, f).values()])
    # the full tier steps down by the same rule whatever else is chosen: never below what the chosen settings take
    declared = t.vram_full(dict(f.params))
    full = max(declared, float(vram)) if declared > 0 else 0.0
    return ResolvedCost(True, float(vram), worst[1], ram, _rating(worst[0], worst[1]), c.vram_measured, c.measured_on, full)


def setting_vram(t: type[NodeDef], f: NodeFacts) -> dict[str, float]:
    """The heavy parameters set to a measured setting -> the most VRAM that setting takes (GB): what the scheduler
    places the node by (engine/cook.py asks for a card by the resolved cost: every declared setting is offered, the
    scheduler finds a card that holds it, never lowers it). An empty value (自动) or a setting that changes the time only
    adds nothing."""
    out = {}
    for spec in t.param_specs():
        value = f.params.get(spec["name"])
        if spec["measured"] and value is WIRED:  # not known yet: the costliest setting it may come to (may_hold)
            if (most := [gb for gb in spec["measured"].values() if gb is not None]):
                out[spec["name"]] = max(most)
        elif spec["measured"] and value is not None:
            gb = spec["measured"].get(option_key(value))
            if gb is not None:
                out[spec["name"]] = gb
    return out


def may_hold(trait, f: NodeFacts) -> bool:
    """Whether a trait (a licence, a cost) may apply to the node as it will run: it holds, or what it asks of is not
    known yet (a parameter driven by a wire whose value is still to be cooked: WIRED, holds None) — then the strictest
    licence and the costliest setting it may come to, never the value typed under the wire. The one rule licence and
    cost both read (resolve_licence, resolve_cost); once the wired value is known the facts carry it (engine/
    evaluation.py facts) and the trait holds or not by it."""
    return tr_holds if (tr_holds := trait.when.holds(f)) is not None else True


def resolve_licence(t: type[NodeDef], f: NodeFacts) -> ResolvedLicence:
    from . import tags

    traits = traits_of(t)
    found = tags.node_tags(t) | {tr.licence for tr in traits if tr.licence and may_hold(tr, f)}
    if any(tr.registration and may_hold(tr, f) for tr in traits):
        found |= {tags.REGISTRATION}
    # the words said about it: the node's own when its licence differs or it has more to say, else its extension's
    # summary — the one text, never restated on the node in other words (MatAnyone's once said 只能研究用 of a 非商用)
    ext = t.project.extension
    from . import text

    own = (text.word(t, "licence", "note") or "") if t.licence.note else ""
    note = own or (ext.license.summary if ext is not None else "")
    return ResolvedLicence(frozenset(found), tags.commercial(frozenset(found)), note)


def resolve(t: type[NodeDef], f: NodeFacts) -> Resolved:
    """Everything the declarations say about one node, from what is known of it (module docstring)."""
    params = resolve_subjects(param_conditions(t), f, t) + resolve_subjects(option_conditions(t), f, t)
    # 「复制到 Nuke」 is a subject like any other (availability.py: a button is a subject): the node says when it
    # writes something pasteable, the server resolves it, the page greys the button with why instead of hiding it
    if getattr(t, "clipboard", "") and getattr(t, "clipboard_when", None) is not None:
        params += resolve_subjects({CLIPBOARD: t.clipboard_when}, f, t)
    for name, conds in table_conditions(t).items():  # each row of a table on its own fields: "layers[2].scale"
        rows = f.params.get(name)
        if not isinstance(rows, (list, tuple)):
            continue
        row_type = _Row(next((p["items"] or [] for p in t.param_specs() if p["name"] == name), []))
        for i, row in enumerate(rows):
            params += resolve_subjects(conds, NodeFacts(row if isinstance(row, Mapping) else {}), row_type, prefix=f"{name}[{i}].")
    outputs = output_ports(t, f.params)
    kinds = {p.name: (f.own[p.kinds_from].value if p.kinds_from in f.own else None) if p.kinds_from else None for p in outputs}
    waiting = tuple(p for p in all_outputs(t) if level(p.when, f) is not AVAILABLE)
    # Output applicability: only for ports present (`when` has already removed the absent ones), using the same
    # resolver as parameters and inputs.
    out_off = resolve_subjects({p.name: p.applies for p in outputs if p.applies is not None}, f, t).inactive
    return Resolved(outputs, waiting, kinds, params, out_off, resolve_cost(t, f), resolve_licence(t, f))


def facts_for(t: type[NodeDef], params: Mapping[str, Any], wired: Mapping[str, tuple[str, ...]] | None = None,
              promoted_wired: tuple[str, ...] = (), wired_out=None, incoming_facts=None,
              wired_data: Mapping[str, tuple[bool | None, ...]] | None = None) -> NodeFacts:
    """NodeFacts from a node's own parameters: the parameters driven by a wire (`promoted_wired`) as WIRED, the node's
    facts from its parameters (NodeDef.facts), which of its outputs are wired on (`wired_out`; None: not known), and
    the facts of what is wired into each input (`incoming_facts`; Graph._incoming)."""
    shown = {**params, **{name: WIRED for name in promoted_wired}}
    return NodeFacts(shown, dict(wired or {}), t.facts(dict(params)), incoming=dict(incoming_facts or {}),
                     wired_out=None if wired_out is None else frozenset(wired_out), wired_data=dict(wired_data or {}))


def resolve_params(t: type[NodeDef], params: Mapping[str, Any], connected=()) -> Resolved:
    """resolve() when only the parameters and which inputs are connected are known (no types: a catalogue, a template
    listing, a worker job's memory)."""
    return resolve(t, facts_for(t, params, {p: ("",) for p in connected}))


def effective_params(t: type[NodeDef], params: Mapping[str, Any], resolved: Resolved) -> dict:
    """The parameters as the node is cooked: the ones that do nothing here at their defaults (so they neither change
    nor re-cook the result)."""
    inactive = resolved.params.inactive
    if not inactive:
        return dict(params)
    from .params import param_defaults

    defaults = param_defaults(t.Params)  # a table row's field ("layers[2].scale") is the row's own: shown or not, kept
    return {**params, **{k: defaults[k] for k in inactive if k in defaults}}


def declared_cost(t: type[NodeDef]) -> dict:
    """What a node type declares running it costs, for the catalogue's lookup table (the web page, before a status):
    whether it always runs on a GPU, the rating its own measured numbers give (a choice's own is in option_traits)."""
    c = t.cost
    may_gpu = c.gpu or any(tr.gpu for tr in t.traits)
    return {"gpu": c.gpu,
            "rating": _rating(c.vram_gb, c.seconds_per_frame) if may_gpu else None}


def option_key(value: Any) -> str:
    """A choice's value as the lookup table keys it: as the web page writes a value with String() (true, 8, "large")."""
    return str(value).lower() if isinstance(value, bool) else str(value)


def option_traits(t: type[NodeDef]) -> dict[str, dict[str, dict]]:
    """The lookup table the catalogue gives for the choices that change something (a trait on Param(x).one_of): parameter
    -> str(value) -> {"gpu", "licence", "registration", "rating"} ("licence": the licence class the choice switches to,
    "" none; "registration": the choice needs a model each user registers for; the rating that choice alone gives,
    None when it measures nothing of its own)."""
    out: dict[str, dict[str, dict]] = {}
    for tr in traits_of(t):
        if not isinstance(tr.when, ParamIn):
            continue
        for value in tr.when.values:
            row = out.setdefault(tr.when.name, {}).setdefault(option_key(value),
                                                              {"gpu": False, "licence": "", "registration": False,
                                                               "rating": None})
            row["gpu"] = row["gpu"] or bool(tr.gpu)
            row["licence"] = row["licence"] or tr.licence
            row["registration"] = row["registration"] or bool(tr.registration)
            if tr.vram_gb is not None or tr.seconds_per_frame is not None:
                row["rating"] = _rating(tr.vram_gb if tr.vram_gb is not None else t.cost.vram_gb,
                                        tr.seconds_per_frame if tr.seconds_per_frame is not None else t.cost.seconds_per_frame)
    return out


def licensed_choices(t: type[NodeDef]) -> dict[str, dict]:
    """Parameter -> value -> the tags that value switches the node to (its traits: NONCOMMERCIAL, RESEARCH, and
    REGISTRATION for a choice that needs a model each user registers for). One value may carry several (the core's
    「标准人」 SMPL-X choice is noncommercial and needs registration); what a login may not use hides them
    (server/access.py _hidden_choices)."""
    from . import tags

    out: dict[str, dict] = {}
    for tr in traits_of(t):
        chosen = frozenset(filter(None, [tr.licence, tags.REGISTRATION if tr.registration else ""]))
        if chosen and isinstance(tr.when, ParamIn):
            for v in tr.when.values:
                out.setdefault(tr.when.name, {}).setdefault(v, set()).update(chosen)
    return out


# ------------------------------------------------------------------ what a node assumes of the picture's lens

# NodeDef.lens: how the node treats the plate's lens.
# "pinhole": it takes the picture as shot by a lens without distortion (a camera solved from a distorted plate is off);
# "any": it handles a distorted lens itself (UniK3D's per-pixel rays, AnyCalib estimating the distortion);
# "solves": it solves the lens itself as part of the solve (COLMAP), the model being its own parameter; it then
# declares `pinhole_when`, the condition under which it solves no distortion and is a pinhole node after all;
# "given": the camera comes from values, the picture gives only its size (创建相机). "" the node makes no camera from
# a picture. Every node that makes a camera from a picture declares one.
LENS_PINHOLE, LENS_ANY, LENS_GIVEN, LENS_SOLVES = "pinhole", "any", "given", "solves"
LENSES = ("", LENS_PINHOLE, LENS_ANY, LENS_GIVEN, LENS_SOLVES)


def standing_notices(t: type[NodeDef]) -> list[Msg]:
    """What the node says whatever its parameters and inputs, before any cook and again with its result: a node whose
    output decides its own size says what that size will be. A pinhole node carries no standing notice about
    undistorted input: production artists know whether their plate was undistorted."""
    said: list[Msg] = []
    for port in all_outputs(t):
        if port.shape.window == "node" and port.shape.said and port.shape.said not in {m.code for m in said}:
            said.append(Msg(port.shape.said))
    return said


def overscan_notices(t: type[NodeDef], node: str, inputs: list[tuple[str, str, object]]) -> list[tuple[str, Msg]]:
    """(input port, N-COOK-OVERSCAN) for every picture a node that can only work on the plate frame
    (keeps_overscan = False: it unprojects it, renders into it, writes a format without windows) gets with pixels past
    that frame: it says what it leaves out. `node`: how messages point at it (GNode.label); `inputs`: (port, its label, a packet) per packet the node cooks with."""
    if t.keeps_overscan:
        return []
    from ..data.windows import Window

    return [(port, Msg("N-COOK-OVERSCAN", node=node, input=label, left=o[0], top=o[1], right=o[2], bottom=o[3]))
            for port, label, packet in inputs if "data_window" in packet.meta
            and any(o := Window.of(packet.meta).overscan)]


def standing_marks(t: type[NodeDef]) -> list[dict]:
    """What the node says before any status, for the catalogue: every standing notice (standing_notices), each with
    `mark` (the few words the node's bottom row shows) and `text`, the whole sentence, which is what the 数据信息
    card and the parameter panel show. The short words are the message's own (`<CODE>.short` in the catalogue, at most
    messages.SHORT_WIDTH full-width characters): a node's bottom row is one line and never cuts text, so the limit
    is held in the catalogue, not by the page cutting anything.
    How loud each one is, is its level's business (a production risk is red, an I is not). Only on the node: a
    template card says nothing of them."""
    return [{**m.json(), "mark": m.short} for m in standing_notices(t)]


def check_declarations(t: type[NodeDef]) -> None:
    """When a node class is made: every condition the node declares, wherever it is declared, names parameters,
    inputs, outputs and facts the node really has; its lens assumption is one of LENSES, and only one that solves the
    lens says when it solves none.

    The check must cover every declaration site. A misspelled name in a condition (a renamed parameter, a removed
    input) otherwise fails silently: no exception, no page error, the condition never holds, and a greyed reason may
    name a raw port that does not exist on the page. `conditions_of()` therefore
    lists every site (a parameter's `applies`, `option_applies`, an input's `applies`, an output's `when` and
    `applies`, traits, handles, `pinhole_when`, `clipboard_when`); a new declaration site is added there as one line."""
    if t.ops:  # declared ops must exist in the catalogue (a wrong id fails at class creation, not at cook time)
        from ..ops import CATALOG

        unknown = [o for o in t.ops if o not in CATALOG]
        if unknown:
            raise TypeError(f"{t.__name__}: ops {unknown} are not in the algorithm catalogue (lab2shot/ops/ops.toml)")
    if t.lens not in LENSES:
        raise TypeError(f"{t.__name__}: lens {t.lens!r} is not one of {LENSES}")
    if (t.pinhole_when is not None) != (t.lens == LENS_SOLVES):
        raise TypeError(f"{t.__name__}: pinhole_when belongs to a node with lens {LENS_SOLVES!r}, and every one of them needs it")
    from .params import _entry_model

    for name, f in t.Params.model_fields.items():  # a table row's conditions name only the row's own fields
        entry = _entry_model(f.annotation)
        for field, cond in (model_conditions(entry) if entry is not None else {}).items():
            ps, qs, fs = cond.names()
            if ps - set(entry.model_fields) or qs or fs:
                raise TypeError(f"{t.__name__}: table {name}'s field {field} has a condition on something its row does not have")
    params = set(t.Params.model_fields)
    # an input-making table (「多层 EXR 输出设置」's 图层) stands for all its rows: WiredType("layers", ...) asks what any
    # row has wired into it (Graph.facts)
    ports = {p.name for p in t.inputs} | ({t.ports_from} if t.ports_from and t.ports_from_side == "inputs" else set())
    facts = set(t.fact_labels)
    outputs = {p.name for p in all_outputs(t)}
    for where, cond in conditions_of(t):
        if any(leaf.kind == CAPABILITY for leaf in cond.leaves()):
            raise TypeError(f"{t.__name__}: {where}'s condition asks a capability: a node does not know who looks at it")
        ps, qs, fs = cond.names()
        missing = [f"parameter {x}" for x in ps - params] + [f"input {x}" for x in qs - ports] + [f"fact {x}" for x in fs - facts]
        # an output a parameter shapes must be one the node has: one it lacks makes the parameter never apply
        # (lost_output_params takes such inherited parameters off before this check)
        missing += [f"output {x}" for x in {c.port for c in cond.leaves() if isinstance(c, WiredOut)} - outputs]
        if missing:
            raise TypeError(f"{t.__name__}: {where}'s condition names {', '.join(missing)}, which the node does not have")
    for p in t.outputs:
        # an output comes and goes with this node's own parameters only (「导入 USD」's 相机 once one is chosen,
        # Sapiens2 的「前景」 while 精细抠像 is on): a front end points at that parameter (Port.describe when), and the
        # graph can work out whether the port is there from the parameters alone, before anything is cooked
        if p.when is not None and (p.when.names()[1] or p.when.names()[2]):
            raise TypeError(f"{t.__name__}: output {p.name} is there by this node's parameters, not by what is wired "
                            f"into it or what it found out: {p.when!r}")
        if p.kinds_from and p.kinds_from not in facts:
            raise TypeError(f"{t.__name__}: output {p.name}'s kinds come from fact {p.kinds_from}, which the node does not name")
    # The "at least one wired" ports must be the node's own declared inputs and all optional (required ports can never
    # be empty anyway). The "alternative wirings" declaration must be well formed: at least two alternatives, at least
    # one port each, no name shared between groups, and every name a declared optional input of the node.
    optional_inputs = {p.name for p in t.inputs if p.optional}
    if t.input_choice:
        if len(t.input_choice) < 2 or any(not g for g in t.input_choice):
            raise TypeError(f"{t.__name__}: input_choice needs at least two ways, each with at least one input: {t.input_choice!r}")
        flat = t.choice_inputs()
        if len(set(flat)) != len(flat):
            raise TypeError(f"{t.__name__}: input_choice repeats an input between its ways: {t.input_choice!r}")
        for name in flat:
            if name not in optional_inputs:
                raise TypeError(f"{t.__name__}: input_choice names {name!r}, which is not one of its optional inputs")
    # An input table whose rows the node names itself (NodeDef.ports_from_names: 「切换」's a…j): one default label per
    # name, no name twice, and the table's own type holds it to them (its row's `name` a Literal of exactly these, the
    # table no longer than they are), so a graph file can never have a row the page could not have added.
    if t.ports_from_names:
        names = t.ports_from_names
        if not (t.ports_from and t.ports_from_side == "inputs"):
            raise TypeError(f"{t.__name__}: ports_from_names belongs to a table that makes inputs")
        if len(set(names)) != len(names):
            raise TypeError(f"{t.__name__}: ports_from_names must be distinct")
        from typing import get_args

        field_ = t.Params.model_fields[t.ports_from]
        row = _entry_model(field_.annotation)
        longest = next((m.max_length for m in field_.metadata if hasattr(m, "max_length")), None)
        if row is None or set(get_args(row.model_fields["name"].annotation)) != set(names) or longest != len(names):
            raise TypeError(f"{t.__name__}: table {t.ports_from}'s rows must be named from ports_from_names only "
                            f"(its name a Literal of them) and hold at most {len(names)} (max_length)")
