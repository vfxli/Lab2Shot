"""Node parameters: the pydantic base, `P()` and the parameter table front ends read.

A node's parameters are a pydantic model. `P()` records, beyond pydantic's field declaration, what front ends read
(group, widget, unit, applicability, wiring, ...); `_fields` / `_param_list` flatten the model into the table read by
the parameter panel and the engine. Node authors import `NodeParams`, `P`, `colorspace_param` and `fp16_param` from
`nodes.base` (the declaration facade).
"""

from __future__ import annotations

import re
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_core import PydanticUndefined

from ..availability import Cond
from ..errors import Invalid
from ..messages import Msg


class NodeParams(BaseModel):
    """Base of every node's parameters: unknown keys are errors (a typo must not be silently ignored)."""

    model_config = ConfigDict(extra="forbid")


def without_params(model: type[NodeParams], names: list[str]) -> type[NodeParams]:
    """The same parameters without `names`, as a new model class: pydantic can add fields to a subclass but not take
    them away, so the others are copied onto NodeParams, in their order. Copied, not shared: every node inheriting one
    Params shares its FieldInfo objects, and this model is this node's own."""
    import copy

    from pydantic import create_model

    if not names:
        return model
    kept = {n: (f.annotation, copy.deepcopy(f)) for n, f in model.model_fields.items() if n not in names}
    return create_model(model.__name__, __base__=NodeParams, __module__=model.__module__, **kept)


def param_defaults(model: type[BaseModel]) -> dict:
    """A node's parameters at their defaults (None for one with none): what a node just added holds, and what a
    graph's own values are laid over (the catalogue, the cook, a worker job's size, licence tags)."""
    return {k: (None if f.is_required() else f.get_default()) for k, f in model.model_fields.items()}


def _entry_model(annotation) -> type[BaseModel] | None:
    """The model of a list-of-entries parameter (list[SomeModel]), else None."""
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) is list and args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
        return args[0]
    return None


def _param_list(model: type[BaseModel], schema: dict) -> list[dict]:
    """Flatten the pydantic schema into what a parameter panel needs, in field order. A list of entries (a table,
    e.g. 读取序列's layers) lists the fields of one entry as `items`; `wire` is the value type a wire into the
    parameter carries ("" it cannot be driven by one: data/values.py param_type)."""
    from ..data.values import param_type

    out = _fields(model, schema)
    for spec in out:
        spec["wire"] = param_type(spec)
        spec["simple"] = simple_kind(spec)
    return out


def range_said(spec: dict) -> Msg:
    """A parameter's range as its message says it, an open end (gt / lt: `open_minimum`, `open_maximum`) as open:
    E-PARAM-BETWEEN / ABOVE / ATLEAST / BELOW / ATMOST, E-PARAM-REFUSED when it has none."""
    lo, hi = spec["minimum"], spec["maximum"]
    lo_open, hi_open = spec.get("open_minimum", False), spec.get("open_maximum", False)
    if lo is not None and hi is not None:
        if lo_open or hi_open:
            return Msg("E-PARAM-INTERVAL", low=Msg("I-PARAM-GT" if lo_open else "I-PARAM-GE", lo=lo),
                       high=Msg("I-PARAM-LT" if hi_open else "I-PARAM-LE", hi=hi))
        return Msg("E-PARAM-BETWEEN", lo=lo, hi=hi)
    if lo is not None:
        return Msg("E-PARAM-ABOVE" if lo_open else "E-PARAM-ATLEAST", lo=lo)
    if hi is not None:
        return Msg("E-PARAM-BELOW" if hi_open else "E-PARAM-ATMOST", hi=hi)
    return Msg("E-PARAM-REFUSED")


_RANGE_ERRORS = {"greater_than", "greater_than_equal", "less_than", "less_than_equal"}


def refusal(specs: list[dict], exc: ValidationError) -> Msg:
    """What a person reads when parameter values do not validate: the first parameter pydantic refused, by its label,
    with the value given and what it takes. This is the only place pydantic's own words would become user text, and
    they never do: they are English and list every option of a choice, including the ones an account may not use
    (server/access.py hides those). A choice not among the options, a value of the wrong kind or a row of a table that
    does not fit is simply refused (E-PARAM-REFUSED); the page offers what fits."""
    error = exc.errors()[0]
    name = str(error["loc"][0]) if error["loc"] else ""
    spec = next((s for s in specs if s["name"] == name), None)
    label = spec["label"] if spec else name
    whole = len(error["loc"]) == 1  # not a cell of a table parameter, whose range is the column's
    reason = range_said(spec) if spec and whole and error["type"] in _RANGE_ERRORS else Msg("E-PARAM-REFUSED")
    value = error.get("input")
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return Msg("E-PARAM-INVALID", name=label, value=text if len(text) <= 40 else text[:39] + "…", reason=reason)


# the most parameters a node declares for its body (NodeDef.on_node): the node stays compact
ON_NODE_MAX = 4
_SIMPLE_WIDGETS = (None, "slider", "select", "vec3")  # multi-option choices always use a dropdown; there is no segmented control


def simple_kind(spec: dict) -> str:
    """How a parameter can show on the node's body, "" when it cannot: a switch ("toggle"), a list of the node's own
    choices ("options"), a number, a three-number vector or a short text. Files, tables, pickers, colour-space lists
    and choices that come from a file or an input stay in the panel only."""
    if spec["widget"] not in _SIMPLE_WIDGETS or spec["choices_from"] or spec["derived_from"]:
        return ""
    if spec["options"]:
        return "options"
    if spec["widget"] == "vec3":
        return "vector"
    return {"boolean": "toggle", "number": "number", "integer": "number", "string": "text"}.get(spec["type"], "")


def _fields(model: type[BaseModel], schema: dict) -> list[dict]:
    props = schema.get("properties", {})
    out = []
    for name, field in model.model_fields.items():
        entry = _entry_model(field.annotation)
        prop = props.get(name, {})
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
        options = None
        target = prop
        for alt in prop.get("anyOf", []):
            if alt.get("type") != "null":
                target = alt
        if "enum" in target:
            options = target["enum"]
        elif "const" in target:  # a Literal of one setting (pydantic writes it as a constant)
            options = [target["const"]]
        default = field.get_default(call_default_factory=True)
        out.append(
            {
                "name": name,
                # the words (label, placeholder, assumed, option_labels, the group's name) are dressed on per
                # language (dress): never declared here
                "label": "",
                "default": None if default is PydanticUndefined else default,
                "type": target.get("type", "string"),
                "nullable": any(a.get("type") == "null" for a in prop.get("anyOf", [])),
                "minimum": target.get("minimum", target.get("exclusiveMinimum")),
                "maximum": target.get("maximum", target.get("exclusiveMaximum")),
                # the bound itself is not allowed (gt / lt): said as 「大于」「小于」, never 「至少」「之间」
                "open_minimum": "minimum" not in target and "exclusiveMinimum" in target,
                "open_maximum": "maximum" not in target and "exclusiveMaximum" in target,
                "multiple_of": target.get("multipleOf"),
                "options": options,
                "option_labels": None,
                "widget": extra.get("widget"),
                "group": extra.get("group", ""),  # its id (group.<id> its name: dress)
                "affects_result": extra.get("affects_result", True),
                "placeholder": "",
                "worker": extra.get("worker", True),
                "accept": list(extra.get("accept", ())),
                "unit": extra.get("unit", ""),
                "derived_from": list(extra.get("derived_from", ())),
                "choices_from": list(extra.get("choices_from", ())),
                "unique": extra.get("unique", False),
                "measured": extra.get("measured") or {},  # setting -> the most VRAM it takes (GB), None time only
                "per_frame": extra.get("per_frame", False),
                # What the parameter amounts to when left empty (its words: .assumed). Empty does not mean inactive:
                # the node still computes with a value, and the bottom row must state it (engine/evaluation.py sources).
                "assumed": "",
                # What the three numbers of widget "vec3" are: X Y Z by default (coordinates, angles), R G B for
                # colours. Not a separate widget: the widget is the same, only the three slot names differ.
                "parts": list(extra.get("parts", ())),
                # How many lines a text parameter uses: 1 = a single-line input (default), 2 or more = a wrapping
                # multi-line text box of that height. Likewise not a separate widget: `simple` stays "text" and the wire
                # is still a text wire; it is only drawn taller and wraps.
                "lines": max(1, int(extra.get("lines", 1) or 1)),
                # False: the parameter does not appear in the panel and is driven only by a wire (its input port
                # remains, and the node menu still finds it)
                "panel": extra.get("panel", True),
                "overrides": list(extra.get("overrides", ())),
                "items": _param_list(entry, entry.model_json_schema()) if entry else None,
                # a family's words for this parameter (a key prefix: <words>.label / .placeholder / .option.<v>), when
                # the family gives it to node types of other extensions: after the node type's own, before the shared
                "words": extra.get("words", ""),
            }
        )
    return out


class Button:
    """A button parameter (widget "button"): a row of the parameter panel like any other parameter — the same three
    marks on its left (对外参数, 提升到节点, 在节点上显示), a full-width button on its right — that holds no value.
    It is not a field of the node's Params: nothing is stored in the graph, nothing reaches the worker, nothing enters
    the fingerprint (param_specs does not list it; NodeDef.interface_specs does, for the page and the parameter
    interface). A click runs the page's action `action` (webui/src/editor/buttonActions.ts: action id -> function), on
    this node. A node declares its own in NodeDef.buttons (output's download); every node has cook (COOK_BUTTON,
    declared once on NodeDef). A template exposes one as any parameter, target "<node id>.<button name>"
    (engine/templates.py); it takes no value (apply_values passes a value given to it over).
    Its words: node.<type>.button.<word>, else the shared button.<word> (`word`: its name, unless several buttons
    share one, as every pick button does). `group`: the id of its group (group.<id>)."""

    def __init__(self, name: str, action: str, group: str = "actions", target: str = "", word: str = "") -> None:
        # target: the parameter the action works on (pick in view: the picks / canvas parameter it picks for)
        self.name, self.action, self.group, self.target = name, action, group, target
        self.word = word or name

    def spec(self, node=None) -> dict:
        """Its row in the parameter table, in the same shape as a parameter's (_fields, dressed): no value, no wire,
        only `action`; `simple` "button" so it can show on the node's body when asked, or by default where the node
        lists it in its `on_node` (output's download). `node`: the node type it is on (its words)."""
        from . import text
        from .. import i18n

        said = (text.word(node, "button", self.word) if node is not None else None) or i18n.lookup(f"button.{self.word}")
        return {
            "name": self.name, "label": said or self.name, "default": None, "type": "button", "nullable": True,
            "minimum": None, "maximum": None, "open_minimum": False, "open_maximum": False, "multiple_of": None,
            "options": None, "option_labels": None,
            "widget": "button", "group": group_label(self.group), "group_id": self.group, "affects_result": False,
            "placeholder": "", "worker": False,
            "accept": [], "unit": "", "unit_label": "", "derived_from": [], "choices_from": [], "unique": False, "measured": {},
            "per_frame": False, "assumed": "", "parts": [], "lines": 1, "panel": True, "overrides": [], "items": None,
            "wire": "", "simple": "button", "action": self.action, **({"target": self.target} if self.target else {}),
        }


# every node's cook button: cook this node (with what it needs), what its right-click cook does
COOK_BUTTON = Button("cook", "cook")

# parameters worked on with the node's handle in the 2D view: each gets its own pick-in-view button, right after it
# (NodeDef.interface_specs). Derived from the handle kinds (nodes/handles.py PICKED_IN_VIEW), the one list: the page
# reads it from the catalogue
from .handles import PICKED_IN_VIEW  # noqa: E402


def pick_button(spec: dict) -> Button:
    """Pick in view for a parameter a 2D handle works on (PICKED_IN_VIEW: picks, outlines, a stick figure): named "<parameter>_pick", action pick_in_view on that parameter
    (the page shows this node in the view so its handle can be used, and ends it on a cook or a mode switch:
    webui/src/editor/viewPicking.ts). A button parameter like any other: exposed, ordered, renamed, conditioned."""
    return Button(f"{spec['name']}_pick", "pick_in_view", group=spec.get("group_id", spec["group"]), target=spec["name"],
                  word="pick")


# ---------------------------------------------------------------- the words of a parameter table


def group_label(group: str) -> str:
    """A parameter group's name in the language now (group.<id>; an id with none, as it is; "" none)."""
    from .. import i18n

    return (i18n.lookup(f"group.{group}") or group) if group else ""


def option_key(value: Any) -> str:
    """An option's value as the last segment of its key (…option.<value>): a switch true / false, else as written."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def dress(specs: list[dict], node: str, scope: str | None) -> list[dict]:
    """A parameter table (_fields) with its words in the language now: each parameter's label, placeholder, assumed,
    its options' names, its group's name (and `group_id`, the group's id) and its unit's (`unit_label`), a table's
    fields alike (…param.<table>.field.<field>.…). The keys: node.<type>.param.<p>.label|placeholder|assumed|
    option.<value>, falling back to param.<p>.… and option.<value> (lab2shot/i18n). A word with none: the label is the
    parameter's name, the others empty."""
    return [_dressed(s, node, scope, ("param", s["name"])) for s in specs]


def _dressed(spec: dict, node: str, scope: str | None, path: tuple[str, ...]) -> dict:
    from .. import i18n
    from ..data.units import unit_label

    def word(*part: str) -> str | None:
        """The node's own key, else the family's words (`words`), else the shared key."""
        key = ".".join((*path, *part))
        if node:
            own = i18n.lookup(f"node.{node}.{key}", scope=scope, exact=True)
            if own is not None:
                return own
            family = i18n.lookup(f"{words}.{'.'.join(part)}") if words else None
            if family is not None:
                return family
            return i18n.node_text(node, *path, *part, in_scope=scope)
        return i18n.lookup(key, scope=scope)

    def key_of(*part: str) -> tuple[str, str | None] | None:
        """The key word() reads its word from (and the extension scope it is in), None when there is none."""
        key = ".".join((*path, *part))
        if node:
            own = f"node.{node}.{key}"
            if i18n.lookup(own, scope=scope, exact=True) is not None:
                return own, scope
            if words and i18n.lookup(f"{words}.{'.'.join(part)}") is not None:
                return f"{words}.{'.'.join(part)}", None
            return (own, scope) if i18n.node_text(node, *path, *part, in_scope=scope) is not None else None
        return (key, scope) if i18n.lookup(key, scope=scope) is not None else None

    words = spec.get("words") or ""
    out = dict(spec)
    # the label is kept by its key (i18n.Word, a str in the language now): a message naming the parameter
    # (E-PARAM-INVALID, E-NODE-NOTWIREABLE, E-VALUES-…) reads in whoever's language follows it
    found = key_of("label")
    out["label"] = i18n.Word(found[0], in_scope=found[1]) if found else spec["name"]
    out["placeholder"] = word("placeholder") or ""
    out["assumed"] = word("assumed") or ""
    if spec["options"]:
        labels = {}
        for v in spec["options"]:
            key = option_key(v)  # keyed as the page and data/values.py option_label read them: the value as text
            said = word("option", key)
            if said is not None:
                labels[key] = said
        out["option_labels"] = labels or None
    out["group_id"] = spec["group"]
    out["group"] = group_label(spec["group"])
    out["unit_label"] = unit_label(spec["unit"]) if spec["unit"] else ""
    if spec.get("items"):
        out["items"] = [_dressed(i, node, scope, (*path, "field", i["name"])) for i in spec["items"]]
    return out


_DRESSED: dict[tuple, list[dict]] = {}


def dressed_specs(cls, specs: list[dict]) -> list[dict]:
    """A node type's parameter table dressed in the language now, worked out once per type and language (forgotten
    when the catalogues change: forget_words)."""
    from .. import i18n
    from . import text

    key = (cls, i18n.current())
    if key not in _DRESSED:
        _DRESSED[key] = dress(specs, getattr(cls, "id", ""), text.scope_of(cls))
    return _DRESSED[key]


def forget_words() -> None:
    """The dressed tables are worked out again (a catalogue changed: nodes/text.py refresh)."""
    _DRESSED.clear()


@dataclass(frozen=True)
class Choice:
    """One option of a parameter whose options come from a registry (P(options_from=)), as the registry declares it:
    its value; the catalogue key of its name ("" the parameter's own option word, …param.<p>.option.<value>); when it
    can be picked (a condition like option_applies'); and what picking it switches the node to (the licence class,
    nodes/tags.py NONCOMMERCIAL / RESEARCH, and 需注册), as an OptionTrait would."""

    value: str
    word: str = ""
    applies: Cond | None = None
    licence: str = ""
    registration: bool = False


class _Extra(dict):
    """A parameter's declarations beyond pydantic's (P): what front ends read, plus the condition under which it
    applies, kept out of the JSON the schema is built from."""

    applies: Cond | None = None
    # {choice: the condition under which that one choice can be picked}, kept out of the JSON like `applies`
    option_applies: dict | None = None
    # () -> the Choices it offers now, read each time (a registry extensions add to as they load), kept out of the JSON
    options_from: Callable[[], Sequence[Choice]] | None = None


def P(default: Any = ..., *, group: str = "", widget: str | None = None, applies: Cond | None = None,
      option_applies: dict | None = None, options_from: Callable[[], Sequence[Choice]] | None = None, **kw) -> Any:
    """A node parameter. Its words are in the catalogues, never here (node.<type>.param.<name>.label / .placeholder /
    .assumed / .option.<value>, falling back to param.<name>.…: dress); `group` is the id of its group (group.<id>).
    Beyond pydantic's own arguments:
    applies: when it does anything (nodes/applies.py: Wired("confidence"), Not(Wired("camera")) (a connected camera
    decides what it would), Param("mode").one_of("video"), Param("gravity").set(), Param("strength").gt(0),
    WiredType("image", "image.3", "image.4"), fact("segments").gt(2), All/AnyOf of them). Where it does nothing it is greyed out in the
    panel with why and cooked at its default (so it neither changes nor re-cooks the result).
    worker=False: the node applies it (units, thresholds, point density, a lens it converts): not sent to the
    worker, so changing it reuses the model's raw results.
    accept: for a file parameter, the file name suffixes it takes (the system's file dialog shows only those).
    unit: shown inside a number field (mm, °, px, EV, 帧): labels carry no brackets.
    A parameter carries no help text: the parameter panel shows no tips (webui/src/platform/tips.ts), so what a
    parameter means is in its label, its unit, its placeholder and its options' names.
    derived_from: parameters the node works this one out from (NodeDef.derive): the editor asks the server when one of
    them changes and sets both at once (读取序列's layers, from its file).
    choices_from: what its options come from (NodeDef.choices): other parameters and input ports (an import node's camera: the
    cameras of the picked file; the in-betweening nodes: the joints of the wired skeleton). Widget "choice" lists
    them as a dropdown, the empty value first (自动: what it comes to here); a table's field with widget "choice" lists
    its table's. The editor asks the server again when one of them changes.
    unique: a node added in the editor gets a value no other node of the graph has (its default, numbered on).
    option_applies: {choice: when that one choice can be picked}, the same conditions as `applies`, one level down
    (nodes/applies.py option_conditions). A choice that is for one shape of data says so with
    incoming("<input>", "<fact>") (「2D 跟踪点输出设置」's CornerPin: exactly four points). The server resolves it and
    the page greys that option with why, instead of letting the cook fail at the end.
    overrides: input ports whose own say this parameter overrides when it is set, typed or wired (a focal length over
    the connected camera's): the opposite of applies=Not(Wired(...)). Empty, the input gives it; the node says which one it uses.
    parts: the names of the three slots of widget "vec3", X, Y, Z by default (coordinates and angles); colours use
    ("R", "G", "B"), since axis names on colour slots are not meaningful to artists. Not a separate widget: the same
    widget with different slot names.
    lines: how many lines a text parameter uses, 1 by default (a single-line input). 2 or more draws a wrapping
    multi-line text box of that height, for parameters such as prompts whose value is a full English sentence that
    would fill one line after a few words. Not a separate widget (the same kind of declaration as `parts` for vec3 and
    `unit` for numbers): the widget is the same text box, `simple` stays "text" and the wire is still a text wire; only
    its height and wrapping differ. The parameter panel and the node body draw the same widget from the same
    declaration (webui/src/ui/controls.tsx TextField).
    per_frame: a number that may change over the shot (a zoom's focal length): a wire bringing one value per frame
    is taken as it is (CookContext.values) instead of being refused when it changes.
    wired=True: its input port is on the node from the start, on every node that has this parameter; the
    parameter declares it once (nodes/lens.py focal_param / filmback_param: 「已知 Focal Length」「Filmback」) instead of every
    node repeating the name in its own `wired_ports` (a copied list is a list that goes stale).
    NodeDef.wired_ports is where this and a node's own list are resolved into the ports.
    panel=False: the parameter is not in the parameter panel: it has no value to type, only a wire drives it
    (a value that only another node's result can give).
    Its input port is there as always (NodeDef.param_port), so a wire dropped on the node still finds it
    (webui/src/graph/rules.ts loosePort), and a graph file that sets it still cooks with that value.
    Any number, switch, vector or free text parameter can be driven by a wire of its value type (NodeDef.param_port).
    options_from: a text parameter whose options are a registry's, not written in the node (「标准人」's 「骨架」: the
    core's body and those extensions register, data/standard_bodies.py): () -> the Choices there are now. They are
    listed like a Literal's (a dropdown on the node and in the panel), each with its name (Choice.word), greyed with
    why while it cannot be picked (Choice.applies, as option_applies) and tagged with the licence it switches the node
    to (Choice.licence / registration, as an OptionTrait: nodes/applies.py traits_of). A value no longer offered is
    the cook's to refuse."""
    extra = _Extra(group=group)
    extra.applies = applies
    extra.option_applies = dict(option_applies or {}) or None
    extra.options_from = options_from
    if widget:
        extra["widget"] = widget
    for key in ("affects_result", "worker", "accept", "unit", "derived_from",
                "choices_from", "unique", "overrides", "per_frame", "measured", "panel", "parts",
                "wired", "lines", "words"):
        if key in kw:
            extra[key] = kw.pop(key)
    return Field(default, json_schema_extra=extra, **kw)


_SOURCES: dict[type, dict[str, Callable[[], Sequence[Choice]]]] = {}


def registered_options(params: type[BaseModel]) -> dict[str, tuple[Choice, ...]]:
    """The parameters of this model whose options come from a registry (P(options_from=)) -> the Choices offered now
    (which parameters have one is worked out once per model, the Choices each time)."""
    if params not in _SOURCES:
        _SOURCES[params] = {name: src for name, f in params.model_fields.items()
                            if (src := getattr(f.json_schema_extra, "options_from", None)) is not None}
    return {name: tuple(src()) for name, src in _SOURCES[params].items()}


def with_registered_options(specs: list[dict], params: type[BaseModel]) -> list[dict]:
    """A dressed parameter table with the registry's options of each P(options_from=) parameter filled in now: its
    values, and their names (Choice.word, else the parameter's own option word already dressed, else the value)."""
    from .. import i18n

    registered = registered_options(params)
    if not registered:
        return specs
    out = []
    for spec in specs:
        choices = registered.get(spec["name"])
        if choices is None:
            out.append(spec)
            continue
        own = spec.get("option_labels") or {}
        labels = {c.value: (i18n.lookup(c.word) if c.word else None) or own.get(c.value) or c.value for c in choices}
        filled = {**spec, "options": [c.value for c in choices], "option_labels": labels or None}
        out.append({**filled, "simple": simple_kind(filled)})
    return out


def always_wired(params: type[BaseModel]) -> tuple[str, ...]:
    """The parameters of this model that declare `P(wired=True)`: their input port is on every node that has them,
    in field order. `NodeDef.wired_ports` is the one place this and a node's own list are resolved into the ports."""
    return tuple(name for name, f in params.model_fields.items()
                 if isinstance(f.json_schema_extra, dict) and f.json_schema_extra.get("wired"))


# ---------------------------------------------------------------- shared fields
# Common parameters are defined once so every node spells and places them the same.


def colorspace_param(**kw) -> Any:
    """The 「色彩空间」 parameter, without an 「自动」 option: the value is always an actual colour space name. The
    default depends on the format and is provided by the node's choices to the web page ("default"), which writes it
    into the parameter so that users can see and change it; the cook applies the same format rule as a fallback (when
    the graph file does not contain it yet). Reader nodes state what the file is; output nodes state what to write."""
    return P(None, group="color", widget="colorspace", **kw)


# Parameters several nodes share: one definition each.


def fp16_param(group: str, default: bool = True) -> Any:
    """Half precision. `default=False`: projects whose upstream does not enable half precision (for example, where the
    upstream documentation states it is numerically unstable) follow upstream; this decision is not overridden."""
    return P(default, group=group)


def typed_list(text: str) -> list[str]:
    """The entries of a list typed into a text parameter (1,3 / Hair, Face_Neck): commas of either width or 、
    separate them."""
    return [t.strip() for t in re.split(r"[,\uff0c\u3001]", text or "") if t.strip()]


def person_ids(text: str) -> set[int]:
    """The person numbers typed into a parameter (1,3)."""
    try:
        return {int(t) for t in typed_list(text)}
    except ValueError:
        raise Invalid(Msg("E-PEOPLE-IDS", text=text)) from None
