"""Node parameters: the pydantic base, `P()` and the parameter table front ends read.

A node's parameters are a pydantic model. `P()` records, beyond pydantic's field declaration, what front ends read
(group, widget, unit, applicability, wiring, ...); `_fields` / `_param_list` flatten the model into the table read by
the parameter panel and the engine. Node authors import `NodeParams`, `P`, `colorspace_param` and `fp16_param` from
`nodes.base` (the declaration facade).
"""

from __future__ import annotations

import re
import typing
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


def _defaults(model: type[BaseModel]) -> dict:
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


# the most parameters a node declares for its body (NodeDef.on_node): the node stays compact
def range_said(spec: dict) -> Msg:
    """A parameter's range as its message says it (E-PARAM-BETWEEN ...; E-PARAM-REFUSED when it has none)."""
    lo, hi = spec["minimum"], spec["maximum"]
    if lo is not None and hi is not None:
        return Msg("E-PARAM-BETWEEN", lo=lo, hi=hi)
    if lo is not None:
        return Msg("E-PARAM-ABOVE", lo=lo) if lo == 0 else Msg("E-PARAM-ATLEAST", lo=lo)
    if hi is not None:
        return Msg("E-PARAM-ATMOST", hi=hi)
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
                "label": field.title or name,
                "default": None if default is PydanticUndefined else default,
                "type": target.get("type", "string"),
                "nullable": any(a.get("type") == "null" for a in prop.get("anyOf", [])),
                "minimum": target.get("minimum", target.get("exclusiveMinimum")),
                "maximum": target.get("maximum", target.get("exclusiveMaximum")),
                "multiple_of": target.get("multipleOf"),
                "options": options,
                "option_labels": extra.get("option_labels"),
                "widget": extra.get("widget"),
                "group": extra.get("group", ""),
                "affects_result": extra.get("affects_result", True),
                "placeholder": extra.get("placeholder", ""),
                "worker": extra.get("worker", True),
                "accept": list(extra.get("accept", ())),
                "unit": extra.get("unit", ""),
                "derived_from": list(extra.get("derived_from", ())),
                "choices_from": list(extra.get("choices_from", ())),
                "unique": extra.get("unique", False),
                "measured": extra.get("measured") or {},  # setting -> the most VRAM it takes (GB), None time only
                "per_frame": extra.get("per_frame", False),
                # What the parameter amounts to when left empty ("按全画幅 36 mm 算"). Empty does not mean inactive:
                # the node still computes with a value, and the bottom row must state it (engine/evaluation.py sources).
                "assumed": extra.get("assumed", ""),
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
            }
        )
    return out


class _Extra(dict):
    """A parameter's declarations beyond pydantic's (P): what front ends read, plus the condition under which it
    applies, kept out of the JSON the schema is built from."""

    applies: Cond | None = None
    # {choice: the condition under which that one choice can be picked}, kept out of the JSON like `applies`
    option_applies: dict | None = None


def P(default: Any = ..., *, label: str, group: str = "", widget: str | None = None, applies: Cond | None = None,
      option_applies: dict | None = None, **kw) -> Any:
    """A node parameter. Beyond pydantic's own arguments:
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
    Any number, switch, vector or free text parameter can be driven by a wire of its value type (NodeDef.param_port)."""
    extra = _Extra(group=group)
    extra.applies = applies
    extra.option_applies = dict(option_applies or {}) or None
    if widget:
        extra["widget"] = widget
    for key in ("option_labels", "affects_result", "placeholder", "worker", "accept", "unit", "derived_from",
                "choices_from", "unique", "overrides", "per_frame", "measured", "panel", "parts",
                "wired", "assumed", "lines"):
        if key in kw:
            extra[key] = kw.pop(key)
    return Field(default, title=label, json_schema_extra=extra, **kw)


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
    return P(None, label="色彩空间", group="色彩", widget="colorspace", placeholder="按格式", **kw)


# Parameters several nodes share: one definition each.


def fp16_param(group: str, default: bool = True) -> Any:
    """Half precision. `default=False`: projects whose upstream does not enable half precision (for example, where the
    upstream documentation states it is numerically unstable) follow upstream; this decision is not overridden."""
    return P(default, label="半精度", group=group)


def typed_list(text: str) -> list[str]:
    """The entries of a list typed into a text parameter (1,3 / Hair, Face_Neck): commas of either width or 、
    separate them."""
    return [t.strip() for t in re.split(r"[,，、]", text or "") if t.strip()]


def person_ids(text: str) -> set[int]:
    """The person numbers typed into a parameter (1,3)."""
    try:
        return {int(t) for t in typed_list(text)}
    except ValueError:
        raise Invalid(Msg("E-PEOPLE-IDS", text=text)) from None
