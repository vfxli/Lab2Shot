"""A tool's contract as the panel and the job read it — the same for every tool of every source:

- input slots (`inputs` of the tool's signature): each a data type and a fixed outside name (input, cam_fbx_path,
  char_fbx_path …); a slot whose deliveries need a menu value (`when`) gets that value set when something is bound to
  it (binding a camera to cam_fbx_path picks 「自己的相机（FBX）」 on cam_src);
- parameters (`exposed`): in the interface's order and groups, hidden and greyed by the interface's own conditions
  (lab2shot_conditions: the server's rules, never a copy), the inputs' own file parameters and what is worked out from
  their files (a camera picked from the file: derived_from) left out — the server works those out from the file;
- the scene's context: values of fixed names the host knows (first_frame, fps, focal …) fill parameters of those
  names, marked 「来自场景」, unless the user set them; a value the parameter would not take is not filled;
- what is sent: the values the user or the scene set, each checked as the server checks it, plus every bound input.
"""

from __future__ import annotations

import json

from . import paths


def _cond():
    return paths.conditions_module()


def input_keys(tool: dict) -> set[str]:
    return {i.get("key") for i in tool.get("inputs") or []} | {i.get("param") for i in tool.get("inputs") or []}


def _input_nodes(tool: dict) -> dict[str, str]:
    """node id of each input -> its file parameter (to tell what is derived from the file)."""
    return {str(i.get("key", "")).split(".", 1)[0]: str(i.get("key", "")).split(".", 1)[-1] for i in tool.get("inputs") or []}


def parameters(tool: dict) -> list[dict]:
    """The exposed parameters the panel shows (buttons, the inputs' files and what derives from them left out)."""
    keys = input_keys(tool)
    by_node = _input_nodes(tool)
    out = []
    for x in tool.get("exposed") or []:
        spec = x.get("param") or {}
        targets = x.get("target") or []  # the node parameters it drives (a list: engine/templates.py exposed_params)
        if spec.get("widget") == "button" or x.get("name") in keys or any(k in keys for k in targets):
            continue
        node, _, param = str(targets[0] if targets else "").partition(".")
        if node in by_node and by_node[node] in (spec.get("derived_from") or []):
            continue
        out.append(x)
    return out


def default_values(tool: dict) -> dict:
    return {x["name"]: x.get("value") for x in tool.get("exposed") or [] if (x.get("param") or {}).get("widget") != "button"}


def current(tool: dict, state: dict) -> dict:
    """Every exposed parameter's value now: the tool's own, then the scene's, then the user's."""
    values = default_values(tool)
    values.update(state.get("scene_values") or {})
    values.update(state.get("values") or {})
    return values


def hidden(x: dict, values: dict) -> bool:
    return _cond().holds(x.get("hide_when"), values)


def disabled(x: dict, values: dict) -> bool:
    return bool(x.get("wired") and not x.get("fallback")) or _cond().holds(x.get("disable_when"), values)


def menu_options(x: dict, values: dict) -> list[tuple]:
    """[(value, label)] of a parameter with a choice: the interface's menu (options its own hide_when hides left
    out), else the parameter's own options."""
    if x.get("widget") == "menu" and isinstance(x.get("options"), list):
        return [(o.get("value"), paths.pick(o.get("label", o.get("value")))) for o in _cond().shown_options(x["options"], values)]
    spec = x.get("param") or {}
    if spec.get("options"):
        labels = spec.get("option_labels") or {}
        return [(o, str(labels.get(str(o), o))) for o in spec["options"]]
    return []


def refuses(x: dict, value) -> str | None:
    """Why the parameter would not take `value` (None: it would) — the server's own rules, never a copy: the
    interface's menu as apply_values holds it, else conditions.value_refused (the very function the server runs), so a
    value the server refuses is never sent."""
    spec = x.get("param") or {}
    if x.get("widget") == "menu" and isinstance(x.get("options"), list):
        if not any(_cond().same(value, o.get("value")) for o in x["options"] if isinstance(o, dict)):
            return paths.text("dcc.contract.not_an_option")
        return None
    if spec.get("type") not in ("boolean", "integer", "number", "string"):
        return None  # a list, a table: the node's own parameter model judges it on the server, as apply_values does
    return _cond().value_refused({"nullable": False, **spec}, value)


def parse(x: dict, text: str):
    """A value typed into a text field, as the parameter's type takes it ("" → none)."""
    spec = x.get("param") or {}
    text = str(text).strip()
    if text == "":
        return None
    kind = spec.get("type")
    if kind == "integer":
        return int(text)
    if kind == "number":
        return float(text)
    if kind in ("array", "object"):
        return json.loads(text)
    return text


def scene_fill(tool: dict, context: dict, colorspaces: list[str] | None = None) -> dict:
    """{name: value} of the context's fixed names the tool exposes and would take. A colour space is filled in only as
    the server names it (`colorspaces`, its OCIO configuration's; matched without case): a DCC's name the server's
    configuration does not have is left to the server's own default rather than sent to fail."""
    out = {}
    known = {c.lower(): c for c in colorspaces or []}
    by_name = {x["name"]: x for x in tool.get("exposed") or []}
    for name, value in (context or {}).items():
        x = by_name.get(name)
        if x is None or value is None or (x.get("wired") and not x.get("fallback")):
            continue
        spec = x.get("param") or {}
        if spec.get("widget") == "colorspace":
            value = known.get(str(value).lower())
            if value is None:
                continue
        if spec.get("type") == "integer" and isinstance(value, float) and value.is_integer():
            value = int(value)
        if spec.get("type") == "number" and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if refuses(x, value) is None:
            out[name] = value
    return out


def when_values(tool: dict, bindings: dict) -> dict:
    """The menu values bound optional inputs need ({cam_src: 1} once a camera is bound to cam_fbx_path)."""
    out = {}
    for i in tool.get("inputs") or []:
        if i.get("param") in bindings and i.get("when"):
            out.update(i["when"])
    return out


def submission(tool: dict, state: dict) -> dict:
    """The values to send: what the scene and the user set (wired ones and buttons never), every bound input's file,
    the menus the bound inputs need. Raises ValueError naming a value the server would refuse."""
    by_name = {x["name"]: x for x in tool.get("exposed") or []}
    values = {}
    chosen = {**(state.get("scene_values") or {}), **(state.get("values") or {})}
    for name, value in chosen.items():
        x = by_name.get(name)
        if x is None or (x.get("param") or {}).get("widget") == "button" or (x.get("wired") and not x.get("fallback")):
            continue
        why = refuses(x, value)
        if why:
            raise ValueError(paths.text("dcc.contract.refused", label=x.get("label", name), why=why))
        values[name] = value
    values.update(when_values(tool, state.get("files") or {}))
    values.update(state.get("files") or {})
    return values
