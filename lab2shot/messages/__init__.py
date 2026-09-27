"""Messages: every sentence Lab2Shot says to a user, by code, with its Chinese in one catalogue.

A message is what happened, what is wrong, what to do — never a label, a tooltip or a stage word (those are UI
vocabulary, declared beside what they name). Code only ever writes a code and its parameters:

    ctx.say("W-COOK-NONCOMMERCIAL", projects=["ViPE"])    # a node, while it cooks (engine/cook.py CookContext.say)
    raise CookError(nid, Msg("E-COOK-MISSINGOUTPUT", node="深度", ports=["depth"]))  # an error a user sees (lab2shot/errors.py)

A code is 类型字母-模块-含义 (`CODE`): E error, W warning, N notice, I information (the log only), B blocked before
anything is cooked, P production-risk (red, the --production-risk token: a possible production accident, never an
ordinary error). The templates live in lab2shot/messages/*.toml (the core's, one folder) and in each extension's
adapters/<name>/messages.toml, whose module is the extension's name in capitals without underscores
(`module_of_extension`); `catalogue()` collects the core's and those of every extension that loaded, so removing an
extension (its folder) removes its messages and nothing else.

A template names its parameters: {name}, or {name:spec} with spec one of FORMATS (the web page formats the same
subset, webui/src/messages/format.ts). A list is joined with 、; a parameter that is a Msg (or an error carrying one)
is its text, so a reason can be a message of its own. `port`, `param` and `fix` are anchors a message is said at (the
input, the parameter, the node that puts it right), never template parameters (ANCHORS).

The catalogue and the code stay in step: every code the code raises is in the catalogue, and no user-facing text is written in code.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

CODE = re.compile(r"^([EWNIBP])-([A-Z][A-Z0-9]*)-([A-Z][A-Z0-9]*)$")
# A message that can end up on a node's one-line footer also declares a short form, 「<CODE>.short」 beside it in the
# same file. The whole sentence then only shows where there is room for it — the node's 数据信息 card, the parameter
# panel, the log. SHORT_WIDTH is the limit, counted in full-width characters (a Latin letter or a digit is half of
# one); every .short text keeps to it, so the footer never has to cut or abbreviate.

SHORT_SUFFIX = ".short"
SHORT_WIDTH = 12
LEVELS = {"E": "error", "W": "warning", "N": "notice", "I": "info", "B": "blocked", "P": "production-risk"}
PLACEHOLDER = re.compile(r"\{(\w+)(?::([^{}]*))?\}")
FORMATS = re.compile(r"^(d|,|g|\.\d+f|\.\d+%)$")
ANCHORS = ("port", "param", "fix")
CORE_DIR = Path(__file__).resolve().parent


class CatalogueError(ValueError):
    """The catalogue does not check out (a malformed code, a template with an unknown format, a code twice): a
    programming error, found when it loads."""


@dataclass(frozen=True, init=False)
class Msg:
    """One message: its code and the parameters its template names."""

    code: str
    params: dict[str, Any] = field(default_factory=dict)

    def __init__(self, code: str, /, **params: Any):
        if not CODE.match(code):
            raise CatalogueError(f"not a message code: {code!r} (LETTER-MODULE-MEANING)")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "params", params)

    @property
    def level(self) -> str:
        return self.code[0]

    @property
    def text(self) -> str:
        return render(self.code, self.params)

    @property
    def short(self) -> str:
        """The few words this message shows where one line is all there is (a node's footer): its own short form when
        it declares one, else the whole sentence. Nothing ever cuts or abbreviates the text itself."""
        text = shorts().get(self.code)
        return PLACEHOLDER.sub(lambda m: format_value(self.params[m.group(1)], m.group(2)), text) if text is not None else self.text

    def __str__(self) -> str:
        return self.text

    def json(self) -> dict:
        """What travels to the page, a client or a log: the code, its level letter, the text, its parameters (as
        plain values)."""
        return {"code": self.code, "level": self.level, "text": self.text, "short": self.short,
                "params": {k: plain(v) for k, v in self.params.items()}}


def worker_params(params: dict[str, Any]) -> dict[str, Any]:
    """worker 用 JSON 交上来的参数：里面嵌的消息（lab2shot_worker.reason 交的 {"message": 编号, "params": {...}}）
    还原成 Msg，这样 worker 也能像核心一样把一条消息当成另一条的参数（模板里的 {why}）。"""
    return {k: _from_worker(v) for k, v in params.items()}


def _from_worker(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"message", "params"} and isinstance(value["message"], str):
        return Msg(value["message"], **worker_params(value["params"]))
    if isinstance(value, list):
        return [_from_worker(v) for v in value]
    return value


def plain(value: Any) -> Any:
    """A parameter as JSON can carry it."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in value]
    return _text(value)


def module_of_extension(name: str) -> str:
    """The module an extension's codes are in: its name in capitals, underscores dropped (mediapipe_face ->
    MEDIAPIPEFACE)."""
    return name.replace("_", "").upper()


def code_of(key: str) -> str | None:
    """The message a catalogue key is about: the key itself, or the code a 「<CODE>.short」 key shortens (None: neither)."""
    base = key[: -len(SHORT_SUFFIX)] if key.endswith(SHORT_SUFFIX) else key
    return base if CODE.match(base) else None


def _read(path: Path) -> dict[str, str]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CatalogueError(f"{path}: {exc}") from None
    out = {}
    for code, template in data.items():
        if code_of(code) is None or not isinstance(template, str):
            raise CatalogueError(f"{path}: {code!r} is not LETTER-MODULE-MEANING = \"template\" (or LETTER-MODULE-MEANING.short)")
        problem = template_problem(template)
        if problem:
            raise CatalogueError(f"{path}: {code}: {problem}")
        out[code] = template
    for key in out:
        if key.endswith(SHORT_SUFFIX) and code_of(key) not in out:
            raise CatalogueError(f"{path}: {key} shortens {code_of(key)}, which this file does not have")
    return out


def template_problem(template: str) -> str:
    """Why a template can't be used ("" it can): a format spec outside FORMATS, an anchor used as a parameter, a
    brace that is no placeholder."""
    for m in PLACEHOLDER.finditer(template):
        if m.group(2) is not None and not FORMATS.match(m.group(2)):
            return f"format {{{m.group(1)}:{m.group(2)}}} is not one of d , g .Nf .N%"
        if m.group(1) in ANCHORS:
            return f"{{{m.group(1)}}} is an anchor, not a parameter"
    if "{" in PLACEHOLDER.sub("", template) or "}" in PLACEHOLDER.sub("", template):
        return "a brace that is not a {placeholder}"
    return ""


def core_files() -> list[Path]:
    return sorted(CORE_DIR.glob("*.toml"))


def merge(sections: list[tuple[str, dict[str, str], str | None]]) -> dict[str, str]:
    """One catalogue of (where, its templates, the module it must keep to — None: any core module) sections; a code
    twice, or an extension's code outside its module, is an error."""
    out: dict[str, str] = {}
    where: dict[str, str] = {}
    for name, templates, module in sections:
        for code, template in templates.items():
            if module is not None and CODE.match(code_of(code)).group(2) != module:
                raise CatalogueError(f"{name}: {code} is not in its extension's module {module}")
            if code in out:
                raise CatalogueError(f"{code} is in both {where[code]} and {name}")
            out[code], where[code] = template, name
    return out


def core_catalogue() -> dict[str, str]:
    """The core's codes alone (no extension, and no 「<CODE>.short」 keys)."""
    return {k: v for k, v in merge([(p.name, _read(p), None) for p in core_files()]).items() if not k.endswith(SHORT_SUFFIX)}


def extension_section(name: str, folder: Path) -> dict[str, str]:
    """An extension's own messages (adapters/<name>/messages.toml; none: {})."""
    path = folder / "messages.toml"
    return _read(path) if path.is_file() else {}


def extension_folders() -> dict[str, Path]:
    """Every adapter folder (adapters/<name>/ with an extension.py, lab2shot/adapters.py's rule), by name: an extension
    removed is a folder gone, and its messages with it. Read here from the folder itself, not through the adapter
    loader, so the catalogue sits below everything that says a message (the layering)."""
    from ..config import ADAPTERS_DIR

    return {f.name: f for f in sorted(ADAPTERS_DIR.iterdir()) if (f / "extension.py").is_file()} if ADAPTERS_DIR.is_dir() else {}


@cache
def _all() -> dict[str, str]:
    """Every key of every section: the codes and the 「<CODE>.short」 forms beside them."""
    sections = [(p.name, _read(p), None) for p in core_files()]
    core_modules = {CODE.match(code_of(c)).group(2) for _, t, _ in sections for c in t}
    for name, folder in extension_folders().items():
        module = module_of_extension(name)
        if module in core_modules:
            raise CatalogueError(f"extension {name}: its module {module} is a core module")
        sections.append((f"adapters/{name}/messages.toml", extension_section(name, folder), module))
    return merge(sections)


@cache
def catalogue() -> dict[str, str]:
    """Code -> template: the core's and every extension's own section."""
    return {k: v for k, v in _all().items() if not k.endswith(SHORT_SUFFIX)}


@cache
def shorts() -> dict[str, str]:
    """Code -> its short form, for the codes that declare one (Msg.short)."""
    return {k[: -len(SHORT_SUFFIX)]: v for k, v in _all().items() if k.endswith(SHORT_SUFFIX)}


def short_width(text: str) -> float:
    """How wide a short form is, in full-width characters: a Latin letter, a digit or a space is half of one."""
    return sum(0.5 if ord(c) < 0x2000 else 1.0 for c in text)


def template(code: str) -> str:
    try:
        return catalogue()[code]
    except KeyError:
        raise CatalogueError(f"message {code} is not in the catalogue (lab2shot/messages/*.toml, adapters/*/messages.toml)") from None


def names(template_text: str) -> set[str]:
    """The parameters a template names."""
    return {m.group(1) for m in PLACEHOLDER.finditer(template_text)}


def _text(value: Any) -> str:
    if isinstance(value, Msg):
        return value.text
    said = getattr(value, "message", None)
    if isinstance(said, Msg):
        return said.text
    if isinstance(value, (list, tuple, set, frozenset)):
        return "、".join(_text(v) for v in value)
    return str(value)


def format_value(value: Any, spec: str | None) -> str:
    if spec is None or isinstance(value, (Msg, str)) or getattr(value, "message", None) is not None:
        return _text(value)
    return format(value, spec)


def render(code: str, params: dict[str, Any]) -> str:
    """The message's text: its template with the parameters in. A parameter the template names and the call does not
    give is a programming error, said loudly; one it gives and the template does not name, too."""
    text = template(code)
    wanted = names(text)
    missing, extra = wanted - set(params), set(params) - wanted
    if missing or extra:
        raise CatalogueError(f"message {code}: parameters {sorted(missing)} missing, {sorted(extra)} not in its template")
    return PLACEHOLDER.sub(lambda m: format_value(params[m.group(1)], m.group(2)), text)


def said_line(message: dict, node_label: str = "") -> str:
    """How a message goes into a log: [code] 「node」 text."""
    where = f"「{node_label}」" if node_label else ""
    return f"[{message['code']}] {where}{message['text']}"
