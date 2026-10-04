"""Messages: every sentence Lab2Shot says to a user, by code, its words in one catalogue per language (lab2shot/i18n).

A message is what happened, what is wrong, what to do — never a label, a tooltip or a stage word (those are UI
vocabulary, i18n words by key). Code only ever writes a code and its parameters:

    ctx.say("W-COOK-NONCOMMERCIAL", projects=["ViPE"])    # a node, while it cooks (engine/cook.py CookContext.say)
    raise CookError(nid, Msg("E-COOK-MISSINGOUTPUT", node=..., ports=["depth"]))  # an error a user sees (lab2shot/errors.py)

A code is LETTER-MODULE-MEANING (`CODE`): E error, W warning, N notice, I information (the log only), B blocked before
anything is cooked, P production-risk (red, the --production-risk token: a possible production accident, never an
ordinary error). The templates live in lab2shot/i18n/<lang>/messages/*.toml (the core's) and in each extension's
adapters/<name>/i18n/<lang>.toml, whose module is the extension's name in capitals without underscores
(`module_of_extension`); `catalogue()` collects the core's and those of every extension that loaded, so removing an
extension (its folder) removes its messages and nothing else.

A Msg is its code and parameters, never its words: `text` and `short` are rendered in the language now
(i18n.current: the request's, the command line's; the log's is always English, logs.say). What is kept or sent on
later (a job's events, a packet's manifest) keeps the code and parameters with the text it had; `localized` renders
such a record again in the language of whoever reads it (the server does so on every answer: server/routes.py).

A message may also say itself to whoever uses a card (the web page's app mode: no nodes, no wires, no node names):
「<CODE>.app」 beside it in the same file (lab2shot/i18n APP_SUFFIX). `Msg.app` is those words (None when the code has
none), and `json()` carries them as "app" beside "text": the page shows them while it shows a card (webui/src/messages
message.ts textOf). It may name a parameter the plain words leave out (a root cause the node mode points at by its
node), and nested messages in it are said in their own app words too (`phrasing`). A message without app words of its
own whose plain words name a node, one of its inputs or outputs, or a line of the graph (a parameter in NODE_PARAMS)
is never said to them as it is: they hear its level's general sentence (「<L>-APP-INSIDE」: this card has a step that
…, switch to the node mode to see which), so no node name and no wire ever reaches a card (`inside`). Messages a card's
user can act on have app words of their own, saying what to do on the card.

A template names its parameters: {name}, or {name:spec} with spec one of FORMATS (the web page formats the same
subset, webui/src/messages/format.ts). A list is joined with the language's separator (i18n list.sep); a parameter
that is a Msg (or an error carrying one) is its text, so a reason can be a message of its own. `port`, `param` and
`fix` are anchors a message is said at (the input, the parameter, the node that puts it right), never template
parameters (ANCHORS).

The catalogue and the code stay in step: every code the code raises is in the catalogue (`lab2shot check messages`), and no
user-facing text is written in code.
"""

from __future__ import annotations

import re
import tomllib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Callable

from lab2shot_shared.protocol import CODE  # the one rule, shared with the workers

from .. import i18n
from ..i18n import Both, Word  # noqa: F401 (a word kept in both languages: messages.Both is i18n.Both)

# A message that can end up on a node's one-line footer also declares a short form, 「<CODE>.short」 beside it in the
# same file. The whole sentence then only shows where there is room for it — the node's 数据信息 card, the parameter
# panel, the log. SHORT_WIDTH is the limit, counted in full-width characters (a Latin letter or a digit is half of
# one); every .short text keeps to it, in each language, so the footer never has to cut or abbreviate.

SHORT_SUFFIX = i18n.SHORT_SUFFIX
APP_SUFFIX = i18n.APP_SUFFIX
# which words a message is said in now: its plain ones, or those for whoever uses a card (「<CODE>.app」, Msg.app);
# nested messages follow, so a reason inside an app sentence is said in its app words too
_PHRASING: ContextVar[str] = ContextVar("lab2shot_phrasing", default="")
SHORT_WIDTH = 12
LEVELS = {"E": "error", "W": "warning", "N": "notice", "I": "info", "B": "blocked", "P": "production-risk"}
PLACEHOLDER = i18n.PLACEHOLDER
FORMATS = i18n.FORMATS
ANCHORS = ("port", "param", "fix")
# the parameters that name a node, one of its inputs or outputs, or a line of the graph: a message whose plain words
# name one says itself to whoever uses a card in its own app words, or in its level's general one (above). The page
# keeps the same set (webui/src/messages/format.ts NODE_PARAMS)
NODE_PARAMS = frozenset({"node", "nodes", "source", "target", "input", "output", "root", "chain", "chains", "block",
                         "inner", "outer", "ports"})
INSIDE = "{level}-APP-INSIDE"  # the general app sentence of each level (web.toml)
CORE_DIR = i18n.lang_dir(i18n.DEFAULT) / "messages"  # the Chinese messages: every code is here (the full set)


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
        """Its words in the language now (i18n.current)."""
        return render(self.code, self.params)

    @property
    def app(self) -> str | None:
        """Its words for whoever uses a card (「<CODE>.app」, in the language now; its level's general sentence when it
        has none and its plain words name a node, `inside`), None when it needs none."""
        if self.code not in _apps(i18n.normal(None) or i18n.current()) and not inside(self.code):
            return None
        with phrasing("app"):
            return render(self.code, self.params)

    @property
    def short(self) -> str:
        """The few words this message shows where one line is all there is (a node's footer): its own short form when
        it declares one, else the whole sentence. Nothing ever cuts or abbreviates the text itself."""
        text = _form(_short_forms(i18n.normal(None) or i18n.current()), self.code, self.params)
        return PLACEHOLDER.sub(lambda m: format_value(self.params[m.group(1)], m.group(2)), text) if text is not None else self.text

    def __str__(self) -> str:
        return self.text

    def json(self) -> dict:
        """What travels to the page, a client or a log: the code, its level letter, the text and short form (in the
        language now), its parameters (as plain values), and, when a parameter is a message itself, `args`: the
        parameters as messages (`wire`), so `localized` can say it again in another language."""
        out = {"code": self.code, "level": self.level, "text": self.text, "short": self.short,
               "params": {k: plain(v) for k, v in self.params.items()}}
        if (app := self.app) is not None:
            out["app"] = app
        if any(_nested(v) for v in self.params.values()):
            out["args"] = {k: wire(v) for k, v in self.params.items()}
        return out


def _said(value: Any) -> Msg | None:
    if isinstance(value, Msg):
        return value
    said = getattr(value, "message", None)
    return said if isinstance(said, Msg) else None


def _both(value: Any) -> "Both | None":
    """`value` as a word kept in both languages: a Both, or an error raised with one (ValueError(Both(...)))."""
    if isinstance(value, Both):
        return value
    if isinstance(value, BaseException) and len(value.args) == 1 and isinstance(value.args[0], Both):
        return value.args[0]
    return None


def _word(value: Any) -> "Word | None":
    """`value` as a catalogue word kept by its key: a Word, or an error raised with one (ValueError(Word(...)))."""
    if isinstance(value, Word):
        return value
    if isinstance(value, BaseException) and len(value.args) == 1 and isinstance(value.args[0], Word):
        return value.args[0]
    return None


def _nested(value: Any) -> bool:
    if _said(value) is not None or _both(value) is not None or _word(value) is not None:
        return True
    return isinstance(value, (list, tuple, set, frozenset)) and any(_nested(v) for v in value)


def wire(value: Any) -> Any:
    """A parameter as JSON can carry it with its messages kept as messages ({"message": code, "params": {...}}, the
    workers' form, read back by worker_params)."""
    said = _said(value)
    if said is not None:
        return {"message": said.code, "params": {k: wire(v) for k, v in said.params.items()}}
    if _both(value) is not None:
        return {"word": dict(_both(value).texts)}
    if (word := _word(value)) is not None:  # a catalogue word kept by its key: said in whoever's language reads it
        return {"said": word.key, "params": {k: wire(v) for k, v in word.params.items()}, **({"scope": word.scope} if word.scope else {})}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [wire(v) for v in value]
    return plain(value)


def worker_params(params: dict[str, Any]) -> dict[str, Any]:
    """A worker's parameters as it sent them in JSON, with every message inside them (lab2shot_worker.reason's
    {"message": code, "params": {...}}) made a Msg again, so a worker can pass one message as another's parameter
    (a template's {why}) the way the core does."""
    return {k: _from_worker(v) for k, v in params.items()}


def _from_worker(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"message", "params"} and isinstance(value["message"], str):
        return Msg(value["message"], **worker_params(value["params"]))
    if isinstance(value, dict) and set(value) == {"word"} and isinstance(value["word"], dict):
        return Both({k: str(v) for k, v in value["word"].items()})
    if isinstance(value, dict) and set(value) - {"scope"} == {"said", "params"} and isinstance(value["said"], str) \
            and i18n.has(value["said"], scope=value.get("scope")):
        return Word(value["said"], in_scope=value.get("scope"), **worker_params(value["params"]))
    if isinstance(value, list):
        return [_from_worker(v) for v in value]
    return value


def plain(value: Any) -> Any:
    """A parameter as JSON can carry it."""
    if isinstance(value, (Both, Word)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in value]
    return _text(value)


def again(said: dict) -> Msg | None:
    """A message as it travelled (Msg.json(): code, params, args) made a Msg again, None when it cannot be (no code, a
    code no catalogue has, its parameters not its template's)."""
    code = said.get("code")
    params = said.get("args") if isinstance(said.get("args"), dict) else said.get("params")
    if not isinstance(code, str) or not CODE.match(code) or not isinstance(params, dict) or code not in catalogue():
        return None
    msg = Msg(code, **worker_params(params))
    try:
        render(code, msg.params)
    except (CatalogueError, KeyError, ValueError, TypeError):
        return None
    return msg


def localized(value: Any) -> Any:
    """`value` (an answer, an event, a record: JSON) with every message in it ({code, text, params …}) said again in
    the language now. What was made elsewhere (a job's events on the farm's thread, a manifest written last week)
    reads in the language of whoever asks; a message that cannot be said again keeps the text it had. Containers are
    copied only where something changed: the rest is `value` itself."""
    if isinstance(value, list):
        out_list = None
        for i, v in enumerate(value):
            nv = localized(v) if isinstance(v, (dict, list)) else v
            if nv is not v:
                if out_list is None:
                    out_list = list(value)
                out_list[i] = nv
        return value if out_list is None else out_list
    if not isinstance(value, dict):
        return value
    out = None
    for k, v in value.items():
        if isinstance(v, (dict, list)):
            nv = localized(v)
            if nv is not v:
                if out is None:
                    out = dict(value)
                out[k] = nv
    cur = value if out is None else out
    if isinstance(cur.get("text"), str) and isinstance(cur.get("code"), str) and "params" in cur:
        msg = again(cur)
        if msg is not None:
            text = msg.text
            short = msg.short if "short" in cur else None
            app = msg.app
            if text != cur["text"] or (short is not None and short != cur.get("short")) or app != cur.get("app"):
                cur = dict(cur)
                cur["text"] = text
                if short is not None:
                    cur["short"] = short
                if app is not None:
                    cur["app"] = app
                else:
                    cur.pop("app", None)
    # a word said with it: "<field>_word" = {"key", "scope", "params"} (a cooking stage's, CookContext.stage) says
    # "<field>" again in the language now
    for k, v in list(cur.items()):
        if k.endswith(WORD_SUFFIX) and isinstance(v, dict) and isinstance(v.get("key"), str):
            field = k[: -len(WORD_SUFFIX)]
            said = said_word(v)
            if said is not None and said != cur.get(field):
                cur = dict(cur) if cur is value else cur
                cur[field] = said
    return cur


WORD_SUFFIX = "_word"


def word_of(key: str, scope: str | None = None, **params: Any) -> dict:
    """A word kept to be said later, in whoever's language reads it (localized): {"key", "scope", "params"}."""
    return {"key": key, "scope": scope, "params": {k: wire(v) for k, v in params.items()}}


def msg_word(msg: "Msg") -> dict:
    """A message kept the way word_of keeps a word ({"key": its code, "params", "message": True}): said again, in
    whoever's language reads it, by said_word / localized (a cooking stage or progress said by a message)."""
    return {"key": msg.code, "scope": None, "params": {k: wire(v) for k, v in msg.params.items()}, "message": True}


def said_word(word: dict) -> str | None:
    """A word kept by word_of (or a message kept by msg_word), in the language now (its key's fallbacks too); None
    when it has no words."""
    if word.get("message"):
        try:
            return Msg(word["key"], **worker_params(dict(word.get("params") or {}))).text
        except Exception:  # a message that cannot be said again keeps the text it had
            return None
    params = worker_params(dict(word.get("params") or {}))  # a word or message among them (wire) said in the language now
    template = i18n.template_for(word["key"], params, scope=word.get("scope"))
    if template is None:
        return None
    try:
        return i18n.fill(template, params)
    except i18n.TextError:
        return None


def module_of_extension(name: str) -> str:
    """The module an extension's codes are in: its name in capitals, underscores dropped (mediapipe_face ->
    MEDIAPIPEFACE)."""
    return name.replace("_", "").upper()


def code_of(key: str) -> str | None:
    """The message a catalogue key is about: the key itself, the code a 「<CODE>.short」 key shortens, or the code of a
    plural form (CODE.one, CODE.other, CODE.short.one, CODE.short.other; i18n PLURAL_FORMS); None: none of them."""
    return i18n.message_code(key)


def split_key(key: str) -> tuple[str, str | None, str | None]:
    """A message key as (code, its kind of words, plural form or None): the kind None for the plain words, "short"
    (「.short」) or "app" (「.app」): "<CODE>.short.one" -> ("<CODE>", "short", "one")."""
    head, dot, form = key.rpartition(".")
    plural = form if dot and form in i18n.PLURAL_FORMS else None
    base = head if plural else key
    for kind, suffix in (("short", SHORT_SUFFIX), ("app", APP_SUFFIX)):
        if base.endswith(suffix):
            return base[: -len(suffix)], kind, plural
    return base, None, plural


def _checked(path: Path | str, data: dict) -> dict[str, str]:
    out = {}
    for code, template in data.items():
        if code_of(code) is None or not isinstance(template, str):
            raise CatalogueError(f"{path}: {code!r} is not LETTER-MODULE-MEANING = \"template\" (or LETTER-MODULE-MEANING.short)")
        problem = template_problem(template)
        if problem:
            raise CatalogueError(f"{path}: {code}: {problem}")
        out[code] = template
    return out


def _read(path: Path) -> dict[str, str]:
    """A file of messages only (i18n/<lang>/messages/*.toml); missing: {}."""
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CatalogueError(f"{path}: {exc}") from None
    out = _checked(path, data)
    _shorts_have_codes(path, out)
    return out


def _shorts_have_codes(path: Path | str, out: dict[str, str]) -> None:
    """A short form (or an app form) is of a code of the same file; a code is written either plain or in its plural
    forms (with .other), never both."""
    have = {split_key(k)[0] for k in out if not split_key(k)[1]}
    for key in out:
        code, kind, plural = split_key(key)
        if kind and code not in have:
            raise CatalogueError(f"{path}: {key} is a form of {code}, which this file does not have")
        base = key[: -len(plural) - 1] if plural else key
        if plural and base in out:
            raise CatalogueError(f"{path}: {base} is written both plain and in plural forms")
        if plural == "one" and f"{base}.other" not in out:
            raise CatalogueError(f"{path}: {key} has no {base}.other")


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


def core_files(lang: str = i18n.DEFAULT) -> list[Path]:
    """The core's message files of one language."""
    return i18n.message_files(lang)


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


def extension_section(name: str, folder: Path, lang: str = i18n.DEFAULT) -> dict[str, str]:
    """An extension's own messages of one language (the message keys of adapters/<name>/i18n/<lang>.toml; none: {})."""
    path = i18n.extension_file(folder, lang)
    if not path.is_file():
        return {}
    try:
        got = {k: v for k, v in i18n.read(path).items() if i18n.is_message_key(k)}
    except i18n.TextError as exc:
        raise CatalogueError(str(exc)) from None
    out = _checked(path, got)
    _shorts_have_codes(path, out)
    return out


def extension_folders() -> dict[str, Path]:
    """Every adapter folder (adapters/<name>/ with an extension.py), by name: an extension removed is a folder gone,
    and its messages with it."""
    return i18n.extension_folders()


@cache
def written(lang: str) -> dict[str, str]:
    """Every key of every section of one language as written (no fallback): the codes and the 「<CODE>.short」 forms."""
    sections = [(f"i18n/{lang}/messages/{p.name}", _read(p), None) for p in core_files(lang)]
    core_modules = {CODE.match(code_of(c)).group(2) for _, t, _ in sections for c in t}
    for name, folder in extension_folders().items():
        module = module_of_extension(name)
        if module in core_modules:
            raise CatalogueError(f"extension {name}: its module {module} is a core module")
        sections.append((f"adapters/{name}/i18n/{lang}.toml", extension_section(name, folder, lang), module))
    return merge(sections)


@cache
def _split(lang: str) -> tuple[dict[str, dict[str | None, str]], dict[str, dict[str | None, str]], dict[str, dict[str | None, str]]]:
    """A language's catalogue as ({code: {form: template}}, {code: {form: short template}}, {code: {form: app
    template}}): form None for a code written plain, "one" / "other" for a plural one (i18n PLURAL_FORMS)."""
    full: dict[str, dict[str | None, str]] = {}
    short: dict[str, dict[str | None, str]] = {}
    app: dict[str, dict[str | None, str]] = {}
    for key, text in written(lang).items():
        code, kind, plural = split_key(key)
        {None: full, "short": short, "app": app}[kind].setdefault(code, {})[plural] = text
    return full, short, app


def _apps(lang: str) -> dict[str, dict[str | None, str]]:
    return _split(lang)[2]


@cache
def inside(code: str) -> bool:
    """Whether `code` has no app words of its own and its plain words (the full catalogue's, every plural form) name a
    node, a port or a wire (NODE_PARAMS): said to whoever uses a card, it is its level's general sentence instead."""
    if code in _apps(i18n.DEFAULT):
        return False
    forms = _split(i18n.DEFAULT)[0].get(code, {})
    return any(names(f) & NODE_PARAMS for f in forms.values())


@contextmanager
def phrasing(kind: str):
    """Say messages in `kind`'s words for this stretch ("app": their 「.app」 forms where they have one; "": plain)."""
    token = _PHRASING.set(kind)
    try:
        yield
    finally:
        _PHRASING.reset(token)


def _form(table: dict[str, dict[str | None, str]], code: str, params: dict[str, Any]) -> str | None:
    """The template of `code` these parameters take: its only one, or the plural form of its number."""
    forms = table.get(code)
    if forms is None:
        return None
    if None in forms:
        return forms[None]
    other = forms["other"]
    lang = i18n.current()
    return forms.get("one", other) if i18n.plural_form(lang, i18n.plural_count(other, params)) == "one" else other


def catalogue(lang: str | None = None) -> dict[str, str]:
    """Code -> template in `lang` (None: the language now): the core's and every extension's own section; a plural
    message's is its `.other` form (the one that names every parameter)."""
    return _codes(i18n.normal(lang) or i18n.current())


def shorts(lang: str | None = None) -> dict[str, str]:
    """Code -> its short form, for the codes that declare one (a plural one's `.other` form; Msg.short picks)."""
    return _shorts(i18n.normal(lang) or i18n.current())


def _short_forms(lang: str) -> dict[str, dict[str | None, str]]:
    return _split(lang)[1]


@cache
def _codes(lang: str) -> dict[str, str]:
    return {c: f.get(None, f.get("other", "")) for c, f in _split(lang)[0].items()}


@cache
def _shorts(lang: str) -> dict[str, str]:
    return {c: f.get(None, f.get("other", "")) for c, f in _split(lang)[1].items()}


def short_width(text: str) -> float:
    """How wide a short form is, in full-width characters: a Latin letter, a digit or a space is half of one."""
    return sum(0.5 if ord(c) < 0x2000 else 1.0 for c in text)


def template(code: str, lang: str | None = None) -> str:
    try:
        return catalogue(lang)[code]
    except KeyError:
        raise CatalogueError(f"message {code} is not in the catalogue (lab2shot/i18n/<lang>/messages/*.toml, adapters/*/i18n/<lang>.toml)") from None


def names(template_text: str) -> set[str]:
    """The parameters a template names."""
    return {m.group(1) for m in PLACEHOLDER.finditer(template_text)}


def _text(value: Any) -> str:
    said = _said(value)
    if said is not None:
        return said.text
    if isinstance(value, (list, tuple, set, frozenset)):
        return i18n.separator().join(_text(v) for v in value)
    return str(value)


def format_value(value: Any, spec: str | None) -> str:
    if spec is None or isinstance(value, (Msg, str)) or getattr(value, "message", None) is not None:
        return _text(value)
    return format(value, spec)


def render(code: str, params: dict[str, Any], lang: str | None = None) -> str:
    """The message's text in `lang` (None: the language now): its template with the parameters in. A parameter the
    template names and the call does not give is a programming error, said loudly; one it gives and the template does
    not name, too."""
    with i18n.using(lang):  # nested messages and list separators in the same language
        template(code)  # in the catalogue, or said loudly
        full, _short, app = _split(i18n.current())
        forms = full[code]
        text = _form(app if _PHRASING.get() == "app" and code in app else full, code, params)
        if _PHRASING.get() == "app" and code not in app and inside(code):  # no node or wire said to a card's user
            text = _form(full, INSIDE.format(level=code[0]), {})
        wanted = set().union(*(names(f) for f in (*forms.values(), *app.get(code, {}).values())))
        missing, extra = wanted - set(params), set(params) - wanted
        if missing or extra:
            raise CatalogueError(f"message {code}: parameters {sorted(missing)} missing, {sorted(extra)} not in its template")
        return PLACEHOLDER.sub(lambda m: format_value(params[m.group(1)], m.group(2)), text)


def said_line(message: dict, node_label: str = "") -> str:
    """How a message goes into a log: [code] 「node」 text, in the log's language (English)."""
    with i18n.using(i18n.LOG_LANG):
        text = localized(message).get("text", "") if isinstance(message, dict) else str(message)
    where = f'"{node_label}" ' if node_label else ""
    return f"[{message['code']}] {where}{text}"
