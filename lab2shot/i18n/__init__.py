"""Languages: every word Lab2Shot shows, by key, in one catalogue per language; the code only ever writes keys.

The back end is English (names, ids, keys, logs); what a person reads is looked up here in the language of whoever is
asking. Two languages, the same keys in both:

    lab2shot/i18n/<lang>/messages/<module>.toml   the messages (lab2shot/messages: "CODE" = "template", "CODE.short")
    lab2shot/i18n/<lang>/**/*.toml                 every other word, by structured key (below)
    adapters/<name>/i18n/<lang>.toml               an extension's own: its messages and its words, removed with it

Keys are ASCII and structured; a file's keys are written in full (the file's name only groups them):

    node.<type>.subtitle / .description            a node type's subtitle (under its type name) and description
    node.<type>.param.<p>.label|placeholder|tip    a parameter      -> falls back to param.<p>.<…>
    node.<type>.param.<p>.option.<value>           an option        -> param.<p>.option.<value> -> option.<value>
    node.<type>.port.<port>.label|help|waits       a port           -> port.<port>.<…>
    node.<type>.stage.<id>                         a cooking stage  -> stage.<id> (the extension's, then the core's)
    node.<type>.button.<id>, node.<type>.fact.<id> a button, a fact -> button.<id>, fact.<id>
    group.<id>, type.<data type>, unit.<id>, category.<id>.label|tip, setting.<key>.label|help, right.<id>,
    route.<name>.summary, cli.<…>, dcc.<…>, ui.<area>.<…>, lang.<code>, list.sep

A type name holds dots (`fbx.import`): in TOML it is a quoted key (`[node."fbx.import".param.scale]`); flattened, the
keys read `node.fbx.import.param.scale.label`. An extension may define the shared keys (param.*, port.*, option.*) for
its own nodes only: looked up with `scope=<extension>` they come before the core's.

The language of the work being done is a context variable (`current`): the server sets it per request (server/access.py
Guard: the X-Lab2Shot-Lang header, the account's choice, the `lang` cookie, Accept-Language, zh), the command line once
for the process (cli/base.py: --lang, LAB2SHOT_LANG, LANG/LC_ALL), and `using(lang)` for a stretch of code (the log is
always written in English: logs.say). A thread started without one works in the process's language (`set_process`).

Every key is in both languages (`lab2shot check i18n`): a key one language lacks is a problem, never read in the other.

A sentence whose words follow a number (English "1 frame" / "2 frames") is written in its plural forms, the key with
`.one` and `.other` after it: English writes both, Chinese only `.other` (it has one form). The code uses the key
without the form (`t("ui.queue.jobs", count=n)`, `Msg("N-QUEUE-BEHIND", ahead=n)`); the form is chosen by the number
(`plural_count`): the parameter `count`, else the first placeholder of the `.other` template given a number. A message
is the same: "CODE.one" / "CODE.other" (and "CODE.short.one" / "CODE.short.other").

A word or a message may also say itself to whoever uses a card (the web page's app mode, a DCC panel: no nodes, no
wires, no node names), the key with `.app` after it — quoted in TOML (`"cook_hint.app" = "…"`, `"E-X-Y.app" = "…"`),
in both languages like any key. The page reads a word's `.app` form while it shows a card (webui/src/i18n/words.ts
phrasing); a message carries its `.app` form beside its text (lab2shot/messages Msg.app). A word's `.app` form names
only placeholders its plain form names (the caller gives those); a message's may name a parameter its plain form
leaves out (the code gives every parameter either form names).
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from functools import cache
from pathlib import Path
from typing import Any

from lab2shot_shared.protocol import CODE

LANGS: tuple[str, ...] = ("zh", "en")
DEFAULT = "zh"  # what a person who has said nothing gets (the design: Chinese first)
LOG_LANG = "en"  # the server's log: one language, the back end's
PLURAL_FORMS = ("one", "other")  # a key's plural forms (key.one / key.other), the CLDR names; Chinese writes only other
HEADER = "X-Lab2Shot-Lang"  # a client's (DCC, CLI) choice, first in the server's order
COOKIE = "lang"  # a browser's choice before (or without) a login
DIR = Path(__file__).resolve().parent
SHORT_SUFFIX = ".short"
APP_SUFFIX = ".app"  # a key's words for whoever uses a card (above)
PLACEHOLDER = re.compile(r"\{(\w+)(?::([^{}]*))?\}")
FORMATS = re.compile(r"^(d|,|g|\.\d+f|\.\d+%)$")

_current: ContextVar[str | None] = ContextVar("lab2shot_lang", default=None)
_process = DEFAULT


class TextError(ValueError):
    """A catalogue that does not read (a file that is not TOML, a key twice, a value that is not text), or a key that is
    in no catalogue: a programming error, said loudly."""


# ------------------------------------------------------------------ the language now


def normal(lang: object) -> str | None:
    """`lang` as one of LANGS, or None: zh, zh-CN, zh_CN.UTF-8, ZH-hans all read zh; en, en-US, en_GB read en."""
    if not isinstance(lang, str):
        return None
    head = re.split(r"[-_.@]", lang.strip().lower(), maxsplit=1)[0]
    return head if head in LANGS else None


def current() -> str:
    """The language of the work being done now: the request's, the stretch's (`using`), else the process's."""
    return _current.get() or _process


def set_process(lang: str) -> None:
    """The language of this process when nothing more particular says one (the command line's --lang)."""
    global _process
    _process = normal(lang) or DEFAULT


def set_current(lang: str | None) -> Token:
    """Set the language for the rest of this context (a request); returns the token `reset` takes."""
    return _current.set(normal(lang))


def reset(token: Token) -> None:
    _current.reset(token)


@contextmanager
def using(lang: str | None) -> Iterator[str]:
    """Do a stretch of work in `lang` (None, or one not in LANGS: the language now)."""
    token = _current.set(normal(lang) or current())
    try:
        yield current()
    finally:
        _current.reset(token)


def from_accept_language(header: str | None) -> str | None:
    """The first language of LANGS a browser's Accept-Language asks for, by its weights (None: none of them)."""
    best: list[tuple[float, int, str]] = []
    for i, part in enumerate((header or "").split(",")):
        tag, _, rest = part.strip().partition(";")
        q = 1.0
        m = re.search(r"q\s*=\s*([0-9.]+)", rest)
        if m:
            try:
                q = float(m.group(1))
            except ValueError:
                q = 0.0
        lang = normal(tag)
        if lang and q > 0:
            best.append((-q, i, lang))
    return min(best)[2] if best else None


# ------------------------------------------------------------------ the catalogues


def message_code(key: str) -> str | None:
    """The code a message key is about: CODE, CODE.short, CODE.app, and their plural forms CODE.one / CODE.other /
    CODE.short.one / CODE.short.other / CODE.app.one / CODE.app.other; None for a word's key."""
    head, dot, form = key.rpartition(".")
    base = head if dot and form in ("one", "other") else key
    for suffix in (SHORT_SUFFIX, APP_SUFFIX):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base if CODE.match(base) else None


def is_message_key(key: str) -> bool:
    """A message's key (its code, 「CODE.short」, 「CODE.app」, or a plural form of one) rather than a word's."""
    return message_code(key) is not None


def flatten(table: dict, where: str, prefix: str = "") -> dict[str, str]:
    """A TOML table as {dotted key: text}; a key that comes out twice, or a value that is not text, is an error."""
    out: dict[str, str] = {}
    for k, v in table.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            for kk, vv in flatten(v, where, key + ".").items():
                if kk in out:
                    raise TextError(f"{where}: {kk} twice")
                out[kk] = vv
        elif isinstance(v, str):
            if key in out:
                raise TextError(f"{where}: {key} twice")
            out[key] = v
        else:
            raise TextError(f"{where}: {key} is not text")
    return out


def read(path: Path) -> dict[str, str]:
    """One catalogue file, flattened (missing: {})."""
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise TextError(f"{path}: {exc}") from None
    return flatten(data, str(path))


def lang_dir(lang: str) -> Path:
    return DIR / lang


def message_files(lang: str) -> list[Path]:
    """The core's message files of one language (i18n/<lang>/messages/*.toml)."""
    return sorted((lang_dir(lang) / "messages").glob("*.toml"))


def word_files(lang: str) -> list[Path]:
    """The core's word files of one language: every .toml under i18n/<lang>/ outside messages/."""
    root = lang_dir(lang)
    return sorted(p for p in root.rglob("*.toml") if "messages" not in p.relative_to(root).parts[:1])


def extension_folders() -> dict[str, Path]:
    """Every adapter folder (adapters/<name>/ with an extension.py, lab2shot/adapters.py's rule), by name, read from
    the folder itself so the catalogue sits below everything that says a word (the layering)."""
    from ..config import ADAPTERS_DIR

    if not ADAPTERS_DIR.is_dir():
        return {}
    return {f.name: f for f in sorted(ADAPTERS_DIR.iterdir()) if (f / "extension.py").is_file()}


def extension_file(folder: Path, lang: str) -> Path:
    return folder / "i18n" / f"{lang}.toml"


@cache
def _core_words(lang: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in word_files(lang):
        got = read(path)
        for k in got:
            if is_message_key(k):
                raise TextError(f"{path}: {k} is a message: it goes in i18n/{lang}/messages/")
            if k in out:
                raise TextError(f"{path}: {k} is in another file of i18n/{lang}/ too")
        out.update(got)
    return out


@cache
def _extension_words(lang: str) -> dict[str, dict[str, str]]:
    """Each extension's own words (its messages are the messages module's), by extension."""
    return {name: {k: v for k, v in read(extension_file(folder, lang)).items() if not is_message_key(k)}
            for name, folder in extension_folders().items()}


@cache
def words(lang: str) -> dict[str, str]:
    """Every word of one language as written (no fallback): the core's and every extension's, the shared keys an
    extension defines for itself left out (they are `scope`d)."""
    out = dict(_core_words(lang))
    for name, own in _extension_words(lang).items():
        for k, v in own.items():
            if _shared(k):
                continue
            if k in out:
                raise TextError(f"adapters/{name}/i18n/{lang}.toml: {k} is defined elsewhere too")
            out[k] = v
    return out


def scoped(lang: str, scope: str) -> dict[str, str]:
    """An extension's own shared keys (param.*, port.*, option.*)."""
    return {k: v for k, v in _extension_words(lang).get(scope, {}).items() if _shared(k)}


SHARED = ("param.", "port.", "option.", "stage.", "button.", "fact.")


def _shared(key: str) -> bool:
    return key.startswith(SHARED)


def clear() -> None:
    """Forget what was read (a test, or a catalogue rewritten while running)."""
    for f in (_core_words, _extension_words, words):
        f.cache_clear()


# ------------------------------------------------------------------ looking a word up


# the segments a node key goes on with after its type name
NODE_PARTS = ("param", "port", "subtitle", "description", "stage", "button", "fact", "official", "cost", "licence", "needs",
              "row", "no_file")


def node_key_parts(key: str) -> tuple[str, list[str]] | None:
    """node.<type>.<rest…> as (type, rest), the type holding dots: it ends before the first segment of NODE_PARTS
    (`param`, `port`, `subtitle`, `description`, …; no type name has one of them as a segment: lab2shot check i18n)."""
    if not key.startswith("node."):
        return None
    seg = key.split(".")
    for i in range(2, len(seg)):
        if seg[i] in NODE_PARTS:
            return ".".join(seg[1:i]), seg[i:]
    return None


def fallbacks(key: str) -> list[str]:
    """The keys a key falls back to, in order (itself first): a node's parameter, option and port to the shared ones."""
    out = [key]
    parts = node_key_parts(key)
    if parts is not None:
        _, rest = parts
        if rest[0] == "param" and len(rest) >= 3:
            out.append(".".join(rest))  # param.<p>.<…>
            if len(rest) >= 4 and rest[2] == "option":
                out.append("option." + ".".join(rest[3:]))
        elif rest[0] in ("port", "stage", "button", "fact") and len(rest) >= 2:
            out.append(".".join(rest))  # port.<name>.<…>, stage.<id>, button.<id>, fact.<id>
    elif key.startswith("param.") and ".option." in key:
        out.append("option." + key.split(".option.", 1)[1])
    return out


def chain(lang: str | None = None) -> list[str]:
    """The languages a lookup tries: the one asked for (every key is in both: lab2shot check i18n)."""
    return [normal(lang) or current()]


def lookup(key: str, lang: str | None = None, scope: str | None = None, exact: bool = False) -> str | None:
    """The word for `key` (its template, placeholders unfilled), with its fallbacks (`exact`: without); None when there
    is none."""
    for one in chain(lang):
        table = words(one)
        own = scoped(one, scope) if scope else {}
        for k in ([key] if exact else fallbacks(key)):
            if k in own:
                return own[k]
            if k in table:
                return table[k]
    return None


def has(key: str, lang: str | None = None, scope: str | None = None) -> bool:
    return lookup(key, lang, scope) is not None


def fill(template: str, params: dict[str, Any]) -> str:
    """A word's template with its parameters in ({name} or {name:spec}, the messages' formats)."""
    def one(m: re.Match) -> str:
        value = params[m.group(1)]
        spec = m.group(2)
        return format(value, spec) if spec and isinstance(value, (int, float)) else str(value)

    missing = {m.group(1) for m in PLACEHOLDER.finditer(template)} - set(params)
    if missing:
        raise TextError(f"{template!r}: parameters {sorted(missing)} missing")
    return PLACEHOLDER.sub(one, template)


def plural_form(lang: str, n: Any) -> str:
    """Which plural form a number takes in a language (PLURAL_FORMS): English one for exactly 1, else other; Chinese
    always other."""
    return "one" if lang == "en" and as_number(n) == 1 else "other"


NUMBER_TEXT = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def as_number(value: Any) -> float | None:
    """A parameter as a number, when it is one: an int or float, or a number already written out ("1", "1,234")."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and NUMBER_TEXT.match(value.strip()):
        return float(value.strip().replace(",", ""))
    return None


def plural_count(template: str, params: Mapping[str, Any]) -> Any:
    """The number a plural sentence follows: the parameter `count`, else the first placeholder of its `.other`
    template given a number (a number written out as text counts too: "1,234"); None: none."""
    if "count" in params:
        return as_number(params["count"])
    for m in PLACEHOLDER.finditer(template):
        n = as_number(params.get(m.group(1)))
        if n is not None:
            return n
    return None


def plural_base(key: str) -> str | None:
    """The key a plural form belongs to (ui.queue.jobs.one -> ui.queue.jobs), None when `key` is no plural form."""
    head, dot, form = key.rpartition(".")
    return head if dot and form in PLURAL_FORMS and head else None


def template_for(key: str, params: Mapping[str, Any] | None = None, lang: str | None = None,
                 scope: str | None = None, exact: bool = False) -> str | None:
    """The template `key` reads with these parameters: its own word, else (a plural sentence) the form its number
    takes (plural_form); None when there is none."""
    found = lookup(key, lang, scope, exact)
    if found is not None:
        return found
    other = lookup(f"{key}.other", lang, scope, exact)
    if other is None:
        return None
    if plural_form(normal(lang) or current(), plural_count(other, params or {})) == "one":
        return lookup(f"{key}.one", lang, scope, exact) or other
    return other


RESERVED_PARAMS = ("in_lang", "in_scope")  # t()'s / node_text()'s own keywords: no placeholder may be named so (i18n lint)


def t(key: str, /, *, in_lang: str | None = None, in_scope: str | None = None, **params: Any) -> str:
    """The word for `key` in the language now (or `in_lang`), its parameters filled in (a plural sentence in the form
    its number takes). A key in no catalogue is a programming error (lab2shot check i18n finds it before it runs).
    The language and scope keywords are named so no placeholder is (RESERVED_PARAMS): `{lang}` is a parameter."""
    template = template_for(key, params, in_lang, in_scope)
    if template is None:
        raise TextError(f"{key} is in no catalogue (lab2shot/i18n/<lang>/, adapters/<name>/i18n/<lang>.toml)")
    return fill(template, params) if params or "{" in template else template


def node_text(type_id: str, *path: str, in_scope: str | None = None, in_lang: str | None = None,
              **params: Any) -> str | None:
    """A node type's word: node_text("fbx.import", "subtitle"), node_text(t, "param", "scale", "label"),
    node_text(t, "param", "mode", "option", "auto"), node_text(t, "port", "camera"), with the shared fallbacks; None
    when there is none (the caller decides: a node's name falls back to its type name)."""
    found = template_for(".".join(("node", type_id, *path)), params, in_lang, in_scope)
    return None if found is None else (fill(found, params) if params else found)


def separator(lang: str | None = None) -> str:
    """How a list is joined in a sentence (、 in Chinese, ", " in English)."""
    return lookup("list.sep", lang) or "、"


class Both(str):
    """A word a message is said with (a node's parameter or input name, an option's, a data type's) kept in every
    language: it reads in the language now wherever it is said (a str in the language it was made in otherwise), and
    travels as {"word": {lang: text}} so `localized` says the message again with it in another language."""

    texts: dict[str, str]

    def __new__(cls, texts: dict[str, str]):
        self = super().__new__(cls, texts.get(current()) or next(iter(texts.values()), ""))
        self.texts = dict(texts)
        return self

    def __str__(self) -> str:
        return self.texts.get(current()) or str.__str__(self)

    def __reduce__(self):  # pickled and copied with its words in every language (a str subclass made from a dict)
        return (Both, (dict(self.texts),))

    @classmethod
    def of(cls, make: Callable[[], Any]) -> "Both":
        """`make()` said in each language (LANGS)."""
        texts = {}
        for lang in LANGS:
            with using(lang):
                texts[lang] = str(make())
        return cls(texts)


class Word(str):
    """A word of the catalogue said when it is shown, not when it is made: `key` (and its parameters) kept, the words
    looked up in the language now each time it is said (str()). The one form of a word a message is said with whose
    key is known (a node pointed at: engine/naming.py node_ref; a parameter's, a setting's, a role's name; a fixed place
    such as 未分类): a message kept to be read later (an audit row, a log line, a cached result) reads in its reader's
    language. It travels as {"said": key, "params": {...}} (messages.wire) and is made again from that when read
    (messages.worker_params). Both is only for words whose key is not known here (a worker's own text).

    It is a str (its words in the language it was made in) so it stands wherever text did: joined, measured, dumped to
    JSON; said again (str(), format) it is in the language now."""

    key: str
    params: dict[str, Any]
    scope: str | None  # the extension whose catalogue has it (t's in_scope): a node type's own words (nodes/text.py)

    def __new__(cls, key: str, /, *, in_scope: str | None = None, **params: Any):
        self = super().__new__(cls, _said(key, params, in_scope))
        self.key = key
        self.params = params
        self.scope = in_scope
        return self

    def __str__(self) -> str:
        return _said(self.key, self.params, self.scope)

    def __format__(self, spec: str) -> str:
        return format(str(self), spec)

    def __repr__(self) -> str:
        return f"Word({self.key!r})"

    def __reduce__(self):  # pickled and copied as its key and parameters (a str subclass made from them)
        return (_word, (self.key, dict(self.params), self.scope))

    def __len__(self) -> int:  # a word stands where text did: what measured or cut the text measures its words now
        return len(str(self))

    def __getitem__(self, index):
        return str(self)[index]


def _said(key: str, params: Mapping[str, Any], scope: str | None = None) -> str:
    return t(key, in_scope=scope, **{k: str(v) if isinstance(v, str) else v for k, v in params.items()})


def _word(key: str, params: dict[str, Any], scope: str | None = None) -> Word:
    return Word(key, in_scope=scope, **params)


# ------------------------------------------------------------------ text with both languages in it


def pick(value: Any, lang: str | None = None) -> str:
    """A display field that may hold both languages (a built-in template's `{"zh": …, "en": …}`): a string as it is
    (a user's own words, one language), an object as its text in the language now, else in the other one; anything
    else as "" (None) or its str."""
    if isinstance(value, dict):
        for one in [normal(lang) or current(), *LANGS]:
            got = value.get(one)
            if isinstance(got, str) and got:
                return got
        return ""
    return "" if value is None else value if isinstance(value, str) else str(value)


def is_both(value: Any) -> bool:
    """A display field written in both languages ({"zh": text, "en": text}, both non-empty)."""
    return isinstance(value, dict) and set(value) == set(LANGS) and all(isinstance(v, str) and v.strip() for v in value.values())


class Words(Mapping):
    """A table of ids whose words are looked up in the language now, `prefix` + id (a module's {id: word} table kept
    as ids only): Words("unit.kind.", ("length", …))["length"] is unit.kind.length's word."""

    def __init__(self, prefix: str, ids) -> None:
        self.prefix, self.ids = prefix, tuple(ids)

    def __getitem__(self, key: str) -> str:
        if key not in self.ids:
            raise KeyError(key)
        return t(self.prefix + key)

    def __iter__(self):
        return iter(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __hash__(self) -> int:
        return hash((self.prefix, self.ids))

    def __eq__(self, other) -> bool:
        return isinstance(other, Words) and (self.prefix, self.ids) == (other.prefix, other.ids)
