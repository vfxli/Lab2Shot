"""What `lab2shot check i18n` holds the code and the catalogues to (lab2shot/i18n).

Rules, each a problem when broken:
    keys          both languages have the same keys (words and messages); a plural sentence is the exception: English
                  writes <key>.one and <key>.other, Chinese only <key>.other (lab2shot/i18n PLURAL_FORMS)
    placeholders  a key written in both languages names the same {placeholders} in both (an English .one may leave
                  out what its .other names: "One frame"); no placeholder is named like t()'s own keywords
                  (RESERVED_PARAMS: in_lang, in_scope), which would take its value instead of filling it
    plurals       an English sentence where a number is followed by a plural noun ("{count} frames": "1 frames") is
                  written in plural forms; a key is written plain or in plural forms, never both
    unused        a word under ui.*, cli.*, dcc.* is used by some code (its key written out whole: a key is never
                  put together from pieces, except the families derived from data, DERIVED); its `.app` form (said to
                  whoever uses a card, lab2shot/i18n APP_SUFFIX) is used with it
    app           a word's `.app` form names no placeholder its plain form does not (the caller gives only those)

Counts, each held to lab2shot/i18n/baseline.toml (it may only go down; tools/i18n_baseline.py lowers it), by area — the
P1 blocks that move the words (AREAS):
    cjk           Chinese written in code: string literals and JSX text holding CJK, comments and docstrings left
                  out (a typer command's docstring is its --help: counted), by line
    missing       keys the Chinese catalogue has and the English one has not (also a problem of `keys`)
    fields        display fields of the built-in templates written in one language only (not {"zh": …, "en": …})

Every area is done (P1): its counts are 0 and stay there."""

from __future__ import annotations

import ast
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import APP_SUFFIX, DIR, LANGS, DEFAULT, PLACEHOLDER, RESERVED_PARAMS, extension_file, extension_folders, is_message_key, plural_base, read, word_files

CJK = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")
BASELINE = DIR / "baseline.toml"

# Who moves what (设计_国际化底座调查.md 3.1): the first prefix that matches a file's path (from the repository root)
# names its area. B1/B2: the extensions by the first letter of their folder.
AREAS: tuple[tuple[str, str], ...] = (
    ("webui/src/admin/", "D"), ("webui/src/", "C"),
    ("lab2shot/nodes/", "A"), ("lab2shot/formats/", "A"), ("lab2shot/ops/", "A"), ("lab2shot/data/", "A"),
    ("lab2shot/cli/", "E"), ("lab2shot/client.py", "E"), ("clients/", "E"), ("tools/", "E"), ("setup.sh", "E"),
    ("lab2shot/installer/", "E"), ("lab2shot/extensions/", "E"),
    ("worker_sdk/", "B1"),
    ("lab2shot/", "D"),
)
# the catalogue files' areas (whose words they hold); an extension's by its folder like its code
CATALOGUE_AREAS = {
    "messages/nodes.toml": "A", "messages/data.toml": "A", "messages/lens.toml": "A", "messages/nuke.toml": "A",
    "messages/web.toml": "C", "messages/client.toml": "E", "messages/sdk.toml": "E", "messages/tools.toml": "E",
    "cli.toml": "E", "dcc.toml": "E", "extensions.toml": "E", "ui/admin.toml": "D", "settings.toml": "D", "roles.toml": "D",
    "categories.toml": "D", "server.toml": "D", "engine.toml": "D", "nodes.toml": "A", "types.toml": "A", "units.toml": "A",
}
CATALOGUE_DEFAULT = {"messages/": "D", "ui/": "C"}

# Not code that shows words, or words that are data (each with its reason)
SKIP_PARTS = {"node_modules", "__pycache__", "third_party", "work", "dist", "tests", ".venv", "generated"}
SKIP_PREFIXES = ("lab2shot/i18n/",)  # the catalogues' own code (its Chinese is in comments and examples)
SKIP_FILES = {
    "lab2shot/data/skeleton_recognition.py": "skeleton recognition: Chinese joint names are what it recognises (data)",
    "lab2shot/data/joints.py": "joint names in Chinese rigs are recognised as written (data)",
}
GENERATED_BLOCKS = {"lab2shot/client.py": ("# ---- messages: generated", "# ---- end of the generated messages")}
# words whose keys are made from data, never written out whole in code: checked against what makes them (P1)
DERIVED = ("node.", "param.", "port.", "option.", "group.", "type.", "unit.", "category.", "setting.", "settings_page.",
           "right.", "route.", "lang.", "list.")
CHECKED_USE = ("ui.", "cli.", "dcc.")


def area_of(rel: str) -> str:
    if rel.startswith("adapters/"):
        name = rel.split("/")[1]
        return "B1" if name[:1].lower() <= "m" else "B2"
    for prefix, area in AREAS:
        if rel.startswith(prefix):
            return area
    return "other"


def catalogue_area(path: Path) -> str:
    """The area of a catalogue file (its words), by where it is."""
    parts = path.parts
    if "adapters" in parts:
        name = parts[parts.index("adapters") + 1]
        return "B1" if name[:1].lower() <= "m" else "B2"
    rel = path.relative_to(DIR / path.relative_to(DIR).parts[0]).as_posix()
    if rel in CATALOGUE_AREAS:
        return CATALOGUE_AREAS[rel]
    return next((a for p, a in CATALOGUE_DEFAULT.items() if rel.startswith(p)), "root")


# ------------------------------------------------------------------ Chinese in code


def _skipped(rel: str) -> bool:
    parts = rel.split("/")
    name = parts[-1]
    if rel in SKIP_FILES or SKIP_PARTS & set(parts[:-1]) or rel.startswith(SKIP_PREFIXES):
        return True
    return name.startswith("test_") or ".test." in name or ".fixture." in name or name.startswith("generated")


def code_files(root: Path) -> list[Path]:
    out = []
    for folder in ("lab2shot", "adapters", "worker_sdk", "clients", "tools"):
        out += [p for p in (root / folder).rglob("*.py")]
    out += [p for p in (root / "webui" / "src").rglob("*.ts")] + [p for p in (root / "webui" / "src").rglob("*.tsx")]
    out += list(root.glob("*.sh")) + [p for p in (root / "tools").rglob("*.sh")]
    return sorted(p for p in out if not _skipped(p.relative_to(root).as_posix()))


def _command(fn: ast.AST) -> bool:
    """A typer command or callback (its docstring is its --help)."""
    for d in getattr(fn, "decorator_list", ()):
        target = d.func if isinstance(d, ast.Call) else d
        if isinstance(target, ast.Attribute) and target.attr in ("command", "callback"):
            return True
    return False


def python_lines(text: str) -> set[int]:
    """The lines of Python source with Chinese in a string literal, docstrings left out (a command's counted)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    docs: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                if not _command(node):
                    docs.add(id(first.value))
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs and CJK.search(node.value):
            out.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return out


def strip_ts_comments(text: str) -> str:
    """TypeScript source with its comments blanked (strings, template literals and JSX text kept): a scanner that
    knows quotes, template literals with ${…} in them, and comments; a regular expression literal is read as code."""
    out = []
    i, n = 0, len(text)
    stack: list[str] = []  # "`" inside a template, "{" a ${ … } inside one
    while i < n:
        c = text[i]
        top = stack[-1] if stack else ""
        if top == "`":
            if c == "\\":
                out.append(text[i:i + 2]); i += 2; continue
            if c == "`":
                stack.pop(); out.append(c); i += 1; continue
            if text.startswith("${", i):
                stack.append("{"); out.append("${"); i += 2; continue
            out.append(c); i += 1; continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if ch == "\n" else " " for ch in text[i:j]))
            i = j
            continue
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1]); i = j + 1; continue
        if c == "`":
            stack.append("`"); out.append(c); i += 1; continue
        if c == "{" and top == "{":
            stack.append("{")
        elif c == "}" and top == "{":
            stack.pop()
        out.append(c); i += 1
    return "".join(out)


# Chinese words written as backslash-u escapes in TypeScript source: a run of escapes (not one escaped itself, as in
# "\\u4E00" in a regular expression's text) that reads as two Chinese characters or more. A single escaped character
# or a range ("[\u3400-\u9FFF]", punctuation in a splitter) is data, not words
_TS_ESCAPES = re.compile(r"(?:(?<!\\)\\u(?:[0-9a-fA-F]{4}|\{[0-9a-fA-F]{1,6}\}))+")
_HAN_WORD = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")


def _escaped_chinese(line: str) -> bool:
    """Whether the line writes Chinese words as escapes (_TS_ESCAPES): what the plain CJK search cannot see."""
    def read(run: str) -> str:
        return re.sub(r"\\u\{?([0-9a-fA-F]+)\}?", lambda m: chr(int(m.group(1), 16)), run)
    def ranged(m: re.Match) -> bool:  # an end of a character range ("\\u4DBF\\u4E00" inside [\\u3400-…-\\u9FFF])
        return line[m.start() - 1:m.start()] == "-" or line[m.end():m.end() + 1] == "-"
    return any(_HAN_WORD.search(read(m.group(0))) and not ranged(m) for m in _TS_ESCAPES.finditer(line))


def ts_lines(text: str) -> set[int]:
    return {i for i, line in enumerate(strip_ts_comments(text).splitlines(), 1) if CJK.search(line) or _escaped_chinese(line)}


# English written into the page's markup where words are shown (a JSX text, or a label / title / placeholder /
# aria-label / alt given as a plain string): two English words or more. A <code> element's text is code, not words
_TSX_ATTR = re.compile(r"""\b(?:label|title|placeholder|aria-label|alt)=(?:"([^"]*)"|\{\s*"([^"]*)"\s*\}|\{\s*'([^']*)'\s*\})""")
_TSX_TEXT = re.compile(r"<([A-Za-z][\w.]*)\b[^<>]*>\s*([^<>{}=;()]*?[A-Za-z]{2,}[ \t]+[A-Za-z]{2,}[^<>{}=;()]*?)\s*</")
_WORDS = re.compile(r"[A-Za-z]{2,}[\s-]+[A-Za-z]{2,}")


def english_problems(root: Path) -> list[str]:
    """Display words in English written in the page's markup (the rule against written words covers both languages:
    the CJK count catches Chinese, this the English)."""
    from . import t

    out = []
    for p in sorted((root / "webui" / "src").rglob("*.tsx")):
        rel = p.relative_to(root).as_posix()
        if "node_modules" in rel or ".test." in p.name:
            continue
        text = strip_ts_comments(p.read_text(encoding="utf-8", errors="replace"))
        found = [(m.start(), next(g for g in m.groups() if g is not None)) for m in _TSX_ATTR.finditer(text)]
        found += [(m.start(2), m.group(2)) for m in _TSX_TEXT.finditer(text) if m.group(1) != "code"]
        for at, words in sorted(found):
            if _WORDS.search(words):
                out.append(t("cli.check.i18n.english", text=words.strip(), where=f"{rel}:{text.count(chr(10), 0, at) + 1}"))
    return out


_SAID = re.compile(r'\b(?:say|ask\s+\w+)\s+"[^"]*"\s+"[^"]*"')
BOTH_LANGUAGES = "# both languages"  # a shell line that shows the two languages side by side (the language picker)


def shell_lines(text: str) -> set[int]:
    """Lines of a shell script with Chinese a user would see in one language only. A shell script runs before Python,
    so it cannot read the catalogues: its words go through `say "<zh>" "<en>"` / `ask <var> "<zh>" "<en>"` (setup.sh),
    whose statements (with their backslash continuations) are allowed; comments are not words; a line ending in
    BOTH_LANGUAGES shows both at once."""
    out: set[int] = set()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        start = i
        while lines[i].rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
        statement = [(n + 1, lines[n]) for n in range(start, i + 1)]
        head = statement[0][1].strip()
        if not head.startswith(("say ", "ask ")):
            for n, line in statement:
                code = line if line.rstrip().endswith(BOTH_LANGUAGES) else line.split(" #", 1)[0]
                if line.lstrip().startswith("#") or line.rstrip().endswith(BOTH_LANGUAGES):
                    continue
                if CJK.search(_SAID.sub("", code)):  # a say/ask in the middle of a line (a case branch) is fine too
                    out.add(n)
        i += 1
    return out


def _generated(rel: str, text: str) -> set[int]:
    begin_end = GENERATED_BLOCKS.get(rel)
    if not begin_end:
        return set()
    lines = text.splitlines()
    try:
        a = next(i for i, l in enumerate(lines, 1) if l.startswith(begin_end[0]))
        b = next(i for i, l in enumerate(lines, 1) if l.startswith(begin_end[1]))
    except StopIteration:
        return set()
    return set(range(a, b + 1))


def cjk_counts(root: Path) -> dict[str, int]:
    """Lines with Chinese in code, by area."""
    out: dict[str, int] = {}
    for path in code_files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = (python_lines(text) if path.suffix == ".py" else shell_lines(text) if path.suffix == ".sh"
                 else ts_lines(text))
        lines -= _generated(rel, text)
        if lines:
            area = area_of(rel)
            out[area] = out.get(area, 0) + len(lines)
    return out


# ------------------------------------------------------------------ the catalogues side by side


@dataclass
class Catalogues:
    """Every key of every catalogue file, per language: key -> (text, file); and each file's own keys (`files`: an
    extension's shared keys, stage.* and the like, are in several files under one key)."""

    keys: dict[str, dict[str, tuple[str, Path]]] = field(default_factory=dict)
    files: dict[str, dict[Path, dict[str, str]]] = field(default_factory=dict)

    @classmethod
    def read(cls) -> Catalogues:
        from ..messages import _read, extension_section

        out = cls({lang: {} for lang in LANGS}, {lang: {} for lang in LANGS})
        for lang in LANGS:
            def add(path: Path, got: dict[str, str]) -> None:
                out.keys[lang].update({k: (v, path) for k, v in got.items()})
                out.files[lang][path] = got

            for path in sorted((DIR / lang / "messages").glob("*.toml")):
                add(path, _read(path))
            for path in word_files(lang):
                add(path, read(path))
            for name, folder in extension_folders().items():
                add(extension_file(folder, lang), read(extension_file(folder, lang)))
                extension_section(name, folder, lang)  # its messages read as messages (a malformed one raises)
        return out


def placeholders(text: str) -> list[str]:
    return sorted(m.group(1) for m in PLACEHOLDER.finditer(text))


def reserved_problems(c: "Catalogues") -> list[str]:
    """Keys naming a placeholder t() takes as its own keyword (RESERVED_PARAMS): `t(key, in_lang=x)` would pick the
    language, not fill the placeholder."""
    from . import t
    out = []
    for lang in LANGS:
        for key, (text, path) in sorted(c.keys[lang].items()):
            for name in sorted(set(placeholders(text)) & set(RESERVED_PARAMS)):
                out.append(t("cli.check.i18n.reserved", key=key, name=name, language=lang))
    return out


# a message's words said once, in the language of the moment, then carried as plain text: a message read later in
# another language (a cached result, the log, a DCC plugin) would mix languages. What names a thing the catalogue has
# words for goes as i18n.Word (or a label that is one: a parameter's, a setting's, GNode.label); a node pointed at is
# its node_ref (name and type), never its subtitle (a type to insert, `via`, is not a node)
EAGER_CALLS = frozenset({"t", "pick", "node_text", "lookup", "subtitle", "description"})
SAYING = frozenset({"Msg", "say"})


def eager_word_problems(root: Path) -> list[str]:
    """Msg(...) / ctx.say(...) keyword arguments that are words said now (a call of t / pick / node_text / lookup) or
    a node named by its subtitle (`node=cls.subtitle`)."""
    from . import t

    out = []
    for path in code_files(root):
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fn = n.func
            if (fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else "") not in SAYING:
                continue
            for k in n.keywords:
                v = k.value
                call = v.func if isinstance(v, ast.Call) else None
                name = (call.id if isinstance(call, ast.Name) else call.attr if isinstance(call, ast.Attribute) else "") if call else ""
                if name in EAGER_CALLS or (k.arg == "node" and isinstance(v, ast.Attribute) and v.attr == "subtitle"):
                    out.append(t("cli.check.i18n.eager", where=f"{_rel(path, root)}:{n.lineno}", name=k.arg or "**"))
    return out


# ------------------------------------------------------------------ plural sentences

# "{count} frames": a number followed by a plural noun reads wrong for 1 ("1 frames"): the sentence is written in plural
# forms. A placeholder is taken for a number unless its format says a fraction (.1f, g, %: "1.0 seconds" is English) or
# its name holds text (TEXTUAL); the word after it is a plural noun when it ends in s and is no verb (VERBS) or unit.
FRACTION = re.compile(r"^(g|\.\d+[f%])$")
TEXTUAL = frozenset("""
name names node nodes path file title type kind model project where what label id value source side part joint input
output port username user role section module extension key text rig host code color config area archive channel chunk
compiler deliverable fact folder gpu gpus job lang layer line member number operation param plugin preset repo requested
root segment sid slot subject target tool unit version which who why written given generate idle index inbox height
mask_height job_height hi listen y last rank confidence expected ext pid person frame mode reason detail error message
""".split())
VERBS = frozenset("""
is was has does needs writes takes uses cooks gives belongs applies differs supports declares reads exposes opens calls
references changes expands records breaks shows matches holds restates repeats says lacks becomes sends spans fails
accepts wants goes drives comes tracks selects crosses shares extends falls means brings as gets this its leaves leads
""".split())
PLURAL_WORD = re.compile(r"^(?:[A-Za-z]{2,}[^sSuU\W][sS]|people|children)$")  # not -ss, -us (class, corpus)
SINGULAR_S = frozenset("lens series species analysis".split())
RANGE_END = re.compile(r"\}(?: to |\s*[–-]\s*)$")


def count_phrases(text: str) -> list[str]:
    """The "{number} nouns" pieces of an English template that read wrong for 1 ("{count} frames")."""
    out = []
    for m in PLACEHOLDER.finditer(text):
        name, spec = m.group(1), m.group(2)
        if (spec and FRACTION.match(spec)) or name in TEXTUAL:
            continue
        if RANGE_END.search(text[:m.start()]):  # "{min} to {max} letters": the end of a range is plural whatever it is
            continue
        after = re.match(r" ([A-Za-z]+)", text[m.end():])
        word = after.group(1).lower() if after else ""
        if after and word not in VERBS and word not in SINGULAR_S and PLURAL_WORD.match(after.group(1)):
            out.append(text[m.start():m.end() + after.end()])
    return out


def plural_problems(c: Catalogues, root: Path) -> list[str]:
    """The plural forms' rules (module docstring: keys, placeholders, plurals), file by file."""
    out: list[str] = []
    for lang, files in c.files.items():
        for path, keys in files.items():
            zh = c.files[DEFAULT].get(_other_lang(path, DEFAULT), {})
            en = c.files["en"].get(_other_lang(path, "en"), {})
            out += _file_plurals(lang, path, keys, zh, en, root)
    return list(dict.fromkeys(out))


def _other_lang(path: Path, lang: str) -> Path:
    """The same catalogue file in another language (i18n/<lang>/…, adapters/<name>/i18n/<lang>.toml)."""
    if path.parent.name == "i18n":
        return path.with_name(f"{lang}.toml")
    rel = path.relative_to(DIR)
    return DIR / lang / Path(*rel.parts[1:])


def _file_plurals(lang: str, path: Path, keys: dict[str, str], zh: dict[str, str], en: dict[str, str], root: Path) -> list[str]:
    from . import t

    out: list[str] = []
    where = _rel(path, root)
    for key, text in sorted(keys.items()):
        base = plural_of(key, keys)
        if base is None:
            if any(k in keys for k in (f"{key}.one", f"{key}.other")):
                out.append(t("cli.check.i18n.plural_twice", key=key, language=lang, where=where))
            if lang == "en":
                for said in count_phrases(text):
                    out.append(t("cli.check.i18n.plural_needed", key=key, language=lang, said=said, where=where))
            continue
        if lang == DEFAULT and key.endswith(".one"):
            out.append(t("cli.check.i18n.plural_zh_one", key=base, where=where))
        elif lang == "en" and not ({f"{base}.one", f"{base}.other"} <= set(en) and f"{base}.other" in zh):
            out.append(t("cli.check.i18n.plural_pair", key=base, language=lang, where=where))
        if lang == "en" and key.endswith(".one") and f"{base}.other" in en \
                and not set(placeholders(text)) <= set(placeholders(en[f"{base}.other"])):
            out.append(t("cli.check.i18n.plural_one_names", key=base, where=where))
    return out


def plural_of(key: str, keys) -> str | None:
    """The sentence a plural form belongs to (key.one / key.other -> key), None for any other key: a table of ids
    may have one named "other" (cli.setup.claim.other beside .missing, .ours), and it is no plural form."""
    base = plural_base(key)
    if base is None:
        return None
    prefix = base + "."
    # its short and app words beside it (「<CODE>.short」, 「<CODE>.app」 and their own plural forms) are no other forms
    siblings = {k[len(prefix):] for k in keys if k.startswith(prefix)} - {"short", "app", "short.one", "short.other", "app.one", "app.other"}
    return base if siblings <= {"one", "other"} else None


def _english_only(key: str, zh: dict) -> bool:
    """An English key the Chinese catalogue rightly lacks: a plural sentence's .one (Chinese has only .other)."""
    base = plural_base(key)
    return base is not None and key.endswith(".one") and f"{base}.other" in zh


def missing_counts(c: Catalogues) -> dict[str, int]:
    out: dict[str, int] = {}
    for lang in LANGS:
        if lang == DEFAULT:
            continue
        for key, (_, path) in c.keys[DEFAULT].items():
            if key not in c.keys[lang]:
                area = catalogue_area(path)
                out[area] = out.get(area, 0) + 1
    return out


def used_keys(root: Path) -> str:
    """Every code file's text in one (a key is used when it is written out whole in one of them)."""
    texts = []
    for folder, globs in (("lab2shot", ("*.py",)), ("adapters", ("*.py",)), ("clients", ("*.py",)), ("tools", ("*.py",)),
                          ("webui/src", ("*.ts", "*.tsx"))):
        for g in globs:
            for p in (root / folder).rglob(g):
                rel = p.relative_to(root).as_posix()
                if "node_modules" in rel or "/i18n/zh/" in rel or "/i18n/en/" in rel or "generated" in p.name:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
                texts.append(strip_ts_comments(text) if p.suffix in (".ts", ".tsx") else text)
    return "\n".join(texts)


# a word key written out whole where it is said (t / tr / Word / i18n.t / i18n.Word with a literal key): it must be in
# the catalogue, or the call throws where it is shown (the page: t.ts; Python: TextError) instead of being caught here
SAID_KEY = re.compile(r"""(?<![\w.])(?:i18n\.)?(?:t|tr|Word)\(\s*(?:"([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)"|'([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)')""")


def unknown_key_problems(c: Catalogues, root: Path) -> list[str]:
    """Keys said in code by a literal (SAID_KEY) that no catalogue has (in the default language; a plural sentence is
    said by its key without the form)."""
    from . import t

    zh = c.keys[DEFAULT]
    out = []
    for folder, globs in (("lab2shot", ("*.py",)), ("adapters", ("*.py",)), ("clients", ("*.py",)), ("tools", ("*.py",)),
                          ("webui/src", ("*.ts", "*.tsx"))):
        for g in globs:
            for p in sorted((root / folder).rglob(g)):
                rel = p.relative_to(root).as_posix()
                if "node_modules" in rel or "/i18n/zh/" in rel or "/i18n/en/" in rel or "generated" in p.name or ".test." in p.name:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace")
                text = strip_ts_comments(text) if p.suffix in (".ts", ".tsx") else text
                for m in SAID_KEY.finditer(text):
                    key = m.group(1) or m.group(2)
                    if key not in zh and f"{key}.other" not in zh:
                        line = text.count("\n", 0, m.start()) + 1
                        out.append(t("cli.check.i18n.unknown", key=key, where=f"{rel}:{line}"))
    return out


# ------------------------------------------------------------------ templates


TEMPLATE_FIELDS = ("label", "note")


def template_fields(root: Path) -> int:
    """Display fields of the built-in templates (templates/, adapters/<name>/templates/: the presets) not written in
    both languages: meta.name, meta.intro, every label / note of the parameter interface (groups, parameters,
    options) and the group boxes' labels. A plain string, or a {zh, en} missing one or with one empty, counts; a
    user's own templates (work/) are not looked at: what they write is theirs, in whatever language (read with pick,
    either form)."""
    from . import is_both

    count = 0

    def one(v) -> None:
        nonlocal count
        if (isinstance(v, str) and v.strip()) or (isinstance(v, dict) and not is_both(v)):
            count += 1

    def walk(entries) -> None:
        for x in entries if isinstance(entries, list) else ():
            if not isinstance(x, dict):
                continue
            for k in TEMPLATE_FIELDS:
                one(x.get(k))
            for o in x.get("options") or () if isinstance(x.get("options"), list) else ():
                if isinstance(o, dict):
                    one(o.get("label"))
            walk(x.get("children"))

    files = sorted((root / "templates").glob("*.json")) + sorted((root / "adapters").glob("*/templates/*.json"))
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        meta = data.get("meta") or {}
        one(meta.get("name"))
        one(meta.get("intro"))
        walk(meta.get("exposed") or data.get("exposed"))
        for b in data.get("boxes") or ():  # a group box's name (the page shows it in its language: graph/naming.ts boxLabel)
            if isinstance(b, dict):
                one(b.get("label"))
    return count


# ------------------------------------------------------------------ the baseline


def baseline() -> dict[str, dict[str, int]]:
    if not BASELINE.is_file():
        return {}
    return {k: {a: int(n) for a, n in v.items()} for k, v in tomllib.loads(BASELINE.read_text(encoding="utf-8")).items()}


def counts(root: Path, c: Catalogues | None = None) -> dict[str, dict[str, int]]:
    return {"cjk": cjk_counts(root), "missing": missing_counts(c or Catalogues.read()), "fields": {"F": template_fields(root)}}


def write_baseline(now: dict[str, dict[str, int]], old: dict[str, dict[str, int]] | None) -> str:
    """The baseline file's text: `now`, never above `old` where old has a number (it only goes down)."""
    lines = ["# lab2shot check i18n: what is still to move, by area (lab2shot/i18n/lint.py AREAS). Only ever lowered:",
             "# uv run python tools/i18n_baseline.py writes what is there now where that is less. 0 = that area is done.", ""]
    for kind in ("cjk", "missing", "fields"):
        lines.append(f"[{kind}]")
        areas = sorted(set(now.get(kind, {})) | set((old or {}).get(kind, {})))
        for a in areas:
            n = now.get(kind, {}).get(a, 0)
            if old is not None and a in old.get(kind, {}):
                n = min(n, old[kind][a])
            lines.append(f'"{a}" = {n}')
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ the check


def run(r, root: Path) -> None:
    """`lab2shot check i18n` (r: cli/check.py's Report): the rules, then the counts against the baseline."""
    from . import t
    from ..messages import written

    c = Catalogues.read()
    zh = c.keys[DEFAULT]
    for lang in LANGS:
        if lang == DEFAULT:
            continue
        for key, (text, path) in sorted(c.keys[lang].items()):
            if key not in zh:
                if not _english_only(key, zh):
                    r.bad(t("cli.check.i18n.extra", language=lang, key=key, where=_rel(path, root)))
            elif placeholders(text) != placeholders(zh[key][0]):
                r.bad(t("cli.check.i18n.placeholders", key=key, zh=" ".join(placeholders(zh[key][0])) or "-",
                        en=" ".join(placeholders(text)) or "-"))
        for key, (_, path) in sorted(zh.items()):
            if key not in c.keys[lang]:
                r.bad(t("cli.check.i18n.lacking", language=lang, key=key, where=_rel(path, root)))
    for problem in reserved_problems(c):
        r.bad(problem)
    for problem in eager_word_problems(root):
        r.bad(problem)
    for problem in plural_problems(c, root):
        r.bad(problem)
    for problem in unknown_key_problems(c, root):
        r.bad(problem)
    for problem in english_problems(root):
        r.bad(problem)
    code = used_keys(root)
    for key, (_, path) in sorted(zh.items()):
        base = plural_of(key, zh) or key  # a plural sentence is used by its key without the form
        base = base[: -len(APP_SUFFIX)] if base.endswith(APP_SUFFIX) else base  # an app form by its key without it
        used = {key, base}
        if key.startswith(CHECKED_USE) and not key.startswith(DERIVED) and not any(f'"{u}"' in code or f"'{u}'" in code for u in used):
            r.bad(t("cli.check.i18n.unused", key=key, where=_rel(path, root)))
    for key, (text, _) in sorted(zh.items()):  # a word's app form: only the placeholders its plain form names
        base = key[: -len(APP_SUFFIX)] if key.endswith(APP_SUFFIX) else None
        if base and not is_message_key(key) and base in zh and not set(placeholders(text)) <= set(placeholders(zh[base][0])):
            r.bad(t("cli.check.i18n.app_placeholders", key=key, base=base))
    words = sum(1 for k in zh if not is_message_key(k))
    r.ok(t("cli.check.i18n.catalogues", langs=" / ".join(LANGS), words=words, messages=len(written(DEFAULT))))

    for problem in node_problems(c):
        r.bad(problem)

    now, most = counts(root, c), baseline()
    over = False
    kinds = {"cjk": t("cli.check.i18n.cjk"), "missing": t("cli.check.i18n.missing"), "fields": t("cli.check.i18n.fields")}
    for kind, what in kinds.items():
        for area in sorted(set(now[kind]) | set(most.get(kind, {}))):
            n, m = now[kind].get(area, 0), most.get(kind, {}).get(area, 0)
            if n > m:
                over = True
                r.bad(t("cli.check.i18n.over", area=area, what=what, now=n, most=m))
            elif n < m:
                r.info(t("cli.check.i18n.under", area=area, what=what, now=n, most=m))
    if not over:
        r.ok(t("cli.check.i18n.counted", cjk=sum(now["cjk"].values()), missing=sum(now["missing"].values()),
               fields=sum(now["fields"].values())))


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ the core nodes' words, worked out from the registry


def node_problems(c: Catalogues) -> list[str]:
    """What the core node types (and the data types, units, groups) lack or have too much of, worked out from what is
    registered: every core node has a subtitle and a description, every parameter a label and every port a label in
    every language (by its own key, a family's or a shared one); every data type, unit and parameter group in use has
    its words; a node key in the core catalogue names a node type there is, and no node type's name holds a segment
    a node key goes on with (NODE_PARTS)."""
    from . import NODE_PARTS, node_key_parts, t, using
    from ..data.types import DATA_TYPES
    from ..data.units import UNITS
    from ..nodes.applies import all_outputs
    from ..nodes.registry import node_types

    out: list[str] = []
    types = node_types()
    for type_id in types:
        if set(type_id.split(".")) & set(NODE_PARTS):
            out.append(t("lint.node.part_in_name", type=type_id))
    groups: set[str] = set()
    for lang in LANGS:
        have = c.keys[lang]
        with using(lang):
            for type_id, cls in sorted(types.items()):
                if cls.runtime != "core":
                    continue
                for part in ("subtitle", "description"):
                    if f"node.{type_id}.{part}" not in have:
                        out.append(t("lint.node.missing", language=lang, key=f"node.{type_id}.{part}"))
                for spec in cls.interface_specs():
                    groups.add(spec.get("group_id") or "")
                    if spec["label"] == spec["name"]:
                        out.append(t("lint.node.missing", language=lang, key=f"node.{type_id}.param.{spec['name']}.label"))
                for port in (*cls.inputs, *all_outputs(cls)):
                    if not (port.text or port.own_word("label", "")):
                        out.append(t("lint.node.missing", language=lang, key=f"node.{type_id}.port.{port.name}.label"))
            for key in [*(f"type.{d}.label" for d in DATA_TYPES), *(f"type.{d}.description" for d in DATA_TYPES),
                        *(f"unit.{u}" for u in UNITS), *(f"group.{g}" for g in sorted(groups) if g)]:
                if key not in have:
                    out.append(t("lint.node.missing", language=lang, key=key))
    core = {lang: DIR / lang / "nodes.toml" for lang in LANGS}
    for lang, path in core.items():
        for key, (_, where) in c.keys[lang].items():
            parts = node_key_parts(key) if where == path else None
            if parts is not None and parts[0] not in types:
                out.append(t("lint.node.unknown_type", language=lang, key=key, type=parts[0]))
    return list(dict.fromkeys(out))
