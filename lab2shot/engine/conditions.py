"""条件表达式：模板参数界面里一项参数什么时候藏起、什么时候置灰（节点图 exposed 参数项的 `hide_when` / `disable_when`，
Houdini 的 Hide When / Disable When；更早的文件写的是 `when`，意思相反，读作 disable_when = not (when)：
engine/templates.py disable_when）。

一个极小的语法，模板作者自己写；服务端（这里）和网页（webui/src/platform/conditions.ts）各实现一份，规则逐条相同，
改规则要两边一起改。
不做任何函数调用，不用 Python eval。

规则：
- 操作数：公开参数的对外名字（exposed 参数项的 `name`，取它的当前值），或字面量：数字（`3`、`-1.5`）、`true` / `false`、
  带引号的字符串（`"abc"` 或 `'abc'`，反斜杠转义下一个字符）。名字由 ASCII 字母、中文（CJK 统一汉字及扩展 A，NAME_LETTERS）、
  ASCII 数字、下划线组成，不以数字开头（两边逐个相同：不按各自运行时的 Unicode 版本判「字母」）；
  `and or not in true false` 是保留字。
- 运算：`==`、`!=`；`in [a, b]`、`not in [a, b]`；`<`、`<=`、`>`、`>=`（两边都是数字才可能为真）；`and`、`or`、`not`；括号。
  优先级从低到高：`or` < `and` < `not` < 比较。比较不连写（`a < b < c` 是写法错误）。
- 相等：布尔只等于布尔，数字按数值比（1 == 1.0），字符串逐字比，空值（null）只等于空值，列表逐项比，其他都不相等。
  布尔和数字不互通：`true == 1` 为假。
- 单独一个操作数（`use_cam`、`not use_cam`）按「有没有」算：布尔取本身，数字非零，字符串非空，列表非空，空值为假。
- 空白：SPACE 列出的字符（两边逐个相同）。数字：整数在服务端是精确的，网页里是双精度，超过 2^53 的整数两边可能
  比出不同的结果；模板里的数远到不了这么大，两边不另做处理。
- 用法是 Houdini 的 Hide When / Disable When：成立 = 藏起 / 置灰。判「成立」用 `holds`：空表达式（没有条件）、写法错误、
  用到了不存在的参数名都不成立（照常显示、可改，参数界面编辑器把错的标红并说明：`problem`）。`judge` 是共用用例的
  口径，对空表达式答 True，不能直接拿来判 Hide / Disable（那样没写条件的项会全被当成成立而藏起）。
"""

from __future__ import annotations

from typing import Any


def say(key: str, **params: Any) -> str:
    """A sentence of this module in the language now: the words are the catalogue's, the page's own (ui.conditions.*,
    lab2shot/i18n/<lang>/ui/conditions.toml: webui/src/platform/conditions.ts says the same sentences with the same
    keys, word for word), never written here. This file is also loaded on its own (a DCC plugin, clients/common/lab2shot_dcc
    paths.conditions_module): there the lab2shot package may be missing, and a plugin sets `say` to its own lookup;
    until it does, the key and its parameters are what comes back."""
    try:
        from lab2shot import i18n
    except ImportError:
        return key + (" " + " ".join(f"{k}={v}" for k, v in params.items()) if params else "")
    return i18n.t(key, **params)


def joined(items) -> str:
    """A list in a sentence, joined the language's way (list.sep). Like `say`, a plugin that loads this file on its own
    sets it to its own lookup (clients/common/lab2shot_dcc/paths.py conditions_module)."""
    try:
        from lab2shot import i18n
    except ImportError:
        return ", ".join(items)
    return i18n.separator().join(items)


KEYWORDS = {"and", "or", "not", "in", "true", "false"}
# what a value of a kind must be, by the kind a parameter takes (the page's valueRefused says the same keys)
WANT = {"boolean": "ui.conditions.want_boolean", "integer": "ui.conditions.want_integer",
        "number": "ui.conditions.want_number", "string": "ui.conditions.want_string"}
# the longest a number may be written (sign and point included): past it, written wrong on both sides (a longer one
# is no value a parameter holds, and the two languages would read it differently)
MOST_DIGITS = 30
# what separates tokens, spelled out the same on both sides (the page's conditions.ts SPACE): str.isspace and a
# JavaScript \s differ on a few characters (\ufeff, \x1c-\x1f, \x85), so neither is used
SPACE = frozenset("\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
                  "\u2028\u2029\u202f\u205f\ufeff" + chr(0x3000))  # (the ideographic space by its number: no CJK text in code)
COMPARE = ("==", "!=", "<=", ">=", "<", ">")
# what a name is made of, spelled out the same on both sides (conditions.ts nameStart / nameChar): ASCII letters and _,
# the CJK unified ideographs (with extension A), ASCII digits after the first; never str.isalpha, whose Unicode
# version is the runtime's (the page's \p{L} is another one)
NAME_LETTERS = ((0x41, 0x5A), (0x61, 0x7A), (0x5F, 0x5F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF))


def _name_start(c: str) -> bool:
    return any(lo <= ord(c) <= hi for lo, hi in NAME_LETTERS)


def _name_char(c: str) -> bool:
    return _name_start(c) or "0" <= c <= "9"


class ConditionError(ValueError):
    """写法错误：`at` 是出错的位置（第几个字，从 0 数），`why` 是说明（its catalogue key, ui.conditions.<…> written out
    whole, said in the language now when read)."""

    def __init__(self, key: str, at: int, **params: Any) -> None:
        super().__init__(key)
        self.key = key
        self.params = params
        self.at = at

    @property
    def why(self) -> str:
        return say(self.key, **self.params)


# ------------------------------------------------------------------ 分词


def _tokens(text: str) -> list[tuple[str, Any, int]]:
    """[(kind, value, at)]：kind 为 num / str / name / kw / op / end。"""
    out: list[tuple[str, Any, int]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in SPACE:
            i += 1
            continue
        start = i
        if "0" <= c <= "9" or (c == "-" and i + 1 < n and "0" <= text[i + 1] <= "9"):
            i += 1
            while i < n and "0" <= text[i] <= "9":
                i += 1
            if i < n and text[i] == ".":
                if not (i + 1 < n and "0" <= text[i + 1] <= "9"):
                    raise ConditionError("ui.conditions.decimal_digits", i)
                i += 1
                while i < n and "0" <= text[i] <= "9":
                    i += 1
            word = text[start:i]
            if len(word) > MOST_DIGITS:  # (Python's int refuses past 4300 digits with a plain ValueError)
                raise ConditionError("ui.conditions.number_too_long", start, most=MOST_DIGITS)
            out.append(("num", float(word) if "." in word else int(word), start))
            continue
        if c in "\"'":
            i += 1
            chars = []
            while i < n and text[i] != c:
                if text[i] == "\\" and i + 1 < n:
                    i += 1
                chars.append(text[i])
                i += 1
            if i >= n:
                raise ConditionError("ui.conditions.unclosed_quote", start)
            i += 1
            out.append(("str", "".join(chars), start))
            continue
        if _name_start(c):
            while i < n and _name_char(text[i]):
                i += 1
            word = text[start:i]
            out.append(("kw" if word in KEYWORDS else "name", word, start))
            continue
        two = text[i:i + 2]
        if two in ("==", "!=", "<=", ">="):
            out.append(("op", two, start))
            i += 2
            continue
        if c in "<>()[],":
            out.append(("op", c, start))
            i += 1
            continue
        if c == "=":
            raise ConditionError("ui.conditions.double_equals", start)
        raise ConditionError("ui.conditions.unknown_char", start, char=c)
    out.append(("end", None, n))
    return out


# ------------------------------------------------------------------ 解析（结果是嵌套的元组）


# how deep brackets and not may nest: what a person writes stays far below it; past it the text is refused as written
# wrong, so neither parsing nor judging it ever runs out of Python's stack
MOST_NESTED = 32


class _Parser:
    def __init__(self, text: str) -> None:
        self.toks = _tokens(text)
        self.i = 0
        self.depth = 0

    def deeper(self, at: int) -> None:
        self.depth += 1
        if self.depth > MOST_NESTED:
            raise ConditionError("ui.conditions.too_deep", at, most=MOST_NESTED)

    def peek(self, k: int = 0) -> tuple[str, Any, int]:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def take(self) -> tuple[str, Any, int]:
        t = self.toks[self.i]
        self.i += 1
        return t

    def is_(self, kind: str, value: Any = None, k: int = 0) -> bool:
        t = self.peek(k)
        return t[0] == kind and (value is None or t[1] == value)

    def expect(self, kind: str, value: str, why: str) -> None:
        if not self.is_(kind, value):
            raise ConditionError(why, self.peek()[2])
        self.take()

    def whole(self) -> tuple:
        node = self.or_()
        if not self.is_("end"):
            raise ConditionError("ui.conditions.misplaced", self.peek()[2], token=_said(self.peek()))
        return node

    def or_(self) -> tuple:
        parts = [self.and_()]
        while self.is_("kw", "or"):
            self.take()
            parts.append(self.and_())
        return parts[0] if len(parts) == 1 else ("or", parts)

    def and_(self) -> tuple:
        parts = [self.not_()]
        while self.is_("kw", "and"):
            self.take()
            parts.append(self.not_())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def not_(self) -> tuple:
        if self.is_("kw", "not"):
            self.deeper(self.take()[2])
            node = ("not", self.not_())
            self.depth -= 1
            return node
        return self.compare()

    def compare(self) -> tuple:
        left = self.operand()
        if self.is_("op") and self.peek()[1] in COMPARE:
            op = self.take()[1]
            return ("cmp", op, left, self.operand())
        if self.is_("kw", "in"):
            self.take()
            return ("in", left, self.items(), False)
        if self.is_("kw", "not") and self.is_("kw", "in", 1):
            self.take()
            self.take()
            return ("in", left, self.items(), True)
        return left

    def items(self) -> list:
        self.expect("op", "[", "ui.conditions.in_list")
        out = []
        if self.is_("op", "]"):
            self.take()
            return out
        while True:
            out.append(self.operand())
            if self.is_("op", ","):
                self.take()
                continue
            self.expect("op", "]", "ui.conditions.list_unclosed")
            return out

    def operand(self) -> tuple:
        kind, value, at = self.peek()
        if kind in ("num", "str"):
            self.take()
            return ("lit", value)
        if kind == "kw" and value in ("true", "false"):
            self.take()
            return ("lit", value == "true")
        if kind == "name":
            self.take()
            return ("name", value)
        if kind == "op" and value == "(":
            self.deeper(self.take()[2])
            node = self.or_()
            self.expect("op", ")", "ui.conditions.paren_unclosed")
            self.depth -= 1
            return node
        if kind == "end":
            raise ConditionError("ui.conditions.unfinished", at)
        raise ConditionError("ui.conditions.misplaced", at, token=_said(self.peek()))


def _said(tok: tuple[str, Any, int]) -> str:
    kind, value, _at = tok
    if kind == "str":
        return f'"{value}"'
    return say("ui.conditions.end") if kind == "end" else str(value)


def blank(text) -> bool:
    """No condition written: nothing but SPACE (the one list both sides use; never str.strip, whose idea of white space
    is not the page's)."""
    return all(c in SPACE for c in str(text or ""))


def parse(text: str) -> tuple | None:
    """解析；空表达式（或只有空白）为 None（没有条件）。写法错误抛 ConditionError。"""
    if blank(text):
        return None
    return _Parser(str(text)).whole()


def names(node: tuple | None) -> list[str]:
    """表达式用到的参数名（按出现先后，不重复）。"""
    out: list[str] = []

    def walk(x: Any) -> None:
        if not isinstance(x, tuple):
            return
        if x[0] == "name":
            if x[1] not in out:
                out.append(x[1])
            return
        for part in x[1:]:
            for y in part if isinstance(part, list) else [part]:
                walk(y)

    walk(node)
    return out


def compared(node: tuple | None) -> list[tuple[str, Any]]:
    """表达式里拿参数名和字面量比相等的地方（`==`、`!=`、`in [...]`、`not in [...]`，名字在哪一边都算）：
    [(参数名, 字面量)]，按出现先后。保存模板时用它查字面量是不是这个参数可能取到的值（engine/templates.py
    check_exposed：`cam_src != 7` 永远成立，多半是写错了）。"""
    out: list[tuple[str, Any]] = []

    def walk(x: Any) -> None:
        if not isinstance(x, tuple):
            return
        if x[0] == "cmp" and x[1] in ("==", "!="):
            a, b = x[2], x[3]
            if a[0] == "name" and b[0] == "lit":
                out.append((a[1], b[1]))
            elif a[0] == "lit" and b[0] == "name":
                out.append((b[1], a[1]))
        elif x[0] == "in" and x[1][0] == "name":
            out.extend((x[1][1], y[1]) for y in x[2] if y[0] == "lit")
        for part in x[1:]:
            for y in part if isinstance(part, list) else [part]:
                walk(y)

    walk(node)
    return out


# ------------------------------------------------------------------ 求值


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def same(a: Any, b: Any) -> bool:
    """相等（规则见模块说明）。"""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _number(a) and _number(b):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if a is None and b is None:
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return False


def truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if _number(v):
        return v != 0
    if isinstance(v, (str, list, dict)):
        return len(v) > 0
    return False


def _value(node: tuple, values: dict[str, Any]) -> Any:
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "name":
        return values.get(node[1])
    return _truth(node, values)


def _truth(node: tuple, values: dict[str, Any]) -> bool:
    kind = node[0]
    if kind == "or":
        return any(_truth(x, values) for x in node[1])
    if kind == "and":
        return all(_truth(x, values) for x in node[1])
    if kind == "not":
        return not _truth(node[1], values)
    if kind == "cmp":
        op, a, b = node[1], _value(node[2], values), _value(node[3], values)
        if op == "==":
            return same(a, b)
        if op == "!=":
            return not same(a, b)
        if not (_number(a) and _number(b)):
            return False
        return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
    if kind == "in":
        a = _value(node[1], values)
        found = any(same(a, _value(x, values)) for x in node[2])
        return not found if node[3] else found
    return truthy(_value(node, values))


def judge(text: str, values: dict[str, Any]) -> bool | str:
    """一条表达式在这些参数值下的结果：True / False；写法错误为 "parse"，用到了 `values` 里没有的名字为 "unknown"。
    两边比的就是这个。"""
    try:
        node = parse(text)
    except ConditionError:
        return "parse"
    if node is None:
        return True
    if any(n not in values for n in names(node)):
        return "unknown"
    return _truth(node, values)


def holds(text: str | None, values: dict[str, Any]) -> bool:
    """Hide When / Disable When 成立吗（网页 platform/conditions.ts conditionHolds 的同一规则）：只有读得通、用到的名字
    都在、结果为真才成立；空表达式（没有条件）、写法错误、用到不存在的名字都不成立。不要拿 judge 判：它对空表达式答 True。"""
    if blank(text):
        return False
    return judge(str(text), values) is True


def shown_options(options: list, values: dict[str, Any]) -> list:
    """公开下拉里现在列出的选项（网页 platform/conditions.ts shownOptions 的同一规则）：
    自己那一项的 hide_when 成立的不列。参数的值恰好是被藏起的那一项时落到第一个列出的（settled）；命令行 / 插件
    直接设成被藏起的那一项时报错（engine/templates.py apply_values）。"""
    return [o for o in options if isinstance(o, dict) and not holds(o.get("hide_when"), values)]


def settled(menus: list, values: dict[str, Any]) -> list[list]:
    """公开下拉的当前值被它自己那一项的 hide_when 藏起时落到哪儿（网页 platform/conditions.ts settledMenus 的同一规则，
    两边同一规则）：落到按当时的值第一个列出的选项；全被藏起的不动（没有可落的）。一个落了
    位会让别的条件变，所以按新值再看，直到没有一个停在被藏起的项上（不动点）；落位互相牵动、回到见过的一组值时（循环），
    在落过位的那几个下拉里按界面顺序、选项顺序找第一组谁都不停在被藏起项上的值（有的话）。`menus`：[{"name", "options"}]，按界面顺序；答 [[名字, 原来的值, 落到的值]]（落回原值的不列），按第一次
    落位的先后。网页在使用者的修改之后、打开文档时写回（graph/exposedTree.ts hiddenChoices），命令行 / 插件设值之后写回
    （engine/templates.py apply_values）。"""
    now, was, seen = dict(values), {}, set()
    while (state := tuple(repr(now.get(m["name"])) for m in menus)) not in seen:
        seen.add(state)
        for menu in menus:
            name, options = menu["name"], [o for o in menu["options"] if isinstance(o, dict)]
            option = next((o for o in options if same(o.get("value"), now.get(name))), None)
            if option is None or not holds(option.get("hide_when"), now) or not (shown := shown_options(options, now)):
                continue
            was.setdefault(name, now.get(name))
            now[name] = shown[0].get("value")
    if any(_stuck(m, now) for m in menus):  # a cycle: the first values of the menus that moved that leave none stuck
        moved = [m for m in menus if m["name"] in was]
        for pick in _choices(moved):
            if not any(_stuck(m, {**now, **pick}) for m in menus):
                now.update(pick)
                break
    return [[name, old, now[name]] for name, old in was.items() if not same(old, now[name])]


def _stuck(menu: dict, values: dict[str, Any]) -> bool:
    """Its value is an option its own hide_when hides, while another is listed."""
    options = [o for o in menu["options"] if isinstance(o, dict)]
    option = next((o for o in options if same(o.get("value"), values.get(menu["name"]))), None)
    return option is not None and holds(option.get("hide_when"), values) and bool(shown_options(options, values))


def _choices(menus: list, most: int = 4096) -> list[dict]:
    """Every assignment of these menus' options, the first menu's first option first (at most `most`)."""
    picks: list[dict] = [{}]
    for menu in menus:
        picks = [{**p, menu["name"]: o.get("value")} for p in picks for o in menu["options"] if isinstance(o, dict)][:most]
    return picks


def disable_when_of(entry: dict) -> str | None:
    """一项公开参数的 Disable When（为真时置灰）。更早的文件写 `when`，意思相反（为真时可以改）：只有文件里没有
    `disable_when` 这个键时才按 `not (when)` 换算；有这个键（哪怕是 null，新版保存的）就以它为准。网页读文件时同一规则
    （platform/conditions.ts disableWhenOf）。"""
    if "disable_when" in entry:
        value = entry["disable_when"]
        return value if isinstance(value, str) else None
    old = entry.get("when")
    return f"not ({old})" if isinstance(old, str) and not blank(old) else None


def problem(text: str | None, known) -> str | None:
    """条件的问题（in the language now: `say`），没有问题为 None。`known`：所有公开参数的对外名字。"""
    try:
        node = parse(text or "")
    except ConditionError as exc:
        return say("ui.conditions.syntax", at=exc.at + 1, why=exc.why)
    missing = [n for n in names(node) if n not in set(known)]
    if missing:
        return say("ui.conditions.unknown_names", names=joined(missing))
    return None


# ------------------------------------------------------------------ 参数值规则（一份：服务器 engine/templates.py 与 DCC 插件
# 同用这里；网页 platform/conditions.ts valueRefused 逐字相同，lab2shot check conditions 管着）


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def value_refused(spec: dict, value: Any) -> str | None:
    """Why the target parameter (its interface spec: type, nullable, options, minimum, maximum) would not take
    `value` (None: it would). The one rule: apply_values on the server and a DCC plugin before it sends anything."""
    if value is None:
        return None if spec["nullable"] else say("ui.conditions.not_null")
    kind = spec["type"]
    fits = {"boolean": isinstance(value, bool), "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": _is_number(value), "string": isinstance(value, str)}.get(kind, False)
    if not fits:
        return say(WANT.get(kind, "ui.conditions.want_menu_unfit"))
    if spec.get("options") and not any(value == o and type(value) is type(o) or (_is_number(value) and _is_number(o) and value == o)
                                       for o in spec["options"]):
        return say("ui.conditions.one_of", options=" / ".join(map(str, spec["options"])))
    if _is_number(value):
        if spec.get("minimum") is not None and value < spec["minimum"]:
            return say("ui.conditions.below_min", min=format(spec["minimum"], "g"))
        if spec.get("maximum") is not None and value > spec["maximum"]:
            return say("ui.conditions.above_max", max=format(spec["maximum"], "g"))
    return None
