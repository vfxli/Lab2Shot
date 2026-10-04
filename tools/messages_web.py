"""生成网页端的消息表与界面词表，两种语言（lab2shot/i18n/<lang>/messages/web.toml、i18n/<lang>/ui/*.toml -> webui/src/messages/）。

网页自身产生的消息（提交前的拦截、页面自身操作遇到的问题）与服务器一样按编号引用，网页的界面字按键引用（`t("ui.…")`，
webui/src/i18n/t.ts）；文字统一维护在语言目录中，本工具据此生成下面两个文件，生成文件不得手工编辑。
`lab2shot check messages` 运行 `--check` 校验二者一致。

每个文件里按语言各一张表（zh、en），两种语言的键相同（lab2shot check i18n）；随数量变的句子按复数形写成
`<键>.one` / `<键>.other`（`"CODE.one"` / `"CODE.other"`），原样进表，网页按数量选（i18n/words.ts）。
完整目录超过 20 KB，而登录门仅使用其中十余条，登录前无需全部下发。因此按登录门静态可达的文件拆分为两份：

  - `generatedGateCatalogue.ts`  登录门实际使用的消息编号与界面键、语言名与列表连接符（lang.*、list.*），以及级别表；
  - `generatedCatalogue.ts`      其余消息与界面字，登录后由 site.tsx 一次性注册（addCatalogue、addWords）。

登录门可达的文件由 `main.tsx` 出发沿静态 import 遍历得出：计入 `import ... from`、`export ... from`、`import "x.css"`；
`import()` 为按需加载，不计入。登录门新增使用的消息或界面键会使 `--check` 报告两个文件已过期。

    uv run python tools/messages_web.py          # 写入两个文件
    uv run python tools/messages_web.py --check  # 文件未更新时以状态码 1 退出
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab2shot_shared.protocol import CODE, CODE_CANDIDATE  # noqa: E402  (the one rule; the finder is looser, see there)

WEB_SRC = ROOT / "webui" / "src"
TARGET = WEB_SRC / "messages" / "generatedCatalogue.ts"
GATE_TARGET = WEB_SRC / "messages" / "generatedGateCatalogue.ts"
GATE_ENTRY = "main.tsx"  # 登录前浏览器加载的入口（webui/index.html）
WORD_PREFIXES = ("ui.", "lang.", "list.", "button.")  # the page's own words (the rest of its words come from the server); button.*: the shared button words an exposed entry may name (ExposedParam.word)
GATE_ALWAYS = ("lang.", "list.")  # the language menu and every message's list need them from the start

STATIC = re.compile(r"(?:^|\n)\s*(?:import|export)\s+(?:type\s+)?(?:[\s\S]*?\sfrom\s+)?[\"']([^\"']+)[\"']")
KEY_CANDIDATE = re.compile(r"""["'`]((?:ui|lang|list)\.[A-Za-z0-9_.-]+)["'`]""")


# 页面在提交前自行给出的服务器拒绝消息（编号、文本与模板均与服务器相同）。这些消息归服务器目录所有，
# web.toml 不重复定义，在生成时从服务器目录复制。页面代码（webui/src）使用而页面目录缺失的编号会导致
# `msg()` 在运行时抛出异常（format.ts），因此页面开始自行使用某个服务器编号时，必须将其加入此元组。
# 例如 `B-WIRE-OUTINACTIVE`（engine.toml）：webui/src/graph/edit.ts 在向未激活的输出端口提交连线前使用该消息。
FROM_SERVER: tuple[str, ...] = ("B-WIRE-OUTINACTIVE",)


def _resolve(from_file: Path, spec: str) -> Path | None:
    if not spec.startswith("."):
        return None  # 包导入
    p = (from_file.parent / spec).resolve()
    for c in (p, p.with_suffix(p.suffix + ".ts"), p.with_suffix(p.suffix + ".tsx"), p / "index.ts", p / "index.tsx"):
        if c.is_file() and c.suffix in (".ts", ".tsx"):  # 仅跟踪代码文件；样式与图片中不含消息编号
            return c
    return None


def gate_files() -> list[Path]:
    """返回登录前静态加载的全部文件（自 main.tsx 起沿 import 遍历）。"""
    seen: set[Path] = set()
    todo = [WEB_SRC / GATE_ENTRY]
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        text = f.read_text(encoding="utf-8")
        for m in STATIC.finditer(re.sub(r"/\*[\s\S]*?\*/|//.*$", "", text, flags=re.M)):
            if m.group(0).strip().startswith("import type"):
                continue
            if (r := _resolve(f, m.group(1))) is not None:
                todo.append(r)
    return sorted(seen)


def _gate_texts() -> list[str]:
    return [f.read_text(encoding="utf-8") for f in gate_files() if f.name not in (TARGET.name, GATE_TARGET.name)]


def gate_codes() -> set[str]:
    """返回登录门相关文件中出现的消息编号（不含两个生成文件本身）。"""
    return {c for text in _gate_texts() for c in CODE_CANDIDATE.findall(text) if CODE.match(c)}


def gate_keys() -> set[str]:
    """返回登录门相关文件中写出的界面键（注释里举的例子不算）。"""
    from lab2shot.i18n.lint import strip_ts_comments

    return {k for text in _gate_texts() for k in KEY_CANDIDATE.findall(strip_ts_comments(text))}


def templates(lang: str) -> dict[str, str]:
    """网页的消息（一种语言）：web.toml 的，加上 FROM_SERVER 从服务器目录复制的。英文没写的不在英文表里。"""
    from lab2shot import i18n
    from lab2shot.messages import _read

    source = i18n.lang_dir(lang) / "messages" / "web.toml"
    said = _read(source)
    server = {code: text for path in i18n.message_files(lang) if path != source for code, text in _read(path).items()}
    if lang == i18n.DEFAULT:
        from lab2shot.messages import code_of

        missing = [code for code in FROM_SERVER if code not in {code_of(k) for k in server}]
        if missing:
            raise SystemExit(f"FROM_SERVER names codes the server's catalogue does not have: {missing}")
    from lab2shot.messages import code_of

    return {**said, **{key: text for key, text in server.items() if code_of(key) in FROM_SERVER}}


def words(lang: str) -> dict[str, str]:
    from lab2shot import i18n

    return {k: v for k, v in sorted(i18n.words(lang).items()) if k.startswith(WORD_PREFIXES)}


def _table(name: str, kind: str, per_lang: dict[str, dict[str, str]]) -> list[str]:
    out = [f"export const {name}: Readonly<Record<Lang, Readonly<Record<string, string>>>> = {{"]
    for lang, items in per_lang.items():
        out.append(f"  {lang}: {{")
        out += [f"    {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in items.items()]
        out.append("  },")
    out.append("};")
    return out


def generated() -> tuple[str, str]:
    """返回（登录门文件, 其余文件）的内容。"""
    from lab2shot import i18n
    from lab2shot.messages import LEVELS, code_of

    codes, keys = gate_codes(), gate_keys()
    msgs = {lang: templates(lang) for lang in i18n.LANGS}
    ws = {lang: words(lang) for lang in i18n.LANGS}
    in_gate = lambda k: (i18n.plural_base(k) or k) in keys or k in keys or k.startswith(GATE_ALWAYS)  # noqa: E731
    head = ["// Generated by tools/messages_web.py from lab2shot/i18n/<lang>/messages/web.toml and i18n/<lang>/ui/*.toml:",
            "// do not edit (edit the .toml, run the tool). One table per language, the same keys in both; a sentence that",
            "// follows a number is in its plural forms (<key>.one / <key>.other: i18n/words.ts picks)."]
    gate_text = "\n".join([
        *head,
        'import type { Lang } from "../i18n/lang";',
        "",
        "// What the gate (the files main.tsx reaches statically) really uses, and the level table: all that is sent before a login.",
        *_table("GATE_CATALOGUE", "messages", {l: {c: t for c, t in m.items() if code_of(c) in codes} for l, m in msgs.items()}),
        "",
        *_table("GATE_WORDS", "words", {l: {k: v for k, v in w.items() if in_gate(k)} for l, w in ws.items()}),
        "",
        "// The message levels, the server's own table (lab2shot/messages LEVELS): a code's first letter and what it is.",
        "export const LEVELS = {",
        *(f"  {letter}: {json.dumps(name)}," for letter, name in LEVELS.items()),
        "} as const;",
        "",
    ])
    rest_text = "\n".join([
        *head,
        'import type { Lang } from "../i18n/lang";',
        "",
        "// The page's own messages by code and words by key; a message from the server comes with its text.",
        "// Sent only after a login: site.tsx registers it when it loads (messages/format.ts addCatalogue, i18n/words.ts addWords).",
        *_table("CATALOGUE", "messages", {l: {c: t for c, t in m.items() if code_of(c) not in codes} for l, m in msgs.items()}),
        "",
        *_table("WORDS", "words", {l: {k: v for k, v in w.items() if not in_gate(k)} for l, w in ws.items()}),
        "",
    ])
    return gate_text, rest_text


def main() -> int:
    gate_text, rest_text = generated()
    wrote = ((GATE_TARGET, gate_text), (TARGET, rest_text))
    if "--check" in sys.argv:
        stale = [p for p, text in wrote if not p.is_file() or p.read_text(encoding="utf-8") != text]
        print("up to date" if not stale else
              "not up to date, run uv run python tools/messages_web.py: " + ", ".join(p.relative_to(ROOT).as_posix() for p in stale))
        return 0 if not stale else 1
    for path, text in wrote:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
