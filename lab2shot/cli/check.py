"""`lab2shot check`: the project's invariants, run as code instead of trusted as comments.

Each check is one rule the code base relies on that nothing else enforces (a rule that lives only in a comment can
quietly stop being true). Run it after any change; it is fast (seconds), needs no server, no GPU, no network, and
touches nothing in work/.

    official     every third-party node's declared ports refer to real ports/params, and every upstream symbol it
                 cites is found in the cited lines (nodes/official.py)
    nodes        every node shows at most ON_NODE_MAX parameters on its body, each a real parameter (nodes/params.py);
                 its button parameters are named apart from its parameters and each other, and hold no value
    workers      every extension node runs its worker through WorkerNode (or its family), an import or an output
                 setting, and no extension class overrides cook
    categories   the two category trees and the node placements read and agree (lab2shot/categories.py)
    messages     every message code the Python code uses exists in a catalogue; the page's generated catalogues
                 are up to date (tools/messages_web.py --check); every .short fits a node's bottom line
    templates    every card loads, wires only existing node types/ports/params, passes the graph checks except
                 for the inputs a user must fill, its parameter interface (exposed) reads, it keeps the conventions of
                 templates/_conventions.md (engine/conventions.py), and its meta matches the rules
    channels     the channel-name mapping for pictures referenced as they are (view/frames.py channel_in_file)
    deletion     cache entries are deleted in exactly one place (data/packet.py remove); every other rmtree in the
                 code is on a known list of non-cache folders
    releases     the release notes (CHANGELOG.toml) read: an https address, every version complete, newest first,
                 no name twice (lab2shot/releases.py)
    imports      every `from lab2shot... import name`, the ones inside functions too, names something that is there
    empties      every output a node's code gives empty by name declares may_be_empty (nodes/port.py)
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import typer

from lab2shot_shared.protocol import CODE, CODE_CANDIDATE

from .base import app, console

# rmtree outside data/packet.py remove(): each is a folder that is NOT a cache entry (installer checkouts, a node's own
# scratch, uploads/outputs the transfer layer owns, database backups, feedback bundles). An rmtree in any other file
# means cache folders are deleted by a second path, which the cache design forbids.
RMTREE_ALLOWED = {
    "lab2shot/cli/accounts.py", "lab2shot/database/__init__.py", "lab2shot/engine/cook.py", "lab2shot/engine/external.py",
    "lab2shot/farm/disk.py", "lab2shot/feedback.py", "lab2shot/installer/envbuild.py", "lab2shot/installer/run.py",
    "lab2shot/installer/sources.py", "lab2shot/library.py", "lab2shot/transfer/outputs.py", "lab2shot/transfer/uploads.py",
    "lab2shot/transfer/tasks.py",  # a task's folder (its graph, footage links, outputs), never a cache entry
}
# fresh_dir (a packet folder about to be written) is called under the entry's lock only: by a cook on its outputs and by
# data/packet.py produce(); anyone else writes a packet through produce()
FRESH_DIR_ALLOWED = {"lab2shot/data/packet.py", "lab2shot/engine/cook.py"}
# route parameters that name one account's data (a packet, a job, a task, a template): a user route with one declares
# whose it is (owned=, server/owners.py)
OWNED_PARAMS = {"fp", "job_id", "task_id", "gid"}
# graph checks a fresh card is allowed to fail: the inputs a user fills before submitting
TEMPLATE_WAITS = ("B-READ-", "B-IMPORT-", "B-WIRE-WAIT")


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.checked: list[str] = []
        self.notes: list[str] = []

    def ok(self, what: str) -> None:
        self.checked.append(what)

    def bad(self, what: str) -> None:
        self.problems.append(what)

    def info(self, what: str) -> None:
        """Worth a line, not a problem (nodes or cards still 未分类, a placing that waits for an extension)."""
        self.notes.append(what)


# ------------------------------------------------------------------ the checks


def check_official(r: Report) -> None:
    """The rule of nodes/official.py for every node that declares it: each name it declares is a real port (takes: an
    input port or a parameter; gives / ours: an output port), each of its ports is declared (an input in takes, an
    output in gives or ours: an undeclared one is something added), and each upstream symbol is in the cited lines."""
    from ..nodes.registry import node_types

    n = 0
    for tid, node in sorted(node_types().items()):
        official = getattr(node, "official", None)
        if official is None:
            continue
        n += 1
        ins = {p.name for p in node.inputs} | {p["name"] for p in node.param_specs()}
        outs = {p.name for p in node.outputs}
        for port in official.takes:
            if port not in ins:
                r.bad(f"official {tid}: takes[{port!r}] 不是该节点的输入端口或参数")
        for port in (*official.gives, *official.ours):
            if port not in outs:
                r.bad(f"official {tid}: gives/ours[{port!r}] 不是该节点的输出端口")
        for p in node.inputs:
            if p.name not in official.takes:
                r.bad(f"official {tid}: 输入端口 {p.name!r} 没有写进 takes：上游不接受它，就是自行添加的")
        for p in node.outputs:
            if p.name not in official.gives and p.name not in official.ours:
                r.bad(f"official {tid}: 输出端口 {p.name!r} 没有写进 gives 或 ours：上游不输出它，就是自行添加的")
        try:
            for m in official.missing_symbols():
                r.bad(f"official {tid}: {m}")
        except Exception as exc:  # noqa: BLE001 (a missing cited file is also a problem)
            r.bad(f"official {tid}: 无法读取引文：{exc}")
    r.ok(f"official：{n} 个第三方节点的端口与上游引文")


def check_nodes(r: Report) -> None:
    """Every node's 「shown on the body」 list keeps to its limit: at most ON_NODE_MAX parameters (nodes/params.py), and
    each names a real parameter of the node (a typo would silently show nothing). Its button parameters
    (nodes/params.py Button, NodeDef.interface_specs: its own, 「在视图里点选」, 「计算」) are named apart from its
    parameters and from each other (an exposed one is found by name) and hold no value: an action, no default, not in
    the fingerprint, not sent to the worker."""
    from ..nodes.params import ON_NODE_MAX
    from ..nodes.registry import node_types

    for tid, node in sorted(node_types().items()):
        params = {p["name"] for p in node.param_specs()}
        if len(node.on_node) > ON_NODE_MAX:
            r.bad(f"on_node {tid}: 节点体上放了 {len(node.on_node)} 个参数，超过 {ON_NODE_MAX}（{'、'.join(node.on_node)}）")
        # 节点体上默认显示的可以是参数，也可以是按钮参数（「输出」的「下载」）
        if unknown := [p for p in node.on_node if p not in {s["name"] for s in node.interface_specs()}]:
            r.bad(f"on_node {tid}: {unknown} 不是该节点的参数")
        # 按钮参数（nodes/params.py Button）：名字不和参数重、彼此不重（公开时 target 按名字找），且只有动作、没有值
        buttons = [b for b in node.interface_specs() if b["widget"] == "button"]
        names = [b["name"] for b in buttons]
        if clash := sorted(set(names) & params) + sorted({n for n in names if names.count(n) > 1}):
            r.bad(f"buttons {tid}: 按钮名 {clash} 和参数或别的按钮重名")
        if bad := [b["name"] for b in buttons if not b.get("action") or b["default"] is not None or b["affects_result"] or b["worker"]]:
            r.bad(f"buttons {tid}: 按钮 {bad} 要有动作、没有值、不进指纹和 worker")
    r.ok(f"on_node：{len(node_types())} 个节点体上的参数都不超过 {ON_NODE_MAX} 个；按钮参数不和参数重名、没有值")
    # 画面还是数值（Port.data）：三或四通道、或任意通道的图像输入口必须说清楚收哪一种（False 画面 / True 数值 /
    # EITHER 都收），不按缺省猜，否则数值图和照片会混接而无法报错；
    # 一或两通道的口由类型定为数值，不能声明成只收画面
    from ..nodes.port import kind_declaration

    undeclared, contrary = [], []
    for tid, node in sorted(node_types().items()):
        for p in node.inputs:
            said = kind_declaration(p)
            if said == "undeclared":
                undeclared.append(f"{tid}.{p.name}")
            elif said == "contrary":
                contrary.append(f"{tid}.{p.name}")
    if undeclared:
        r.bad(f"data: 图像输入口没说收画面还是数值（Port.data：False / True / EITHER）：{undeclared}")
    if contrary:
        r.bad(f"data: 一或两通道的输入口声明成只收画面：{contrary}")
    if not undeclared and not contrary:
        r.ok("data：每个三 / 四通道和任意通道的图像输入口都声明了收画面、数值还是都收")
    # 每种数据类型（连同它的上级类型）在包的说明书表里都有一行（data/contracts.py META）：缺一行时，这种包一算出来
    # 核对就 KeyError，整个计算中断，而不是报在节点上
    from ..data.contracts import META, lineage
    from ..data.types import DATA_TYPES

    if missing := sorted({t for d in DATA_TYPES for t in lineage(d) if t not in META}):
        r.bad(f"contracts: 数据类型 {missing} 在 data/contracts.py META 里没有说明书行")
    else:
        r.ok(f"contracts：{len(DATA_TYPES)} 种数据类型都有说明书行")


def check_workers(r: Report) -> None:
    """An extension's node runs its worker through one template: WorkerNode (prepare -> worker -> convert; families
    inherit it), or the two framework families that run workers their own way — an import (formats.py ImportNode:
    the file's scene arrays) and an output setting (nodes/output.py OutputSettings: write()). No class an extension
    defines overrides cook: a second way to run a worker would skip missing-frame handling, streaming and the
    lens record (nodes/families/base.py)."""
    from ..nodes.core import CORE_NODES
    from ..nodes.families.base import WorkerNode
    from ..nodes.formats import ImportNode
    from ..nodes.output import OutputSettings
    from ..nodes.registry import node_types

    core = {n.id for n in CORE_NODES}
    count = 0
    for tid, t in sorted(node_types().items()):
        if tid in core:
            continue
        count += 1
        if not issubclass(t, (WorkerNode, ImportNode, OutputSettings)):
            r.bad(f"workers {tid}: 接入层节点要继承 WorkerNode（或它的家族）、ImportNode 或 OutputSettings，现在是 {t.__mro__[1].__name__}")
        own = [c.__name__ for c in t.__mro__ if c.__module__.startswith("adapters.") and "cook" in vars(c)]
        if own:
            r.bad(f"workers {tid}: 接入层的类不重写 cook（{'、'.join(own)}）：发什么写在 prepare，读回写在 convert")
    r.ok(f"workers：{count} 个接入层节点都经 WorkerNode / 导入 / 输出设置跑 worker，没有自己重写 cook")


def check_categories(r: Report) -> None:
    """The two category trees and the node placements are data files (lab2shot/categories.py): each must read, every
    subcategory must hang under a first-level one, every placed node must be a node type placed somewhere the tree has,
    and every card's place must be one the templates tree has (a place it lacks is 未分类, reported here rather than hidden)."""
    from .. import categories as cats
    from ..nodes.registry import node_types

    for name, tree in (("templates", cats.templates), ("menu", cats.menu)):
        if problem := tree.problem():
            r.bad(f"categories {name}: {problem}")
            continue
        rows = tree.rows()
        for k, row in rows.items():
            if not cats.ID.match(k):
                r.bad(f"categories {name}: 代号 {k!r} 不符合命名规则")
            parent = str(row.get("parent") or "")
            if parent and (parent not in rows or rows[parent].get("parent")):
                r.bad(f"categories {name}: {k} 的上级 {parent!r} 不是一级分类")
            if not parent and tree.sections and str(row.get("section") or "") not in tree.sections:
                r.bad(f"categories {name}: 一级分类 {k} 未指定所属区域（{'、'.join(tree.sections)}）")
    if problem := cats.nodes.problem():
        r.bad(f"categories nodes: {problem}")
    types = node_types()
    placed = cats.nodes.rows()
    gone = sorted(t for t in placed if t not in types)
    if gone:  # an extension not loaded now (uninstalled, or failing to load): its placing waits for it, not an error
        r.info(f"categories nodes: {len(gone)} 条归类记录对应的节点当前不存在（扩展包未安装或未加载）：{'、'.join(gone[:6])}{'……' if len(gone) > 6 else ''}")
    for tid, where in placed.items():
        if not cats.menu.known(where):
            r.bad(f"categories nodes: {tid} 归入 {where!r}，但菜单树中没有该分类")
    loose = sorted(t for t in types if t not in placed)
    if loose:
        r.info(f"categories nodes: {len(loose)} 个节点未分类（可在菜单中拖放到分类上）：{'、'.join(loose[:8])}{'……' if len(loose) > 8 else ''}")
    from ..nodes import text

    for p in [*text.problems(), *text.limit_problems()]:
        r.bad(f"categories 节点文字: {p}")
    missing = sorted(t for t, cls in types.items() if not text.entries(text.file_for(cls)).get(t, {}).get("label"))
    if missing:
        r.info(f"categories 节点文字: {len(missing)} 个节点缺少名称（菜单中以类型 id 显示；可在节点菜单中点击「编辑」补充）：{'、'.join(missing[:6])}{'……' if len(missing) > 6 else ''}")
    for f in text.files():
        for t, words in text.entries(f).items():
            # the page shows a node's words as plain text (the node menu, its info card): Markdown shows as typed
            if any("**" in str(v) for v in words.values()):
                r.bad(f"categories 节点文字 {f.parent.name}/{f.name}: {t} 的文字里有「**」：节点文字按纯文字显示，不会加粗，只会原样露出星号")
            if t not in types:
                r.info(f"categories 节点文字 {f.parent.name}/{f.name}: 节点类型 {t} 当前不存在（扩展包未安装、未加载，或 id 已更改）")
    r.ok(f"categories：模板树 {len(cats.templates.rows())} 条、菜单树 {len(cats.menu.rows())} 条、{len(placed)} 个节点已归类、{len(text.files())} 份节点文字")


def check_messages(r: Report, root: Path) -> None:
    from ..messages import SHORT_WIDTH, catalogue, short_width, shorts

    over = {c: short_width(t) for c, t in shorts().items() if short_width(t) > SHORT_WIDTH}
    for code, width in sorted(over.items()):
        r.bad(f"messages: {code}.short 有 {width:g} 个全角宽，超过 {SHORT_WIDTH}（节点底行放不下，会被截断）")
    r.ok(f"messages：{len(shorts())} 个 .short 简写都不超过 {SHORT_WIDTH} 个全角宽")
    known = set(catalogue())
    used: dict[str, list[str]] = {}
    for folder in ("lab2shot", "adapters", "worker_sdk"):
        for path in (root / folder).rglob("*.py"):
            if path.name == "check.py" and path.parent.name == "cli":
                continue  # this file lists the prefixes and examples itself
            text = path.read_text(encoding="utf-8", errors="replace")
            for code in CODE_CANDIDATE.findall(text):
                used.setdefault(code, []).append(str(path.relative_to(root)))
    for code, where in sorted(c for c in used.items() if not CODE.match(c[0])):
        r.bad(f"messages: {code} 不是消息编号的写法（类型字母-模块-含义，lab2shot_shared/protocol.py CODE；{where[0]}）")
    missing = {c: v for c, v in used.items() if c not in known and CODE.match(c)}
    for code, where in sorted(missing.items()):
        r.bad(f"messages: {code} 在目录中没有条目（{where[0]}）")
    r.ok(f"messages：已使用的 {len(used)} 个编号均在目录中（目录共 {len(known)} 条）")
    gen = root / "tools" / "messages_web.py"
    if gen.is_file():
        out = subprocess.run([sys.executable, str(gen), "--check"], capture_output=True, text=True, cwd=root)
        if out.returncode != 0:
            r.bad("messages: 网页的消息目录不是最新版本（请运行 uv run python tools/messages_web.py）")
        else:
            r.ok("messages：网页目录与 web.toml 一致")


def check_templates(r: Report) -> None:
    from ..engine import templates as T
    from ..engine.graph import Graph
    from ..errors import MessageError
    from ..nodes.registry import node_types

    from ..engine import conventions
    from ..library import TEMPLATES_DIR

    types = node_types()
    cards = T.templates()
    # the conventions for people (templates/_conventions.md), checked where a graph can tell (engine/conventions.py)
    rules = conventions.read_table(TEMPLATES_DIR / "_conventions.md") if (TEMPLATES_DIR / "_conventions.md").is_file() else None
    prefixes = conventions.read_prefixes(TEMPLATES_DIR / "_conventions.md") if rules is not None else []
    for said in conventions.table_problems(rules) if rules is not None else ():  # a row that would check nothing
        r.bad(f"templates: {said}")
    # how a row of several names is read: 「a / b」 against one group of targets each, else said (a sample table)
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "t.md"
        sample.write_text("| 名 | 义 | 目标 | 备注 |\n|---|---|---|---|\n"
                          "| `a` / `b` | | `core.math.operation` / `core.value_int.value` `unit` | |\n"
                          "| `c` / `d` | | `core.math.operation` `core.value_int.value` | |\n", encoding="utf-8")
        got = conventions.read_table(sample)
        if (got["a"]["targets"], got["b"]["targets"]) != ({("core.math", "operation")}, {("core.value_int", "value"), ("core.value_int", "unit")}) \
                or not got["c"]["unpaired"] or got["a"]["unpaired"]:
            r.bad(f"templates: 固定名表的「名字 / 名字」对「目标组 / 目标组」读错了：{got}")
        # rules 1 / 2 and a menu's meanings, on a sample card: the core node is 「数学」's project (core); an
        # unprefixed name on another project's node is said, a prefixed or table one is not; a value meaning something
        # else than the table says is said, a card the notes name as the exception is not
        sample.write_text("**通则**\n1. 不带前缀\n2. 辅助节点加前缀：`aux_`\n\n| 名 | 义 | 目标 | 备注 |\n|---|---|---|---|\n"
                          "| `src` | 来源 | `core.value_int.value` | 1 甲 / 2 乙（Odd 例外：2 = 丙） |\n", encoding="utf-8")
        s_table, s_prefixes = conventions.read_table(sample), conventions.read_prefixes(sample)
        aux = next(t for t in types.values() if t.runtime != "core" and "width" not in {p["name"] for p in t.param_specs()}
                   and len(t.param_specs()) >= 2)
        a1, a2 = [p["name"] for p in aux.param_specs()][:2]
        card = {"meta": {"name": "样卡", "project": "core"},
                "nodes": [{"id": "v", "type": "core.value_int", "params": {}}, {"id": "m", "type": "core.math", "params": {}},
                          {"id": "c", "type": aux.id, "params": {}}],
                "edges": [],
                "exposed": [{"name": "src", "target": "v.value", "widget": "menu", "options": [{"value": 1, "label": "甲"}, {"value": 2, "label": "丙"}]},
                            {"name": "operation", "target": "m.operation"}, {"name": f"aux_{a1}", "target": f"c.{a1}"}, {"name": a2, "target": f"c.{a2}"}]}
        types_of = {n["id"]: types[n["type"]] for n in card["nodes"]}
        said = conventions._prefixed(card, card["exposed"], types_of, s_table, s_prefixes) + conventions._meanings(card, card["exposed"], s_table)
        odd = conventions._meanings({**card, "meta": {"name": "Odd 卡"}}, card["exposed"], s_table)
        if len(said) != 2 or a2 not in said[0] or "丙" not in said[1] or odd:
            r.bad(f"templates: 通则 1 / 2 或菜单取值含义的判定不对：{said}；例外卡 {odd}")
        # 「<内容>_name」 passes only as an output-settings node's name (rule 4): on another node's parameter it is judged
        # by rules 1 / 2 (a node's character name is no delivery's)
        sample.write_text("**通则**\n2. 辅助节点加前缀：`aux_`\n4. 同类输出两个以上时按内容 `<内容>_name`\n", encoding="utf-8")
        named = conventions.read_prefixes(sample)
        writer = next(t for t in types.values() if "name" in {p["name"] for p in t.param_specs()} and t.id.startswith("core.output_"))
        card2 = {"meta": {"name": "样卡", "project": "core"},
                 "nodes": [{"id": "o", "type": writer.id, "params": {}}, {"id": "c", "type": aux.id, "params": {}}], "edges": [],
                 "exposed": [{"name": "plate_name", "target": "o.name"}, {"name": f"{a2}_name", "target": f"c.{a2}"}]}
        got = conventions._prefixed(card2, card2["exposed"], {n["id"]: types[n["type"]] for n in card2["nodes"]}, {}, named)
        if len(got) != 1 or f"{a2}_name" not in got[0]:
            r.bad(f"templates: 「<内容>_name」只应放行输出设置的 name：{got}")
    for t in cards:
        data = t["graph"]
        name = t["name"]
        for n in data["nodes"]:
            if n["type"] not in types:
                r.bad(f"templates {name}: 节点类型 {n['type']} 不存在")
                continue
            specs = {p["name"] for p in types[n["type"]].param_specs()}
            for k in n.get("params", {}):
                if k not in specs:
                    r.bad(f"templates {name}: 节点 {n['id']} 参数 {k} 不存在")
        try:
            g = Graph.from_json(data)
        except Exception as exc:  # noqa: BLE001
            r.bad(f"templates {name}: 无法构建节点图：{exc}")
            continue
        for n in data["nodes"]:
            try:
                g.check_inputs(n["id"])
            except MessageError as exc:
                if not str(getattr(exc, "code", "")).startswith(TEMPLATE_WAITS):
                    r.bad(f"templates {name}: 节点 {n['id']}：{exc}")
            except Exception as exc:  # noqa: BLE001
                r.bad(f"templates {name}: 节点 {n['id']} 检查时抛出异常：{type(exc).__name__}: {exc}")
        for m in T.check_exposed(data):  # 参数界面（exposed 树）：结构、目标参数、控件、下拉的值、条件
            r.bad(f"templates {name}: {m.text}")
        for said in T.route_problems(data, TEMPLATE_WAITS):  # 每个公开选择的取值下（与 route_tags 同一枚举）都能读图、规划
            r.bad(f"templates {name}: {said}")
        for said in T.menu_routes(data):  # 驱动切换的下拉：每个取值都落在切换的某一路上
            r.bad(f"templates {name}: {said}")
        for said in conventions.problems(data, rules, prefixes) if rules is not None else ():  # 对外参数名与按钮的约定
            r.bad(f"templates {name}: {said}")
        # a declared main project (meta.project, templates.core_project) is one of the card's own nodes' projects
        if (declared := (data.get("meta") or {}).get("project")) is not None and declared not in {
                types[n["type"]].runtime for n in data["nodes"] if n["type"] in types}:
            r.bad(f"templates {name}: meta.project={declared!r} 不是卡上任何节点的项目")
        # the card's name and intro fit what the library takes when an administrator saves one (library.NAME_CHARS /
        # TEXT_CHARS): a file written by hand is held to the same, or saving it again from the page would refuse it
        from .. import library

        for what, text, most in (("名字", t["name"], library.NAME_CHARS), ("简介", t["intro"], library.TEXT_CHARS)):
            if len(text) > most:
                r.bad(f"templates {name}: {what} {len(text)} 个字，超过 {most}")
        if t.get("deliverable") and not t.get("category"):
            r.bad(f"templates {name}: meta.deliverable={t.get('deliverable')!r}，模板树中没有该分类（卡将显示在「未分类」中）")
    for said in T.shared_interfaces([(t["name"], t["graph"]) for t in cards]):  # 同一块参数在各卡上一致（NodeDef.same_on_cards）
        r.bad(f"templates {said}")
    loose = [t["name"] for t in cards if not t.get("deliverable")]
    if loose:
        r.info(f"templates: {len(loose)} 张卡未分类：{'、'.join(loose[:6])}{'……' if len(loose) > 6 else ''}")
    if rules is None:
        r.info("templates: 没有 templates/_conventions.md，这次不查对外参数名的约定")
    r.ok(f"templates：{len(cards)} 张卡的节点、参数、连线、参数界面{'、对外参数名约定' if rules is not None else ''}与分类")


def check_channels(r: Report) -> None:
    from ..view.frames import channel_in_file

    cases = [(["R", "G", "B", "A"], "G", "G"), (["Y"], "G", "Y"), (["Y", "A"], "B", "Y"), (["Y", "A"], "A", "A"),
             (["R", "G"], "G", "G"), (["R", "G"], "B", "B"),
             ([], "R", "R"), (["Z"], "A", "A"), (["R", "G", "B"], "A", "A")]
    for have, name, want in cases:
        got = channel_in_file(have, name)
        if got != want:
            r.bad(f"channels: 文件 {have} 请求 {name} → {got!r}，应为 {want!r}")
    r.ok(f"channels：{len(cases)} 种通道映射")


def check_deletion(r: Report, root: Path) -> None:
    found = set()
    for path in (root / "lab2shot").rglob("*.py"):
        if path.name == "check.py" and path.parent.name == "cli":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "rmtree(" in line and not line.lstrip().startswith("#"):
                found.add(str(path.relative_to(root)))
    extra = sorted(found - RMTREE_ALLOWED - {"lab2shot/data/packet.py"})
    for f in extra:
        r.bad(f"deletion: {f} 中出现了 rmtree：缓存条目只允许经 data/packet.py remove() 删除；删除其他文件夹时，必须将该文件登记到 RMTREE_ALLOWED 并注明原因")
    writers = set()
    for path in (root / "lab2shot").rglob("*.py"):
        if path.name == "check.py" and path.parent.name == "cli":
            continue
        if any("fresh_dir(" in line and not line.lstrip().startswith("#") for line in path.read_text(encoding="utf-8", errors="replace").splitlines()):
            writers.add(str(path.relative_to(root)))
    for f in sorted(writers - FRESH_DIR_ALLOWED):
        r.bad(f"deletion: {f} 中直接调用了 fresh_dir()：cook 之外写入缓存包只允许经 data/packet.py produce()（持有该条目的锁）")
    r.ok(f"deletion：rmtree 仅出现在已登记的 {len(RMTREE_ALLOWED) + 1} 个文件中；fresh_dir 仅出现在 cook 与 produce 中")


# A function of one of the page's own TypeScript files, run by Node on arguments given as JSON (one list of arguments per
# case): the file, and the page's own files it imports by relative path (model/places.ts uses model/math3d.ts), are
# turned into JavaScript by the page's own build tool (vite, installed by npm ci) into a temporary folder, so any Node
# the page supports runs them. Such a file must import only other pure page files (no packages).
_WEB_JS = r"""
import { readFileSync, writeFileSync, mkdtempSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve, relative } from "node:path";
import { pathToFileURL } from "node:url";
import { transformWithOxc } from "vite";
const [file, name, cases] = [process.argv[1], process.argv[2], JSON.parse(process.argv[3])];
const src = resolve(dirname(file), "..");
const out = mkdtempSync(join(tmpdir(), "l2s-web-"));
const done = new Set();
async function emit(path) {
  if (done.has(path)) return;
  done.add(path);
  const { code } = await transformWithOxc(readFileSync(path, "utf8"), path, { lang: "ts" });
  const deps = [...code.matchAll(/from\s+["'](\.{1,2}\/[^"']+)["']/g)].map((m) => m[1]);
  const fixed = code.replace(/from\s+["'](\.{1,2}\/[^"']+)["']/g, (_, p) => `from "${p}.mjs"`);
  const to = join(out, relative(src, path)).replace(/\.ts$/, ".mjs");
  const { mkdirSync } = await import("node:fs");
  mkdirSync(dirname(to), { recursive: true });
  writeFileSync(to, fixed);
  for (const d of deps) {
    const next = resolve(dirname(path), d + ".ts");
    if (existsSync(next)) await emit(next);
  }
}
await emit(resolve(file));
const mod = await import(pathToFileURL(join(out, relative(src, resolve(file)).replace(/\.ts$/, ".mjs"))).href);
console.log(JSON.stringify(cases.map((args) => mod[name](...args))));
"""


# placements the check runs: translations (cm), rotations about each axis alone and all three (degrees, also past 180
# and negative, where an order mistake shows), scales
_PLACED = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0), ((12.5, -3.0, 250.0), (0.0, 0.0, 0.0), 1.0),
           ((0.0, 0.0, 0.0), (30.0, 0.0, 0.0), 1.0), ((0.0, 0.0, 0.0), (0.0, -45.0, 0.0), 1.0),
           ((0.0, 0.0, 0.0), (0.0, 0.0, 190.0), 1.0), ((5.0, 6.0, 7.0), (10.0, 20.0, 30.0), 1.0),
           ((-100.0, 20.0, 3.5), (-75.0, 135.0, -60.0), 2.5), ((1.0, 2.0, 3.0), (89.9, 0.1, -179.0), 0.4)]


# people boxes (x1, y1, x2, y2 per frame) and clicks [frame, x, y] the check runs: overlapping boxes (the smaller wins),
# two of equal area (the smaller id wins, listed in the other order), a click on an edge, outside every box, and on
# frames where a person has no box (the nearest frame's box, equal distance: the earlier), and a person with no box
_PEOPLE = [
    {"id": 3, "boxes": {"0": [100, 100, 300, 500], "2": [120, 100, 320, 500], "4": [140, 100, 340, 500]}},
    {"id": 1, "boxes": {"0": [150, 150, 250, 350], "1": [150, 150, 250, 350]}},
    {"id": 5, "boxes": {"0": [600, 100, 700, 200], "1": [600, 100, 700, 200], "2": [600, 100, 700, 200]}},
    {"id": 2, "boxes": {"0": [650, 150, 750, 250], "1": [650, 150, 750, 250], "2": [650, 150, 750, 250]}},
    {"id": 7, "boxes": {"5": [900, 900, 1000, 1000], "9": [950, 950, 1100, 1100]}},
    {"id": 8, "boxes": {}},
]
_CLICKS = [[0, 200, 250], [0, 110, 120], [0, 300, 500], [0, 675, 175], [1, 675, 175], [0, 500, 500], [3, 200, 250],
           [3, 130, 110], [2, 200, 250], [7, 960, 960], [7, 1050, 1050], [0, 960, 960], [12, 1090, 1090], [1, 99.5, 300],
           [4, 330, 450], [6, 330, 450], [5, 1000, 1000]]


class _AllPlaces:
    """Resources for the engine cases: every node gets a place at once (no GPU, never expired)."""

    class _Ticket:
        granted, gpu, gpu_name, expired = True, "", "", None

    def ask(self, need, stop, woken):
        woken()
        return self._Ticket()

    def done(self, ticket) -> None:
        pass


def check_imports(r: Report, root: Path) -> None:
    """Every `from lab2shot... import name` in lab2shot/ names something that module has, the ones inside functions
    too: a name removed from a module is otherwise found only when that function first runs (a route that answers 500)."""
    import ast
    import importlib

    n = 0
    for f in sorted((root / "lab2shot").rglob("*.py")):
        rel = f.relative_to(root)
        module = ".".join(rel.with_suffix("").parts)
        package = module.removesuffix(".__init__") if f.name == "__init__.py" else module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if not target.startswith("lab2shot"):
                continue
            try:
                found = importlib.import_module(target)
            except Exception as exc:  # noqa: BLE001 (reported)
                r.bad(f"imports {rel}:{node.lineno}: 导入 {target} 失败：{type(exc).__name__}: {exc}")
                continue
            for alias in node.names:
                n += 1
                if alias.name == "*" or hasattr(found, alias.name):
                    continue
                try:
                    importlib.import_module(f"{target}.{alias.name}")
                except ImportError:
                    r.bad(f"imports {rel}:{node.lineno}: {target} 里没有 {alias.name}")
    r.ok(f"imports：lab2shot 里 {n} 个 from … import 的名字（含函数里延迟导入的）都存在")


def check_releases(r: Report) -> None:
    """The release notes file is edited by hand at every release and read by the server only when it starts: a mistake
    in it would first show as a broken 「更新说明」 dialog, so the same reading runs here."""
    from rich.markup import escape  # the problems quote TOML, whose [[release]] the console would read as markup

    from .. import releases

    notes = releases.read()
    for p in notes.problems:
        r.bad(f"releases {releases.FILE.name}: {escape(p)}")
    if not notes.problems:
        r.ok(f"releases：{releases.FILE.name} 的 {len(notes.releases)} 个版本（最新：{notes.releases[0]['name']} {notes.releases[0]['date']}）")


def check_empties(r: Report) -> None:
    """Every output a node type's code gives empty by name (empty_packet(ctx, "port") in its class) declares
    may_be_empty: the engine and the page treat such a port so (no N-COOK-NOTHINGIN downstream, the parameter it
    drives left editable); a port given empty by a name worked out at run time is held to the same by empty_packet
    itself when the cook gives it."""
    import inspect
    import re

    from ..nodes.applies import all_outputs
    from ..nodes.registry import node_types

    gives = re.compile(r"empty_packet\(\s*[\w.]+\s*,\s*[\"']([^\"']+)[\"']")
    bad, seen = [], 0
    for type_id, t in sorted(node_types().items()):
        try:
            source = inspect.getsource(t)
        except (OSError, TypeError):
            continue
        declared = {p.name: p for p in all_outputs(t)}
        for port in sorted(set(gives.findall(source))):
            seen += 1
            if port in declared and not declared[port].may_be_empty:
                bad.append(f"{type_id}.{port}")
    if bad:
        r.bad(f"empties: 这些输出在计算里会给空包却没有声明 may_be_empty：{'、'.join(bad)}")
    else:
        r.ok(f"empties：节点代码里按名字给空包的 {seen} 个输出都声明了 may_be_empty")


# ------------------------------------------------------------------ the command


CHECKS = ("official", "nodes", "workers", "categories", "messages", "templates", "channels", "deletion", "imports", "releases", "empties")


@app.command()
def check(only: str = typer.Argument("", help="仅运行指定的一项：" + "、".join(CHECKS))) -> None:
    """项目的静态不变量检查：端口与引文、节点体参数、分类、消息编号、模板、通道映射、删除入口、导入的名字都存在、更新说明。
    修改代码后应运行；耗时为秒级，不修改 work/。"""
    from ..config import ROOT

    wanted = [only] if only else list(CHECKS)
    if only and only not in CHECKS:
        console.print(f"[red]不存在此检查项：{only}[/red]（可选项：{'、'.join(CHECKS)}）")
        raise typer.Exit(2)
    r = Report()
    t0 = time.time()
    for name in wanted:
        fn = globals()[f"check_{name}"]
        try:
            fn(r, ROOT) if name in ("messages", "deletion", "imports") else fn(r)
        except Exception as exc:  # noqa: BLE001 (a check that fails to run is also reported as a problem)
            r.bad(f"{name}: 检查未能完成：{type(exc).__name__}: {exc}")
    for line in r.checked:
        console.print(f"[green]✓[/green] {line}")
    for line in r.notes:
        console.print(f"[dim]·[/dim] {line}")
    for line in r.problems:
        console.print(f"[red]✗[/red] {line}")
    console.print(f"{len(r.problems)} 个问题，用时 {time.time() - t0:.1f} 秒")
    if r.problems:
        raise typer.Exit(1)
