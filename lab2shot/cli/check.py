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
    categories   the two category trees and the node placements read and agree; the node menu's algorithms band is the
                 templates tree (lab2shot/categories.py)
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
    gates        a node that blocks (gate) is one input, one output following it (type_from), may be empty, and a
                 route of 「take it / take nothing」; a node that collects has only optional inputs (nodes/base.py)
    formats      every output-settings node declares the format it writes (nodes/output.py Format), no name twice: a 3D
                 one's name is what a client asks for (engine/deliver_formats.py)
    reads        every reading node declares `reads` (nodes/base.py Reads) with a file parameter it has, and what it
                 gives and reads with the file are real (engine/node_tools.py, engine/external.py go by it)
    routes, layers, digests, tests  the architecture's rules and the unit tests (cli/check_arch.py)
    extensions   every extension states generative; a base and what runs in it (runs_in) agree (cli/check_extensions.py)
    dcc          the DCC plugins write nothing of the user's scene (static), their pure-Python tests pass (cli/check_dcc.py)
    tips         no hover tip in the page restates its control's own words (webui/src/ui/Button.tsx: a tip only says why
                 a control is off, the full text of a cut-off one, an icon's name, or what an action costs)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import typer
from rich.markup import escape

from lab2shot_shared.protocol import CODE, CODE_CANDIDATE

from .. import i18n
from .base import app, console
from .check_arch import check_digests, check_layers, check_routes, check_tests  # noqa: F401  (run by name: CHECKS)
from .check_dcc import check_dcc  # noqa: F401  (run by name: CHECKS)
from .check_extensions import check_extensions  # noqa: F401  (run by name: CHECKS)
from .check_naming import check_naming  # noqa: F401  (run by name: CHECKS)

# rmtree outside data/packet.py remove(): each is a folder that is NOT a cache entry (installer checkouts, a node's own
# scratch, uploads/outputs the transfer layer owns, database backups, feedback bundles). An rmtree in any other file
# means cache folders are deleted by a second path, which the cache design forbids.
RMTREE_ALLOWED = {
    "lab2shot/cli/accounts.py", "lab2shot/database/__init__.py", "lab2shot/engine/cook.py", "lab2shot/engine/external.py",
    "lab2shot/farm/disk.py", "lab2shot/site/feedback.py", "lab2shot/installer/envbuild.py", "lab2shot/installer/run.py",
    "lab2shot/installer/sources.py", "lab2shot/site/library.py", "lab2shot/transfer/outputs.py", "lab2shot/transfer/uploads.py",
    "lab2shot/transfer/tasks.py",  # a task's folder (its graph, footage links, outputs), never a cache entry
}
# fresh_dir (a packet folder about to be written) is called under the entry's lock only: by a cook on its outputs and by
# data/packet.py produce(); anyone else writes a packet through produce()
FRESH_DIR_ALLOWED = {"lab2shot/data/packet.py", "lab2shot/engine/cook.py"}
# graph checks a fresh card is allowed to fail: the inputs a user fills before submitting
TEMPLATE_WAITS = ("B-READ-", "B-IMPORT-", "B-WIRE-WAIT")
# syntax, not words: the exception word of the fixed-name table's notes (templates/_conventions.md 「TRAM 例外」, read by
# engine/conventions.py _meanings), and the full-width colon and bracket a page tip may follow its control's words with
EXCEPTION_WORD = chr(0x4F8B) + chr(0x5916)
TIP_SEPARATORS = (chr(0xFF1A), ":", chr(0xFF08), "(")


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
                r.bad(i18n.t("cli.check.official.takes", type=tid, port=repr(port)))
        for port in (*official.gives, *official.ours):
            if port not in outs:
                r.bad(i18n.t("cli.check.official.gives", type=tid, port=repr(port)))
        for p in node.inputs:
            if p.name not in official.takes:
                r.bad(i18n.t("cli.check.official.input_undeclared", type=tid, port=repr(p.name)))
        for p in node.outputs:
            if p.name not in official.gives and p.name not in official.ours:
                r.bad(i18n.t("cli.check.official.output_undeclared", type=tid, port=repr(p.name)))
        try:
            for m in official.missing_symbols():
                r.bad(f"official {tid}: {m}")
        except Exception as exc:  # noqa: BLE001 (a missing cited file is also a problem)
            r.bad(i18n.t("cli.check.official.unreadable", type=tid, error=exc))
    r.ok(i18n.t("cli.check.official.ok", count=n))


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
            r.bad(i18n.t("cli.check.nodes.on_node_many", type=tid, count=len(node.on_node), most=ON_NODE_MAX,
                         names=i18n.separator().join(node.on_node)))
        # 节点体上默认显示的可以是参数，也可以是按钮参数（「输出」的「下载」）
        if unknown := [p for p in node.on_node if p not in {s["name"] for s in node.interface_specs()}]:
            r.bad(i18n.t("cli.check.nodes.on_node_unknown", type=tid, names=unknown))
        # 按钮参数（nodes/params.py Button）：名字不和参数重、彼此不重（公开时 target 按名字找），且只有动作、没有值
        buttons = [b for b in node.interface_specs() if b["widget"] == "button"]
        names = [b["name"] for b in buttons]
        if clash := sorted(set(names) & params) + sorted({n for n in names if names.count(n) > 1}):
            r.bad(i18n.t("cli.check.nodes.button_clash", type=tid, names=clash))
        if bad := [b["name"] for b in buttons if not b.get("action") or b["default"] is not None or b["affects_result"] or b["worker"]]:
            r.bad(i18n.t("cli.check.nodes.button_value", type=tid, names=bad))
    r.ok(i18n.t("cli.check.nodes.ok", count=len(node_types()), most=ON_NODE_MAX))
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
        r.bad(i18n.t("cli.check.nodes.data_undeclared", ports=undeclared))
    if contrary:
        r.bad(i18n.t("cli.check.nodes.data_contrary", ports=contrary))
    if not undeclared and not contrary:
        r.ok(i18n.t("cli.check.nodes.data_ok"))
    # 每种数据类型（连同它的上级类型）在包的说明书表里都有一行（data/contracts.py META）：缺一行时，这种包一算出来
    # 核对就 KeyError，整个计算中断，而不是报在节点上
    from ..data.contracts import META, lineage
    from ..data.types import DATA_TYPES

    if missing := sorted({t for d in DATA_TYPES for t in lineage(d) if t not in META}):
        r.bad(i18n.t("cli.check.nodes.contracts_missing", types=missing))
    else:
        r.ok(i18n.t("cli.check.nodes.contracts_ok", count=len(DATA_TYPES)))
    # 深度与摄影机家族的「是否度量」（DepthCamera.metric）没有默认值：每个成员都要声明 True / False 或按参数的条件，
    # 忘了写会让相对尺度的方法按米交付（或反过来），且没有任何报错
    from ..availability import Cond
    from ..nodes.families.depth_camera import DepthCamera

    depth = [(tid, n) for tid, n in sorted(node_types().items()) if issubclass(n, DepthCamera)]
    unsaid = [tid for tid, n in depth if not isinstance(n.metric, (bool, Cond))]
    for tid in unsaid:
        r.bad(i18n.t("cli.check.nodes.metric_undeclared", type=tid))
    if not unsaid:
        r.ok(i18n.t("cli.check.nodes.metric_ok", count=len(depth)))


def check_workers(r: Report) -> None:
    """An extension's node runs its worker through one template: WorkerNode (prepare -> worker -> convert; families
    inherit it), or the two framework families that run workers their own way — an import (formats.py ImportNode:
    the file's scene arrays) and an output setting (nodes/output.py OutputSettings: write()). No class an extension
    defines overrides cook: a second way to run a worker would skip missing-frame handling, streaming and the
    lens record (nodes/families/base.py)."""
    from ..nodes.families.base import WorkerNode
    from ..nodes.formats import ImportNode
    from ..nodes.output import OutputSettings
    from ..nodes.registry import core_nodes, node_types

    core = {n.id for n in core_nodes()}
    count = 0
    for tid, t in sorted(node_types().items()):
        if tid in core:
            continue
        count += 1
        if not issubclass(t, (WorkerNode, ImportNode, OutputSettings)):
            r.bad(i18n.t("cli.check.workers.base", type=tid, now=t.__mro__[1].__name__))
        own = [c.__name__ for c in t.__mro__ if c.__module__.startswith("adapters.") and "cook" in vars(c)]
        if own:
            r.bad(i18n.t("cli.check.workers.cook", type=tid, classes=i18n.separator().join(own)))
    r.ok(i18n.t("cli.check.workers.ok", count=count))


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
                r.bad(i18n.t("cli.check.categories.bad_id", tree=name, id=repr(k)))
            parent = str(row.get("parent") or "")
            if parent and (parent not in rows or rows[parent].get("parent")):
                r.bad(i18n.t("cli.check.categories.parent", tree=name, id=k, parent=repr(parent)))
            if not parent and tree.sections and str(row.get("section") or "") not in tree.sections:
                r.bad(i18n.t("cli.check.categories.section", tree=name, id=k, sections=i18n.separator().join(tree.sections)))
    # the node menu's algorithms band is the templates tree itself (lab2shot/categories.py MenuTree): the tools file
    # holds the tools band only, and every row of the templates tree shows in the menu under the algorithms band
    own = cats.menu.own_rows()
    stray = sorted(k for k in own if cats.MenuTree._band(own, k) != cats.TOOLS)
    if stray:
        r.bad(i18n.t("cli.check.categories.stray", ids=i18n.separator().join(stray[:6])))
    if not cats.menu.problem():
        merged = cats.menu.rows()
        alone = cats.MenuTree._alone(cats.templates.rows())  # templates-only (「工作流」): never in the node menu
        for k in alone:
            if k in merged:
                r.bad(i18n.t("cli.check.categories.alone_in_menu", id=k))
        for tid, where in cats.nodes.rows().items():
            if where in alone:
                r.bad(i18n.t("cli.check.categories.node_in_alone", type=tid, id=where))
        for k, row in cats.templates.rows().items():
            if k in alone:
                continue
            if k not in merged or cats.MenuTree._band(merged, k) != cats.ALGORITHMS:
                r.bad(i18n.t("cli.check.categories.not_in_band", id=k))
    if problem := cats.nodes.problem():
        r.bad(f"categories nodes: {problem}")
    types = node_types()
    placed = cats.nodes.rows()
    gone = sorted(t for t in placed if t not in types)
    if gone:  # an extension not loaded now (uninstalled, or failing to load): its placing waits for it, not an error
        r.info(i18n.t("cli.check.categories.gone", count=len(gone), types=i18n.separator().join(gone[:6]),
                        more=i18n.t("cli.check.common.more") if len(gone) > 6 else ""))
    for tid, where in placed.items():
        if not cats.menu.known(where):
            r.bad(i18n.t("cli.check.categories.unknown_place", type=tid, id=repr(where)))
    loose = sorted(t for t in types if t not in placed)
    if loose:
        r.info(i18n.t("cli.check.categories.loose", count=len(loose), types=i18n.separator().join(loose[:8]),
                        more=i18n.t("cli.check.common.more") if len(loose) > 8 else ""))
    from ..nodes import text

    for p in [*text.problems(), *text.limit_problems()]:
        r.bad(i18n.t("cli.check.categories.text_problem", problem=p))
    missing = sorted(t for t, cls in types.items() if not text.entries(text.file_for(cls)).get(t, {}).get("subtitle"))
    if missing:
        r.info(i18n.t("cli.check.categories.text_unnamed", count=len(missing), types=i18n.separator().join(missing[:6]),
                        more=i18n.t("cli.check.common.more") if len(missing) > 6 else ""))
    for f in text.files():
        for t, words in text.entries(f).items():
            # the page shows a node's words as plain text (the node menu, its info card): Markdown shows as typed
            if any("**" in str(v) for v in words.values()):
                r.bad(i18n.t("cli.check.categories.text_bold", file=f"{f.parent.name}/{f.name}", type=t))
            if t not in types:
                r.info(i18n.t("cli.check.categories.text_gone", file=f"{f.parent.name}/{f.name}", type=t))
    r.ok(i18n.t("cli.check.categories.ok", templates=len(cats.templates.rows()), menu=len(cats.menu.rows()), placed=len(placed),
                files=len(text.files())))


def check_messages(r: Report, root: Path) -> None:
    from .. import i18n
    from ..messages import SHORT_WIDTH, catalogue, short_width, shorts, written

    for lang in i18n.LANGS:  # each language's own short forms (an English one not written yet is the Chinese one)
        own = {k[: -len(".short")]: v for k, v in written(lang).items() if k.endswith(".short")}
        over = {c: short_width(t) for c, t in own.items() if short_width(t) > SHORT_WIDTH}
        for code, width in sorted(over.items()):
            r.bad(i18n.t("cli.check.messages.short_wide", code=code, lang=lang, width=f"{width:g}", most=SHORT_WIDTH))
    r.ok(i18n.t("cli.check.messages.short_ok", count=len(shorts(i18n.DEFAULT)), most=SHORT_WIDTH))
    known = set(catalogue(i18n.DEFAULT))
    used: dict[str, list[str]] = {}
    for folder in ("lab2shot", "adapters", "worker_sdk"):
        for path in (root / folder).rglob("*.py"):
            if path.name == "check.py" and path.parent.name == "cli":
                continue  # this file lists the prefixes and examples itself
            text = path.read_text(encoding="utf-8", errors="replace")
            for code in CODE_CANDIDATE.findall(text):
                used.setdefault(code, []).append(str(path.relative_to(root)))
    for code, where in sorted(c for c in used.items() if not CODE.match(c[0])):
        r.bad(i18n.t("cli.check.messages.bad_code", code=code, where=where[0]))
    missing = {c: v for c, v in used.items() if c not in known and CODE.match(c)}
    for code, where in sorted(missing.items()):
        r.bad(i18n.t("cli.check.messages.no_entry", code=code, where=where[0]))
    r.ok(i18n.t("cli.check.messages.codes_ok", used=len(used), known=len(known)))
    gen = root / "tools" / "messages_web.py"
    if gen.is_file():
        out = subprocess.run([sys.executable, str(gen), "--check"], capture_output=True, text=True, cwd=root)
        if out.returncode != 0:
            r.bad(i18n.t("cli.check.messages.web_stale"))
        else:
            r.ok(i18n.t("cli.check.messages.web_ok"))
    gen = root / "tools" / "messages_client.py"
    if gen.is_file():  # the DCC client's generated words (lab2shot/client.py), both languages
        out = subprocess.run([sys.executable, str(gen), "--check"], capture_output=True, text=True, cwd=root)
        if out.returncode != 0:
            r.bad(i18n.t("cli.check.messages.client_stale"))
        else:
            r.ok(i18n.t("cli.check.messages.client_ok"))
    # the DCC panel's look, generated from the page's own tokens and icons (设计_DCC新面板.md §2): never a copy that
    # drifts. The icons need node and the page's packages: left out where there are none (a server without webui deps)
    stale = []
    gen = root / "tools" / "dcc_theme.py"
    if gen.is_file() and subprocess.run([sys.executable, str(gen), "--check"], capture_output=True, cwd=root).returncode:
        stale.append("theme.qss / tokens.json")
    gen = root / "tools" / "dcc_icons.mjs"
    if gen.is_file() and shutil.which("node") and (root / "webui" / "node_modules" / "vite").is_dir() and \
            subprocess.run(["node", str(gen), "--check"], capture_output=True, cwd=root).returncode:
        stale.append("icons/*.svg")
    if stale:
        r.bad(i18n.t("cli.check.messages.dcc_look_stale", what=", ".join(stale)))
    else:
        r.ok(i18n.t("cli.check.messages.dcc_look_ok"))


def check_i18n(r: Report, root: Path) -> None:
    """Every word in both languages' catalogues, and what is still to move held to its baseline (lab2shot/i18n/lint.py)."""
    from ..i18n import lint

    lint.run(r, root)


def check_templates(r: Report) -> None:
    from ..engine import templates as T
    from ..engine.graph import Graph
    from ..errors import MessageError
    from ..nodes.registry import node_types

    from ..engine import conventions
    from ..site.library import TEMPLATES_DIR, presets

    types = node_types()
    cards = presets()
    # the conventions for people (templates/_conventions.md), checked where a graph can tell (engine/conventions.py)
    rules = conventions.read_table(TEMPLATES_DIR / "_conventions.md") if (TEMPLATES_DIR / "_conventions.md").is_file() else None
    for said in conventions.table_problems(rules) if rules is not None else ():  # a row that would check nothing
        r.bad(f"templates: {said}")
    # how a row of several names is read: 「a / b」 against one group of targets each, else said (a sample table)
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "t.md"
        sample.write_text("| name | meaning | target | notes |\n|---|---|---|---|\n"
                          "| `a` / `b` | | `math.operation` / `value_int.value` `unit` | |\n"
                          "| `c` / `d` | | `math.operation` `value_int.value` | |\n", encoding="utf-8")
        got = conventions.read_table(sample)
        if (got["a"]["targets"], got["b"]["targets"]) != ({("math", "operation")}, {("value_int", "value"), ("value_int", "unit")}) \
                or not got["c"]["unpaired"] or got["a"]["unpaired"]:
            r.bad(i18n.t("cli.check.templates.pairs", got=got))
        # rules 1 / 2 / 6 and a menu's meanings, on a sample card: a table name passes, every other name must be the
        # one its target derives (node id_parameter; cook / download on the delivering node, cook_<id> elsewhere), a
        # terse node id is said, and a value meaning something else than the table says is said unless the notes name
        # the card as the exception
        sample.write_text("| name | meaning | target | notes |\n|---|---|---|---|\n"
                          f"| `src` | source | `value_int.value` | 1 Alpha / 2 Beta (Odd {EXCEPTION_WORD}: 2 = Gamma) |\n",
                          encoding="utf-8")
        s_table = conventions.read_table(sample)
        aux = next(t for t in types.values() if t.runtime != "core" and len(t.param_specs()) >= 2)
        a1, a2 = [p["name"] for p in aux.param_specs()][:2]
        card = {"meta": {"name": "Sample"},
                "nodes": [{"id": "choice", "type": "value_int", "params": {}}, {"id": "solve", "type": aux.id, "params": {}},
                          {"id": "m", "type": "math", "params": {}}],
                "edges": [],
                "exposed": [{"name": "src", "target": "choice.value", "widget": "menu", "options": [{"value": 1, "label": "Alpha"}, {"value": 2, "label": "Gamma"}]},
                            {"name": f"solve_{a1}", "target": f"solve.{a1}"}, {"name": a2, "target": f"solve.{a2}"},
                            {"name": "cook_solve", "target": "solve.cook"}, {"name": "m_operation", "target": "m.operation"}]}
        said = conventions.problems(card, s_table)
        odd = conventions._meanings({**card, "meta": {"name": "Odd card"}}, card["exposed"], s_table)
        wrong_name = [x for x in said if f"solve_{a2}" in x]
        terse = [x for x in said if "'m'" in x]
        meaning = [x for x in said if "Gamma" in x]
        if len(said) != 3 or not (wrong_name and terse and meaning) or odd:
            r.bad(i18n.t("cli.check.templates.derive", said=said, odd=odd))
    internal = 0  # node parameters the cards list in meta.internal: need no entry in app mode (T.route_problems)
    for t in cards:
        data = t["graph"]
        name = t["name"]
        for n in data["nodes"]:
            if n["type"] not in types:
                r.bad(i18n.t("cli.check.templates.no_type", name=name, type=n["type"]))
                continue
            specs = {p["name"] for p in types[n["type"]].param_specs()}
            for k in n.get("params", {}):
                if k not in specs:
                    r.bad(i18n.t("cli.check.templates.no_param", name=name, node=n["id"], param=k))
        try:
            g = Graph.from_json(data)
        except Exception as exc:  # noqa: BLE001
            r.bad(i18n.t("cli.check.templates.no_graph", name=name, error=exc))
            continue
        for n in data["nodes"]:
            try:
                g.check_inputs(n["id"])
            except MessageError as exc:
                if not str(getattr(exc, "code", "")).startswith(TEMPLATE_WAITS):
                    r.bad(i18n.t("cli.check.templates.node_said", name=name, node=n["id"], said=exc))
            except Exception as exc:  # noqa: BLE001
                r.bad(i18n.t("cli.check.templates.node_raised", name=name, node=n["id"], error=f"{type(exc).__name__}: {exc}"))
        for m in T.check_exposed(data):  # 参数界面（exposed 树）：结构、目标参数、控件、下拉的值、条件
            r.bad(f"templates {name}: {m.text}")
        # 每个公开选择的取值下（与 route_tags 同一枚举）都能读图、规划，路线上要使用者给的（文件、视图里点）都有公开入口
        for said in T.route_problems(data, TEMPLATE_WAITS):
            r.bad(f"templates {name}: {said}")
        for said in T.menu_routes(data):  # 驱动切换的下拉：每个取值都落在切换的某一路上
            r.bad(f"templates {name}: {said}")
        for said in conventions.problems(data, rules) if rules is not None else ():  # 对外参数名与按钮的约定
            r.bad(f"templates {name}: {said}")
        internal += len((data.get("meta") or {}).get("internal") or ())
        # the card's name and intro fit what the library takes when an administrator saves one (library.NAME_CHARS,
        # characters / TEXT_WIDTH, display width): a file written by hand is held to the same, or saving it again from
        # the page would refuse it; a built-in card's in every language it is written in
        from ..site import library

        meta = t["graph"].get("meta") or {}
        for lang in i18n.LANGS:
            text = i18n.pick(meta.get("name"), lang)
            if len(text) > library.NAME_CHARS:
                r.bad(i18n.t("cli.check.templates.too_long", name=name, what=i18n.t("cli.check.templates.what_name"),
                             count=len(text), most=library.NAME_CHARS))
            wide = library.width(i18n.pick(meta.get("intro"), lang))
            if wide > library.TEXT_WIDTH:
                r.bad(i18n.t("cli.check.templates.too_wide", name=name, what=i18n.t("cli.check.templates.what_intro"),
                             lang=lang, width=wide, most=library.TEXT_WIDTH))
        if t.get("deliverable") and not t.get("category"):
            r.bad(i18n.t("cli.check.templates.no_category", name=name, deliverable=repr(t.get("deliverable"))))
    for said in T.shared_interfaces([(t["name"], t["graph"]) for t in cards]):  # 同一块参数在各卡上一致（NodeDef.same_on_cards）
        r.bad(f"templates {said}")
    loose = [t["name"] for t in cards if not t.get("deliverable")]
    if loose:
        r.info(i18n.t("cli.check.templates.loose", count=len(loose), names=i18n.separator().join(loose[:6]),
                        more=i18n.t("cli.check.common.more") if len(loose) > 6 else ""))
    if internal:
        r.info(i18n.t("cli.check.templates.internal", count=internal))
    if rules is None:
        r.info(i18n.t("cli.check.templates.no_conventions"))
    r.ok(i18n.t("cli.check.templates.ok_conventions" if rules is not None else "cli.check.templates.ok", count=len(cards)))


def check_channels(r: Report) -> None:
    from ..view.frames import channel_in_file

    cases = [(["R", "G", "B", "A"], "G", "G"), (["Y"], "G", "Y"), (["Y", "A"], "B", "Y"), (["Y", "A"], "A", "A"),
             (["R", "G"], "G", "G"), (["R", "G"], "B", "B"),
             ([], "R", "R"), (["Z"], "A", "A"), (["R", "G", "B"], "A", "A")]
    for have, name, want in cases:
        got = channel_in_file(have, name)
        if got != want:
            r.bad(i18n.t("cli.check.channels.wrong", have=have, name=name, got=repr(got), want=repr(want)))
    r.ok(i18n.t("cli.check.channels.ok", count=len(cases)))


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
        r.bad(i18n.t("cli.check.deletion.rmtree", file=f))
    writers = set()
    for path in (root / "lab2shot").rglob("*.py"):
        if path.name == "check.py" and path.parent.name == "cli":
            continue
        if any("fresh_dir(" in line and not line.lstrip().startswith("#") for line in path.read_text(encoding="utf-8", errors="replace").splitlines()):
            writers.add(str(path.relative_to(root)))
    for f in sorted(writers - FRESH_DIR_ALLOWED):
        r.bad(i18n.t("cli.check.deletion.fresh_dir", file=f))
    r.ok(i18n.t("cli.check.deletion.ok", count=len(RMTREE_ALLOWED) + 1))


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
                r.bad(i18n.t("cli.check.imports.failed", file=rel, line=node.lineno, module=target,
                                 error=f"{type(exc).__name__}: {exc}"))
                continue
            for alias in node.names:
                n += 1
                if alias.name == "*" or hasattr(found, alias.name):
                    continue
                try:
                    importlib.import_module(f"{target}.{alias.name}")
                except ImportError:
                    r.bad(i18n.t("cli.check.imports.missing", file=rel, line=node.lineno, module=target, name=alias.name))
    r.ok(i18n.t("cli.check.imports.ok", count=n))


def check_releases(r: Report) -> None:
    """The release notes file is edited by hand at every release and read by the server only when it starts: a mistake
    in it would first show as a broken 「更新说明」 dialog, so the same reading runs here: every text in both languages,
    the lines of each version as many in English as in Chinese."""
    from rich.markup import escape  # the problems quote TOML, whose [[release]] the console would read as markup

    from .. import releases

    notes = releases.read()
    for p in notes.problems:
        r.bad(f"releases {releases.FILE.name}: {escape(p)}")
    if not notes.problems:
        r.ok(i18n.t("cli.check.releases.ok", file=releases.FILE.name, count=len(notes.releases),
                    name=i18n.pick(notes.releases[0]["name"]), date=notes.releases[0]["date"]))


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
        r.bad(i18n.t("cli.check.empties.bad", ports=i18n.separator().join(bad)))
    else:
        r.ok(i18n.t("cli.check.empties.ok", count=seen))


def check_gates(r: Report) -> None:
    """A node type that `blocks` (「阻断」, nodes/base.py) is what the engine takes it for (engine/evaluation.py _blocked,
    engine/routing.py gives_nothing): one data input and one output whose type follows that input (type_from
    "input:<it>"), declared may_be_empty; a route (condition_input + chosen_inputs) whose ways are that input or none,
    with its condition wireable (wired_ports). A node type that `collects` (多层 EXR 输出设置) has only optional inputs
    besides its parameters' (it is blocked when every one is): a required one would make it skipped, not blocked."""
    from ..engine import scopes as sc
    from ..nodes.registry import node_types

    bad: list[str] = []
    gates = collects = 0
    for tid, t in sorted(node_types().items()):
        params = {k: (f.default if not f.is_required() else None) for k, f in t.Params.model_fields.items()}
        if getattr(t, "blocks", False):
            gates += 1
            ins = [p for p in t.input_ports(params) if not p.param]
            outs = list(t.outputs)
            if len(ins) != 1 or len(outs) != 1:
                bad.append(i18n.t("cli.check.gates.one_one", type=tid, ins=len(ins), outs=len(outs)))
                continue
            if outs[0].type_from != f"input:{ins[0].name}":
                bad.append(i18n.t("cli.check.gates.type_from", type=tid, input=ins[0].name, now=repr(outs[0].type_from)))
            if not outs[0].may_be_empty:
                bad.append(i18n.t("cli.check.gates.may_be_empty", type=tid))
            if not sc.chooses(t):
                bad.append(i18n.t("cli.check.gates.route", type=tid))
                continue
            if t.condition_input.removeprefix("param:") not in t.wired_ports:
                bad.append(i18n.t("cli.check.gates.wired", type=tid, condition=t.condition_input))
            for value in (True, False):
                got = set(t.chosen_inputs({**params, t.condition_input.removeprefix("param:"): value}, None))
                if not got <= {ins[0].name}:
                    bad.append(i18n.t("cli.check.gates.chosen", type=tid, input=ins[0].name, got=sorted(got)))
        if getattr(t, "collects", False):
            collects += 1
            if req := [p.name for p in t.input_ports(params) if not p.param and not p.optional]:
                bad.append(i18n.t("cli.check.gates.collects", type=tid, inputs=i18n.separator().join(req)))
    if bad:
        r.bad("gates: " + i18n.t("cli.check.common.sep").join(bad))
    elif not gates:
        r.bad(i18n.t("cli.check.gates.none"))
    else:
        r.ok(i18n.t("cli.check.gates.ok", gates=gates, collects=collects))


def check_formats(r: Report) -> None:
    """Every output-settings node type declares its Format, and no two the same name: the 3D ones' names are the
    outside contract of POST /api/jobs formats (engine/deliver_formats.py), never worked out from an id or a label."""
    from ..nodes.output import Format, OutputSettings
    from ..nodes.registry import node_types

    seen: dict[str, str] = {}
    bad: list[str] = []
    count = 0
    for tid, t in sorted(node_types().items()):
        if not (isinstance(t, type) and issubclass(t, OutputSettings)):
            continue
        count += 1
        f = t.format
        if not isinstance(f, Format) or not f.name or not f.label:
            bad.append(i18n.t("cli.check.formats.undeclared", type=tid))
            continue
        if f.name in seen:
            bad.append(i18n.t("cli.check.formats.twice", type=tid, name=f.name, other=seen[f.name]))
        seen.setdefault(f.name, tid)
    if bad:
        r.bad("formats: " + i18n.t("cli.check.common.sep").join(bad))
    else:
        r.ok(i18n.t("cli.check.formats.ok", count=count))


def check_reads(r: Report) -> None:
    """Every reading node (one that reads the file the user picks: ReadsFile) declares `reads`, its file parameter one
    of its parameters, what it reads with the file too, and each output it says the file is read into a data type."""
    from ..data.types import DATA_TYPES
    from ..nodes.base import Reads, ReadsFile
    from ..nodes.registry import node_types

    bad: list[str] = []
    count = 0
    for tid, t in sorted(node_types().items()):
        if not issubclass(t, ReadsFile):
            continue
        count += 1
        reads = t.reads
        if not isinstance(reads, Reads):
            bad.append(i18n.t("cli.check.reads.undeclared", type=tid))
            continue
        params = set(t.Params.model_fields)
        if reads.file not in params:
            bad.append(i18n.t("cli.check.reads.file", type=tid, file=reads.file))
        if missing := [p for p in reads.goes_with if p not in params]:
            bad.append(i18n.t("cli.check.reads.goes_with", type=tid, names=i18n.separator().join(missing)))
        if wrong := [ty for _port, ty in reads.gives if ty not in DATA_TYPES]:
            bad.append(i18n.t("cli.check.reads.gives", type=tid, types=i18n.separator().join(wrong)))
    if bad:
        r.bad("reads: " + i18n.t("cli.check.common.sep").join(bad))
    else:
        r.ok(i18n.t("cli.check.reads.ok", count=count))


def _opening_tag_end(text: str, at: int) -> int:
    """The index just past the `>` closing the JSX opening tag that contains position `at` (braces skipped: an
    attribute like onClick={() => ...} holds a `>` of its own); -1 when the tag does not close."""
    depth = 0
    quote = ""
    for i in range(at, len(text)):
        c = text[i]
        if quote:
            if c == quote:
                quote = ""
        elif c in "\"'`":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == ">" and depth == 0:
            return i + 1 if text[i - 1] != "/" else -1  # a self-closing tag has no words of its own
    return -1


def restates(tip: str, words: str) -> bool:
    """A tip that is the control's own words, or starts with them and a colon or a bracket (「日志：出现过的…」)."""
    tip, words = tip.strip(), words.strip()
    return bool(words) and (tip == words or any(tip.startswith(words + sep) for sep in TIP_SEPARATORS))


def check_tips(r: Report, root: Path) -> None:
    """The page's hover tips say only what the control cannot: why it is off, the full text of a cut-off one, an exact
    value, an error, a shortcut, what a destructive action loses. Every tip says which (platform/tips.ts TipWhy): a
    component takes a Tip, never a bare text (tsc refuses one), and data-tip written anywhere but platform/tips.ts is
    refused here. A tip that restates the words on the control is noise the pointer has to wait through; this also
    finds the literal ones (a JSX element whose tip starts with its own text, an option whose tip starts with its
    label)."""
    jsx_tip = re.compile(r"\b(?:tip|data-tip)=\"([^\"]+)\"")
    typed_tip = re.compile(r"\btip=\{tipOf\(\"\w+\",\s*\"([^\"]+)\"\)\}")
    option = re.compile(r"\{[^{}]*?\blabel:\s*\"([^\"]+)\"[^{}]*?\btip:\s*(?:tipOf\(\"\w+\",\s*)?\"([^\"]+)\"[^{}]*\}")
    # a tip says why it is there (platform/tips.ts tipOf + tipAttrs, the one place the attribute is written): data-tip
    # written anywhere else is a tip without a why
    raw = re.compile(r"[\"']?\bdata-tip\b[\"']?\s*[=:]")
    found, unsaid, files = [], [], 0
    for path in sorted((root / "webui" / "src").rglob("*.tsx")) + sorted((root / "webui" / "src").rglob("*.ts")):
        if ".test." in path.name or path.name.startswith("generated"):
            continue
        files += 1
        text = path.read_text(encoding="utf-8")
        where = str(path.relative_to(root))
        if path.name != "tips.ts" or path.parent.name != "platform":
            for m in raw.finditer(text):
                unsaid.append(i18n.t("cli.check.tips.where", file=where, line=text.count(chr(10), 0, m.start()) + 1,
                                     words=text[m.start():m.start() + 40].split(chr(10))[0]))
        for m in [*jsx_tip.finditer(text), *typed_tip.finditer(text)]:
            end = _opening_tag_end(text, m.end())
            if end < 0:
                continue
            words = re.match(r"\s*([^<{]*)", text[end:]).group(1)
            if restates(m.group(1), words):
                found.append(i18n.t("cli.check.tips.where", file=where, line=text.count(chr(10), 0, m.start()) + 1,
                                          words=words.strip()))
        for m in option.finditer(text):
            if restates(m.group(2), m.group(1)):
                found.append(i18n.t("cli.check.tips.where", file=where, line=text.count(chr(10), 0, m.start()) + 1,
                                          words=m.group(1)))
    for line in found:
        r.bad(i18n.t("cli.check.tips.restates", where=line))
    for line in unsaid:
        r.bad(i18n.t("cli.check.tips.raw", where=line))
    if not found and not unsaid:
        r.ok(i18n.t("cli.check.tips.ok", count=files))


def joints_cases(cases: list[dict]):
    """The skeleton recognition regression cases (lab2shot/cli/skeleton_corpus.json, from the corpus outside the repo:
    骨架识别语料/export_check.py): every representative skeleton as it is, with its names swapped left for right
    (the positions say otherwise: the positions win), nameless (bone_N), turned 180° about its up, lying Z-up, scaled
    by 0.01, and nameless turned 90° and scaled by 100. Yields (label, names, parents, rest [J,3], truth)."""
    import numpy as np

    turn = np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, -1]])
    zup = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    quarter = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    for c in cases:
        rest = np.asarray(c["rest"], np.float64)
        nameless = [f"bone_{j}" for j in range(len(c["names"]))]
        yield f"{c['name']}", c["names"], c["parents"], rest, c["truth"]
        if c.get("swapped"):
            yield f"{c['name']} swapped", c["swapped"], c["parents"], rest, c["truth"]
        yield f"{c['name']} nameless", nameless, c["parents"], rest, c["truth"]
        yield f"{c['name']} turned", c["names"], c["parents"], rest @ turn.T, c["truth"]
        yield f"{c['name']} z-up", c["names"], c["parents"], rest @ zup.T, c["truth"]
        yield f"{c['name']} scaled 0.01", c["names"], c["parents"], rest * 0.01, c["truth"]
        yield f"{c['name']} nameless turned 90 scaled 100", nameless, c["parents"], (rest * 100) @ quarter.T, c["truth"]


def joints_score(names, parents, rest, truth) -> tuple[int, list[str], int]:
    """(parts right, parts wrong [as "part"], parts left out) of joints.guess against the truth."""
    from ..data.joints import PARTS, guess

    got = guess(list(names), parents, rest=rest)
    right, wrong, missed = 0, [], 0
    for p in PARTS:
        t, g = truth.get(p), got.get(p)
        if t is None and g is None:
            continue
        if t is not None and g is not None and t == g:
            right += 1
        elif g is None:
            missed += 1
        else:
            wrong.append(p)
    return right, wrong, missed


def check_joints(r: Report) -> None:
    """The skeleton recognition engine (data/skeleton_recognition.py, read through data/joints.py guess) on a small
    embedded corpus (lab2shot/cli/skeleton_corpus.json: UniRig's nameless zhanshi, Mixamo, UE5, MMD, VRoid, Rigify,
    Daz, AccuRIG, SMPL-X, HumanIK, a MetaHuman-sized face rig, a quadruped that is no human) and its variants
    (joints_cases): no part given wrongly, and at least as many parts recognised as when the corpus was taken."""
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).with_name("skeleton_corpus.json")).read_text(encoding="utf-8"))
    expect = data.get("expect", {})
    bad, total, cases = [], 0, 0
    for label, names, parents, rest, truth in joints_cases(data["cases"]):
        right, wrong, _ = joints_score(names, parents, rest, truth)
        cases += 1
        total += right
        if wrong:
            bad.append(i18n.t("cli.check.joints.wrong", case=label, parts=i18n.separator().join(wrong[:4])))
        elif right < expect.get(label, 0):
            bad.append(i18n.t("cli.check.joints.fewer", case=label, right=right, before=expect[label]))
    if bad:
        r.bad(i18n.t("cli.check.joints.worse", cases=i18n.t("cli.check.common.sep").join(bad[:6])))
    else:
        r.ok(i18n.t("cli.check.joints.ok", cases=cases, parts=total))


# ------------------------------------------------------------------ the command


CHECKS = ("official", "nodes", "workers", "categories", "messages", "i18n", "templates", "naming", "channels", "deletion", "imports", "releases", "empties",
          "gates", "formats", "reads", "tips", "joints", "routes", "layers", "digests", "tests",
          "extensions", "dcc")


@app.command(help=i18n.t("cli.check.help.text"))
def check(only: str = typer.Argument("", help=i18n.t("cli.check.only", names=i18n.separator().join(CHECKS)))) -> None:
    from ..config import ROOT

    wanted = [only] if only else list(CHECKS)
    if only and only not in CHECKS:
        console.print("[red]" + escape(i18n.t("cli.check.unknown", name=only, names=i18n.separator().join(CHECKS))) + "[/red]")
        raise typer.Exit(2)
    r = Report()
    t0 = time.time()
    for name in wanted:
        fn = globals()[f"check_{name}"]
        try:
            fn(r, ROOT) if name in ("messages", "i18n", "deletion", "imports", "tips", "layers", "digests", "tests", "dcc") else fn(r)
        except Exception as exc:  # noqa: BLE001 (a check that fails to run is also reported as a problem)
            r.bad(i18n.t("cli.check.failed_to_run", name=name, error=f"{type(exc).__name__}: {exc}"))
    for line in r.checked:
        console.print(f"[green]✓[/green] {line}")
    for line in r.notes:
        console.print(f"[dim]·[/dim] {line}")
    for line in r.problems:
        console.print(f"[red]✗[/red] {line}")
    console.print(i18n.t("cli.check.summary", problems=len(r.problems), seconds=time.time() - t0))
    if r.problems:
        raise typer.Exit(1)
