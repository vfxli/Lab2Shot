"""`lab2shot check`: the project's invariants, run as code instead of trusted as comments.

Each check is one rule the code base relies on that nothing else enforces (a rule that lives only in a comment can
quietly stop being true). Run it after any change; it is fast (seconds), needs no server, no GPU, no network, and
touches nothing in work/.

    official     every third-party node's declared ports refer to real ports/params, and every upstream symbol it
                 cites is found in the cited lines (nodes/official.py)
    nodes        every node shows at most ON_NODE_MAX parameters on its body, each a real parameter (nodes/params.py)
    categories   the two category trees and the node placements read and agree (lab2shot/categories.py)
    messages     every message code the Python code uses exists in a catalogue; the page's generated catalogues
                 are up to date (tools/messages_web.py --check); every .short fits a node's bottom line
    templates    every card loads, wires only existing node types/ports/params, passes the graph checks except
                 for the inputs a user must fill, and its meta matches the rules
    channels     the channel-name mapping for pictures referenced as they are (view/frames.py channel_in_file)
    cache        dependency recording and the validity judgement of the packet cache (data/packet.py), and that
                 every account's cache is its own
    identity     an address naming the current generation is cached long, one naming another is 404, one naming
                 none is not cached long (server/wire.py versioned)
    deletion     cache entries are deleted in exactly one place (data/packet.py remove); every other rmtree in the
                 code is on a known list of non-cache folders
    routes       every route declares its access, admin routes are admin-only, and the page entry list holds only
                 pages that exist
    places       the viewer's placement matrix (webui/src/model/places.ts) equals the cook's (nodes/handles.py Places)
    people       the viewer's person-pick rule (webui/src/model/people.ts) picks whom the cook's (ops at_point) picks
    releases     the release notes (CHANGELOG.toml) read: an https address, every version complete, newest first,
                 no name twice (lab2shot/releases.py)
"""

from __future__ import annotations

import json
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
    each names a real parameter of the node (a typo would silently show nothing)."""
    from ..nodes.params import ON_NODE_MAX
    from ..nodes.registry import node_types

    for tid, node in sorted(node_types().items()):
        params = {p["name"] for p in node.param_specs()}
        if len(node.on_node) > ON_NODE_MAX:
            r.bad(f"on_node {tid}: 节点体上放了 {len(node.on_node)} 个参数，超过 {ON_NODE_MAX}（{'、'.join(node.on_node)}）")
        if unknown := [p for p in node.on_node if p not in params]:
            r.bad(f"on_node {tid}: {unknown} 不是该节点的参数")
    r.ok(f"on_node：{len(node_types())} 个节点体上的参数都不超过 {ON_NODE_MAX} 个")


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

    for p in text.problems():
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

    types = node_types()
    cards = T.templates()
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
        if t.get("deliverable") and not t.get("category"):
            r.bad(f"templates {name}: meta.deliverable={t.get('deliverable')!r}，模板树中没有该分类（卡将显示在「未分类」中）")
    loose = [t["name"] for t in cards if not t.get("deliverable")]
    if loose:
        r.info(f"templates: {len(loose)} 张卡未分类：{'、'.join(loose[:6])}{'……' if len(loose) > 6 else ''}")
    r.ok(f"templates：{len(cards)} 张卡的节点、参数、连线与分类")


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


def check_cache(r: Report) -> None:
    from ..data import packet
    from ..data.store import Store, using
    from ..serving import Account, serving

    # a scratch store (never the work folder), as account 1: every cache is an account's own (data/store.py)
    with tempfile.TemporaryDirectory() as tmp, using(Store(Path(tmp) / "work", Path(tmp) / "data")), serving(Account(1)):
        root = packet.cache_root()
        root.mkdir(parents=True)
        a, b, c = (root / ("a" * 24)), (root / ("b" * 24)), (root / ("c" * 24))
        for d in (a, b, c):
            d.mkdir()
        src = Path(tmp) / "src.txt"  # outside the cache directory: a file inside the cache would be recorded as another packet
        src.write_text("x")
        # dependency recording: files as a dict, a list, or with path are all accepted; files of other packets
        # are recorded as packets; plate is soft. References are relative to the packet's folder (file_ref)
        src_ref = packet.file_ref(a, src)
        deps = packet.deps_of(a, {"files": {"1": src_ref}, "plate": "p" * 24}, record=True)
        if [f["ref"] for f in deps["files"]] != [src_ref] or deps["soft"] != ["p" * 24]:
            r.bad(f"cache: deps_of 对字典写法的记录有误：{deps}")
        deps = packet.deps_of(a, {"files": [src_ref, {"path": packet.file_ref(a, b / "f.exr")}]}, record=True)
        if deps["packets"] != ["b" * 24] or len(deps["files"]) != 1:
            r.bad(f"cache: deps_of 对列表写法的记录有误：{deps}")
        # validity: a dependency folder present without .complete is being written and stays valid; a missing
        # folder invalidates; a changed file invalidates
        (a / packet.COMPLETE).touch()
        (a / packet.MANIFEST).write_text(json.dumps({"deps": {"files": [], "packets": ["b" * 24], "soft": []}}))
        if not packet.check(a).ok:
            r.bad("cache: 依赖包正在写入（文件夹存在但无 .complete）时，下游被误判为作废")
        b.rmdir()
        v = packet.check(a)
        if v.ok or v.why != "packet":
            r.bad(f"cache: 依赖包缺失时应判定为 packet 作废，实际得到 {v}")
        st = src.stat()
        (c / packet.COMPLETE).touch()
        (c / packet.MANIFEST).write_text(json.dumps({"deps": {"files": [{"ref": packet.file_ref(c, src), "size": st.st_size, "mtime": int(st.st_mtime)}], "packets": [], "soft": []}}))
        if not packet.check(c).ok:
            r.bad("cache: 外部文件未变化却被判定为作废")
        src.write_text("xy")
        if packet.check(c).ok:
            r.bad("cache: 外部文件已变化（大小）却仍被判定为有效")
        if packet.check(root / ("d" * 24)).why != "missing":
            r.bad("cache: 不存在的包应判定为 missing")
        # a packet can be written, committed and read back as valid: this pins the full round trip. A structural
        # error in packet.py (e.g. a misindented Packet.commit becoming an inner function of another function)
        # passes compilation and every other check while every cook fails at commit; only this check catches it
        e = packet.fresh_dir("e" * 24)
        made = packet.Packet(e, "value.float", {"value": 1.0}).commit("check")
        if not (isinstance(made, packet.Packet) and packet.valid(e) and packet.Packet.load(e).type == "value.float"):
            r.bad("cache: 包在 commit 之后无法读回或被判定为无效")
        got = packet.produce("f" * 24, lambda d: packet.Packet(d, "value.float", {"value": 2.0}).commit("check"))
        if not (packet.valid(root / ("f" * 24)) and got.meta.get("value") == 2.0):
            r.bad("cache: produce() 未完整写入包")
        # every account has a cache of its own: another account never sees these entries, and work done for no
        # account in particular (ANYONE: the command line, a tool) has no cache at all rather than a guessed one
        with serving(Account(2)):
            if packet.cache_root() == root or packet.packet_dir("f" * 24).exists():
                r.bad("cache: 另一个账号看到了这个账号的缓存（缓存必须按账号分开）")
        from ..data.store import NoAccount
        from ..serving import ANYONE

        with serving(ANYONE):
            try:
                packet.cache_root()
                r.bad("cache: 没有指明账号时不应有缓存位置（不能替它猜一个账号）")
            except NoAccount:
                pass
    r.ok("cache：依赖记录与有效性判定、按账号分开")


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
# case): the file is turned into JavaScript by the page's own build tool (vite, installed by npm ci), so any Node the
# page supports runs it. The file must import nothing (webui/src/model/places.ts, model/people.ts)
_WEB_JS = """
import { readFileSync } from "node:fs";
import { transformWithOxc } from "vite";
const [file, name, cases] = [process.argv[1], process.argv[2], JSON.parse(process.argv[3])];
const { code } = await transformWithOxc(readFileSync(file, "utf8"), file, { lang: "ts" });
const mod = await import("data:text/javascript;base64," + Buffer.from(code).toString("base64"));
console.log(JSON.stringify(cases.map((args) => mod[name](...args))));
"""


def _run_web(root: Path, file: str, name: str, cases: list) -> list | str:
    """`name` of webui/src/`file` on each case's arguments: its answers, or why it did not run (a message)."""
    import shutil

    node = shutil.which("node")
    if node is None:
        return "没有找到 node（网页构建也需要它）"
    got = subprocess.run([node, "--input-type=module", "-e", _WEB_JS, str(root / "webui" / "src" / file), name, json.dumps(cases)],
                         capture_output=True, text=True, cwd=root / "webui")
    if got.returncode != 0:
        return (got.stdout + got.stderr).strip()[:300]
    return json.loads(got.stdout)

# placements the check runs: translations (cm), rotations about each axis alone and all three (degrees, also past 180
# and negative, where an order mistake shows), scales
_PLACED = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0), ((12.5, -3.0, 250.0), (0.0, 0.0, 0.0), 1.0),
           ((0.0, 0.0, 0.0), (30.0, 0.0, 0.0), 1.0), ((0.0, 0.0, 0.0), (0.0, -45.0, 0.0), 1.0),
           ((0.0, 0.0, 0.0), (0.0, 0.0, 190.0), 1.0), ((5.0, 6.0, 7.0), (10.0, 20.0, 30.0), 1.0),
           ((-100.0, 20.0, 3.5), (-75.0, 135.0, -60.0), 2.5), ((1.0, 2.0, 3.0), (89.9, 0.1, -179.0), 0.4)]


def check_places(r: Report, root: Path) -> None:
    """The transform handle's matrix is written twice: the cook's (nodes/handles.py Places.matrix, data/scene.py
    trs_matrix) and the viewer's, which shows what the handle places while it is dragged and before 计算
    (webui/src/model/places.ts placeMatrix). Both run here on the same parameters for every node that places what it
    gives; they must agree, or a result would land elsewhere than the display showed."""
    import numpy as np

    from ..nodes.registry import node_types

    cases, want, names = [], [], []
    for tid, t in sorted(node_types().items()):
        if t.places is None:
            continue
        pl = t.places
        for translate, rotate, scale in _PLACED:
            params = {pl.translate: list(translate), pl.rotate: list(rotate), **({pl.scale: scale} if pl.scale else {})}
            cases.append([pl.placement(), params])
            want.append(pl.matrix(params))
            names.append(f"{tid} {params}")
    got = _run_web(root, "model/places.ts", "placeMatrix", cases)
    if isinstance(got, str):
        r.bad(f"places: 视图的变换矩阵没能运行：{got}")
        return
    for name, m, flat in zip(names, want, got):
        # placeMatrix is column-major (three.js Matrix4.fromArray order)
        diff = float(np.max(np.abs(np.asarray(flat, np.float64).reshape(4, 4).T - m)))
        if diff > 1e-9:
            r.bad(f"places: {name}：视图的变换矩阵与计算的不一致（最大差 {diff:.3g}）")
    r.ok(f"places：{len(cases)} 组参数下视图与计算的变换矩阵一致（{len({n.split()[0] for n in names})} 种放置节点）")


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


def check_people(r: Report, root: Path) -> None:
    """The rule of which person a click picks is written twice: the cook's (「选人」 in 点选 mode runs ops people.select,
    rule at_point, lab2shot/ops/run.py) and the viewer's, which accepts a click, lights the person under the pointer and
    lights the picked people before 计算 (webui/src/model/people.ts personAt). Both run here on the same boxes and clicks
    and must give the same person (or none) for every click."""
    from ..ops import run as run_op

    got = _run_web(root, "model/people.ts", "personAt", [[{"people": _PEOPLE}, f, {"x": x, "y": y}] for f, x, y in _CLICKS])
    if isinstance(got, str):
        r.bad(f"people: 视图的点选规则没能运行：{got}")
        return
    for (f, x, y), web in zip(_CLICKS, got):
        picked = run_op("people.select", {"items": _PEOPLE, "rule": "at_point", "picks": [[f, x, y]]})["indices"]
        server = _PEOPLE[picked[0]]["id"] if picked else None
        if web != server:
            r.bad(f"people: 第 {f} 帧点在 ({x}, {y})：视图选 {web}，计算选 {server}")
    r.ok(f"people：{len(_CLICKS)} 次点选中视图与计算选中的人一致")


def check_identity(r: Report) -> None:
    """Address identity (server/wire.py versioned): a matching generation is immutable, a mismatching one is 404, and
    none is served without long-term caching."""
    from types import SimpleNamespace

    from ..errors import NotFound
    from ..server import wire

    p = SimpleNamespace(created="2026-09-26T10:00:00")
    if wire.versioned(p, "2026-09-26T10:00:00") is not True:
        r.bad("identity: 代次一致时应可长期缓存")
    if wire.versioned(p, "") is not False:
        r.bad("identity: 未携带代次的地址不应长期缓存")
    try:
        wire.versioned(p, "2026-01-01T00:00:00")
        r.bad("identity: 代次不一致时应返回 404")
    except NotFound:
        pass
    r.ok("identity：地址代次规则")


def check_routes(r: Report) -> None:
    from ..server import app as server_app  # noqa: F401 (registers all routes)
    from ..server import access, routes

    n = 0
    for key, acc in routes.DECLARED.items():
        n += 1
        method, _, path = key.partition(" ")
        if path.startswith("/api/admin") and acc.level != "admin":
            r.bad(f"routes: {key} 位于 /api/admin 下但不是 admin 级（{acc.level}）")
        if path.startswith("/api/") and acc.level == "page":
            r.bad(f"routes: {key} 是接口但被声明为页面")
    for page in ("/", "/admin", "/admin/x"):
        if not access.PAGE_ENTRY.match(page):
            r.bad(f"routes: 页面入口应放行 {page}")
    for page in ("/help", "/developer", "/datasets"):
        if access.PAGE_ENTRY.match(page):
            r.bad(f"routes: 页面入口仍放行已删除的 {page}")
    r.ok(f"routes：{n} 条路由的访问级别与页面入口")


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


# ------------------------------------------------------------------ the command


CHECKS = ("official", "nodes", "categories", "messages", "templates", "channels", "cache", "identity", "deletion", "routes", "places", "people", "releases")


@app.command()
def check(only: str = typer.Argument("", help="仅运行指定的一项：" + "、".join(CHECKS))) -> None:
    """项目的不变量检查：端口与引文、节点体参数、分类、消息编号、模板、通道映射、缓存判定、地址代次、删除入口、路由权限、
    变换手柄与选人在网页和计算两处的同一规则、更新说明。修改代码后应运行；耗时为秒级，不修改 work/。"""
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
            fn(r, ROOT) if name in ("messages", "deletion", "places", "people") else fn(r)
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
