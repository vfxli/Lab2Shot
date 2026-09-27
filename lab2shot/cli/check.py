"""`lab2shot check`: the project's invariants, run as code instead of trusted as comments.

Each check is one rule the code base relies on that nothing else enforces (a rule that lives only in a comment can
quietly stop being true). Run it after any change; it is fast (seconds), needs no server, no GPU, no network, and
touches nothing in work/.

    official     every third-party node's declared ports refer to real ports/params, and every upstream symbol it
                 cites is found in the cited lines (nodes/official.py)
    categories   the two category trees and the node placements read and agree (lab2shot/categories.py)
    messages     every message code the Python code uses exists in a catalogue; the page's generated catalogues
                 are up to date (tools/messages_web.py --check)
    templates    every card loads, wires only existing node types/ports/params, passes the graph checks except
                 for the inputs a user must fill, and its meta matches the rules
    channels     the channel-name mapping for pictures referenced as they are (view/frames.py channel_in_file)
    cache        dependency recording and the validity judgement of the packet cache (data/packet.py)
    deletion     cache entries are deleted in exactly one place (data/packet.py remove); every other rmtree in the
                 code is on a known list of non-cache folders
    routes       every route declares its access, admin routes are admin-only, and the page entry list holds only
                 pages that exist
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

from .base import app, console

# rmtree outside data/packet.py remove(): each is a folder that is NOT a cache entry (installer checkouts, a node's own
# scratch, uploads/deliveries the transfer layer owns, database backups, feedback bundles). A new file appearing here
# means cache folders are deleted by a second path, which the cache design forbids.
RMTREE_ALLOWED = {
    "lab2shot/cli/accounts.py", "lab2shot/database/__init__.py", "lab2shot/engine/cook.py", "lab2shot/engine/external.py",
    "lab2shot/farm/disk.py", "lab2shot/feedback.py", "lab2shot/installer/envbuild.py", "lab2shot/installer/run.py",
    "lab2shot/installer/sources.py", "lab2shot/library.py", "lab2shot/transfer/deliveries.py", "lab2shot/transfer/uploads.py",
}
# fresh_dir (a packet folder about to be written) is called under the entry's lock only: by a cook on its outputs and by
# data/packet.py produce(); anyone else writes a packet through produce()
FRESH_DIR_ALLOWED = {"lab2shot/data/packet.py", "lab2shot/engine/cook.py"}
# graph checks a fresh card is allowed to fail: the inputs a user fills before submitting
TEMPLATE_WAITS = ("B-DELIVER-NOPATH", "B-READ-", "B-IMPORT-", "B-WIRE-WAIT")
CODE = re.compile(r'"([A-Z]-[A-Z0-9]+-[A-Z0-9]+)"')


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
        try:
            for m in official.missing_symbols():
                r.bad(f"official {tid}: {m}")
        except Exception as exc:  # noqa: BLE001 (a missing cited file is also a problem)
            r.bad(f"official {tid}: 无法读取引文：{exc}")
    r.ok(f"official：{n} 个第三方节点的端口与上游引文")


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
        for t in text.entries(f):
            if t not in types:
                r.info(f"categories 节点文字 {f.parent.name}/{f.name}: 节点类型 {t} 当前不存在（扩展包未安装、未加载，或 id 已更改）")
    r.ok(f"categories：模板树 {len(cats.templates.rows())} 条、菜单树 {len(cats.menu.rows())} 条、{len(placed)} 个节点已归类、{len(text.files())} 份节点文字")


def check_messages(r: Report, root: Path) -> None:
    from ..messages import catalogue

    known = set(catalogue())
    used: dict[str, list[str]] = {}
    for folder in ("lab2shot", "adapters", "worker_sdk"):
        for path in (root / folder).rglob("*.py"):
            if path.name == "check.py" and path.parent.name == "cli":
                continue  # this file lists the prefixes and examples itself
            text = path.read_text(encoding="utf-8", errors="replace")
            for code in CODE.findall(text):
                used.setdefault(code, []).append(str(path.relative_to(root)))
    missing = {c: v for c, v in used.items() if c not in known}
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

    cases = [(["R", "G", "B", "A"], "G", "G"), (["Y"], "G", "Y"), (["R", "G"], "G", "G"), (["R", "G"], "B", "B"),
             ([], "R", "R"), (["Z"], "A", "A"), (["R", "G", "B"], "A", "A")]
    for have, name, want in cases:
        got = channel_in_file(have, name)
        if got != want:
            r.bad(f"channels: 文件 {have} 请求 {name} → {got!r}，应为 {want!r}")
    r.ok(f"channels：{len(cases)} 种通道映射")


def check_cache(r: Report) -> None:
    from ..data import packet

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "cache"
        root.mkdir()
        keep = packet.cache_root
        packet.cache_root = lambda: root  # type: ignore[assignment]
        try:
            a, b, c = (root / ("a" * 24)), (root / ("b" * 24)), (root / ("c" * 24))
            for d in (a, b, c):
                d.mkdir()
            src = Path(tmp) / "src.txt"  # outside the cache directory: a file inside the cache would be recorded as another packet
            src.write_text("x")
            # dependency recording: files as a dict, a list, or with path are all accepted; files of other packets
            # are recorded as packets; plate is soft
            deps = packet.deps_of(a, {"files": {"1": str(src)}, "plate": "p" * 24}, record=True)
            if [f["ref"] for f in deps["files"]] != [str(src)] or deps["soft"] != ["p" * 24]:
                r.bad(f"cache: deps_of 对字典写法的记录有误：{deps}")
            deps = packet.deps_of(a, {"files": [str(src), {"path": str(b / "f.exr")}]}, record=True)
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
            (c / packet.MANIFEST).write_text(json.dumps({"deps": {"files": [{"ref": str(src), "size": st.st_size, "mtime": int(st.st_mtime)}], "packets": [], "soft": []}}))
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
        finally:
            packet.cache_root = keep  # type: ignore[assignment]
    r.ok("cache：依赖记录与有效性判定")


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


def check_ops(r: Report, root: Path) -> None:
    """The operation catalogue has a single source (lab2shot/ops/ops.toml); the browser copy webui/src/ops/ops.gen.ts is
    generated from it and must be current. On a mismatch the server accepts an operation the browser cannot run, which
    at runtime silently falls back to the queue (ops/vocab.py); this check turns that into an error."""
    import subprocess
    import sys

    got = subprocess.run([sys.executable, str(root / "tools" / "gen_ops_ts.py"), "--check"], capture_output=True, text=True, cwd=root)
    if got.returncode != 0:
        r.bad(f"ops: {(got.stdout + got.stderr).strip()[:300]}")
    r.ok("ops：浏览器的算法表与 ops.toml 一致")


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


# ------------------------------------------------------------------ the command


CHECKS = ("official", "categories", "messages", "templates", "channels", "cache", "identity", "deletion", "routes", "ops")


@app.command()
def check(only: str = typer.Argument("", help="仅运行指定的一项：" + "、".join(CHECKS))) -> None:
    """项目的不变量检查：端口与引文、分类、消息编号、模板、通道映射、缓存判定、删除入口、路由权限。修改代码后应运行；耗时为秒级，不修改 work/。"""
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
            fn(r, ROOT) if name in ("messages", "deletion", "ops") else fn(r)
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
