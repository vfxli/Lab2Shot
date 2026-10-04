"""节点图的流程节点：列表、逐项处理、切换与判断。

这些节点与数据类型无关（端口按 data/types.py 的 ANY / ANY_LIST 声明，新增的数据类型自动适用），且不处理像素：

- 列表：「合成列表」将多份数据合成一个列表；「拆成列表」拆分包含多个条目的数据（人物框、场景、2D 跟踪点、分割图）；
  「取一条」取出其中一个条目；「列表合并」将列表合并回单份数据（拆分与合并统一登记在 data/items.py）；
- 逐项处理：「逐项开始」与「逐项结束」界定一个块，块内节点对每个条目各计算一次（引擎实例，engine/scopes.py）；
- 「命名」为数据命名：场景改为 /shot/<名字>，列表按「名字_序号」逐条命名，其他类型将名字记录在数据上，
  供「逐项结束」和写出节点使用；
- 「切换」按条件只计算其中一路；「与或非」「比较」「数学」「取信息」提供其条件与数值。

条目数据不会被复制：列表数据包只记录名字与指纹（data/packet.py），「逐项开始」输出的「条目」即条目本身的数据包。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

from ..port import EITHER  # containers pass either kind on as it is
from ...data.packet import Packet, copy_packet, item_fingerprint, items_meta, items_of, packet_dir, produce
from ...data.types import ANY, ANY_LIST, element_of, is_list, type_label
from ...data.values import BOOL, FLOAT, INT, TEXT, factor, read, unit_problem, value_meta, value_packet
from ...errors import Invalid
from ... import i18n
from ...messages import Msg
from ..applies import Param
from ..base import PARAM, NodeDef, NodeParams, P, Port, empty_packet, typed_list
from ..expects import OneValue

# Flow nodes produce no image and never modify pixels, so none of their outputs is checked against a plate
# (data/contracts.py); the data they pass on keeps its original capture information unchanged.
NO_PICTURE = ""
BOTH = f"{ANY}|{ANY_LIST}"  # port type accepting either a single value or a list


def _block_param() -> str:
    return P("A", group="block", affects_result=False)


def _list_type(ctx, port: str = "list") -> str:
    """Return the list type of an output port as resolved for this graph (for example 图像序列[])."""
    return ctx.output_types[port]


class ItemsOnly:
    """Mixin for the nodes that split data and merge it back (「拆成列表」, 「列表合并」). Both operate only on types
    that contain multiple items; data/items.py is the single place that defines which types these are."""

    @classmethod
    def refuses(cls, data_type: str, kinds: frozenset[str] | None = None):
        from ...data.items import kind_of

        if all(kind_of(element_of(t)) is None for t in data_type.split("|")):
            return Msg("E-LIST-NOITEMS", kind=i18n.Both.of(lambda: type_label(data_type)))
        return ""


def _item_packets(p: Packet) -> list[tuple[str, Packet]]:
    """Return a list packet's items as packets, in order (they are guaranteed to be cached by the contract check, and
    the list was taken as an input only when valid, which covers the packets it references: Packet.exists, whether the
    manifest can be read, is all that is left to ask; engine/presence.py says which test is for what)."""
    out = []
    for name, fp in items_of(p):
        d = packet_dir(fp)
        if not Packet.exists(d):
            raise Invalid(Msg("E-CONTRACT-NOITEM", name=name))
        out.append((name, Packet.load(d)))
    return out


# ------------------------------------------------------------------ 逐项处理


class EachBegin(NodeDef):
    id = "foreach_begin"
    category = "flow"
    picture = NO_PICTURE
    inputs = (Port("list", ANY_LIST, data=EITHER),)
    outputs = (
        Port("item", ANY, type_from="input:list#item"),
        Port("name", TEXT),
        Port("index", INT),
        Port("count", INT),
    )
    main = "item"
    # read by engine/scopes.py: this node begins a 逐项处理 block
    scope_role, scope_kind = "begin", "each"
    item_input, item_output = "list", "item"

    class Params(NodeParams):
        block: str = _block_param()

    @classmethod
    def scope_name(cls, params: dict) -> str:
        return params["block"]

    @classmethod
    def scope_items(cls, params: dict, packet):
        from ...data.items import Item

        return [Item(name, fp) for name, fp in items_of(packet)]

    @classmethod
    def item_outputs(cls, params: dict, item, list_packet: str) -> dict:
        """Return the per-item outputs. 名字 depends only on the item (its key: name and packet, so the same data under
        two names gives two names, never the one cooked first), 序号 on the item and its position, and 条数 on the whole
        list, so appending an item does not invalidate existing items or anything cooked from them."""
        from ...data.items import port_fp

        return {"name": (port_fp(item.key, "name"), item.name),
                "index": (port_fp(item.packet, "index", item.index), item.index + 1),
                "count": (port_fp(list_packet, "count"), item.count)}

    @classmethod
    def cook(cls, ctx) -> dict:
        given = cls.item_outputs(ctx.params, ctx.item, "")
        return {port: value_packet(ctx.outputs[port], TEXT if port == "name" else INT, given[port][1])
                for port in ("name", "index", "count") if port in ctx.wanted}


class EachEnd(NodeDef):
    id = "foreach_end"
    category = "flow"
    picture = NO_PICTURE
    inputs = (Port("result", BOTH, multi=True, data=EITHER),)
    outputs = (Port("list", ANY_LIST, type_from="input:result#list"),)
    scope_role, scope_kind = "end", "each"

    class Params(NodeParams):
        block: str = _block_param()

    @classmethod
    def scope_name(cls, params: dict) -> str:
        return params["block"]

    @classmethod
    def cook(cls, ctx) -> dict:
        parts: list[tuple[str, str]] = []
        for item in ctx.items:  # each item's results, one per wire, paired by the engine (CookContext.item_results)
            mine = list(ctx.item_results.get(item.key, ()))
            if not mine:
                continue
            if len(mine) > 1:  # multiple results for one item are packed into a single scene
                parts.append((item.name, cls._packed(ctx, item.key, mine).fingerprint))
            elif is_list(mine[0].type):  # a list from a nested block is flattened and named 外层/内层
                parts += [(f"{item.name}/{name}", fp) for name, fp in items_of(mine[0])]
            else:
                parts.append((mine[0].meta.get("item_name") or item.name, mine[0].fingerprint))
        return {"list": Packet(ctx.outputs["list"], _list_type(ctx), items_meta(parts))}

    @classmethod
    def several_refused(cls, port: str, types: tuple[str, ...]) -> Msg | None:
        """Several results of one item pack into a single scene: all of them must be 3D data (NodeDef.several_refused).
        A type the graph has open (「图像|遮罩」) counts as 3D data while any of what it may be is."""
        if len(types) > 1 and not all(any(part.startswith("scene") for part in t.split("|")) for t in types):
            return Msg("B-EACH-SEVERAL", kinds=sorted({type_label(t) for t in types}))
        return None

    @classmethod
    def _packed(cls, ctx, key: str, packets: list[Packet]) -> Packet:
        """Pack one item's multiple results into a single scene. All results must be 3D data, since no other types
        can be combined (several_refused, the rule the graph checks before a cook)."""
        from ...data.scene import pack

        if (said := cls.several_refused("result", tuple(p.type for p in packets))) is not None:
            raise Invalid(said)
        fp = item_fingerprint(cls.id, cls.version, ctx.fingerprint, key)
        return produce(fp, lambda d: pack(packets, d).commit(cls.id))  # runs under the entry lock (data/packet.py produce)


# ------------------------------------------------------------------ 列表


class TakeOne(NodeDef):
    id = "select_item"
    category = "list"
    picture = NO_PICTURE
    list_role = "one"  # offered by engine/graph.py when a list is wired into a single-value port
    # a list with nothing in it as expected (a block's end over items that each gave nothing as expected): no item to
    # take, it gives nothing in turn, quietly (engine/cook.py _context); one that should have had items is said as ever
    inputs = (Port("list", ANY_LIST, data=EITHER, takes_empty=False),)
    outputs = (Port("item", ANY, type_from="input:list#item"),)
    on_node = ("by", "index", "name")
    # 算法定义在算法目录（lab2shot/ops/ops.toml）：按序号或名字选取条目由 items.take_one 实现。
    # 未选中时该算法只返回事实（超出范围 / 名字不存在），由本节点决定输出哪条消息
    ops = ("items.take_one",)

    class Params(NodeParams):
        by: Literal["index", "name"] = P("index", group="items")
        index: int = P(1, ge=1, group="items", applies=Param("by").one_of("index"))
        name: str = P("", group="items", applies=Param("by").one_of("name"))

    @classmethod
    def cook(cls, ctx) -> dict:
        from ...ops import run as run_op

        items = _item_packets(ctx.input("list"))
        got = run_op("items.take_one", {"items": [{"name": name} for name, _ in items],
                                        "rule": ctx.params["by"], "index": ctx.params["index"],
                                        "name": ctx.params["name"]})
        if not got["indices"]:  # 未选中：依据返回的事实输出消息（消息定义在消息目录，不在算法中）
            miss = got["missed"][0]
            if miss["rule"] == "name":
                raise Invalid(Msg("E-ITEMS-NONAME", name=miss["name"], kind=i18n.Both.of(lambda: type_label(ctx.input("list").type))))
            raise Invalid(Msg("E-LIST-NOINDEX", index=miss["index"], count=miss["count"]))
        return {"item": copy_packet(items[got["indices"][0]][1], ctx.outputs["item"])}


class MakeList(NodeDef):
    id = "make_list"
    category = "list"
    picture = NO_PICTURE
    list_role = "make"
    inputs = (Port("items", ANY, multi=True, data=EITHER),)
    outputs = (Port("list", ANY_LIST, type_from="input:items#list"),)

    class Params(NodeParams):
        names: str = P("", group="items")

    @classmethod
    def wiring_notes(cls, params: dict, wires: dict[str, int]) -> list[tuple[Msg, str]]:
        """More names than items: the ones past the last item name nothing (NodeDef.wiring_notes)."""
        given, n = typed_list(params.get("names") or ""), wires.get("items", 0)
        return [(Msg("W-LIST-MORENAMES", names=len(given), count=n, extra=given[n:]), "items")] if len(given) > n > 0 else []

    @classmethod
    def cook(cls, ctx) -> dict:
        given = typed_list(ctx.params["names"])
        parts = []
        for i, packet in enumerate(ctx.inputs["items"]):
            name = given[i] if i < len(given) else (packet.meta.get("item_name") or str(i + 1))
            parts.append((name, packet.fingerprint))
        return {"list": Packet(ctx.outputs["list"], _list_type(ctx), items_meta(parts))}


class SplitItems(ItemsOnly, NodeDef):
    id = "split_items"
    category = "list"
    picture = NO_PICTURE
    list_role = "split"
    inputs = (Port("data", ANY, data=EITHER),)
    outputs = (Port("list", ANY_LIST, type_from="input:data#list"),)

    @classmethod
    def cook(cls, ctx) -> dict:
        from ...data.items import as_items

        data = ctx.input("data")
        # 拆分逻辑只在 data/items.py 中实现（as_items）：每种数据类型的拆法都在那里
        parts = as_items(data, cls.id, cls.version, ctx.each_done)
        if not parts:  # no items (e.g. no person detected): an empty list, not an error
            ctx.say("N-LIST-NOTHING", node=ctx.label, kind=i18n.Both.of(lambda: type_label(data.type)))
        return {"list": Packet(ctx.outputs["list"], _list_type(ctx), items_meta(parts))}


class MergeItems(ItemsOnly, NodeDef):
    id = "merge_items"
    # 通用的列表合并，适用于人物框、场景、跟踪点、分割图等任意列表，与「拆成列表」互逆
    category = "list"
    picture = NO_PICTURE
    inputs = (Port("list", ANY_LIST, data=EITHER),)
    outputs = (Port("data", ANY, type_from="input:list#item", may_be_empty=True),)

    @classmethod
    def cook(cls, ctx) -> dict:
        from ...data.items import merge

        parts = _item_packets(ctx.input("list"))
        if not parts:
            return {"data": empty_packet(ctx, "data")}
        return {"data": merge(parts, ctx.outputs["data"], ctx.each_done)}


class NameIt(NodeDef):
    id = "name"
    category = "list"
    picture = NO_PICTURE
    inputs = (Port("data", BOTH, data=EITHER),)
    outputs = (Port("out", BOTH, type_from="input:data"),)
    on_node = ("name",)

    class Params(NodeParams):
        name: str = P("name", group="name")

    @classmethod
    def cook(cls, ctx) -> dict:
        name = str(ctx.params["name"]).strip()
        if not name:
            raise Invalid(Msg("E-NAME-EMPTY"))
        data = ctx.input("data")
        out = ctx.outputs["out"]
        if is_list(data.type):  # name each item as 名字_序号
            parts = [(f"{name}_{i + 1}", fp) for i, (_old, fp) in enumerate(items_of(data))]
            return {"out": Packet(out, data.type, items_meta(parts))}
        if data.type.startswith("scene"):
            from ...data.scene import as_group

            return {"out": as_group(data, name, out)}
        made = copy_packet(data, out)
        made.meta["item_name"] = name
        return {"out": made}


# ------------------------------------------------------------------ 切换和判断


# 「切换」's ways: rows of its 「各路」 table, one input each, a…j in the order they are added (a graph's wires to a, b, c
# keep their meaning however many ways it has), named by their ports' words (node.switch.port.<a…j>.label: 第一路…第十路)
# while a row's own label is empty; two to begin with, ten at most
WAY_NAMES = tuple("abcdefghij")


class SwitchWay(NodeParams):
    """「切换」「各路」表中的一行＝一路：其输入口（a、b、c……，稳定不变，连线和节点图文件引用的是口）和显示名（节点上
    这一口的名字，可随时改，改名不断线）。第几路按行序数：「走哪一路」= 2 走表里第二行的口。"""

    name: Literal[WAY_NAMES] = P(..., widget="fixed")
    # a user's own name (one language, a string) or a built-in card's in both languages ({"zh": …, "en": …}, read with
    # i18n.pick in the language now: made_ports). The string last: the parameter table says its column is text
    # (nodes/params.py _fields takes the last alternative of the schema)
    label: dict[str, str] | str = P(...)


class Switch(NodeDef):
    id = "switch"
    category = "flow"
    picture = NO_PICTURE
    # one input per row of 「各路」 (nodes/base.py made_ports), added on the page with the node's 「＋」 like
    # 「多层 EXR 输出设置」's layers; the ways a graph file names are the rows it lists
    ports_from = "ways"
    ports_from_side = "inputs"
    ports_from_type = BOTH
    ports_from_names = WAY_NAMES
    # the table's name stands for every way it has (engine/graph.py _followed)
    outputs = (Port("out", BOTH, type_from="input:ways#common"),)
    # 「走哪一路」 is the switch's own parameter, its input port always on the node (wired_ports: "param:which"), the
    # port whose value makes the choice. Not wired, the parameter decides; wired, the wire does (a boolean or an
    # integer). A template exposes it as a menu with its own names for the ways (engine/templates.py).
    condition_input = PARAM + "which"
    wired_ports = ("which",)
    on_node = ("which",)

    class Params(NodeParams):
        # no range of its own: 1 to however many ways there are, which Graph.check_inputs holds it to
        # (B-SWITCH-RANGE). Wired (param_port: a boolean or an integer, nothing else gets through the wire's check): an
        # integer n is way n (0 or past the ways: B-SWITCH-WIREDRANGE); a boolean is not a number here, on is the
        # first way and off the second (chosen_inputs, the one rule)
        which: int = P(1, group="switch")
        ways: list[SwitchWay] = P(
            [{"name": n, "label": ""} for n in WAY_NAMES[:2]], widget="table", group="switch", max_length=len(WAY_NAMES), validate_default=True,
            # not in the fingerprint: which way is taken shows in the fingerprints of what comes in on it (only that
            # way's wires are the switch's inputs), and a way's 显示名 changes nothing
            affects_result=False,
        )

    @classmethod
    def param_port(cls, name: str) -> Port:
        """「走哪一路」's input also takes a boolean (on: the first way, off: the second), as the switch always has; a
        number with a unit is not a way's number (Port.plain)."""
        port = super().param_port(name)
        return replace(port, type=f"{BOOL}|{INT}", expects=(*port.expects, OneValue()), plain=True) if name == "which" else port

    @classmethod
    def made_ports(cls, params: dict) -> tuple[Port, ...]:
        """One input per way, named by the row's label in the language now (a built-in card's {zh, en}: i18n.pick)."""
        ways = [{**w, "label": i18n.pick(w.get("label"))} if isinstance(w, dict) else w for w in params.get("ways") or ()]
        return super().made_ports({**params, "ways": ways})

    @staticmethod
    def way_names(params: dict) -> list[str]:
        """Its ways' input ports, in row order: way n is the n-th row."""
        return [row["name"] for row in params.get("ways") or ()]

    @classmethod
    def chosen_inputs(cls, params: dict, condition) -> frozenset[str]:
        """Return the branch selected (engine/scopes.py: a switch): by the wire into 「走哪一路」 when there is one (a
        boolean selects the first way when on and the second when off; an integer selects that way, counting from 1:
        the wire takes nothing else, param_port), else by the parameter's own value (engine/evaluation.py passes None).
        None selected (0, a number past its ways): Graph.check_inputs says so (B-SWITCH-RANGE / B-SWITCH-WIREDRANGE)
        before anything is cooked."""
        ways = cls.way_names(params)
        value = params.get("which", 1) if condition is None else read(condition).one()
        if isinstance(value, bool):
            n = 1 if value else 2
        elif isinstance(value, int):
            n = value
        else:  # never through the wire's check; a file's own odd value: none (B-SWITCH-RANGE)
            return frozenset()
        return frozenset({ways[n - 1]} if 1 <= n <= len(ways) else set())

    @classmethod
    def cook(cls, ctx) -> dict:
        # which way is routing's answer, not the switch's (engine/routing.py taken_ports): the engine hands it that
        # way's packet and nothing else — a way not wired is refused before (B-SWITCH-NOWIRE), one that failed or gave
        # nothing never gets here (Routing.requires)
        (taken,) = [p for way in cls.way_names(ctx.params) for p in ctx.inputs.get(way) or ()]
        return {"out": copy_packet(taken, ctx.outputs["out"])}



class Gate(NodeDef):
    """「阻断」：一个输入一个输出，输出跟输入的类型（type_from）。「通过」开着原样交出接进来的数据；关着（阻断）时它的输入
    不取——和「切换」没走的那一路同一个机制（engine/routing.py taken_ports：一个能选路的节点，这里选的是「走 / 不走」），
    上游不算——它自己和只靠它的下游安静地跳过（engine/evaluation.py outcome：Outcome.blocked，「已跳过（被阻断）」，不报错
    也不出警告）；可选输入、多线输入里的一根、汇总节点的一行接着它，就当没接。

    「通过」是常驻口（wired_ports）：模板用一个布尔数值节点同时开关几个阻断（两条互斥的分支一个开一个关：另一个打开
    「反过来」）。接的是计算前就知道的值（数值节点）时，算之前就知道走不走；要先算出来的值，和「切换」的条件一样等它。"""

    id = "gate"
    category = "flow"
    picture = NO_PICTURE
    # 一个输入；阻断时它不取（chosen_inputs 给空），这与「切换」的路一样：必需输入 + 不走 = 不检查、不算、不进指纹
    inputs = (Port("data", BOTH, data=EITHER),)
    outputs = (Port("data", BOTH, type_from="input:data", may_be_empty=True),)
    condition_input = PARAM + "through"
    wired_ports = ("through",)
    on_node = ("through", "invert")
    # 阻断时它自己没有结果、也不是出错：引擎据此给它「被阻断」的结果（engine/evaluation.py），lab2shot check gates 核对
    blocks = True

    class Params(NodeParams):
        # 勾上 = 通过，不勾 = 阻断（输入不取、上游不算，只靠它的下游安静地跳过）
        through: bool = P(True, group="gate")
        # 打开后反着用：开关为开时阻断、为关时通过（两条互斥的分支接同一个开关，一个打开它）
        invert: bool = P(False, group="gate")

    @classmethod
    def param_port(cls, name: str) -> Port:
        """「通过」的口只收一个布尔值（和「切换」的「走哪一路」一样，Port.plain：带单位的数不是开关）。"""
        port = super().param_port(name)
        return replace(port, type=BOOL, expects=(*port.expects, OneValue()), plain=True) if name == "through" else port

    @classmethod
    def chosen_inputs(cls, params: dict, condition) -> frozenset[str]:
        """走不走：接了线按线上的布尔值，否则按「通过」；「反过来」翻转它。通过 = 取「数据」，阻断 = 什么都不取。"""
        value = params.get("through", True) if condition is None else read(condition).one()
        on = bool(value) != bool(params.get("invert", False))
        return frozenset({"data"}) if on else frozenset()

    @classmethod
    def cook(cls, ctx) -> dict:
        # 只在通过时算（阻断时引擎给它「被阻断」的结果，不来这里）；接进来的是空包时交出空包（输出可以为空）
        got = ctx.inputs.get("data") or ()
        if not got:
            return {"data": empty_packet(ctx, "data")}
        return {"data": copy_packet(got[0], ctx.outputs["data"])}


class Exists(NodeDef):
    """「有没有」：接进来的那根线这次有没有东西——上游是空、被「阻断」、或是没选文件的读取（以及只转手它们的「切换」、
    放行的「阻断」），就是否，其余是是。只看上游在不在，不取上游的内容（它的输入不取，上游不为它计算）：答案在计划阶段
    就有（engine/evaluation.py present，写进它的「有」参数），所以接到「切换」的「走哪一路」/「阻断」的「通过」上，提交前
    页面就知道走哪条路——「上传了参考帧就补帧，没上传就纯生成」这一类卡靠它。"""

    id = "has_data"
    category = "flow"
    picture = NO_PICTURE
    inputs = (Port("data", BOTH, optional=True, data=EITHER),)
    outputs = (Port("value", BOOL),)
    # 引擎读这一口的「在不在」（engine/routing.py 不取它，engine/evaluation.py params 填「有」）
    presence_of = "data"

    class Params(NodeParams):
        # 引擎在计划阶段填（上游在不在），不由人填：不进参数面板；它进指纹，上游从有到没有时下游跟着重算
        present: bool = P(False, group="presence", panel=False)

    @classmethod
    def known_outputs(cls, params):
        return {"value": value_meta(BOOL, bool(params.get("present")))}

    @classmethod
    def cook(cls, ctx) -> dict:
        return {"value": value_packet(ctx.outputs["value"], BOOL, bool(ctx.params["present"]))}

class Logic(NodeDef):
    id = "logic"
    category = "math"
    picture = NO_PICTURE
    inputs = (Port("values", BOOL, multi=True),)
    outputs = (Port("value", BOOL),)
    on_node = ("operation",)

    class Params(NodeParams):
        operation: Literal["and", "or", "not"] = P("and", group="operation")

    @classmethod
    def wiring_notes(cls, params: dict, wires: dict[str, int]) -> list[tuple[Msg, str]]:
        """非 reads the first wire alone: the others are said to do nothing (NodeDef.wiring_notes)."""
        n = wires.get("values", 0)
        return [(Msg("W-LOGIC-NOTONE", count=n), "values")] if params.get("operation") == "not" and n > 1 else []

    @classmethod
    def cook(cls, ctx) -> dict:
        values = [bool(read(p).one()) for p in ctx.inputs["values"]]
        how = ctx.params["operation"]
        got = (not values[0]) if how == "not" else (all(values) if how == "and" else any(values))
        return {"value": value_packet(ctx.outputs["value"], BOOL, got)}


COMPARISONS = i18n.Words("compare.op.", ("gt", "ge", "lt", "le", "eq", "ne"))


class Compare(NodeDef):
    id = "compare"
    category = "math"
    picture = NO_PICTURE
    inputs = (Port("a", "value"), Port("b", "value"))
    outputs = (Port("value", BOOL),)
    on_node = ("operation",)

    class Params(NodeParams):
        operation: Literal["gt", "ge", "lt", "le", "eq", "ne"] = P(
            "gt", group="operation")

    @classmethod
    def param_refuses(cls, data_type: str, params: dict) -> Msg | None:
        """Ordering operations (大于, 小于, ...) apply only to numbers: a wire carrying 文字 or 开关 is rejected for
        them, while equality accepts any type. The problem is reported on the wire before cooking."""
        if params.get("operation") in ("eq", "ne"):
            return None
        if all(t in (BOOL, TEXT) for t in data_type.split("|")):
            return Msg("E-COMPARE-ORDER", how=COMPARISONS[params["operation"]], kind=i18n.Both.of(lambda: type_label(data_type)))
        return None

    @classmethod
    def cook(cls, ctx) -> dict:
        a, b = read(ctx.input("a")), read(ctx.input("b"))
        how = ctx.params["operation"]
        if a.numeric and b.numeric:
            if (why := unit_problem(b.unit, a.unit)) is not None:
                raise Invalid(Msg("E-COMPARE-UNIT", reason=why))
            x, y = float(_number(a.one())), float(_number(b.one())) * factor(b.unit, a.unit)
            got = {"gt": x > y, "ge": x >= y, "lt": x < y, "le": x <= y,
                   "eq": math.isclose(x, y), "ne": not math.isclose(x, y)}[how]
        else:
            if how not in ("eq", "ne"):
                raise Invalid(Msg("E-COMPARE-ORDER", kind=i18n.Both.of(lambda: type_label(ctx.input("a").type)), how=COMPARISONS[how]))
            got = (a.one() == b.one()) if how == "eq" else (a.one() != b.one())
        return {"value": value_packet(ctx.outputs["value"], BOOL, bool(got))}


def _number(value) -> float:
    """Convert a value to a number; a vector yields its length, so positions compare by distance."""
    if isinstance(value, (list, tuple)):
        return float(sum(float(v) * float(v) for v in value) ** 0.5)
    return float(value)


UNARY = ("round", "abs")


class Math(NodeDef):
    id = "math"
    version = 2  # 2: the result also as an integer (rounded to the nearest), for an integer parameter
    category = "math"
    picture = NO_PICTURE
    inputs = (Port("values", f"{FLOAT}|{INT}", multi=True),)
    # the result, and the same rounded to the nearest whole number: a wire into an integer parameter (a radius in px, a
    # frame count) takes only an integer, so a number worked out on a card reaches one through `integer`
    outputs = (Port("value", FLOAT, unit="param:unit"), Port("integer", INT, unit="param:unit"))
    on_node = ("operation", "unit")

    class Params(NodeParams):
        operation: Literal["add", "subtract", "multiply", "divide", "min", "max", "round", "abs"] = P(
            "add", group="operation")
        unit: Literal["", "mm", "cm", "m", "px", "°", "frame", "s", "EV"] = P(
            "", group="operation")

    @classmethod
    def cook(cls, ctx) -> dict:
        want = ctx.params["unit"]
        numbers = []
        for p in ctx.inputs["values"]:
            v = read(p)
            if (why := unit_problem(v.unit, want)) is not None:
                raise Invalid(Msg("E-MATH-UNIT", reason=why))
            numbers.append(float(_number(v.one())) * factor(v.unit, want))
        how = ctx.params["operation"]
        if how in UNARY:
            got = round(numbers[0]) if how == "round" else abs(numbers[0])
        elif how == "add":
            got = sum(numbers)
        elif how == "subtract":
            got = numbers[0] - sum(numbers[1:])
        elif how == "multiply":
            got = math.prod(numbers)
        elif how == "divide":
            if any(n == 0 for n in numbers[1:]):
                raise Invalid(Msg("E-MATH-ZERO"))
            got = numbers[0] / math.prod(numbers[1:]) if len(numbers) > 1 else numbers[0]
        else:
            got = min(numbers) if how == "min" else max(numbers)
        return {"value": value_packet(ctx.outputs["value"], FLOAT, float(got), unit=want),
                "integer": value_packet(ctx.outputs["integer"], INT, int(math.floor(got + 0.5)), unit=want)}


# ------------------------------------------------------------------ 取信息（摘要在 data/summary.py）


class DataInfo(NodeDef):
    id = "data_info"
    category = "value"
    picture = NO_PICTURE
    inputs = (Port("data", BOTH, data=EITHER),)
    outputs = (
        Port("frames", INT, unit="frame", may_be_empty=True),
        Port("first", INT, unit="frame", may_be_empty=True),
        Port("last", INT, unit="frame", may_be_empty=True),
        Port("width", INT, unit="px", may_be_empty=True),
        Port("height", INT, unit="px", may_be_empty=True),
        Port("count", INT, may_be_empty=True),
        Port("value", FLOAT, may_be_empty=True),
        Port("text", TEXT, may_be_empty=True),
    )
    main = "frames"

    class Params(NodeParams):
        item: str = P("", widget="choice", group="info", choices_from=("data",))

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        """Return the lines of the connected data's summary (data/summary.py), which 「取哪一项」 lists."""
        from ...data.summary import describe

        packet = (inputs.get("data") or [None])[0]
        if packet is None:
            return {"item": {"options": [], "empty": i18n.t("choices.wire_data")}}
        lines = describe(packet).get("items") or []
        return {"item": {"options": [line["id"] for line in lines],
                         "labels": {line["id"]: line["label"] or line["text"] for line in lines},
                         "empty": i18n.t("choices.take_none" if lines else "choices.no_more_info")}}

    @classmethod
    def cook(cls, ctx) -> dict:
        from ...data.summary import describe

        data = ctx.input("data")
        meta = data.meta
        frames = list(meta.get("frames") or ())
        given: dict[str, tuple[str, object, str]] = {
            "frames": (INT, len(frames) or None, "frame"),
            "first": (INT, frames[0] if frames else None, "frame"),
            "last": (INT, frames[-1] if frames else None, "frame"),
            "width": (INT, meta.get("width"), "px"),
            "height": (INT, meta.get("height"), "px"),
            "count": (INT, len(items_of(data)) if is_list(data.type) else None, ""),
        }
        out = {}
        for port, (kind, value, unit) in given.items():
            if port not in ctx.wanted:
                continue
            out[port] = (value_packet(ctx.outputs[port], kind, int(value) if kind == INT else float(value), unit=unit)
                         if value is not None else empty_packet(ctx, port))
        line = next((x for x in describe(data).get("items") or () if x["id"] == ctx.params["item"]), None)
        if "text" in ctx.wanted:
            out["text"] = (value_packet(ctx.outputs["text"], TEXT, line["text"]) if line is not None
                           else empty_packet(ctx, "text"))
        if "value" in ctx.wanted:
            number = _one_number(line["value"]) if line is not None else None
            out["value"] = (value_packet(ctx.outputs["value"], FLOAT, number) if number is not None
                            else empty_packet(ctx, "value"))
        return out


def _one_number(value) -> float | None:
    """Return the number held by a summary line (「Focal Length：35 mm」, 「人物：3 个」), or None if it holds none."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        numbers = [v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(numbers) == 1:
            return float(numbers[0])
        for key in ("count", "value"):
            got = value.get(key)
            try:
                return float(got)
            except (TypeError, ValueError):
                continue
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


NODES = (EachBegin, EachEnd, TakeOne, MakeList, SplitItems, MergeItems, NameIt, Switch, Gate, Exists, Logic, Compare, Math, DataInfo)