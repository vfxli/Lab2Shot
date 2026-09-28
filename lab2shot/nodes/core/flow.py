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
from typing import Literal

from ...data.packet import Packet, copy_packet, item_fingerprint, items_meta, items_of, packet_dir, produce
from ...data.types import ANY, ANY_LIST, element_of, is_list, type_label
from ...data.values import BOOL, FLOAT, INT, TEXT, factor, read, unit_problem, value_packet
from ...errors import Invalid
from ...messages import Msg
from ..applies import Param
from ..base import NodeDef, NodeParams, P, Port, empty_packet, typed_list
from ..expects import OneValue

# Flow nodes produce no image and never modify pixels, so none of their outputs is checked against a plate
# (data/contracts.py); the data they pass on keeps its original capture information unchanged.
NO_PICTURE = ""
BOTH = f"{ANY}|{ANY_LIST}"  # port type accepting either a single value or a list


def _block_param() -> str:
    return P("A", label="块名", group="块", affects_result=False)


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
            return Msg("E-LIST-NOITEMS", kind=type_label(data_type))
        return ""


def _item_packets(p: Packet) -> list[tuple[str, Packet]]:
    """Return a list packet's items as packets, in order (they are guaranteed to be cached by the contract check)."""
    out = []
    for name, fp in items_of(p):
        d = packet_dir(fp)
        if not Packet.exists(d):
            raise Invalid(Msg("E-CONTRACT-NOITEM", name=name))
        out.append((name, Packet.load(d)))
    return out


# ------------------------------------------------------------------ 逐项处理


class EachBegin(NodeDef):
    id = "core.each_begin"
    category = "flow"
    picture = NO_PICTURE
    inputs = (Port("list", ANY_LIST, "列表"),)
    outputs = (
        Port("item", ANY, "条目", type_from="input:list#item"),
        Port("name", TEXT, "名字"),
        Port("index", INT, "序号"),
        Port("count", INT, "条数"),
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
        """Return the per-item outputs. 名字 depends only on the item, 序号 on the item and its position, and 条数 on
        the whole list, so appending an item does not invalidate existing items or anything cooked from them."""
        from ...data.items import port_fp

        return {"name": (port_fp(item.packet, "name"), item.name),
                "index": (port_fp(item.packet, "index", item.index), item.index + 1),
                "count": (port_fp(list_packet, "count"), item.count)}

    @classmethod
    def cook(cls, ctx) -> dict:
        given = cls.item_outputs(ctx.params, ctx.item, "")
        return {port: value_packet(ctx.outputs[port], TEXT if port == "name" else INT, given[port][1])
                for port in ("name", "index", "count") if port in ctx.wanted}


class EachEnd(NodeDef):
    id = "core.each_end"
    category = "flow"
    picture = NO_PICTURE
    inputs = (Port("result", BOTH, "结果", multi=True),)
    outputs = (Port("list", ANY_LIST, "列表", type_from="input:result#list"),)
    scope_role, scope_kind = "end", "each"

    class Params(NodeParams):
        block: str = _block_param()

    @classmethod
    def scope_name(cls, params: dict) -> str:
        return params["block"]

    @classmethod
    def cook(cls, ctx) -> dict:
        items, got = list(ctx.items), list(ctx.inputs["result"])
        parts: list[tuple[str, str]] = []
        # The engine supplies one result per item and wire, grouped by wire (engine/evaluation.py wires). All wires
        # cover the same items, so the result of wire w for item i is at index w * items + i.
        wires = len(got) // len(items) if items else 0
        for i, item in enumerate(items):
            mine = [got[w * len(items) + i] for w in range(wires)]
            if len(mine) > 1:  # multiple results for one item are packed into a single scene
                parts.append((item.name, cls._packed(ctx, item.key, mine).fingerprint))
            elif is_list(mine[0].type):  # a list from a nested block is flattened and named 外层/内层
                parts += [(f"{item.name}/{name}", fp) for name, fp in items_of(mine[0])]
            else:
                parts.append((mine[0].meta.get("item_name") or item.name, mine[0].fingerprint))
        return {"list": Packet(ctx.outputs["list"], _list_type(ctx), items_meta(parts))}

    @classmethod
    def _packed(cls, ctx, key: str, packets: list[Packet]) -> Packet:
        """Pack one item's multiple results into a single scene. All results must be 3D data, since no other types
        can be combined."""
        from ...data.scene import pack

        if not all(p.type.startswith("scene") for p in packets):
            raise Invalid(Msg("E-EACH-SEVERAL", kinds=sorted({type_label(p.type) for p in packets})))
        fp = item_fingerprint(cls.id, cls.version, ctx.fingerprint, key)
        return produce(fp, lambda d: pack(packets, d).commit(cls.id))  # runs under the entry lock (data/packet.py produce)


# ------------------------------------------------------------------ 列表


class TakeOne(NodeDef):
    id = "core.take_one"
    category = "list"
    picture = NO_PICTURE
    list_role = "one"  # offered by engine/graph.py when a list is wired into a single-value port
    inputs = (Port("list", ANY_LIST, "列表"),)
    outputs = (Port("item", ANY, "条目", type_from="input:list#item"),)
    on_node = ("by", "index", "name")
    # 算法定义在算法目录（lab2shot/ops/ops.toml）：按序号或名字选取条目由 items.take_one 实现。
    # 未选中时该算法只返回事实（超出范围 / 名字不存在），由本节点决定输出哪条消息
    ops = ("items.take_one",)

    class Params(NodeParams):
        by: Literal["index", "name"] = P("index", label="按", group="条目",
                                         option_labels={"index": "序号", "name": "名字"})
        index: int = P(1, label="序号", ge=1, group="条目", applies=Param("by").one_of("index"))
        name: str = P("", label="名字", group="条目", applies=Param("by").one_of("name"))

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
                raise Invalid(Msg("E-ITEMS-NONAME", name=miss["name"], kind=type_label(ctx.input("list").type)))
            raise Invalid(Msg("E-LIST-NOINDEX", index=miss["index"], count=miss["count"]))
        return {"item": copy_packet(items[got["indices"][0]][1], ctx.outputs["item"])}


class MakeList(NodeDef):
    id = "core.make_list"
    category = "list"
    picture = NO_PICTURE
    list_role = "make"
    inputs = (Port("items", ANY, "条目", multi=True),)
    outputs = (Port("list", ANY_LIST, "列表", type_from="input:items#list"),)

    class Params(NodeParams):
        names: str = P("", label="名字", group="条目", placeholder="按顺序，逗号分开")

    @classmethod
    def cook(cls, ctx) -> dict:
        given = typed_list(ctx.params["names"])
        parts = []
        for i, packet in enumerate(ctx.inputs["items"]):
            name = given[i] if i < len(given) else (packet.meta.get("item_name") or str(i + 1))
            parts.append((name, packet.fingerprint))
        return {"list": Packet(ctx.outputs["list"], _list_type(ctx), items_meta(parts))}


class SplitItems(ItemsOnly, NodeDef):
    id = "core.split_items"
    category = "list"
    picture = NO_PICTURE
    list_role = "split"
    inputs = (Port("data", ANY, "数据"),)
    outputs = (Port("list", ANY_LIST, "列表", type_from="input:data#list"),)

    @classmethod
    def cook(cls, ctx) -> dict:
        from ...data.items import as_items

        data = ctx.input("data")
        # 拆分逻辑只在 data/items.py 中实现（as_items）：每种数据类型的拆法都在那里
        parts = as_items(data, cls.id, cls.version, ctx.each_done)
        if not parts:  # no items (e.g. no person detected): an empty list, not an error
            ctx.say("N-LIST-NOTHING", node=ctx.label, kind=type_label(data.type))
        return {"list": Packet(ctx.outputs["list"], _list_type(ctx), items_meta(parts))}


class MergeItems(ItemsOnly, NodeDef):
    id = "core.merge_items"
    # 通用的列表合并，适用于人物框、场景、跟踪点、分割图等任意列表，与「拆成列表」互逆
    category = "list"
    picture = NO_PICTURE
    inputs = (Port("list", ANY_LIST, "列表"),)
    outputs = (Port("data", ANY, "数据", type_from="input:list#item"),)

    @classmethod
    def cook(cls, ctx) -> dict:
        from ...data.items import merge

        parts = _item_packets(ctx.input("list"))
        if not parts:
            return {"data": empty_packet(ctx, "data")}
        return {"data": merge(parts, ctx.outputs["data"], ctx.each_done)}


class NameIt(NodeDef):
    id = "core.name_item"
    category = "list"
    picture = NO_PICTURE
    inputs = (Port("data", BOTH, "数据"),)
    outputs = (Port("out", BOTH, "结果", type_from="input:data"),)
    on_node = ("name",)

    class Params(NodeParams):
        name: str = P("名字", label="名字", group="名字")

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


class Switch(NodeDef):
    id = "core.switch"
    category = "flow"
    picture = NO_PICTURE
    inputs = (
        Port("condition", f"{BOOL}|{INT}", "条件", expects=(OneValue(),)),
        Port("a", BOTH, "第一路", optional=True),
        Port("b", BOTH, "第二路", optional=True),
        Port("c", BOTH, "第三路", optional=True),
        Port("d", BOTH, "第四路", optional=True),
    )
    outputs = (Port("out", BOTH, "结果", type_from="input:a,b,c,d#common"),)
    WAYS = ("a", "b", "c", "d")
    condition_input = "condition"

    @classmethod
    def chosen_inputs(cls, params: dict, condition) -> frozenset[str]:
        """Return the branch selected by the condition (engine/scopes.py: a switch): a boolean selects the first branch
        when on and the second when off; a number selects that branch, counting from 1."""
        value = read(condition).one()
        if isinstance(value, bool):
            return frozenset({cls.WAYS[0] if value else cls.WAYS[1]})
        n = int(value)
        return frozenset({cls.WAYS[n - 1]} if 1 <= n <= len(cls.WAYS) else set())

    @classmethod
    def cook(cls, ctx) -> dict:
        taken = next((p for way in cls.WAYS for p in ctx.inputs.get(way) or ()), None)
        if taken is None:
            value = read(ctx.input("condition")).one()
            raise Invalid(Msg("E-SWITCH-NOWAY", value=str(value)))
        return {"out": copy_packet(taken, ctx.outputs["out"])}


class Logic(NodeDef):
    id = "core.logic"
    category = "math"
    picture = NO_PICTURE
    inputs = (Port("values", BOOL, "布尔", multi=True),)
    outputs = (Port("value", BOOL, "结果"),)
    on_node = ("operation",)

    class Params(NodeParams):
        operation: Literal["and", "or", "not"] = P("and", label="运算", group="运算",
                                                   option_labels={"and": "与", "or": "或", "not": "非"})

    @classmethod
    def cook(cls, ctx) -> dict:
        values = [bool(read(p).one()) for p in ctx.inputs["values"]]
        how = ctx.params["operation"]
        got = (not values[0]) if how == "not" else (all(values) if how == "and" else any(values))
        return {"value": value_packet(ctx.outputs["value"], BOOL, got)}


COMPARISONS = {"gt": "大于", "ge": "不小于", "lt": "小于", "le": "不大于", "eq": "等于", "ne": "不等于"}


class Compare(NodeDef):
    id = "core.compare"
    category = "math"
    picture = NO_PICTURE
    inputs = (Port("a", "value", "甲"), Port("b", "value", "乙"))
    outputs = (Port("value", BOOL, "结果"),)
    on_node = ("operation",)

    class Params(NodeParams):
        operation: Literal["gt", "ge", "lt", "le", "eq", "ne"] = P(
            "gt", label="运算", group="运算", option_labels=COMPARISONS)

    @classmethod
    def param_refuses(cls, data_type: str, params: dict) -> Msg | None:
        """Ordering operations (大于, 小于, ...) apply only to numbers: a wire carrying 文字 or 开关 is rejected for
        them, while equality accepts any type. The problem is reported on the wire before cooking."""
        if params.get("operation") in ("eq", "ne"):
            return None
        if all(t in (BOOL, TEXT) for t in data_type.split("|")):
            return Msg("E-COMPARE-ORDER", how=COMPARISONS[params["operation"]], kind=type_label(data_type))
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
                raise Invalid(Msg("E-COMPARE-ORDER", kind=type_label(ctx.input("a").type), how=COMPARISONS[how]))
            got = (a.one() == b.one()) if how == "eq" else (a.one() != b.one())
        return {"value": value_packet(ctx.outputs["value"], BOOL, bool(got))}


def _number(value) -> float:
    """Convert a value to a number; a vector yields its length, so positions compare by distance."""
    if isinstance(value, (list, tuple)):
        return float(sum(float(v) * float(v) for v in value) ** 0.5)
    return float(value)


OPERATIONS = {"add": "加", "subtract": "减", "multiply": "乘", "divide": "除", "min": "最小", "max": "最大",
              "round": "取整", "abs": "绝对值"}
UNARY = ("round", "abs")
UNITS = {"": "无", "mm": "mm", "cm": "cm", "m": "m", "px": "px", "°": "°", "帧": "帧", "秒": "秒", "EV": "EV"}


class Math(NodeDef):
    id = "core.math"
    category = "math"
    picture = NO_PICTURE
    inputs = (Port("values", f"{FLOAT}|{INT}", "数值", multi=True),)
    outputs = (Port("value", FLOAT, "结果", unit="param:unit"),)
    on_node = ("operation", "unit")

    class Params(NodeParams):
        operation: Literal["add", "subtract", "multiply", "divide", "min", "max", "round", "abs"] = P(
            "add", label="运算", group="运算", option_labels=OPERATIONS)
        unit: Literal["", "mm", "cm", "m", "px", "°", "帧", "秒", "EV"] = P(
            "", label="单位", group="运算", option_labels=UNITS)

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
        return {"value": value_packet(ctx.outputs["value"], FLOAT, float(got), unit=want)}


# ------------------------------------------------------------------ 取信息（摘要在 data/summary.py）


class DataInfo(NodeDef):
    id = "core.data_info"
    category = "value"
    picture = NO_PICTURE
    inputs = (Port("data", BOTH, "数据"),)
    outputs = (
        Port("frames", INT, "帧数", unit="帧"),
        Port("first", INT, "首帧", unit="帧"),
        Port("last", INT, "末帧", unit="帧"),
        Port("width", INT, "宽", unit="px"),
        Port("height", INT, "高", unit="px"),
        Port("count", INT, "条数"),
        Port("value", FLOAT, "值"),
        Port("text", TEXT, "文字"),
    )
    main = "frames"

    class Params(NodeParams):
        item: str = P("", label="取哪一项", widget="choice", group="信息", choices_from=("data",), placeholder="先接上数据")

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        """Return the lines of the connected data's summary (data/summary.py), which 「取哪一项」 lists."""
        from ...data.summary import describe

        packet = (inputs.get("data") or [None])[0]
        if packet is None:
            return {"item": {"options": [], "empty": "先接上数据"}}
        lines = describe(packet).get("items") or []
        return {"item": {"options": [line["id"] for line in lines],
                         "labels": {line["id"]: line["label"] or line["text"] for line in lines},
                         "empty": "不取" if lines else "这份数据没有别的信息"}}

    @classmethod
    def cook(cls, ctx) -> dict:
        from ...data.summary import describe

        data = ctx.input("data")
        meta = data.meta
        frames = list(meta.get("frames") or ())
        given: dict[str, tuple[str, object, str]] = {
            "frames": (INT, len(frames) or None, "帧"),
            "first": (INT, frames[0] if frames else None, "帧"),
            "last": (INT, frames[-1] if frames else None, "帧"),
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


NODES = (EachBegin, EachEnd, TakeOne, MakeList, SplitItems, MergeItems, NameIt, Switch, Logic, Compare, Math, DataInfo)