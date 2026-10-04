"""算法目录的词汇表：定义支持的运算种类，以及精度与取整规则。

每条算法以数据形式描述（机器可读，见 `ops.toml`），由一个通用执行器按描述执行（`lab2shot/ops/run.py`）。
算法统一定义于本目录，不属于某个节点，任何节点均可引用；只在服务器上执行，浏览器不计算任何节点。

词汇表是封闭的：`KINDS` 只包含三种运算。新增算法若属于其中之一，执行器无需修改，只需在 `ops.toml` 中增加一条描述；
新增第四种运算时先修改本词汇表，再在执行器中加一段。

适用范围：仅限纯确定性、不需要模型权重或 GPU、输入输出均为核心类型的基础算法（选人、选取条目、框转遮罩、图像合成）。
解算器、格式读写与 USD 不属于此层，第三方 adapter 不涉及此层。

## 精度与取整规则

1. 统一使用 float32：执行器的数组 dtype 为 float32。
2. 不隐式钳位：仅当描述中写明 `clamp(...)` 时才钳位。
   （「图像合成」不钳位，相加结果可超过 1，此为预期行为；「人物框转遮罩」的钳位写在其描述中。）
3. 不隐式取整：三种运算均不包含取整；需要取整的算法须在描述中显式写出。
"""

from __future__ import annotations

import re

# ------------------------------------------------------------------ 三种运算（词汇表本体）

KINDS: dict[str, dict] = {
    # 逐像素算术：对每个输入像素计算一条公式。变量为输入名（a、b），
    # 通道数较少的一方自动广播到通道数较多的一方（例如单通道遮罩作用于三通道图像）
    "pixel": {"desc": "per-pixel arithmetic", "fields": {"expr": "expr"}},
    # 按框涂：将若干矩形绘制到画布上。`axis` 定义单条轴上像素被框覆盖的程度
    # （变量：p 为像素坐标，lo/hi 为框在该轴上的两端），`combine` 将两条轴合成为像素覆盖率
    # （变量：x、y），`accumulate` 定义多个框的叠加方式
    "boxes.paint": {"desc": "paint by boxes", "fields": {"axis": "expr", "combine": "expr", "accumulate": "enum:accumulate", "base": "number"}},
    # 挑条目：按一条规则从有序条目表中选出若干条，返回其位置而非内容。
    # 未能选中的情况（编号越界、名称不存在、点选处为空）原样返回给节点，由节点决定提示内容
    # （执行器不产生消息：消息属于消息目录，策略属于节点）
    "items.pick": {"desc": "pick items", "fields": {"rules": "enum_list:rules", "count": "enum:count"}},
}

# 枚举取值：执行器按名称分支。新增取值需在此处加一项，并在执行器中加一个分支
ENUMS: dict[str, tuple[str, ...]] = {
    "accumulate": ("max",),            # 多个框重叠时取较大值（软边互不覆盖）
    "count": ("one", "any"),           # 必须恰好选中一条，或允许多条
    "rules": ("index", "name", "first", "top", "all", "ids", "at_point"),
}

# 挑条目各规则所需的参数（执行器与节点均按名称存取）
RULE_ARGS: dict[str, tuple[str, ...]] = {
    "index": ("index",),      # 序号，从 1 开始
    "name": ("name",),        # 条目名称，须完全相同
    "first": (),              # 第一条（条目表已按显著程度排序，即最显著的人物）
    "top": ("count",),        # 前 N 条（按显著程度排序，N 由用户设定）
    "all": (),                # 全部
    "ids": ("ids",),          # 编号属于给定列表的条目
    "at_point": ("picks",),   # 2D 视图中点选的条目（picks 的每项为 [帧, x, y]）
}


# ------------------------------------------------------------------ 公式文法

# 仅包含数字、变量、+ - * /、取负、括号及三个函数。
# 文法仅需满足目录中算法的需要，不作为通用语言
FUNCTIONS: dict[str, int] = {"min": 2, "max": 2, "clamp": 3}

_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?)|([A-Za-z_][A-Za-z_0-9]*)|([-+*/(),]))")


class BadOp(Exception):
    """算法目录本身有误（公式错误、未知枚举取值）。属于开发错误而非用户错误，因此不纳入消息目录。"""


def tokens(text: str) -> list[tuple[str, str]]:
    out, i = [], 0
    while i < len(text):
        m = _TOKEN.match(text, i)
        if m is None:
            if text[i:].strip() == "":
                break
            raise BadOp(f"unrecognised characters in the formula: {text[i:]!r} (whole: {text!r})")
        i = m.end()
        num, name, sym = m.groups()
        out.append(("num", num) if num else ("name", name) if name else ("sym", sym))
    return out


def parse(expr: str) -> tuple:
    """将公式解析为语法树：
    ("num", 值) / ("var", 名字) / ("bin", 运算符, 左, 右) / ("neg", 子) / ("call", 函数名, [参数…])。"""
    ts = tokens(expr)
    pos = 0

    def peek() -> tuple[str, str] | None:
        return ts[pos] if pos < len(ts) else None

    def eat(sym: str) -> None:
        nonlocal pos
        if peek() != ("sym", sym):
            raise BadOp(f"formula {expr!r} is missing {sym!r}")
        pos += 1

    def additive() -> tuple:
        node = multiplicative()
        while peek() in (("sym", "+"), ("sym", "-")):
            op = ts[pos][1]
            pos_add()
            node = ("bin", op, node, multiplicative())
        return node

    def pos_add() -> None:
        nonlocal pos
        pos += 1

    def multiplicative() -> tuple:
        node = unary()
        while peek() in (("sym", "*"), ("sym", "/")):
            op = ts[pos][1]
            pos_add()
            node = ("bin", op, node, unary())
        return node

    def unary() -> tuple:
        if peek() == ("sym", "-"):
            pos_add()
            return ("neg", unary())
        if peek() == ("sym", "+"):
            pos_add()
            return unary()
        return primary()

    def primary() -> tuple:
        tok = peek()
        if tok is None:
            raise BadOp(f"formula {expr!r} is incomplete")
        kind, text = tok
        if kind == "num":
            pos_add()
            return ("num", float(text))
        if kind == "name":
            pos_add()
            if peek() == ("sym", "("):
                if text not in FUNCTIONS:
                    raise BadOp(f"formula {expr!r} uses unknown function {text} (known: {sorted(FUNCTIONS)})")
                eat("(")
                args = [additive()]
                while peek() == ("sym", ","):
                    pos_add()
                    args.append(additive())
                eat(")")
                if len(args) != FUNCTIONS[text]:
                    raise BadOp(f"{text} takes {FUNCTIONS[text]} arguments, formula {expr!r} gives {len(args)}")
                return ("call", text, args)
            return ("var", text)
        if tok == ("sym", "("):
            pos_add()
            node = additive()
            eat(")")
            return node
        raise BadOp(f"{text!r} is out of place in formula {expr!r}")

    tree = additive()
    if pos != len(ts):
        raise BadOp(f"formula {expr!r} has something extra at its end")
    return tree


def variables(tree: tuple) -> set[str]:
    """返回语法树中使用的变量名，供执行器检查节点是否提供了全部输入。"""
    if tree[0] == "var":
        return {tree[1]}
    if tree[0] == "bin":
        return variables(tree[2]) | variables(tree[3])
    if tree[0] == "neg":
        return variables(tree[1])
    if tree[0] == "call":
        return set().union(*(variables(a) for a in tree[2]))
    return set()


def check(op_id: str, desc: dict) -> None:
    """校验一条算法描述。目录加载时逐条校验，有误即报错。"""
    kind = desc.get("kind")
    if kind not in KINDS:
        raise BadOp(f"op {op_id}: kind={kind!r} is not in the vocabulary (known: {sorted(KINDS)})")
    fields = KINDS[kind]["fields"]
    missing = [f for f in fields if f not in desc]
    if missing:
        raise BadOp(f"op {op_id} ({kind}) is missing {missing}")
    for field, how in fields.items():
        value = desc[field]
        if how == "expr":
            parse(value)
        elif how == "number":
            float(value)
        elif how.startswith("enum_list:"):
            allowed = ENUMS[how.split(":")[1]]
            bad = [v for v in value if v not in allowed]
            if bad:
                raise BadOp(f"op {op_id}: {bad} in {field} is not one of {list(allowed)}")
        elif how.startswith("enum:"):
            allowed = ENUMS[how.split(":")[1]]
            if value not in allowed:
                raise BadOp(f"op {op_id}: {field}={value!r} is not one of {list(allowed)}")
