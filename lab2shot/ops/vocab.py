"""算法目录的词汇表：定义支持的运算种类，以及精度与取整规则。

每条算法以数据形式描述（机器可读，见 `ops.toml`），服务器与浏览器各有一个通用执行器按同一份描述执行
（`lab2shot/ops/run.py`、`webui/src/ops/run.ts`）。算法统一定义于本目录，不属于某个节点，任何节点均可引用。
本设计不采用以下做法：以伪代码为文档再分别用 Python 和 TypeScript 各实现一遍（两份实现容易产生偏差）；
以单一语言实现后在另一端运行（浏览器需加载数 MB 运行时）。

词汇表是封闭的：`KINDS` 只包含三种运算。新增算法若属于其中之一，两个执行器均无需修改，只需在 `ops.toml`
中增加一条描述。只有新增第四种运算时才需要修改两个执行器，且应先修改本词汇表。

适用范围：仅限纯确定性、不需要模型权重或 GPU、输入输出均为核心类型的基础算法，例如选人、拆分列表、框转遮罩、
图像合成、数学运算、比较与切换。解算器、格式读写与 USD 均在服务器端处理，第三方 adapter 不涉及此层。

浏览器端实现的覆盖范围（以 `KINDS` 为准）：

| 类别 | 浏览器实现 | 示例 |
|---|---|---|
| 逐像素独立计算 | 有（`pixel`、`boxes.paint`、`items.pick`，即 `KINDS`） | 色彩变换、范围映射、遮罩运算、合成、通道提取、选人、拆分列表、框转遮罩 |
| 依赖邻域像素 | 无 | 模糊、变形、STMap 重采样 |

「无」表示回退到服务器计算，并非不支持。邻域采样需要在两个执行器中各实现一份（`KINDS` 需新增一种运算），
目前尚未实现；这是能力限制，而非设计约束。

与节点的对应关系：节点能否在浏览器中计算，由两处共同决定：服务器端 `NodeDef.browser_ops`（重写即表示可计算）
与浏览器端 `webui/src/ops/recipe.ts` 的 `WIRING`（端口与算法参数的对应）；判定逻辑仅位于
`webui/src/graph/rules.ts browserCanCompute`。两处必须一致，且不一致时不会报错：若服务器声明可计算而浏览器
未配置对应关系，将静默回退为排队计算，用户仅会观察到修改参数后需要等待。

## 精度与取整规则（两个执行器均须遵守）

1. 统一使用 float32：服务器端数组 dtype 为 float32；浏览器端以 float64 计算并存入 `Float32Array`。
   两端最终均为 float32 精度，差异仅来自中间步骤的舍入。
2. 不隐式钳位：仅当描述中写明 `clamp(...)` 时才钳位。
   （「图像合成」不钳位，相加结果可超过 1，此为预期行为；「人物框转遮罩」的钳位写在其描述中。）
3. 不隐式取整：三种运算均不包含取整；需要取整的算法须在描述中显式写出。
4. 一致性判据为 `DISPLAY_TOLERANCE`（视图仅用于预览，不要求逐位一致）：同一算法、同一输入下，两端像素最大差
   ≤ 1/255 即视为一致，不得为末位差异反复调整参数。结构性差异不属于误差，例如软边变为硬边、整片区域偏移、
   一个像素的位移，这些表明两端未按同一描述执行，必须修复。
"""

from __future__ import annotations

import re

# 一致性判据：8 位显示的一个量化级。见模块说明「精度与取整规则」第 4 条
DISPLAY_TOLERANCE = 1.0 / 255.0


# ------------------------------------------------------------------ 三种运算（词汇表本体）

KINDS: dict[str, dict] = {
    # 逐像素算术：对每个输入像素计算一条公式。变量为输入名（a、b），
    # 通道数较少的一方自动广播到通道数较多的一方（例如单通道遮罩作用于三通道图像）
    "pixel": {"desc": "每像素算术", "fields": {"expr": "expr"}},
    # 按框涂：将若干矩形绘制到画布上。`axis` 定义单条轴上像素被框覆盖的程度
    # （变量：p 为像素坐标，lo/hi 为框在该轴上的两端），`combine` 将两条轴合成为像素覆盖率
    # （变量：x、y），`accumulate` 定义多个框的叠加方式
    "boxes.paint": {"desc": "按框涂", "fields": {"axis": "expr", "combine": "expr", "accumulate": "enum:accumulate", "base": "number"}},
    # 挑条目：按一条规则从有序条目表中选出若干条，返回其位置而非内容。
    # 未能选中的情况（编号越界、名称不存在、点选处为空）原样返回给节点，由节点决定提示内容
    # （执行器不产生消息：消息属于消息目录，策略属于节点）
    "items.pick": {"desc": "挑条目", "fields": {"rules": "enum_list:rules", "count": "enum:count"}},
}

# 枚举取值：执行器按名称分支。新增取值需在此处加一项，并在两个执行器中各加一个分支
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


# ------------------------------------------------------------------ 公式文法（两端完全一致）

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
            raise BadOp(f"公式里有认不出的字符：{text[i:]!r}（整条：{text!r}）")
        i = m.end()
        num, name, sym = m.groups()
        out.append(("num", num) if num else ("name", name) if name else ("sym", sym))
    return out


def parse(expr: str) -> tuple:
    """将公式解析为语法树，树的结构在两端一致：
    ("num", 值) / ("var", 名字) / ("bin", 运算符, 左, 右) / ("neg", 子) / ("call", 函数名, [参数…])。"""
    ts = tokens(expr)
    pos = 0

    def peek() -> tuple[str, str] | None:
        return ts[pos] if pos < len(ts) else None

    def eat(sym: str) -> None:
        nonlocal pos
        if peek() != ("sym", sym):
            raise BadOp(f"公式 {expr!r} 里少了 {sym!r}")
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
            raise BadOp(f"公式 {expr!r} 没写完")
        kind, text = tok
        if kind == "num":
            pos_add()
            return ("num", float(text))
        if kind == "name":
            pos_add()
            if peek() == ("sym", "("):
                if text not in FUNCTIONS:
                    raise BadOp(f"公式 {expr!r} 用了没有的函数 {text}（有的是 {sorted(FUNCTIONS)}）")
                eat("(")
                args = [additive()]
                while peek() == ("sym", ","):
                    pos_add()
                    args.append(additive())
                eat(")")
                if len(args) != FUNCTIONS[text]:
                    raise BadOp(f"{text} 要 {FUNCTIONS[text]} 个参数，公式 {expr!r} 给了 {len(args)} 个")
                return ("call", text, args)
            return ("var", text)
        if tok == ("sym", "("):
            pos_add()
            node = additive()
            eat(")")
            return node
        raise BadOp(f"公式 {expr!r} 里 {text!r} 放错了地方")

    tree = additive()
    if pos != len(ts):
        raise BadOp(f"公式 {expr!r} 末尾有多余的东西")
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
        raise BadOp(f"算法 {op_id} 的 kind={kind!r} 不在词汇表里（有的是 {sorted(KINDS)}）")
    fields = KINDS[kind]["fields"]
    missing = [f for f in fields if f not in desc]
    if missing:
        raise BadOp(f"算法 {op_id}（{kind}）少了 {missing}")
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
                raise BadOp(f"算法 {op_id} 的 {field} 里 {bad} 不在 {list(allowed)} 里")
        elif how.startswith("enum:"):
            allowed = ENUMS[how.split(":")[1]]
            if value not in allowed:
                raise BadOp(f"算法 {op_id} 的 {field}={value!r} 不在 {list(allowed)} 里")
