"""第三方项目解算器节点的官方输入输出声明：解算器节点的输入输出即上游官方要求的输入输出，不额外添加任何内容；
需要添加的功能一律做成显式的独立工具节点。

本声明是该规则可由机器检查的形式，核对对象是上游仓库本身。声明中写的不是「官方有相机输入」这类判断，
而是上游源码中的符号名（函数参数名、模型输出的字段名）以及其所在文件和行号。`lab2shot check official`（lab2shot/cli/check.py check_official）完成四项工作：

  1. 声明中的每个端口名都必须是节点真有的端口（takes：输入端口或参数；gives / ours：输出端口）：写错即判为失败；
  2. 节点的每个端口都必须写进声明（输入端口在 takes 中，输出端口在 gives 或 ours 中）：没写的就是额外添加的，判为失败；
  3. 声明中的每个上游符号都必须能在对应的上游文件中找到：找不到即为虚构，判为失败；
  4. `cite` 所指的上游文件必须存在：否则无法核对，判为失败。

第 3 条是关键：它使「该端口来自官方」这一说法有不可更改的依据（上游源码），而不是由声明者自行核对。

写法：

    official = Official(
        cite="third_party/hamer/repo/hamer/models/hamer.py:107-126",
        takes={"image": "batch['img']"},              # 节点的端口名 -> 上游源码中的符号
        gives={"camera": "pred_cam", "mano": "pred_mano_params"},
    )
    # 给人看的上游说明写在语言目录 node.<type>.official.note（可选）

自行添加的内容一律不写入此处，应做成显式的独立工具节点（如「相机空间转换」「提取骨架」）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CITE = re.compile(r"^(third_party/[\w.-]+/[\w./-]+?):(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class Official:
    """第三方节点的官方输入输出，按上游源码中的符号名书写（见模块说明）。

    `cite` 可以是一条或多条（元组）：上游常把输入和输出写在两个文件中
    （HaMeR 的入口在 `demo.py`、MANO 参数在 `models/hamer.py`；SMIRK 的编码器在 `smirk_encoder.py`、
    网格在 `FLAME/FLAME.py`）。若只能写一条，另一个文件中的符号会被检查误报为找不到；
    符号在任一条引用的行中找到即视为通过。"""

    cite: str | tuple[str, ...]  # third_party/<项目>/repo/<文件>:<行> 或 :<起>-<止>；多条时写为元组
    takes: dict[str, str] = field(default_factory=dict)  # 本节点的输入端口 -> 上游符号
    gives: dict[str, str] = field(default_factory=dict)  # 本节点的输出端口 -> 上游符号
    # 原样输出节点自身参数的端口 -> 对应参数。它不是上游结果，因此不写入 gives；
    # 但也不是隐式添加的内容，必须在此声明，检查才会放行。
    # 例：镜头节点的「Filmback」：Focal Length 的毫米值按它换算，将其传给下游是为了避免同一数值在
    # 两个节点上分别填写且不一致。
    # 给人看的说明在语言目录 node.<type>.official.ours.<port>（ours_of）
    ours: dict[str, str] = field(default_factory=dict)
    # 上游说明给人看的文字在语言目录：node.<type>.official.note（note_of）

    def __post_init__(self) -> None:
        for one in self.cites:
            if not CITE.match(one):
                raise ValueError(f"cite must read third_party/<project>/.../<file>:<line> or :<from>-<to>, got {one!r}")

    @property
    def cites(self) -> tuple[str, ...]:
        return (self.cite,) if isinstance(self.cite, str) else tuple(self.cite)

    def source(self) -> tuple[Path, str]:
        """(第一条引用的上游文件, 所有引用行原文的拼接)。文件不存在时抛出异常：无法核对时不得视为通过。"""
        first, texts = None, []
        for one in self.cites:
            m = CITE.match(one)
            path = ROOT / m.group(1)
            if not path.is_file():
                raise FileNotFoundError(f"{one}: the upstream file is not there, nothing to check against (is third_party/ installed?)")
            first = first or path
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            a = int(m.group(2))
            b = int(m.group(3) or m.group(2))
            texts.append("\n".join(lines[max(a - 1, 0):b]))
        return first, "\n".join(texts)

    def missing_symbols(self) -> list[str]:
        """声明中写出但在上游引用行中找不到的符号：表示声明与上游不符。"""
        _path, text = self.source()
        out = []
        for port, symbol in {**self.takes, **self.gives}.items():
            needle = symbol.split("[")[0].strip().strip("'\"")
            if needle and needle not in text:
                out.append(f"{port} -> {symbol} (not in the cited lines {', '.join(self.cites)}: no {needle})")
        return out


def note_of(node_type) -> str:
    """A node type's note on how it follows upstream, in the language now: node.<type>.official.note of the catalogue
    (its extension's own, then the core's); "" when it has none."""
    from .. import i18n

    official = getattr(node_type, "official", None)
    if official is None:
        return ""
    runtime = getattr(node_type, "runtime", "core")
    return i18n.node_text(node_type.id, "official", "note", in_scope=None if runtime == "core" else runtime) or ""


def ours_of(node_type, port: str) -> str:
    """What an `ours` port of a node type says it passes on, in the language now: node.<type>.official.ours.<port>,
    "" when it has none."""
    from .. import i18n

    official = getattr(node_type, "official", None)
    if official is None:
        return ""
    runtime = getattr(node_type, "runtime", "core")
    return i18n.node_text(node_type.id, "official", "ours", port, in_scope=None if runtime == "core" else runtime) or ""
