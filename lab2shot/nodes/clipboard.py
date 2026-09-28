"""节点与外部软件之间的文本交换：将结果复制到外部软件，以及将外部软件的数据粘贴为节点参数。

复制到外部软件，分三层，各层职责不重复：

    节点声明     `Pasteable`：目标软件（clipboard）及存放文本的输出端口（clipboard_port）
    计算阶段     节点将文本写入该端口的数据包（put_clipboard），并在包的 meta 中记录目标软件与文件名
    页面         节点上及「数据信息」中各有一个按钮，点击后向服务器请求该文本（server/packets.py clipboard）

声明位于本模块而非 `OutputSettings`：任何节点均可声明其某个结果可粘贴到外部软件（例如输出镜头参数的节点
既不是输出设置节点，也不写交付文件），`OutputSettings` 只是其中一类。

从外部软件粘贴：

    节点声明     `paste`：可读取的软件数据（如 "nuke"）
    节点实现     `read_pasted(text) -> {参数名: 值}`：由节点自行解析格式，无法解析时抛出 Invalid 并说明原因
    服务器       由一条路由将文本交给节点类，返回一组参数值（server/app.py paste）
    页面         面板上的按钮依次执行：读取剪贴板、请求参数值、一次性写入（作为单步撤销）

该机制是通用的：粘贴文本并一次性填写一组参数的需求不限于镜头，相机、跟踪点等同样适用。
节点只需实现 `paste` 与 `read_pasted` 即可支持。

核心层不识别任何软件或格式：软件名由节点声明，文本由格式模块读写（lab2shot/formats/nuke/）。
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..data.packet import Packet
from ..messages import Msg, plain


class Pasteable:
    """Mixin for a node whose result can be pasted into another application as text.

    `clipboard`      the application's name ("nuke"), which the page spells on the button
    `clipboard_port` the output port whose packet holds the text ("": the node's main result)
    `clipboard_file` the suffix of the file holding it, for a node that writes it among its delivered files
    `clipboard_when` the condition under which the node writes pasteable text (None: always). For example,
                     「2D 跟踪点输出设置」 writes a Nuke node only for its Tracker / CornerPin settings; for 3DE / CSV
                     the button is disabled with an explanation rather than hidden, so that it stays discoverable.
    """

    clipboard: ClassVar[str] = ""
    clipboard_port: ClassVar[str] = ""
    clipboard_file: ClassVar[str] = ".nk"
    clipboard_when: ClassVar[Any] = None  # a nodes/applies.py condition
    # 可粘贴进本节点的软件数据（""：不支持）。声明时必须实现 read_pasted
    paste: ClassVar[str] = ""

    @classmethod
    def read_pasted(cls, text: str) -> dict[str, Any]:
        """将粘贴的文本解析为本节点的一组参数值（参数名 → 值）。无法解析时抛出 Invalid 并说明原因，
        例如文本中没有可识别的内容、存在多个候选或版本不匹配。格式由节点与格式模块解析，核心层不参与。"""
        raise NotImplementedError

    @classmethod
    def clipboard_output(cls) -> str:
        """Which output port carries the text: what it declared, else its main result."""
        return cls.clipboard_port or cls.main_output()  # type: ignore[attr-defined]

    @classmethod
    def describe(cls) -> dict[str, Any]:
        # Only nodes that write for another application carry these fields: every browser downloads the catalogue
        # before the editor opens, so empty fields on every node would be wasted payload. `clipboard_port` is
        # likewise included only when it differs from the main output the page already knows.
        got = super().describe()  # type: ignore[misc]
        if not cls.clipboard:
            return {**got, **({"paste": cls.paste} if cls.paste else {})}
        port = cls.clipboard_output()
        return {**got, "clipboard": cls.clipboard,
                **({"clipboard_port": port} if port != got.get("main") else {}),
                **({"paste": cls.paste} if cls.paste else {})}


def clipboard_meta(app: str, file: str, said: "Msg | None" = None) -> dict:
    """What a packet's meta records of the text it holds for another application: the one shape the server reads.

    `said`: a message shown to the user when copying, such as what the target application cannot represent of this
    result. It is stored with the text and shown by the page after the copy rather than at cook time, so that it
    does not appear on nodes whose output is never copied."""
    got = {"app": str(app), "file": str(file)}
    if said is not None:
        got["said"] = {"code": said.code, "params": {k: plain(v) for k, v in said.params.items()}}
    return {"clipboard": got}


def put_clipboard(packet: Packet, app: str, file: str, text: str, said: "Msg | None" = None) -> Packet:
    """Write `text` into `packet` as `file` and record in its meta that it is the text for `app`. The packet is still
    being cooked, so its meta is written along with it. The file is stored beside the data and is not a delivered
    file (「输出」 delivers only what output-settings nodes write)."""
    packet.path(file).write_text(text, encoding="utf-8")
    packet.meta.update(clipboard_meta(app, file, said))
    return packet
