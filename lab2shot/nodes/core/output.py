"""输出: the one node that delivers. The output-settings nodes wired into it (nodes/output.py: each format module's
「… 输出设置」) wrote their files as their results; 「输出」 hands them to the user, one sub-folder per 名字, as one tar
(tar.gz) or into a folder, through the delivery sink its cook is given (CookContext.delivery: nodes/services.py
DeliverySink, lab2shot/transfer/deliveries.py Sink). It knows no format."""

from __future__ import annotations

from typing import Literal

from ..base import NodeDef, NodeParams, P, Port
from ..output import FILES

SAVE_AS = {"tar": "tar", "tar.gz": "tar.gz", "folder": "文件夹"}


class Output(NodeDef):
    id = "core.output"
    category = "out_deliver"
    inputs = (Port("files", FILES, "文件", multi=True),)
    on_node = ("mode",)
    delivers = True  # a click hands the files over; showing it only shows what they were made from

    class Params(NodeParams):
        mode: Literal["tar", "tar.gz", "folder"] = P(
            "tar", label="保存成", group="保存", option_labels=SAVE_AS, option_needs={"folder": "folders"},
            help="tar：一个不压缩的文件，最快，任何浏览器下载都最稳。\n"
                 "tar.gz：压缩的文件。只有没压缩的 EXR 或文本格式（USDA、CSV、JSON）才值得；EXR、PNG、USDC 本来就压缩过，"
                 "几乎不会变小，大序列还要多花服务器的计算时间。\n"
                 "文件夹：不打包，每个子文件夹直接写进你选的文件夹。网页里要 Chrome 或 Edge 才能直接写文件夹；"
                 "DCC 插件和命令行写进给的文件夹",
        )
        path: str = P("", label="保存到", widget="deliver", group="保存", accept=(".tar", ".tar.gz"),
                      help="tar、tar.gz：用系统的保存对话框选一个文件，算完存成这一个文件；文件夹：选一个文件夹，算完每个子文件夹"
                           "写进去。DCC 插件和命令行给一个路径（tar 的文件或一个文件夹）")
        # 面板上没有它：包名就是「保存到」选的文件名，写两遍是冗余。
        # 只有一根线能驱动它：逐项处理块里把「逐项开始」的「名字」接过来，每个条目各交付一个包。
        name: str = P("", label="名字", group="保存", placeholder="按「保存到」的文件名", panel=False,
                      help="交付的包叫什么。默认按「保存到」选的文件名；放在逐项处理块里时把「逐项开始」的「名字」"
                           "接到这里，每个条目就按它的名字各交付一个包（sh010.tar、sh020.tar）")

    @classmethod
    def cook(cls, ctx):
        p = ctx.params
        # the package's name: what 「名字」 says (a wire from 「逐项开始」's 名字 names every item's own package),
        # else the file or folder chosen in 「保存到」
        record = ctx.delivery.deliver(ctx.node_id, ctx.label, ctx.inputs["files"], p["name"].strip() or p["path"], p["mode"],
                                      item="/".join(ctx.path))  # inside a 逐项处理 block: one package per item
        if not record["commercial"]:
            ctx.say("W-COOK-NONCOMMERCIAL", projects=sorted({s["project"] for s in ctx.provenance["sources"] if not s["commercial"]}))
        return {}


NODES = (Output,)
