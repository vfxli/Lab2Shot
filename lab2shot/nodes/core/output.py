"""输出: the one node that hands results to the user. The output-settings nodes wired into it (nodes/output.py: each
format module's 「… 输出设置」) wrote their files as their results; 「输出」's own cook is 整理 + 压缩: it collects them
into its task's folder, one sub-folder per 名字 (inside a 逐项处理 block, one sub-folder per item), and packs that
folder into one zip, through the collector its cook is given (CookContext.collector: nodes/services.py OutputSink,
lab2shot/transfer/outputs.py Collector). Only once that is done can the user download it (the node's only control);
DCC plugins fetch single files of the folder. It has no parameters (only the button 「下载」, beside every node's 「计算」), and knows no format."""

from __future__ import annotations

from ...errors import Invalid
from ... import i18n
from ...messages import Msg
from ..base import Button, NodeDef, NodeParams, Port
from ..output import FILES, name_key


class Output(NodeDef):
    id = "output"
    category = "out_deliver"
    inputs = (Port("files", FILES, multi=True),)
    delivers = True  # its cook collects and packs the files for the user; showing it only shows what they were made from
    # 「下载」: download the zip this 「输出」 packed (the page's action "download", webui/src/editor/buttonActions.ts);
    # greyed while there is none. Packing is its 「计算」 (every node's button): cooking an 「输出」 collects and packs.
    # A button parameter, not a value (nodes/params.py Button)
    buttons = (Button("download", "download"),)
    # the node's body shows its 「下载」 row by default: the only download control on the graph (none in the footer or at
    # the top of the parameter panel)
    on_node = ("download",)

    class Params(NodeParams):
        pass

    @classmethod
    def cook(cls, ctx):
        # 各输出设置的名字各占一个子文件夹：图上写着的名字在提交前就比过（engine/graph.py check_delivery），由连线给的
        # 名字（「文字」节点接到「名字」的口）到这时才知道，在这里用同一条规则再比一次
        named: dict[str, list[str]] = {}
        for p in ctx.inputs["files"]:
            named.setdefault(name_key(str(p.meta.get("name", ""))), []).append(p.meta.get("name", ""))
        same = [Msg("B-DELIVER-NAMED", outputs=i18n.Word("output.settings_count", count=len(names)), name=names[0])
                for names in named.values() if len(names) > 1]
        if same:
            raise Invalid(Msg("B-DELIVER-SAMENAME", node=ctx.label, same=same))
        # it only collects: the engine packs once every instance of it has (engine/cook.py Engine.cook)
        found = ctx.collector.collect(ctx.node_id, ctx.label, ctx.inputs["files"], ctx.stop, ctx.path, ctx.names)
        if not found["commercial"]:
            ctx.say("W-COOK-NONCOMMERCIAL", projects=sorted({s["project"] for s in ctx.provenance["sources"] if not s["commercial"]}))
        return {}


NODES = (Output,)
