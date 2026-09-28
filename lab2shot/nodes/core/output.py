"""输出: the one node that hands results to the user. The output-settings nodes wired into it (nodes/output.py: each
format module's 「… 输出设置」) wrote their files as their results; 「输出」's own cook is 整理 + 压缩: it collects them
into its task's folder, one sub-folder per 名字 (inside a 逐项处理 block, one sub-folder per item), and packs that
folder into one zip, through the collector its cook is given (CookContext.collector: nodes/services.py OutputSink,
lab2shot/transfer/outputs.py Collector). Only once that is done can the user download it (the node's only control);
DCC plugins fetch single files of the folder. It has no parameters, and knows no format."""

from __future__ import annotations

from ..base import NodeDef, NodeParams, Port
from ..output import FILES


class Output(NodeDef):
    id = "core.output"
    category = "out_deliver"
    inputs = (Port("files", FILES, "文件", multi=True),)
    delivers = True  # its cook collects and packs the files for the user; showing it only shows what they were made from

    class Params(NodeParams):
        pass

    @classmethod
    def cook(cls, ctx):
        # it only collects: the engine packs once every instance of it has (engine/cook.py Engine.cook)
        found = ctx.collector.collect(ctx.node_id, ctx.label, ctx.inputs["files"], ctx.stop, ctx.path, ctx.names)
        if not found["commercial"]:
            ctx.say("W-COOK-NONCOMMERCIAL", projects=sorted({s["project"] for s in ctx.provenance["sources"] if not s["commercial"]}))
        return {}


NODES = (Output,)
