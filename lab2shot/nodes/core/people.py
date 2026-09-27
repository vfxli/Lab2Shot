"""人物：选人（通用，适用于任何检测节点输出的人物框）。"""

from __future__ import annotations

from typing import Literal

from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port, parse_picks, person_ids
from ..expects import KnownPeople
from ..handles import Handle
from ..applies import Param

# 参数上的四种选择方式 → 算法目录中的四条规则（lab2shot/ops/ops.toml 的 people.select）。
# 「最大」即第一条：人物框已按画面显著程度排序，1 号最显著。
RULES = {"largest": "first", "top": "top", "all": "all", "ids": "ids", "picked": "at_point"}


class SelectPeople(NodeDef):
    id = "core.select_people"
    version = 5  # 输出为一份「人物框」，而非「人物框[]」
    category = "mask_edit"
    inputs = (Port("boxes", "boxes", "人物框", expects=(KnownPeople(),), takes_empty=False),)  # 未检测到任何人时无可选择
    # 输出为一份人物框：该类型本身可容纳多人，选择一人即为只含一人的一份，可直接接入解算器的「人物框」端口。
    # 不输出「人物框[]」（每人一条的列表）：否则接入解算器前还需合并，先拆分再合并不符合直觉。
    # 确需逐人处理时，显式连接「拆成列表」→「逐项开始」。
    outputs = (Port("boxes", "boxes", "人物框"),)
    handles = (Handle("person", {"picks": "picks"}, source="boxes", when=Param("mode").one_of("picked")),)
    on_node = ("mode", "count", "ids")
    # 算法定义不在此处，而在算法目录（lab2shot/ops/ops.toml）中：各选择方式是 people.select 的规则，
    # 浏览器按同一份描述执行（webui/src/ops/run.ts），因此选人之后画面立即更新，无需提交计算。
    ops = ("people.select",)

    class Params(NodeParams):
        mode: Literal["largest", "top", "all", "ids", "picked"] = P(
            "largest", label="选择", group="人物",
            option_labels={"largest": "最大", "top": "前几个", "all": "全部", "ids": "编号", "picked": "点选"},
            help="最大：画面里最显眼的一个人；前几个：最显眼的前 N 个人（N 填在「人数」里）；全部：所有检测到的人；编号：按下面填的编号选；点选：在 2D 视图里点画面上的人",
        )
        # 按显著程度取前 N 个：有几十个路人的镜头只计算最显著的几个
        count: int = P(3, label="人数", ge=1, group="人物", applies=Param("mode").one_of("top"),
                       help="选择方式为「前几个」时用：按画面存在感取最显眼的前几个人（1 号最显眼）")
        ids: str = P("", label="编号", help="选择方式为「编号」时用：填人物编号，多个用逗号，如 1,3。编号按画面存在感排序，1 号最显眼，2D 视图里框上有标", group="人物", placeholder="1,3", applies=Param("mode").one_of("ids"))
        picks: list[str] = P([], label="点选", help="选择方式为「点选」时用：在 2D 视图里点谁就选谁，自动记下帧号和位置；点错了在这里删掉", widget="picks", group="人物", placeholder="在 2D 视图里点人", applies=Param("mode").one_of("picked"))

    @classmethod
    def browser_ops(cls, params, types):
        """选人即从整段中挑选若干人物框：每帧只有少量浮点数，由浏览器计算，无需提交。"""
        mode = params["mode"]
        # 只在使用相应方式时读取其参数：编号填写错误、点选位置偏离等由该方式自身报告
        args: dict = {"rule": RULES[mode]}
        if mode == "top":
            args["count"] = int(params["count"])
        elif mode == "ids":  # 框中不存在的编号由端口的用法检查提示
            args["ids"] = sorted(person_ids(params["ids"]))
        elif mode == "picked":
            args["picks"] = [[frame, x, y] for frame, x, y in parse_picks(params["picks"])]
        return ({"op": "people.select", "args": args},)

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import read_boxes, write_boxes
        from ...ops import run as run_op

        src = ctx.input("boxes")
        people = read_boxes(src)
        # 参数到算法参数的转换只在 browser_ops 一处定义：服务器与浏览器使用同一结果，不会各自实现
        call = cls.browser_ops(ctx.params, {})[0]
        got = run_op(call["op"], {"items": people, **call["args"]})
        for miss in got["missed"]:  # 点击位置为空：算法返回事实，提示内容由节点决定（消息位于消息目录中）
            ctx.say("N-PEOPLE-PICKMISS", frame=miss["frame"], x=miss["x"], y=miss["y"], param="picks")
        chosen = [people[i] for i in got["indices"]]  # 保持输入中的顺序（write_boxes 写入 meta["people"] 时即为此顺序）
        if not chosen:  # 参数指向不存在的人（编号填写错误、点击位置为空）：属于使用者输入问题，需明确说明
            raise Invalid(Msg("E-PEOPLE-NONE"))
        m = src.meta
        return {"boxes": write_boxes(ctx.outputs["boxes"], chosen, m["width"], m["height"], m["frames"], chosen=True)}


NODES = (SelectPeople,)
