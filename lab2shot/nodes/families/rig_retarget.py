"""学习式重定向（解算器）：「重定向预处理」交来的源和目标进模型，模型原生输出什么就交出什么。

三段式的中间一段（core/retarget.py 是前后两段）。对应关系、参考姿态、哪些骨骼不送进模型都在预处理里定好、写在两侧
骨架上（kit/retarget_prepare.py），解算器只认带这些标注的输入；每个成员声明自己收什么骨骼（`needs`，
kit/retarget_needs.py），送进来的不合就报错，指回预处理——解算器自己不丢骨骼。结果写回目标自己的层级：模型给的
世界姿态换成目标的局部（保留目标自己的骨长、轴和缩放：这是格式换算，不是修正），忽略的关节保持参考姿态的局部、跟着
父骨骼走，模型的根位移原样。不做位移、髋高、约束：那些在「重定向后处理」里明着做。
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np

from ...data.animation import placement_at, skeleton_animation
from ...data.scene_arrays import scene_arrays
from ...errors import Invalid
from ...messages import Msg
from ..applies import Cost
from ..base import Info, NodeParams, P, Port
from ..kit.retarget_needs import JointNeeds, check
from .base import Job, RawOutput, WorkerNode
from ... import i18n


class RigRetargetParams(NodeParams):
    """所有成员共有的参数：只有推理设置；对应、参考姿态、忽略都在「重定向预处理」里。"""

    fps: int = P(24, group="motion", ge=1, le=120, unit="fps")


class RigRetarget(WorkerNode):
    """学习式重定向家族的共同部分：两个输入（预处理的「源」「目标」），一个骨架动画输出；prepare 读两侧标注、按成员的
    `needs` 检查、写 worker 的输入（lab2shot_worker.rig_retarget.write_job），convert 把模型的结果写回目标层级。"""

    version = 6  # 输入改为预处理的标注；忽略的关节保持参考姿态的局部；不再输出适配预览
    category = "scene_convert"
    inputs = (Port("source", "scene.skeleton|scene.character"),
              Port("target", "scene.character|scene.skeleton"))
    outputs = (Port("skeleton", "scene.skeleton"),)
    Params = RigRetargetParams
    cost = Cost(gpu=True)
    # 成员收什么骨骼：规则 id -> 需求（kit/retarget_needs.py）。预处理的「忽略规则」下拉列出它们；输入合其中任何一条即可
    needs: ClassVar[dict[str, JointNeeds]] = {}

    @classmethod
    def required_mesh_sides(cls, params) -> tuple[str, ...]:
        """按参数才定下来的网格要求（运动学精修的 geo 阶段两侧都要）：不属于忽略，是节点自己的输入检查。固定的网格要求
        写在 `needs` 里。"""
        return ()

    @classmethod
    def plan_refusals(cls, params, comes):
        """按参数要网格的一侧（required_mesh_sides）接进来的只有骨架（这一路上没有「角色」：纯骨架 FBX、BVH）：算之前
        就知道，「计算」置灰并写原因（B-LEARNEDRETARGET-NOMESH）。角色带不带网格要读了才知道，那时由
        prepare 拒绝（E-LEARNEDRETARGET-MESH）。"""
        out = []
        for side, port in (("src", "source"), ("dst", "target")):
            if side in cls.required_mesh_sides(params) and comes(port) and not comes(port, "scene.character"):
                out.append((Msg("B-LEARNEDRETARGET-NOMESH", side=i18n.Word("role.source" if side == "src" else "role.target")), port))
        return out

    @classmethod
    def describe(cls) -> dict:
        """目录里带上成员的忽略规则 id（声明顺序，第一条是默认）：页面接线时就能把预处理的「忽略规则」预选成它，不必等
        手柄数据。"""
        return {**super().describe(), "retarget_rules": list(cls.needs)}

    @classmethod
    def info(cls, params, inputs):
        """帧来自「源」：目标常常只有一帧的静止姿势，它的帧不属于结果。"""
        return Info.merge(inputs.get("source", []))

    @classmethod
    def _checked(cls, src, dst) -> None:
        """两侧合不合成员的规则：合任何一条就过；都不合时报问题最少的那一条（E-LEARNEDRETARGET-NEEDS）。"""
        if not cls.needs:
            return
        found = [(check(needs, src, dst), needs) for needs in cls.needs.values()]
        problems, needs = min(found, key=lambda got: len(got[0]))
        if problems:
            raise Invalid(Msg("E-LEARNEDRETARGET-NEEDS", rule=needs.label, problems=problems))

    @classmethod
    def prepare(cls, ctx) -> Job:
        from lab2shot_worker.rig_retarget import write_job

        from ..kit.retarget_prepare import bases, prepared

        p = ctx.params
        src = prepared(ctx.input("source"), side=i18n.Word("role.source"), least_frames=2)
        dst = prepared(ctx.input("target"), side=i18n.Word("role.target"))
        mapping, base = bases(src, dst)
        if mapping.hands:
            raise Invalid(Msg("E-LEARNEDRETARGET-BODY"))
        cls._checked(src, dst)
        arrays = {}
        for side, port, rig in (("src", "source", src.rig), ("dst", "target", dst.rig)):
            characters = scene_arrays(ctx.input(port)).items["character"]
            item = next((it for it in characters if rig.path == str(it["path"]) + "/skeleton"
                         or rig.path == str(it["path"])), None)
            if item is None:
                item = next((it for it in characters if str(rig.path).startswith(str(it["path"]) + "/")), None)
            if side in cls.required_mesh_sides(p) and (item is None or not item.get("meshes")):
                raise Invalid(Msg("E-LEARNEDRETARGET-MESH", side=i18n.Word("role.source" if side == "src" else "role.target")))
            arrays[side] = item
        ctx.stage("prepare_retarget_input")
        path = write_job(ctx.work / "retarget.npz", src.rig, dst.rig, base, mapping, arrays, fps=p["fps"],
                         src_ignored=src.ignored_joints, dst_ignored=dst.ignored_joints)
        return Job(None, inputs={"retarget": path}, notes={"source": src, "target": dst, "mapping": mapping})

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job):
        from lab2shot_shared import motion as mo
        from lab2shot_worker.rig_retarget import read_result

        from ..kit.retarget import Result, target_locals

        src, dst, mapping = (job.notes[k] for k in ("source", "target", "mapping"))
        source, goal = src.rig, dst.rig
        result = read_result(raw)
        world = np.asarray(result["world"], np.float64)
        if world.shape != (len(source.frames), len(goal.names), 4, 4) or not np.isfinite(world).all():
            raise Invalid(Msg("E-LEARNEDRETARGET-RESULT", frames=len(source.frames), joints=len(goal.names)))
        cls._say_sent(ctx, result["info"])
        hip = mapping.dst["hips"][0]
        skip = set(dst.ignored_joints)
        placement = placement_at(ctx.input("target"), goal.path, source.frames)
        # 世界姿态换成目标自己的局部（与「动作重定向」同一种写回：保留目标的骨长、缩放和镜像，Skeleton prim 带缩放或镜像也对）；
        # 模型给的髋的轨迹原样放上去。忽略的关节不在转动之列，写回从参考帧的局部起步：把它们参考帧的局部换成参考
        # 姿态的，它们就整个保持参考姿态的局部、跟着父骨骼走，而且是在放髋之前（忽略的根骨也算在髋的位置里）
        first = np.array(goal.local[:1], np.float64)
        first[0, sorted(skip)] = dst.reference_local[sorted(skip)]
        native = Result(rotations={j: mo.orthonormal(world[:, j, :3, :3]) for j in range(len(goal.names)) if j not in skip},
                        placed={hip: world[:, hip, :3, 3]}, scale=1.0, by="none", factor=1.0, measured_cm=(0.0, 0.0, 0.0),
                        target_cm=0.0, joints=len(goal.names) - len(skip),
                        rest=mo.world_from_local(dst.reference_local, dst.parents))
        local = target_locals(replace(goal, local=first), source.frames, placement, native)
        ctx.stage("write_target_animation")
        rig = replace(goal, frames=list(source.frames))
        info = {"extension": cls.runtime, "retarget": {**result["info"], "from": source.path}}
        return {"skeleton": skeleton_animation(ctx.input("target"), ctx.outputs["skeleton"], rig, local, info)}

    @staticmethod
    def _say_sent(ctx, info: dict) -> None:
        """实际送进模型的（适配层在 result.json 里报的，lab2shot_worker.rig_retarget.sent）：两侧的关节数和它补出的标记点。"""
        sent, markers = info.get("sent"), info.get("markers")
        if not isinstance(sent, dict) or not isinstance(markers, dict):
            return
        ctx.say("I-LEARNEDRETARGET-SENT", src=sent.get("src", 0), dst=sent.get("dst", 0),
                src_markers=list(markers.get("src") or []) or i18n.Word("list.none"), dst_markers=list(markers.get("dst") or []) or i18n.Word("list.none"))
