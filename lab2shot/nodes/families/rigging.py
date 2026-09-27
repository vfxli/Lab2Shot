"""绑定：为没有骨骼的模型自动生成骨架并计算蒙皮权重，输出可直接驱动的蒙皮角色。

本家族的工作对应 CG 中绑定师的第一步：扫描、生成或雕刻得到的静态网格本身无法运动；
要使其运动，需要先有一副骨架（关节的层级和位置），再将每个顶点绑定到骨架上（蒙皮权重）。

只有两个端口，两端都是已有的核心类型：

    模型 scene.model  →  蒙皮角色 scene.character

家族共有的三项工作在此实现一次，其他自动绑定项目接入时只需编写自己的参数：

* 网格如何交给模型：一个「模型」数据包中可能有多个网格（身体、衣服、头发片），自动绑定处理的是整个
  角色，因此先按世界坐标将其合并为一张三角网格（`one_mesh`），并记录顶点来自哪个网格（`Meshes.owner`）；
* 权重如何放回：按 `Meshes.owner` 将逐顶点权重拆回各个网格，写入原有的 USD
  （`data/scene.py bind_skin`），UV、法线、分区、材质均保持不变，只增加一副骨架和蒙皮；
* 骨骼如何命名和定轴：经过 `data/skeleton.py character_of_model`（与解算器输出蒙皮角色使用同一处），
  可识别的部位换为 CG 名称，关节轴按 CG 惯例摆正。

每点受影响的骨骼数（Maya 的 maxInfluences）也是家族共有的参数：它是交付的属性，而非某个模型的属性。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...data.packet import Packet
from ...errors import Invalid
from ...messages import Msg
from ...data.scene import Meshes
from ..applies import Cost
from ..base import NodeParams, P, Port
from .base import Job, RawOutput, WorkerNode


def one_mesh(src: Packet, frame: int | None = None) -> Meshes:
    """一个「模型」数据包中所有网格合并成的一张三角网格（世界坐标、厘米、Y 向上）。

    `frame`：逐帧变形的点缓存取哪一帧的形状（默认为第一帧）。四边形和多边形按 USD 自身的方式三角化
    （io/usd.py triangulated），点的编号不变，因此权重可以原样放回。"""
    from ...data.scene import mesh_arrays

    return mesh_arrays(src, frame)


def influences_param():
    """「影响骨骼数」：每个顶点最多受几根骨骼驱动（Maya 的 maxInfluences、USD 的 elementSize）。"""
    return P(4, label="影响骨骼数", widget="choice", group="蒙皮", option_labels={"1": "1 · 刚性", "2": "2", "4": "4 · 常用", "8": "8 · 精细"},
             help="每个顶点最多由几根骨骼带动（Maya 的 maxInfluences）。权重最大的几根留下，其余归零后重新归一。"
                  "4 是影视和游戏里最常用的；关节窝、肩膀这些地方要更细腻就用 8；1 是刚性绑定，整块跟着一根骨头走")


class AutoRigParams(NodeParams):
    """所有自动绑定节点共有的参数；各节点另外添加其模型自身的参数。"""

    influences: Literal[1, 2, 4, 8] = influences_param()
    character: str = P("", label="角色名", group="蒙皮", placeholder="按接进来的模型",
                  help="交付的蒙皮角色在 USD 里叫什么（Maya、Houdini 里看到的那个名字）。留空就用接进来的模型自己的名字")


class AutoRig(WorkerNode):
    """自动绑定：模型 → 蒙皮角色。

    worker 返回 raw/rig.npz（本家族的 worker 约定，`lab2shot_worker.rigging`）：
      joints [J,3] 关节位置（厘米、Y 向上，与输入模型同一坐标系）、parents [J]（根为 -1，父节点在子节点之前）、
      names [J] 模型给出的骨骼名、weights [V,J] 每个顶点对每根骨骼的权重（V 与 `one_mesh` 给出的顶点数相同）。
    节点侧只做三件事：选出影响最大的若干骨骼、按 CG 惯例为骨骼命名和定轴、写回原有的 USD。"""
    # 不读取画面，输入为已有的网格
    inputs = (Port("model", "scene.model", "模型"),)
    outputs = (Port("character", "scene.character", "蒙皮角色"),)
    cost = Cost(gpu=True)

    @classmethod
    def prepare(cls, ctx) -> Job:
        """将网格合并为一张交给 worker；notes["meshes"] 供 convert 将权重拆回各网格。"""
        from lab2shot_worker.rigging import write_mesh_job

        src = ctx.input("model")
        frames = [int(f) for f in src.meta["frames"]]
        meshes = one_mesh(src, frames[0])
        if len(frames) > 1:
            ctx.say("N-RIG-DEFORMING", frame=frames[0], frames=len(frames))
        ctx.stage("整理网格")
        path = write_mesh_job(ctx.work / "mesh.npz", meshes.points, meshes.faces)
        return Job(None, inputs={"mesh": path}, notes={"meshes": meshes, "model": src})

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        from lab2shot_worker.rigging import read_rig

        from ...data.scene import bind_skin
        from ...data.skeleton import rig_of_model

        meshes: Meshes = job.notes["meshes"]
        src: Packet = job.notes["model"]
        rig = read_rig(raw)
        weights = np.asarray(rig["weights"], np.float64)
        if weights.shape[0] != len(meshes.points):
            raise Invalid(Msg("E-RIG-VERTEXCOUNT", got=int(weights.shape[0]), want=len(meshes.points)))
        ctx.stage("整理蒙皮")
        idx, w = top_influences(weights, int(ctx.params["influences"]))
        joints = np.asarray(rig["joints"], np.float64)
        parents = np.asarray(rig["parents"], np.int64)
        bind = np.repeat(np.eye(4)[None], len(joints), 0)
        bind[:, :3, 3] = joints
        # 骨骼名和关节轴经过项目中唯一的转换入口（data/skeleton.py），与解算器输出的蒙皮角色完全一致
        names, bind, _ = rig_of_model([str(n) for n in rig["names"]], parents, bind, bind[None])
        ctx.stage("写出蒙皮角色")
        name = str(ctx.params["character"]).strip()
        packet = bind_skin(src, ctx.outputs["character"], joint_names=names, parents=parents, bind_world=bind,
                           joint_indices=meshes.split(idx), joint_weights=meshes.split(w), name=name,
                           info={"extension": cls.runtime, **rig["info"]})
        ctx.say("I-RIG-DONE", joints=len(joints), vertices=len(meshes.points))
        return {"character": packet}


def top_influences(weights: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """逐顶点保留权重最大的 `k` 根骨骼（Maya 的 maxInfluences），归一化后返回 (骨骼号 [V,k], 权重 [V,k])。
    未受任何骨骼影响的顶点（模型给出全 0 的一行）归属第 0 根骨骼，不留下无归属的点。"""
    k = min(int(k), weights.shape[1])
    idx = np.argsort(-weights, axis=1)[:, :k]
    w = np.take_along_axis(weights, idx, 1)
    total = w.sum(1, keepdims=True)
    lost = (total <= 0).reshape(-1)
    w[lost] = 0.0
    w[lost, 0] = 1.0
    idx[lost, 0] = 0
    total = np.where(total <= 0, 1.0, total)
    return idx.astype(np.int64), w / total
