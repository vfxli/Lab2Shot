"""绑定家族的 worker 契约（nodes/families/rigging.py AutoRig）：一张没有骨骼的三角网格进，一副骨架和
逐顶点的蒙皮权重出。族里每个项目的 worker 写的是同一份原始结果，节点那边只有一段读法。

节点写 job 的输入文件 `mesh`：

    mesh.npz    points [V,3] float32（厘米、Y 向上、世界坐标），faces [T,3] int32（三角形，索引进 points）

worker 写 raw/rig.npz：

    joints  [J,3] float32   每根骨骼的起点（关节位置），和 points 同一个坐标系、同一个单位
    parents [J]   int32     父骨骼的下标，根是 -1，父一定排在子前面
    names   [J]   str       模型自己给的骨骼名（认得出部位的，节点那边会换成 CG 的名字）
    weights [V,J] float32   每个顶点对每根骨骼的权重，每行和为 1（全 0 的行由节点接手）

加上 raw/result.json（`write_result`）：这次用了什么模型、花了多久、显存峰值，节点写进交付物的来源信息里。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import fail
from .run import Run


def write_mesh_job(path: Path, points, faces) -> Path:
    """节点这边：把拼好的网格写成 worker 读的 mesh.npz。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, points=np.asarray(points, np.float32), faces=np.asarray(faces, np.int32))
    return path


def read_mesh_job(path) -> tuple[np.ndarray, np.ndarray]:
    """worker 这边：读回 (points [V,3] float64, faces [T,3] int64)。"""
    d = np.load(path)
    return d["points"].astype(np.float64), d["faces"].astype(np.int64)


def write_rig(run: Run, joints, parents, names, weights, **info) -> None:
    """worker 这边：写 raw/rig.npz 和 raw/result.json（worker 自己的字段加 Run 的标准计时、显存字段）。
    形状对不上当场停下，不把错的东西交给节点。"""
    joints = np.asarray(joints, np.float32).reshape(-1, 3)
    parents = np.asarray(parents, np.int32).reshape(-1)
    weights = np.asarray(weights, np.float32)
    names = [str(n) for n in names]
    if not (len(joints) == len(parents) == len(names) == weights.shape[1]):
        fail("E-RIGGING-SHAPES", joints=len(joints), parents=len(parents), names=len(names), columns=int(weights.shape[1]))
    if (parents[1:] >= np.arange(1, len(parents))).any() or parents[0] != -1:
        fail("E-RIGGING-ORDER")
    raw = run.job.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    np.savez(raw / "rig.npz", joints=joints, parents=parents, names=np.array(names), weights=weights)
    run.finish([], **info)  # a rig has no frames: the standard fields say so


def read_rig(raw) -> dict:
    """节点这边：raw/rig.npz + raw/result.json 读成一份字典。"""
    d = raw.arrays("rig.npz")
    return {"joints": d["joints"], "parents": d["parents"], "names": [str(n) for n in d["names"]],
            "weights": d["weights"], "info": raw.result()}
