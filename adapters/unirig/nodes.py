"""Nodes provided by the UniRig extension."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import Official, AutoRig, AutoRigParams, Cost, P


class UniRigAutoRig(AutoRig):
    id = "unirig.auto_rig"
    # 引的是上游自己的数据结构 RawData（run.py 的 --input 读进来的网格、两个阶段写出的 predict_skeleton.npz /
    # predict_skin.npz 都是它）：进去的是 vertices / faces，出来的是 joints / skin / parents / names。
    official = Official(
        cite="third_party/unirig/repo/src/data/raw_data.py:14-54",
        takes={"model": "vertices"},
        gives={"character": "skin"},
    )
    runtime = "unirig"
    version = 4  # 4：回退 3 的 Z 向上送入（实测两个标准人形更差，worker.py 注释）；3：试过按官方 Z 向上送入；2：骨骼按项目命名规范改名（识别引擎按形状认部位，data/bone_names.py），参数「骨骼命名」可选原始名
    # docs.md：一张网格算一次，和帧数无关；面数超过 5 万先按官方做法降面再算，权重铺回原网格每一个顶点
    # vram_gb / whole: RTX 4090 上测得（docs.md）
    cost = Cost(gpu=True, vram_gb=3.7, whole=True)

    class Params(AutoRigParams):
        seed: int = P(12345, group="model", ge=0)
        # upstream run.py 的 --cls：生成的起始 token。人形选 VRoid 模板，骨骼带名字（J_Bip_C_Hips……），重定向直接认得
        template: Literal["auto", "vroid"] = P("auto", group="model")


NODES = (UniRigAutoRig,)
