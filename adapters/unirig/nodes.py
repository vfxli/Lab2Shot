"""Nodes provided by the UniRig extension."""

from __future__ import annotations

from lab2shot.sdk import Official, AutoRig, AutoRigParams, Cost, P


class UniRigAutoRig(AutoRig):
    id = "unirig.auto_rig"
    # 引的是上游自己的数据结构 RawData（run.py 的 --input 读进来的网格、两个阶段写出的 predict_skeleton.npz /
    # predict_skin.npz 都是它）：进去的是 vertices / faces，出来的是 joints / skin / parents / names。
    official = Official(
        cite="third_party/unirig/repo/src/data/raw_data.py:14-54",
        takes={"model": "vertices"},
        gives={"character": "skin"},
        note="「人物」这一个口对应上游第二阶段写出的 predict_skin.npz（RawSkin：skin / vertices / joints，"
             "src/system/skin.py:277-278）加第一阶段的 predict_skeleton.npz（joints / parents / names，"
             "src/system/ar.py:343-345）——官方就是靠这两份 npz 合成一副能动的绑定，没有别的产物。",
    )
    runtime = "unirig"
    # docs.md：一张网格算一次，和帧数无关；面数超过 5 万先按官方做法降面再算，权重铺回原网格每一个顶点
    # vram_gb / whole: RTX 4090 上测得（docs.md）
    cost = Cost(gpu=True, vram_gb=3.8, whole="一块网格算一次，不按帧算：骨架和权重两个阶段各跑一遍，约一分钟")

    class Params(AutoRigParams):
        seed: int = P(12345, label="随机种子", group="模型", ge=0,
                      help="骨架是一根一根生成出来的，同一个种子得到同一副骨架。骨头长错地方、少了尾巴或翅膀，"
                           "就换一个数字重算，挑一副最好的（官方默认 12345）")


NODES = (UniRigAutoRig,)
