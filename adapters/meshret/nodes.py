"""MeshRet: official learned motion retargeting."""
from typing import Literal
from lab2shot.sdk import RigRetarget, RigRetargetParams, Official, P, Cost, JointNeeds
from lab2shot_worker.rig_retarget import MESHRET_FULL_PARTS

class MeshretRetarget(RigRetarget):
    id = "meshret.retarget"
    runtime = "meshret"
    cost = Cost(gpu=True, vram_gb=3.0)
    version = 10  # 10：只为让半途代码的缓存重算；9：只为让 8 的缓存重算一次（试过的头顶估法已撤回，结果不变）；8：固定 65 槽部位表，手指槽可缺（optional）
    # 官方输入是 65 关节的扩展 Mixamo 骨架，含手指（已发布模型只驱动其中 25 个身体关节，手指仍是输入：掌心的表面
    # 传感器按手腕到中指根算）。所以规则收身体和手指，固定部位表就是这 65 槽（worker fixed_body 按它投影）；手指槽
    # 可缺（AccuRIG 每指 3 节、没有手指的骨架），缺的由适配层补标记，末端点（-1）也由适配层补。两侧都要蒙皮网格
    needs = {"meshret": JointNeeds(frozenset({"body", "fingers"}),
                                   MESHRET_FULL_PARTS, meshes=("src", "dst"), optional=frozenset({"fingers"}))}
    official = Official(cite="third_party/meshret/repo/run/demo.py:63-74", takes={"source": "x", "target": "y"},
                        gives={"skeleton": "x_hat"})
    class Params(RigRetargetParams):
        window: Literal[30] = P(30, group="model", widget="fixed", unit="frame")


NODES = (MeshretRetarget,)
