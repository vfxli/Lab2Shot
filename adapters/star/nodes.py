"""STaR: official learned motion retargeting."""
from typing import Literal
from lab2shot.sdk import RigRetarget, RigRetargetParams, Official, P, Cost, JointNeeds
from lab2shot_worker.rig_retarget import STAR_PARTS

class StarRetarget(RigRetarget):
    id = "star.retarget"
    runtime = "star"
    cost = Cost(gpu=True, vram_gb=3.0)
    version = 8  # 8：只为让半途代码的缓存重算；7：收什么骨骼改由预处理的忽略规则定；不再输出适配预览
    # 官方 22 关节 BVH：固定的身体部位表（链多出的中间节在预处理里忽略），两侧都要蒙皮网格（形状编码器读表面点）
    needs = {"star": JointNeeds(frozenset({"body"}), STAR_PARTS, meshes=("src", "dst"))}
    official = Official(cite="third_party/star/repo/method/network.py:283-344", takes={"source": "quatA_norm_cp", "target": "skelB_norm"},
                        gives={"skeleton": "quatB_rt"})
    class Params(RigRetargetParams):
        window: Literal[120] = P(120, group="model", widget="fixed", unit="frame")
        seed: int = P(42, group="model", ge=0)


NODES = (StarRetarget,)
