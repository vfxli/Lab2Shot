"""Kinematic Refinement: official learned motion retargeting."""
from typing import Literal
from lab2shot.sdk import RigRetarget, RigRetargetParams, Official, P, Cost, Param, JointNeeds

class KinematicRefinementRetarget(RigRetarget):
    id = "kinematic_refinement.retarget"
    runtime = "kinematic_refinement"
    cost = Cost(gpu=True, vram_gb=3.0)
    version = 9  # 9：geo 阶段只要目标的蒙皮网格（官方 geo 模型只对目标网格做几何修正，源只经运动学编码器，源网格不读）；8：只为让半途代码的缓存重算；7：「模型骨架范围」改为预处理的两条忽略规则；不再输出适配预览
    # 官方人体图收躯干和四肢；加手指是实验。geo 阶段目标要蒙皮网格，按参数定，见 required_mesh_sides
    needs = {"kinref": JointNeeds(frozenset({"body"})),
             "kinref_fingers": JointNeeds(frozenset({"body", "fingers"}))}
    official = Official(cite="third_party/kinematic_refinement/repo/src/kinref/geo_test.py:134-142", takes={"source": "src_batch", "target": "tgt_batch"},
                        gives={"skeleton": "out_motion_list"})
    class Params(RigRetargetParams):
        # 默认 geo：论文的最终方法（几何感知精修，在 kin 先验上精修），目标要蒙皮网格；kin 只用先验阶段
        stage: Literal["geo", "kin"] = P("geo", group="model")
        geo_batch_frames: int = P(16, group="model", ge=1, le=128,
            applies=Param('stage').one_of('geo'))

    @classmethod
    def required_mesh_sides(cls, params):
        # geo_model.GeoModel.forward: the source goes only through the kinematic encoder (kin_model.encode, skeleton and
        # pose); the mesh terms (static features, skinning, penetration) read the target graph alone. A source with
        # no mesh (a BVH) is as good as one with
        return ("dst",) if params["stage"] == "geo" else ()

NODES = (KinematicRefinementRetarget,)
