"""SATA: official learned motion retargeting."""
from typing import Literal
from lab2shot.sdk import RigRetarget, RigRetargetParams, Official, P, Cost, JointNeeds

class SataRetarget(RigRetarget):
    id = "sata.retarget"
    runtime = "sata"
    cost = Cost(gpu=True, vram_gb=3.0)
    version = 8  # 8：只为让半途代码的缓存重算（预处理的参考姿态改为站在动画的地面上）；7：「模型骨架范围」改为预处理的两条忽略规则；不再输出适配预览
    # 官方人体图收躯干和四肢（解算器不自己丢骨骼：手指、面部、辅助骨在「重定向预处理」里按规则忽略）；加手指是实验
    needs = {"sata": JointNeeds(frozenset({"body"})),
             "sata_fingers": JointNeeds(frozenset({"body", "fingers"}))}
    official = Official(cite="third_party/sata/repo/src/sata/mymodel.py:884-900", takes={"source": "src_graph", "target": "tgt_graph"},
                        gives={"skeleton": "hatD"})
    class Params(RigRetargetParams):
        source_toe_semantics: Literal['auto', 'base', 'big_toe'] = P('auto', group='joint_semantics')
        target_toe_semantics: Literal['auto', 'base', 'big_toe'] = P('auto', group='joint_semantics')
        model: Literal["vae_human", "vae_merge", "vae_animo", "rvq_human"] = P("vae_human", group="model")
        window: int = P(64, group="model", ge=8, le=256, unit="frame")
        seed: int = P(0, group="model", ge=0)


NODES = (SataRetarget,)
