"""Nodes provided by the SMIRK extension (MIT code; FLAME 2020 and the training data research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import rgb_port, Official, CameraLensParams, P, Port, WorldHumans, curves_packet, Cost


class Face(WorldHumans):
    id = "smirk.face"
    # 引的是官方的解算器本体 SmirkEncoder：一张裁好的脸（img）进去，FLAME 的一套参数出来
    # （pose_params / cam / shape_params / expression_params / eyelid_params / jaw_params）。
    official = Official(
        cite=("third_party/smirk/repo/src/smirk_encoder.py:34-133",
              "third_party/smirk/repo/src/FLAME/FLAME.py:160-190"),
        takes={"image": "img"},
        gives={"character": "shape_params", "expressions": "expression_params"},
        # 「相机」不是上游的输出：它是本节点把弱透视换成针孔时用的那台相机（原点、不动，Focal Length 为节点上的「Focal
        # Length」，不填按 50 mm），交出来让三维视图透过它看背板、交付时头和画面对得上
        ours={"camera": "放脸用的针孔相机：原点、不动，焦距 = 「Focal Length」（不填按全画幅 50 mm），主点在画面中心"},
        note="① **官方的整套 FLAME 参数就是「蒙皮角色」这个口**，没有另立类型、也没有另加口（SMPL / SMPL-X / MANO / FLAME / MHR "
             "这类参数化人体就是「蒙皮 + 权重 + 骨架动画」，装成「蒙皮角色」，不另立数据类型）。"
             "逐项对上（worker.py face_rig）：`shape_params`（300 个，整段锁成中位数）变成这张脸的静止网格和"
             "静止骨架（npz rest_vertices / rest_joints），配 FLAME 自己的蒙皮权重（npz skin_weights）；"
             "`pose_params`（头的旋转）变成骨架根关节每帧的旋转（npz local_rotations[:, 0]，含 FLAME 空间转"
             "相机空间的那一次固定旋转）；`jaw_params` 变成下巴关节的旋转（local_rotations[:, 2]）；"
             "`expression_params`（50 个）和 `eyelid_params`（2 个）变成「蒙皮角色」上的 52 条 blendShape"
             "（npz blendshapes / blendshape_names / blendshape_weights），同一份数字另外走「表情曲线」这个口。"
             "**一个参数都没丢。**"
             "② `cam` 不在 gives 里，因为它**不是**官方的相机：它是 224×224 裁切上的弱透视三个数"
             "（缩放 + 平移，smirk_encoder.py:43），我们只拿它算出头离镜头多远（worker.py perspective）。"
             "「相机」口交出的是放脸用的那台针孔相机（原点、不动，焦距 = 节点的「Focal Length」，不填按 50 mm），"
             "登记在 ours：三维视图透过它看背板，头和画面对得上。"
             "头就留在相机空间；要摆进某台相机的世界，"
             "接核心节点「相机空间转换」（core.camera_space）。"
             "③ 官方的网格在另一个文件 "
             "`src/FLAME/FLAME.py:310-314` 的 flame_output['vertices']（demo.py:110 flame.forward(outputs)），"
             "是 FLAME 拿这些参数跑出来的，不是编码器直接吐的。",
    )
    on_node = ("focal_mm", "crop")
    # 脸够大、正脸到大半侧脸；每帧单独计算，没有时序平滑；
    # Focal Length 只决定头离镜头的远近（不填按全画幅 50 mm 估算）。
    # 只有「RGB」一个输入口：官方的 SmirkEncoder 只吃一张裁好的脸，相机一个字节都进不去。
    # 「相机」输出口（家族的 plate_camera）：放脸用的那台原点静止针孔相机，见 official 的 ours
    inputs = (rgb_port(),)
    plate_camera = True
    outputs = WorldHumans.outputs + (Port("expressions", "curves", "表情曲线"),)
    runtime = "smirk"
    camera_to_worker = None
    default_focal_mm = 50.0  # SMIRK's camera is orthographic: the pinhole it is turned into needs a lens
    # RTX 4090
    cost = Cost(gpu=True, vram_gb=0.6, seconds_per_frame=0.13)

    class Params(CameraLensParams):
        crop: Literal["auto", "none"] = P(
            "auto", label="裁切", group="面部",
            option_labels={"auto": "自动找脸", "none": "整幅画面"},
        )

    @classmethod
    def convert(cls, ctx, raw, job):
        """The world-humans outputs, plus the face's 52 expression curves (held over frames without a face)."""
        import numpy as np

        image = job.plate
        out = super().convert(ctx, raw, job)
        d = raw.arrays("person_01.npz")
        frames, own = image.meta["frames"], d["frames"]
        weights = d["blendshape_weights"]
        values = np.stack([np.interp(frames, own, weights[:, k]) for k in range(weights.shape[1])], 1)  # frames without a face: held
        out["expressions"] = curves_packet(ctx.outputs["expressions"], frames,
                                           [str(n) for n in d["blendshape_names"]], values, extension=cls.runtime)
        return out


NODES = (Face,)
