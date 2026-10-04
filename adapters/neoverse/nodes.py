from typing import Literal
from lab2shot.sdk import GaussianReconstruction, Measured, Official, P, Cost, measured_param


class Reconstruct(GaussianReconstruction):
    id = "neoverse.reconstruct"
    version = 5  # 5：use_motion=False（官方 inference.py:33）；4：分段尺度取重叠帧的场景深度比（不再取几毫米的相机基线）；3：轴向与单位烘进高斯本体，分段对齐连同球谐一起烘；分段内参按画面缩放比还原
    runtime = "neoverse"
    category = "gaussian"
    official = Official(cite=("third_party/neoverse/repo/diffsynth/auxiliary_models/worldmirror/models/models/worldmirror.py:191-340",
                              "third_party/neoverse/repo/diffsynth/auxiliary_models/worldmirror/models/models/rasterization.py:545-553"),
                        takes={"image": "imgs"}, gives={"gaussians": "splats", "camera": "rendered_extrinsics"})
    cost = Cost(gpu=True, vram_gb=18.3, measured_on="RTX 5090 32 GB", note=True)  # 1280x720、每段 32 帧、官方 560 实测峰值

    class Params(GaussianReconstruction.Params):
        resolution: Literal[280, 420, 560] = P(560, group="solve")  # 官方 560x336（inference.py:142-145）
        # 超过上限分段解算，按重叠帧的相机对齐到同一世界。显存随每段帧数近乎线性增长：只给测过的档（measured_param）；
        # 32 是实测档，更少帧不会更多显存
        max_frames: Literal[8, 16, 32] = measured_param({8: Measured(below=16), 16: Measured(below=32), 32: Measured(gb=18.3)},
                                                        default=32, group="solve")


NODES = (Reconstruct,)
