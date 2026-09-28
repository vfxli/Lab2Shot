"""Nodes provided by the VideoMaMa extension (code CC-BY-NC-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import Official, measured_param, GuidedMatte, NodeParams, P, fp16_param, Cost, Licence, Measured


# 默认处理分辨率 1024 下实测的显存峰值（docs.md：1080×1920 60 帧 14.8 GB，864×480 12.2 GB 取高的那个）。
# 节点的 Cost 和「处理分辨率」1024 那一档都用它
DEFAULT_VRAM_GB = 14.8


class Matte(GuidedMatte):
    id = "videomama.matte"
    # 上游 inference_onestep_folder.py：pipeline.run(cond_frames=画面, mask_frames=每帧粗遮罩) -> generated_frames
    official = Official(
        cite="third_party/videomama/repo/inference_onestep_folder.py:359-363",
        takes={"image": "cond_frames", "mask": "mask_frames"},
        gives={"alpha": "generated_frames"},
        note="上游每一帧都要一张 mask_frames（load_image_sequence 找不到配对的遮罩就直接报错，第 78-79 行）。",
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   mask：实测（CRGNN 实拍、VideoMatte 绿幕共 8 个人像镜头）：粗遮罩每帧都要有，只给第一帧时后面的帧会跟丢（SAD 468）；BiRefNet 的遮罩比 SAM 3 的好（8 个镜头 7 个更好、1 个一样，SAD 8.1 对 9.8）
    # docs.md：任何主体都行；每次算 16 帧、相邻段重叠 4 帧交叉淡化，显存只和这 16 帧有关、和镜头长度无关；
    # 边缘过渡最柔和（人物边缘 alpha 跳动 0.030，三者最稳）
    runtime = "videomama"
    # vram_gb: RTX 4090 上测得（docs.md），默认处理尺寸 1024（DEFAULT_VRAM_GB）
    cost = Cost(gpu=True, vram_gb=DEFAULT_VRAM_GB, seconds_per_frame=0.42)
    licence = Licence(note="代码是 CC-BY-NC-4.0，只能研究用。模型权重是 Stability AI Community License"
        "（商用需注册、年收入低于 100 万美元、注明 Powered by Stability AI），但因代码许可整体只能研究用。")

    class Params(NodeParams):
        # 只测过官方尺寸 1024（12-15 GB）；2048 官方也没验证过效果，不开放：只给测过、确认安全的范围
        resolution: Literal[512, 768, 1024] = measured_param(
            "处理分辨率", {512: Measured(below=1024), 768: Measured(below=1024), 1024: Measured(gb=DEFAULT_VRAM_GB)},
            default=1024, group="抠像")
        erode_dilate: int = P(0, label="遮罩扩缩", unit="px", ge=-50, le=50, group="抠像")
        fp16: bool = fp16_param("抠像")  # 关掉时用 bfloat16，显存多很多


NODES = (Matte,)
