"""Nodes provided by the VGGT extension (code: VGGT License; weights CC-BY-NC-4.0 or VGGT License)."""

from __future__ import annotations

from typing import Literal

from .vggt_models import OPTION_LICENCES
from lab2shot.sdk import (licence_traits, Official, Confidence, P, WholeShotDepthCamera, WholeShotParams, loops_param, max_frames_param,
                          resolution_param, Cost, Licence, Measured)


# 默认设置（原版权重、518、每段 130 帧）下实测的显存峰值：1080×1920 竖幅 150 帧分两段（docs.md 实测表）。
# 节点的 Cost 和「处理分辨率」518 那一档都用它；一整段 130 帧竖幅一次算完的 20.0 GB 记在「每段最多帧数」上
DEFAULT_VRAM_GB = 14.4


class Reconstruct(WholeShotDepthCamera):
    id = "vggt.reconstruct"
    metric = False  # its units are arbitrary: the 「尺度」 says how many centimetres one is (DepthCamera.metric)
    # 官方的输入等于解算器的输入：VGGT.forward 只吃 images，所以不声明 `takes_mask`，节点上没有遮罩口
    # （那是给真的吃遮罩的上游留的，如 MonST3R 的 dynamic_mask_path）。想只重建画面的一部分，在送进去之前把其余部分涂黑：
    # 「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个「RGB」口，图上一眼看得见
    native_points = "points"  # worker 交出 VGGT 自己的 world_points（放在每帧的相机空间里）
    # 结果变了就加一，让旧缓存作废（7：raw 里带官方的 points，运动像素不抹掉）
    version = 7
    # docs.md：有视差的镜头和静止机位都可用（固定机位位移只有场景深度的 0.3%）；尺度是自己归一化的任意单位；
    # 模型看的是整幅画面，运动物体也在里面；每段帧数安全上限 130（按竖画面测得）
    runtime = "vggt"
    # 官方的输入等于解算器的输入、输出等于输出：
    # VGGT.forward(images, query_points=None) 交出 pose_enc / depth / depth_conf / world_points /
    # world_points_conf（+ 给了 query_points 时的 track / vis / conf），vggt.py:29-52。
    official = Official(
        cite="third_party/vggt/repo/vggt/models/vggt.py:29-52",
        takes={"image": "images"},
        gives={"depth": "depth", "camera": "pose_enc", "points": "world_points"},
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # vram_gb: RTX 4090 上测得（docs.md），默认设置（DEFAULT_VRAM_GB）
    cost = Cost(gpu=True, vram_gb=DEFAULT_VRAM_GB, seconds_per_frame=0.2)
    licence = Licence(note=True)
    traits = licence_traits(OPTION_LICENCES)

    class Params(WholeShotParams):
        model: Literal["original", "commercial"] = P("original", group="solve")
        resolution: Literal[280, 392, 518] = resolution_param(
            {280: Measured(below=518), 392: Measured(below=518), 518: Measured(gb=DEFAULT_VRAM_GB)}, default=518)
        # 一次性把整段看完（不是流式），24G 显卡上：竖幅约 130 帧、横幅约 230 帧都在 20 GB 内；按竖幅的更紧上限统一封顶
        max_frames: Literal[32, 64, 130] = max_frames_param(
            {32: Measured(below=130), 64: Measured(below=130), 130: Measured(gb=20.0)},
            default=130)
        loops: bool = loops_param()


NODES = (Reconstruct,)
