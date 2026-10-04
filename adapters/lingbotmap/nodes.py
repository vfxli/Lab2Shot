"""Nodes provided by the LingBot-Map extension (Apache-2.0 code and weights; parts of the code from VGGT)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, P, WholeShotDepthCamera, WholeShotParams, conf_threshold_param,
                          resolution_param, Cost, Licence, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "lingbotmap.reconstruct"
    metric = False  # its units are arbitrary: the 「尺度」 says how many centimetres one is (DepthCamera.metric)
    # 输入只有画面：GCT.forward 的文档串只写 images 和 query_points，签名里那个 mask 参数不是画面遮罩
    # （gct_base.py:294 传下去的是 ordered_video）。家族的默认口里还带着「运动物体遮罩」「人物框」，那是给真的
    # 吃遮罩的 MonST3R 留的，这里筛掉。想只重建画面的一部分，在送进去之前把其余部分涂黑：
    # 「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个「RGB」口
    # 缓存版本：有效位不抹掉运动像素（节点没有遮罩输入口），与含义不同的旧缓存区分开
    version = 10  # 8：竖画面按官方 crop 模式（宽 518、取中间一块）；9：超长镜头 / 设了每段帧数时按官方 windowed 推理（流式）；10：摄影机按模型本来的 camera-to-world 交出（之前多求了一次逆，朝向是反的）
    # 归在「深度与法线」而不是「三维重建」：网络结构上有 world_points 头（gct_base.py:226），但放出的权重没有
    # point_head（见 worker.py），演示用的 GCTStream 也默认 enable_point=False（gct_stream.py:118），
    # 官方查看器是用深度反投影的——不算原生的场景点云
    # 有移动的镜头最好，固定机位 124 帧漂移 1.2%；流式——5084 帧显存仍是 11.7 GB，不随长度增长；
    # 尺度不是米（开头几帧的平均距离当 1 个单位）；模型看的是整幅画面，运动物体也在里面
    runtime = "lingbotmap"
    # GCT.forward(images, query_points=None, …) 交出 pose_enc / depth / depth_conf / world_points /
    # world_points_conf（gct_base.py:309-314）。
    official = Official(
        cite="third_party/lingbotmap/repo/lingbot_map/models/gct_base.py:287-315",
        takes={"image": "images"},
        gives={"depth": "depth", "camera": "pose_enc"},
    )
    confidence = Confidence(
        "exp_plus_one", help=True)  # how its model gives its confidence (CONFIDENCE_SCALES); what it means: node."lingbotmap.reconstruct".port.confidence.help
    # RTX 4090 上量得的显存和速度：流式架构，显存不随帧数增长
    cost = Cost(gpu=True, vram_gb=11.7, seconds_per_frame=0.15)
    licence = Licence(note=True)

    class Params(WholeShotParams):
        model: Literal["long", "balanced"] = P(
            "long", group="solve",
        )
        max_frames: Literal[100, 300, 1000, 3000] | None = measured_param(
            {100: Measured(gb=11.7), 300: Measured(gb=11.7), 1000: Measured(gb=11.7), 3000: Measured(gb=11.7)}, auto=True, group="solve")
        resolution: Literal[280, 392, 518] = resolution_param(
            {280: Measured(below=518), 392: Measured(below=518), 518: Measured(gb=11.5)}, default=518)
        keyframe_interval: Literal[0, 16, 32, 64] = measured_param(
            {0: Measured(gb=11.7), 16: Measured(below=0), 32: Measured(below=0), 64: Measured(below=0)}, default=0, group="solve")
        context_window: Literal[16, 32, 64] = measured_param(
            {16: Measured(below=64), 32: Measured(below=64), 64: Measured(gb=11.5)}, default=64, group="solve")
        conf_threshold: float = conf_threshold_param(1.5, 1000.0)
        mask_sky: bool = P(False, group="scene")


NODES = (Reconstruct,)
