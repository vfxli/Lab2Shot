"""Nodes provided by the LingBot-Map extension (Apache-2.0 code and weights; parts of the code from VGGT)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, P, WholeShotDepthCamera, WholeShotParams, conf_threshold_param,
                          resolution_param, unit_cm_param, Cost, Licence, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "lingbotmap.reconstruct"
    # 输入只有画面：GCT.forward 的文档串只写 images 和 query_points，签名里那个 mask 参数不是画面遮罩
    # （gct_base.py:294 传下去的是 ordered_video）。家族的默认口里还带着「运动物体遮罩」「人物框」，那是给真的
    # 吃遮罩的 MonST3R 留的，这里筛掉。想只重建画面的一部分，在送进去之前把其余部分涂黑：
    # 人物检测 → 人物框转遮罩 → 图像相乘 → 这个「图像」口
    # 7: 去掉「运动物体遮罩」「人物框」输入口之后，结果里不再抹掉运动像素，有效位变了，旧缓存作废
    version = 7
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
        note="相机是官方位姿编码 pose_enc（gct_base.py:310），由上游自己的 pose_encoding_to_extri_intri 换成"
             "外参加内参（gct_base.py:260）。节点没有「运动物体遮罩」「人物框」输入口："
             "forward 里那个 mask 参数是给 ordered_video 用的（gct_base.py:293-296），不是画面遮罩，"
             "画面遮罩进不了模型，只能在算完之后把运动像素从有效位里去掉，这是我们自己加的一步，不放在解算器上。"
             "想局部重建走「人物检测 → 人物框转遮罩 → 图像相乘」。"
             "世界点图 world_points 写在 forward 的文档串里（gct_base.py:312），但**放出来的权重里没有这个头**："
             "实测 lingbot-map.pt / lingbot-map-long.pt 的键只有 aggregator (1211)、"
             "camera_head (69)、depth_head (62)，一个 point_head. 都没有，而流式模型默认也不建它"
             "（gct_stream.py:118 enable_point=False）。所以这个节点没有「点云」口——不是我们丢了官方的结果，"
             "是官方没放这部分权重",
    )
    confidence = Confidence(
        "exp_plus_one",  # how its model gives its confidence (CONFIDENCE_SCALES)
        help="每个像素这一帧的深度有多可信（LingBot-Map 的原始值 1 + exp(x)，换成 0–1）。"
             "这个解算给出的深度本身已经把深度边缘和天空排除在外（「深度」的有效区就是它的可用遮罩），"
             "所以剩下的像素之间比，主要是「越远越不确定」")
    # RTX 4090 上量得的显存和速度：流式架构，显存不随帧数增长
    cost = Cost(gpu=True, vram_gb=11.7, seconds_per_frame=0.15)
    licence = Licence(note="代码和权重 Apache-2.0，可以商用；其中来自 VGGT 的几个代码文件遵守 VGGT License（可商用，禁止军事等用途）。")

    class Params(WholeShotParams):
        model: Literal["long", "balanced"] = P(
            "long", label="模型", group="解算",
            option_labels={"long": "长镜头", "balanced": "均衡"},
            help="长镜头：为长序列训练，相机轨迹更稳；均衡：Focal Length 估得稍准，相机轨迹误差更大",
        )
        max_frames: Literal[100, 300, 1000, 3000] | None = measured_param(
            "每段最多帧数", {100: Measured("300 帧强制分 4 段 50 秒，显存同下", gb=11.7), 300: Measured("每秒约 7 帧", gb=11.7), 1000: Measured("992 帧实测，显存同 300", gb=11.7), 3000: Measured("2976 帧实测，显存同 300、内存 2.9 GB", gb=11.7)}, auto="自动", group="解算",
            help="连续读多少帧后分段再拼接。留空 = 一口气读完（最多 3000 帧，更长自动分段）；分段越多拼接误差越大，只在镜头里场景完全变了时才手动分段。"
                 "它是流式模型，显存不随帧数涨")
        resolution: Literal[280, 392, 518] = resolution_param(
            {280: Measured("比实测的一档省", below=518), 392: Measured("比实测的一档省", below=518), 518: Measured("训练尺寸：124 帧 19 秒", gb=11.5)}, default=518, note="；518 是模型的训练尺寸，最稳")
        keyframe_interval: Literal[0, 16, 32, 64] = measured_param(
            "关键帧间隔", {0: Measured("自动：5084 帧实测", gb=11.7), 16: Measured("存进长期记忆的帧不多于自动（每段最多 3000 帧时自动间隔不超过 10），显存不高于自动", below=0), 32: Measured("同上，更省", below=0), 64: Measured("同上，最省", below=0)}, default=0, group="解算",
            help="每隔几帧把一帧存进长期记忆。0 = 自动（320 帧以内每帧都存，更长的按比例）；很长的镜头调大更快、更省")
        context_window: Literal[16, 32, 64] = measured_param(
            "短期记忆帧数", {16: Measured("比实测的一档省", below=64), 32: Measured("比实测的一档省", below=64), 64: Measured("官方默认：124 帧实测", gb=11.5)}, default=64, group="解算",
            help="每帧参考最近多少帧（官方 64）。快速运动的镜头可以调小；更大的没测过、不给选")
        conf_threshold: float = conf_threshold_param(1.5, 1000.0)
        mask_sky: bool = P(False, label="去掉天空", help="用天空分割模型把天空从深度和拼接里去掉（室外镜头有用）。在 CPU 上跑，每 300 帧多约 50 秒", group="场景")
        unit_cm: float = unit_cm_param("LingBot-Map 的尺度要手动缩放：先按米（100）放，和 ViPE 或实测距离对比后再调")

    @classmethod
    def prepare(cls, ctx):
        return super().prepare(ctx).with_(notes={"scale": "relative"})  # the model's units are arbitrary


NODES = (Reconstruct,)
