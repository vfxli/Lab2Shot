"""Nodes provided by the SegAnyMo extension (MIT; every model it runs may be used commercially)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, NodeParams, P, Port, Segmentation, Cost,
                          Licence, Measured)


class MovingObjects(Segmentation):
    id = "seganymo.moving_objects"
    # 上游 sam2/run_sam2.py main()：args.video_dir（画面目录）-> video_segments（每帧每个物体一张遮罩）
    official = Official(
        cite="third_party/seganymo/repo/sam2/run_sam2.py:595-700",
        takes={"image": "video_dir"},
        gives={"mask": "video_segments", "objects": "out_obj_ids"},
        note="上游整条链（深度、轨迹、DINO 特征、运动分割、SAM 2 分组）都从同一份画面自己算出来，"
             "除了画面不吃别的东西。",
    )
    # 主要用途是给相机解算挡掉运动物体；画面里什么都没动时交出空遮罩并提示；
    # 整段均匀挑 100 帧判断谁在动（遮罩每帧都有）；边来自 SAM 2，适合挡解算、不是精细抠像
    inputs = (Port("image", "image.3", "RGB"),)
    # 两个输出口由分割家族给（families/segmentation.py）：口名 `mask` / `objects` 和「物体分割」这个标签是家族定的，
    # 「运动」是这个节点的事（它就叫「SegAnyMo 运动物体遮罩」），不是口的事
    mask_label = "运动物体遮罩"
    mask_half = True  # 遮罩是 0/1：值域声明出来、写半精度
    runtime = "seganymo"
    # ram_gb: every frame at 1024 x 1024 for SAM 2 (12 MB each) and every object's mask on every frame
    # vram_gb: RTX 4090（1280×534，默认参数，全新进程）：PyTorch 峰值 6.96 GB，整卡上涨 7.68 GB（含约 0.47 GB CUDA 上下文），48 帧和 100 帧一样、不随帧数涨
    cost = Cost(gpu=True, vram_gb=7.2, ram_gb=16, note="多个模型一起跑的流水线")
    licence = Licence(note="代码 MIT；用到的模型 SAM 2、DINOv2、BootsTAPIR、Depth Anything V2 Small 都可以商用。"
        "训练数据含研究数据集，严格的商业交付前建议做一次法务确认。")

    class Params(NodeParams):
        resolution: Literal[640, 1000] = measured_param(
            "处理分辨率", {640: Measured(below=1000), 1000: Measured(gb=7.0)}, default=1000, group="分析")
        analysis_frames: Literal[50, 100] = measured_param(
            "分析帧数", {50: Measured(below=100), 100: Measured(gb=7.0)}, default=100, group="分析")
        query_step: int = P(10, label="查询间隔", ge=1, le=30, group="分析")

    @classmethod
    def found(cls, ctx, raw, objects) -> None:
        info = raw.result()
        if objects:
            ctx.say("I-SEGANYMO-FOUND", count=len(objects), frames=info["analysed_frames"])
        else:
            ctx.say("N-SEGANYMO-EMPTY")

    @classmethod
    def class_list(cls, ctx, objects) -> list[dict]:
        return [{"index": o["id"], "name": f"moving {o['id']}", "name_zh": f"运动物体 {o['id']}"} for o in objects]

    @classmethod
    def label_map(cls, ctx, image, objects):
        """raw/frame_<n>.npz: labels uint8 [H,W]，0 = 不动，k = 第 k 个运动物体（`adapters/seganymo/worker.py` main，写出遮罩那一段）。"""
        import numpy as np

        return lambda d: d["labels"].astype(np.float32)


NODES = (MovingObjects,)
