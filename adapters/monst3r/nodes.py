"""Nodes provided by the MonST3R extension (CC-BY-NC-SA-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, LensWholeShotParams, P, Port, WholeShotDepthCamera,
                          conf_threshold_param, frame_maps, resolution_param, Cost,
                          max_frames_param, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "monst3r.reconstruct"
    version = 3  # 2：SEA-RAFT / SAM 2.1 每个任务只加载一次；3：试过官方 window_wise 续接后退回独立分段 + 拼接（实测更准，见 worker.py 开头），清掉那期间的缓存
    metric = False  # its units are arbitrary: the 「尺度」 says how many centimetres one is (DepthCamera.metric)
    # 没有「人物框」输入口：上游只收一张运动物体遮罩图（dynamic_mask_path，demo.py:107/117），
    # 要挡人就接「人物框转遮罩」，那一步在节点图上看得见。
    takes_mask = True  # 上游真的收一张遮罩图（见 official 的 cite）
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（12 个镜头）：填真实 Focal Length 或接 AnyCalib，相机轨迹都没区别
    #   mask：实测（8 个镜头）：接 SAM 3 遮罩，相机轨迹没区别，输出的运动物体遮罩 J&F 0.310 → 0.347
    on_node = ("focal_mm", "step", "max_frames")
    # 主结果是深度，相机是顺带的（家族默认是相机）：分类按主口判，视图双击也先看深度
    main = "depth"
    # 固定机位上最稳，慢速移动可用，长焦跟拍不可靠；自己找出运动区域并另外输出运动遮罩（0 / 1 的选区，
    # 适合挡解算，不是精细抠像）；尺度是自己的任意单位，整段一致；显存随每段帧数线性增长，默认每段 36 帧
    outputs = WholeShotDepthCamera.outputs + (Port("mask", "image.1"),)
    runtime = "monst3r"
    # 官方 demo.py get_reconstructed_scene(… filelist …) 收一串画面，外加一份运动物体遮罩
    # （--use_gt_davis_masks → dynamic_mask_path → load_images(dynamic_mask_root=…)，demo.py:107-117，
    # 全局对齐那一步 use_self_mask=not use_gt_mask，demo.py:133），交出 poses / K / depth_maps /
    # dynamic_masks / conf（demo.py:156-160）。
    official = Official(
        cite="third_party/monst3r/repo/demo.py:94-165",
        takes={"image": "filelist", "mask": "dynamic_mask_path"},
        gives={"depth": "depth_maps", "mask": "dynamic_masks", "camera": "poses"},
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # RTX 4090，默认每段约 36 帧
    cost = Cost(gpu=True, vram_gb=17.0, seconds_per_frame=3.0)

    class Params(LensWholeShotParams):
        # 24G 显卡：40 帧（288×512）18.9 GB，显存随帧数线性增长；40 是测过的最大点，再多就是外推，不给填
        max_frames: Literal[12, 24, 36, 40] = max_frames_param(
            {12: Measured(below=36), 24: Measured(below=36), 36: Measured(gb=17.0), 40: Measured(gb=18.9)},
            default=36)
        # 18.9 GB 是「每段最多帧数」40 那一档量出来的，不是 512 本身；512 配默认的 36 帧就是 cost 里的 17.0
        resolution: Literal[256, 384, 512] = resolution_param(
            {256: Measured(below=512), 384: Measured(below=512),
             512: Measured(gb=17.0)},
            default=512)
        niter: Literal[100, 300, 500] = measured_param({100: Measured(flat=True), 300: Measured(flat=True), 500: Measured(flat=True)}, default=300, group="solve")
        motion_threshold: float = P(0.35, ge=0.05, le=0.95, group="moving_objects")
        sam2_refine: bool = P(True, group="moving_objects")
        conf_threshold: float = conf_threshold_param(1.1)

    @classmethod
    def convert(cls, ctx, raw, job):
        import numpy as np

        moving = {"mask": ("image.1", lambda d: d["moving"].astype(np.float32), None)}  # reconstructed frames only
        return {**super().convert(ctx, raw, job), **frame_maps(ctx, raw, job.plate, moving, stage="write_moving_mask")}


NODES = (Reconstruct,)
