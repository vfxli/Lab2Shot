"""Nodes provided by the MonST3R extension (CC-BY-NC-SA-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, LensWholeShotParams, P, Port, WholeShotDepthCamera,
                          conf_threshold_param, frame_maps, resolution_param, unit_cm_param, Cost, Licence,
                          max_frames_param, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "monst3r.reconstruct"
    # 没有「人物框」输入口：上游只收一张运动物体遮罩图（dynamic_mask_path，demo.py:107/117），
    # 要挡人就接「人物框转遮罩」，那一步在节点图上看得见。
    takes_mask = "运动物体遮罩"  # 上游真的收一张遮罩图（见 official 的 cite）
    # Benchmark figures shown as the inputs' tooltips.
    measured = {
        "focal_mm": "实测（12 个镜头）：填真实 Focal Length 或接 AnyCalib，相机轨迹都没区别",
        "mask": "实测（8 个镜头）：接 SAM 3 遮罩，相机轨迹没区别，输出的运动物体遮罩 J&F 0.310 → 0.347",
    }
    on_node = ("focal_mm", "step", "max_frames")
    # 主结果是深度，相机是顺带的（家族默认是相机）：分类按主口判，视图双击也先看深度
    main = "depth"
    # 固定机位上最稳，慢速移动可用，长焦跟拍不可靠；自己找出运动区域并另外输出运动遮罩（0 / 1 的选区，
    # 适合挡解算，不是精细抠像）；尺度是自己的任意单位，整段一致；显存随每段帧数线性增长，默认每段 36 帧
    outputs = WholeShotDepthCamera.outputs + (Port("mask", "image.1", "运动物体遮罩"),)
    runtime = "monst3r"
    # 官方 demo.py get_reconstructed_scene(… filelist …) 收一串画面，外加一份运动物体遮罩
    # （--use_gt_davis_masks → dynamic_mask_path → load_images(dynamic_mask_root=…)，demo.py:107-117，
    # 全局对齐那一步 use_self_mask=not use_gt_mask，demo.py:133），交出 poses / K / depth_maps /
    # dynamic_masks / conf（demo.py:156-160）。
    official = Official(
        cite="third_party/monst3r/repo/demo.py:94-165",
        takes={"image": "filelist", "mask": "dynamic_mask_path"},
        gives={"depth": "depth_maps", "mask": "dynamic_masks", "camera": "poses"},
        note="遮罩这一路是官方的：上游本来就能收外来的运动物体遮罩（demo.py:107 dynamic_mask_path）。"
             "上游那一头只有遮罩图这一种形式，没有「框」这种输入，所以「人物框」输入口已删——"
             "要挡人就接「人物框转遮罩」，那一步在节点图上看得见。"
             "「运动物体」输出就是官方 save_dynamic_masks 交的那一张（demo.py:159）",
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # RTX 4090，默认每段约 36 帧
    cost = Cost(gpu=True, vram_gb=17.0, seconds_per_frame=3.0)
    licence = Licence(note="代码和权重 CC-BY-NC-SA-4.0（含 DUSt3R / CroCo 代码），只能研究用；光流 SEA-RAFT（BSD-3）和 SAM 2（Apache-2.0）可以商用。")

    class Params(LensWholeShotParams):
        # 24G 显卡：40 帧（288×512）18.9 GB，显存随帧数线性增长；40 是测过的最大点，再多就是外推，不给填
        max_frames: Literal[12, 24, 36, 40] = max_frames_param(
            {12: Measured("显存随帧数线性降", below=36), 24: Measured("显存随帧数线性降", below=36), 36: Measured("竖幅 512 实测", gb=17.0), 40: Measured("288×512 实测", gb=18.9)},
            default=36, note="。显存随帧数线性涨，更多的没测过、不给选")
        # 18.9 GB 是「每段 40 帧」那一档量出来的，不是 512 本身；512 配默认的 36 帧就是 cost 里的 17.0
        resolution: Literal[256, 384, 512] = resolution_param(
            {256: Measured("比实测的一档省", below=512), 384: Measured("比实测的一档省", below=512),
             512: Measured("训练尺寸：每段 36 帧约 3 秒/帧", gb=17.0)},
            default=512, note="；512 是模型的训练尺寸；调低时每段能放更多帧")
        niter: Literal[100, 300, 500] = measured_param(
            "优化迭代次数", {100: Measured("时间约为 300 次的三分之一，显存不变", flat=True), 300: Measured("官方默认：约 3 秒/帧", flat=True), 500: Measured("时间约为 300 次的 1.7 倍，显存不变", flat=True)}, default=300, group="解算",
            help="每段全局优化的迭代次数（官方 300）。相机抖、没收敛就选 500，很慢；预览可以选 100")
        motion_threshold: float = P(0.35, label="运动判定门槛", help="光流误差超过它的像素算运动物体（官方 0.35）。运动的人没被遮住就调低；静止的地面、树丛被当成运动就调高", ge=0.05, le=0.95, group="运动物体")
        sam2_refine: bool = P(True, label="SAM 2 修整", help="用 SAM 2 把光流找到的运动区域修整成完整的物体遮罩，边缘更干净；关掉更快", group="运动物体")
        conf_threshold: float = conf_threshold_param(1.1)
        unit_cm: float = unit_cm_param("MonST3R 的尺度要手动缩放：先按米（100）放，和 ViPE 或实测距离对比后再调")

    @classmethod
    def prepare(cls, ctx):
        return super().prepare(ctx).with_(notes={"scale": "relative"})  # the model's units are arbitrary

    @classmethod
    def convert(cls, ctx, raw, job):
        import numpy as np

        moving = {"mask": ("image.1", lambda d: d["moving"].astype(np.float32), None)}  # reconstructed frames only
        return {**super().convert(ctx, raw, job), **frame_maps(ctx, raw, job.plate, moving, stage="写出运动遮罩")}


NODES = (Reconstruct,)
