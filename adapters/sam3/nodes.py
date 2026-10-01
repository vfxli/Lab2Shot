"""Nodes provided by the SAM 3 extension (SAM License: commercial use allowed, no military use)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (rgb_port, Official, measured_param, Job, NodeParams, P, Port, people_port, Segmentation,
                          Cost, Licence, Not, Wired, Measured)


class Segment(Segmentation):
    id = "sam3.segment"
    # 上游两条路（worker.py）：提示词走视频预测器 start_session(resource_path=画面)（worker 直接调它内部的
    # model.init_state）+ add_prompt(text_str) -> {"out_obj_ids", "out_binary_masks"}；人物框走它的跟踪器（SAM 2
    # 那一套）：init_state 后每个人 add_new_points_or_box(box=) 一个框提示，段尾还在的人在下一段用上一段的遮罩
    # add_new_mask 接上，丢了的人从他的下一个框重新接
    official = Official(
        cite=("third_party/sam3/repo/sam3/model/sam3_base_predictor.py:119-245",
              "third_party/sam3/repo/sam3/model/sam3_tracking_predictor.py:57-60",
              "third_party/sam3/repo/sam3/model/sam3_tracking_predictor.py:180-190",
              "third_party/sam3/repo/sam3/model/sam3_tracking_predictor.py:343-350"),
        takes={"image": "resource_path", "boxes": "add_new_points_or_box"},
        gives={"mask": "out_binary_masks", "objects": "out_obj_ids"},
        note="文字提示（text_str）是上游预测器的提示，我们做成了节点参数「提示词」而不是输入口；"
             "人物框不是预测器的 bounding_boxes 提示，而是跟踪器的框提示（add_new_points_or_box），每人一个、按人分会话跑。",
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   boxes：实测（CRGNN 实拍、VideoMatte 绿幕共 8 个人像镜头）：用 ViTDet 人物框代替提示词，人物遮罩 J&F 整体没区别（0.967 → 0.970），但各镜头不一：2 个更好、1 个框错了人明显更差
    on_node = ("prompt", "threshold")
    # 按提示词或人物框整段跟踪；它分的是「哪块是这个物体」，不是精细抠像，边是 0 / 1 的选区；
    # 长镜头按物体数自动分段（8 个物体时一段约 800 帧），显存不随镜头变长
    inputs = (rgb_port(), people_port(optional=True))  # a matte object per person
    # 「遮罩」「物体分割」两个输出口由分割家族给（families/segmentation.py）：mask_label 说这个项目选出来的是什么
    mask_label = "遮罩"
    runtime = "sam3"
    # RTX 4090，文字提示词的默认用法
    cost = Cost(gpu=True, vram_gb=6.1, seconds_per_frame=0.23)
    licence = Licence(note="SAM License：可以商用；禁止军事、武器等用途；发表时注明使用了 SAM 3；再分发要附许可证原文。")

    class Params(NodeParams):
        prompt: str = P("person", label="提示词", group="分割", lines=4, applies=Not(Wired("boxes")))
        max_objects: Literal[1, 2, 4, 8] = measured_param(
            "最多物体数", {1: Measured(below=2), 2: Measured(gb=5.1), 4: Measured(below=8), 8: Measured(gb=6.1)}, default=8, group="分割")
        threshold: float = P(0.5, label="检测阈值", ge=0.05, le=0.95, group="分割", widget="slider", applies=Not(Wired("boxes")))

    @classmethod
    def prepare(cls, ctx) -> Job:
        """接了「人物框」就把它一起送进 worker（上游的框提示）。"""
        return Job(ctx.input("image"), inputs=ctx.input_files("boxes"))

    @classmethod
    def found(cls, ctx, raw, objects) -> None:
        if not objects:
            ctx.say("N-SAM3-EMPTY")

    @classmethod
    def class_list(cls, ctx, objects) -> list[dict]:
        import re

        def crypto_name(label: str, oid: int) -> str:
            # Cryptomatte manifest name: prompt (or its label) plus a zero-padded instance number, e.g. person_01
            slug = re.sub(r"\s+", "_", (label or "object").strip()) or "object"
            return f"{slug}_{oid:02d}"

        return [{"index": k + 1, "name": crypto_name(o.get("label") or ctx.params["prompt"], o["id"]), "id": o["id"]}
                for k, o in enumerate(objects)]

    @classmethod
    def label_map(cls, ctx, image, objects):
        """raw/frame_<n>.npz: masks uint8 [K,H,W] 0/255，每帧同一个物体顺序（`adapters/sam3/worker.py` main，写出遮罩那一段）。
        第 k 个物体的编号是 k + 1（0 是背景）；重叠时后面的物体压在前面的上面。"""
        import numpy as np

        size = (image.meta["height"], image.meta["width"])

        def labels(d):
            m = d["masks"]
            m = m if m.ndim == 3 and len(m) else np.zeros((0, *size), np.uint8)
            out = np.zeros(size, np.float32)
            for k in range(len(m)):
                out[m[k] > 127] = k + 1
            return out

        return labels


NODES = (Segment,)
