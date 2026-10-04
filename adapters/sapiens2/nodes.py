"""Nodes provided by the Meta Sapiens2 extension (Sapiens2 License: commercial use with restrictions)."""

from __future__ import annotations

from typing import Literal


from lab2shot.sdk import (rgb_port, Official, normal_port, UNIT, MissingFrames, Matting, Param, camera_normals, WorkerNode, NodeParams, P, Port,
                          foreground_entry, fp16_param, frame_maps, Cost, Licence)

class _Params(NodeParams):
    model_size: Literal["1b", "0.4b"] = P("1b", group="model")
    fp16: bool = fp16_param("model")  # 半精度用 bf16，和全精度的差别在 0.1% 以内


def _inputs():
    # 上游两个演示脚本的 argparse 只有 config / checkpoint / --input / --output / --save_pred / --device：
    # 一张画面进、一张结果出，没有框也没有遮罩（official 里各自写了行号）
    return (rgb_port(),)


class Segment(Matting, WorkerNode):
    missing_frames = MissingFrames.SKIP
    id = "sapiens2.segment"
    version = 2  # 2：缩小送进模型的画面时两个方向都抗锯齿（横向 720p 以前没有）
    # 引的是官方抠像那条演示路径（vis_matting.py）：一张画面进去，一次前向出 4 个通道 —— 前景色 fgr_rgb 和 alpha。
    official = Official(
        cite=("third_party/sapiens2/repo/sapiens/dense/tools/vis/vis_matting.py:50-91",
              "third_party/sapiens2/repo/sapiens/dense/tools/vis/vis_seg.py:69-78"),
        takes={"image": "image"},
        gives={"alpha": "alpha", "foreground": "fgr_rgb", "parts": "pred_labels"},
    )
    on_node = ("model_size", "matte")
    # 只认人；每帧单独算，边缘和部位分界会有轻微闪动；
    # 默认开「精细抠像」，用单独的 1B 抠像模型出 alpha，头发边缘最好（软边）
    inputs = _inputs()
    main = "parts"
    outputs = (Port("parts", "image.1"), Port("alpha", "image.1"))
    # 官方抠像模型一次前向出 4 个通道 [前景 RGB, alpha]（third_party/sapiens2/repo/.../vis_matting.py）：
    # 前景色是它自己算的，不是拿 alpha 乘出来的，所以这一族的「前景」口它有
    foreground = Param("matte").one_of(True)
    runtime = "sapiens2"
    # RTX 4090，默认的 1B 模型
    cost = Cost(gpu=True, vram_gb=6.5, seconds_per_frame=0.23)
    licence = Licence(note=True)

    class Params(_Params):
        matte: bool = P(True, group="model")

    @classmethod
    def convert(cls, ctx, raw, job):
        import json

        image = job.plate
        classes = json.loads(raw.file("classes.json").read_text(encoding="utf-8"))
        maps = {
            "parts": ("image.1", "labels", {"value_range": (0, max(c["index"] for c in classes)), "classes": classes}),
            "alpha": ("image.1", "alpha", {"value_range": UNIT, "half": True}),  # 0..1
        }
        if "foreground" in ctx.wanted:  # 抠像家族的「前景」，写出走家族里的那一份实现
            maps["foreground"] = foreground_entry()
        return frame_maps(ctx, raw, image, maps, stage="write_maps")


class Normal(WorkerNode):
    missing_frames = MissingFrames.SKIP
    id = "sapiens2.normal"
    # 引的是官方法线那条演示路径（vis_normal.py）：一张画面进去，单位化的 normal 出来（存成 .npy）。
    official = Official(
        cite="third_party/sapiens2/repo/sapiens/dense/tools/vis/vis_normal.py:95-125",
        takes={"image": "image"},
        gives={"normal": "normal"},
    )
    on_node = ("model_size",)
    version = 4  # 4：缩小送进模型的画面时两个方向都抗锯齿；3：法线是官方原值，背景不抹掉，没有「遮罩」输出口；更早版本的缓存不能复用
    # 每帧单独算，边缘和部位分界会有轻微闪动；只认人；法线是官方原值（背景处也有值，只是没有意义）
    inputs = _inputs()
    main = "normal"
    outputs = (normal_port(),)
    runtime = "sapiens2"
    # RTX 4090，默认的 1B 模型
    cost = Cost(gpu=True, vram_gb=6.5, seconds_per_frame=0.26)
    licence = Licence(note=True)

    class Params(_Params):
        pass

    @classmethod
    def convert(cls, ctx, raw, job):
        image = job.plate
        # OpenCV camera -> GL camera (Z toward the lens)：唯一的那一步转轴（nodes/kit/maps.py）。
        # 第二个参数留空：官方每个像素都有法线，没有「哪里有值」这回事
        return frame_maps(ctx, raw, image, {"normal": camera_normals("normal", "")}, stage="write_maps")


NODES = (Segment, Normal)
