"""Nodes provided by the Meta Sapiens2 extension (Sapiens2 License: commercial use with restrictions)."""

from __future__ import annotations

from typing import Literal


from lab2shot.sdk import (Official, UNIT, MissingFrames, Matting, Param, RawOutput, camera_normals, NodeDef, NodeParams, P, Port,
                          foreground_entry, fp16_param, frame_maps, Cost, Licence)

LICENSE_NOTE = (
    "Sapiens2 许可证可以商用，但禁止：深度伪造和冒充真人（真人换脸、假替身）、监控、生物特征识别和辨认身份、"
    "推断敏感信息、色情、军事用途；再分发要附许可证，发表时注明使用了 Sapiens。"
)


class _Params(NodeParams):
    model_size: Literal["1b", "0.4b"] = P(
        "1b", label="模型大小", group="模型",
        option_labels={"1b": "1B", "0.4b": "0.4B"},
    )
    fp16: bool = fp16_param("模型")  # 半精度用 bf16，和全精度的差别在 0.1% 以内


def _inputs():
    # 上游两个演示脚本的 argparse 只有 config / checkpoint / --input / --output / --save_pred / --device：
    # 一张画面进、一张结果出，没有框也没有遮罩（official 里各自写了行号）
    return (Port("image", "image.3", "RGB"),)


class Segment(Matting, NodeDef):
    id = "sapiens2.segment"
    # 引的是官方抠像那条演示路径（vis_matting.py）：一张画面进去，一次前向出 4 个通道 —— 前景色 fgr_rgb 和 alpha。
    official = Official(
        cite=("third_party/sapiens2/repo/sapiens/dense/tools/vis/vis_matting.py:50-91",
              "third_party/sapiens2/repo/sapiens/dense/tools/vis/vis_seg.py:69-78"),
        takes={"image": "image"},
        gives={"alpha": "alpha", "foreground": "fgr_rgb", "parts": "pred_labels"},
        note="① 这个节点的口分散在官方两个演示脚本里，所以 cite 写了两条：alpha 和前景色在 vis_matting.py"
             "（一次前向出 4 个通道），「部位分割」在 vis_seg.py（`pred_labels = seg_logits.argmax(dim=1)`，"
             "`--save_pred` 存成 _seg.npy）。两个都是官方的。"
             "② 节点没有「人物框」输入：官方两个演示脚本的 argparse 只有 config / "
             "checkpoint / --input / --output / --save_pred / --device（vis_matting.py:19-29），整幅画面一起算，"
             "按框裁切放大是我们自己加的一步，不放在解算器上。"
             "实测也说明它没带来什么：CRGNN 实拍 + VideoMatte 绿幕共 8 个人像镜头（人在画面里很大），"
             "按人裁切和整幅一起算没区别（J&F 0.992 → 0.987，其中 2 个镜头略差）。",
    )
    on_node = ("model_size", "matte")
    # 只认人；每帧单独算，边缘和部位分界会有轻微闪动；
    # 默认开「精细抠像」，用单独的 1B 抠像模型出 alpha，头发边缘最好（软边）
    inputs = _inputs()
    main = "parts"
    outputs = (Port("parts", "image.1", "部位分割"), Port("alpha", "image.1", "Alpha"))
    # 官方抠像模型一次前向出 4 个通道 [前景 RGB, alpha]（third_party/sapiens2/repo/.../vis_matting.py）：
    # 前景色是它自己算的，不是拿 alpha 乘出来的，所以这一族的「前景」口它有
    foreground = Param("matte").one_of(True)
    foreground_waits = "打开「精细抠像」"
    runtime = "sapiens2"
    # RTX 4090，默认的 1B 模型
    cost = Cost(gpu=True, vram_gb=6.5, seconds_per_frame=0.23)
    licence = Licence(note=LICENSE_NOTE)

    class Params(_Params):
        matte: bool = P(True, label="精细抠像", group="模型")

    @classmethod
    def cook(cls, ctx):
        import json


        image = ctx.input("image")
        raw = RawOutput(ctx.run_worker(image), MissingFrames.SKIP)
        classes = json.loads(raw.file("classes.json").read_text(encoding="utf-8"))
        maps = {
            "parts": ("image.1", "labels", {"value_range": (0, max(c["index"] for c in classes)), "classes": classes}),
            "alpha": ("image.1", "alpha", {"value_range": UNIT, "half": True}),  # 0..1
        }
        if "foreground" in ctx.wanted:  # 抠像家族的「前景」，写出走家族里的那一份实现
            maps["foreground"] = foreground_entry()
        return frame_maps(ctx, raw, image, maps, stage="写出分割和 alpha")


class Normal(NodeDef):
    id = "sapiens2.normal"
    # 引的是官方法线那条演示路径（vis_normal.py）：一张画面进去，单位化的 normal 出来（存成 .npy）。
    official = Official(
        cite="third_party/sapiens2/repo/sapiens/dense/tools/vis/vis_normal.py:95-125",
        takes={"image": "image"},
        gives={"normal": "normal"},
        note="① 节点没有「人物框」输入：官方的 argparse 只有 config / checkpoint / "
             "--input / --output / --seg_dir / --device（vis_normal.py:19-37），整幅画面一起算，"
             "按框裁切放大是我们自己加的一步，不放在解算器上。"
             "② 节点也没有「遮罩」输出：官方那里的 mask 是**读进来的**（--seg_dir 里已有的"
             "分割，vis_normal.py:59-89），而且只用在拼给人看的那张对比图上（:125 normal[mask == 0] = -1），"
             "**存下来的 .npy 法线是没有遮过的**（:122 np.save 在遮之前）。节点上的遮罩只能靠另外跑一次分割"
             "模型得到，属于我们自己造的输出；也不用遮罩把人以外的法线抹成 0——"
             "交出去的就是官方 np.save 的那一份，不加工（和前馈重建家族的深度图同一条规则）。",
    )
    on_node = ("model_size",)
    version = 3  # 3：法线是官方原值，背景不抹掉，没有「遮罩」输出口；更早版本的缓存不能复用
    # 每帧单独算，边缘和部位分界会有轻微闪动；只认人；法线是官方原值（背景处也有值，只是没有意义）
    inputs = _inputs()
    main = "normal"
    outputs = (Port("normal", "image.3", "法线图", means=("space",)),)
    runtime = "sapiens2"
    # RTX 4090，默认的 1B 模型
    cost = Cost(gpu=True, vram_gb=6.5, seconds_per_frame=0.26)
    licence = Licence(note=LICENSE_NOTE)

    class Params(_Params):
        pass

    @classmethod
    def cook(cls, ctx):
        image = ctx.input("image")
        raw = RawOutput(ctx.run_worker(image), MissingFrames.SKIP)
        # OpenCV camera -> GL camera (Z toward the lens)：唯一的那一步转轴（nodes/kit/maps.py）。
        # 第二个参数留空：官方每个像素都有法线，没有「哪里有值」这回事
        return frame_maps(ctx, raw, image, {"normal": camera_normals("normal", "")}, stage="写出法线")


NODES = (Segment, Normal)
