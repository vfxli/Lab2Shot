"""AnyCalib 扩展包提供的节点（代码与权重均为 Apache-2.0）。"""

from __future__ import annotations

import typing
from pathlib import Path


from lab2shot.sdk import (Official, FILMBACK_MM, measured_param, FLOAT, LENS, LENS_TABLE, LensCalibration, NodeParams, P, Packet,
                          distortion, focal_px_to_mm, lens_note, packed_lens, value_packet, Cost, Measured, distorts)


# 「拟合模型」的选项即 AnyCalib 自身的 cam_id；提供哪些模型及其对应的公式表模型，均由 lens.py 的 GROUP 定义
# （「LensDistortion」中 AnyCalib 一组列出的也是这份表，两端保持一致）。
from .lens import GROUP

MODEL_IDS = tuple(GROUP.models)


def lens_distortion(params: dict, names: list[str], values: list[float]) -> dict:
    """将 AnyCalib 的估计结果转换为核心公式表的畸变参数：所请求模型的系数按组内定义的顺序排列，公式表模型的其余参数
    置零。无论何种模型，AnyCalib 都将各项命名为 k1 k2 …；公式表模型自身的参数名取自 GROUP。"""
    gm = GROUP.models[params["fit_model"]]
    got = [float(v) for n, v in zip(names, values, strict=True) if n.startswith("k")]
    return distortion(gm.table, dict(zip(gm.names, got[:len(gm.names)], strict=False)))


class Calibrate(LensCalibration):
    id = "anycalib.calibrate"
    version = 2  # 「镜头内参」的值包含组名和公式表 id（nodes/lens.py packed_lens）；更早版本的缓存不含这些字段
    on_node = ("fit_model",)
    # 视图下方的控件与「COLMAP 相机解算」一致：解算出的三个值已在视图中显示，参数中仅保留镜头模型
    strip = {"fit_model": "镜头模型"}
    # 不提供「复制标定」：本节点输出的是 OpenCV 系数，而 3DE 与 Nuke 仅支持 3DE 自身的三种模型，两者之间不做转换，
    # 复制出去只能携带内参，无法携带 k1 k2。需要对外交付镜头时，应使用「LensDistortion」烘焙 ST-map。
    # 对整段中的若干帧（默认 8 帧）分别估计后取中位数；假定相机固定，仅输出镜头信息；不适用于变焦镜头。
    # 输入「图像」，输出 Focal Length / Filmback / 镜头模型 / 畸变系数，均由「镜头标定」家族声明（families/lens_calibration.py），
    # 与 GeoCalib 一致。本节点仅输出数值；ST-map 的烘焙仅由「LensDistortion」执行，在节点图上可见。
    # 「拟合模型」中实际解出畸变系数的选项（家族的 `distorting_models`）由 GROUP 计算：仅限公式表上带系数的模型。
    # 选择无畸变时，家族将「镜头内参」端口置灰（此时 cook 在该端口输出空包），新增模型时无需修改此处。
    distorting_models = tuple(m for m, gm in GROUP.models.items() if distorts(gm.table))
    # 主点包含在「镜头内参」中：家族声明的三个端口即全部输出
    runtime = "anycalib"
    # 官方接口的输入、输出与解算器一致：AnyCalib.predict(im, cam_id) 返回 pred["intrinsics"]
    # （所选相机模型的参数序列）和 pred["success"]（anycalib_pretrained.py:277-308）。
    official = Official(
        cite="third_party/anycalib/repo/anycalib/model/anycalib_pretrained.py:277-308",
        takes={"image": "im"},
        gives={"focal": "intrinsics", "lens": "intrinsics"},
        ours={"filmback": "filmback_mm"},  # 节点自身的「Filmback」参数原样传给下游
        note="上游只出一串 intrinsics（anycalib_pretrained.py:300-303，按 cam_id 那个相机模型的参数顺序排）："
             "Focal Length、主点、畸变系数都是从这一串里取的；「镜头内参」= 我们请求的 cam_id 对应的镜头表模型 + 这些数。"
             "**单位**：上游那一串 intrinsics 里的 Focal Length 和主点都是**像素**；我们的「Focal Length」「主点 X / Y」"
             "交的是**毫米**（主点在「镜头内参」里），用节点上的「Filmback」参数换算（Focal Length（mm）= Focal Length（px）÷ 画面宽度 × Filmback）。"
             "毫米是 Lab2Shot 的内部标准单位，"
             "下游的「LensDistortion」「相机属性」吃的也是毫米，两边对得上。"
             "「Filmback」输出口给的是节点上那个参数的原值——上游没有这一项，这个口只是把它带给下游，"
             "省得用户在两个节点上各填一遍。"
             "**上游还有、我们没有口的**：pred[\"success\"]（这一帧线性拟合 + 非线性优化成没成功，"
             "anycalib_pretrained.py:171）——worker 已经读它，用来发 W 级警告，没做成口；"
             "还有 pred[\"intrinsics_icovs\"]（内参的逆协方差，:172）和稠密的 fov_field / tangent_coords（:270）。"
             "目前没有对应的输出口",
    )
    # vram_gb：RTX 4090，单个镜头（默认取 8 帧）1.3-1.8 GB
    cost = Cost(gpu=True, vram_gb=1.5, whole="标定用几帧就够，不是逐帧的活，没有秒/帧这个概念")

    class Params(NodeParams):
        # 「拟合模型」是请求（按何种模型拟合），输出端口 `lens_model` 是结果（识别出的模型）：两者名称必须区分，
        # 不得与同一节点的输出端口重名。选项名采用核心镜头表中的名称，与「LensDistortion」一侧保持一致
        fit_model: typing.Literal[MODEL_IDS] = P(  # type: ignore[valid-type]
            "simple_radial:1", label="镜头模型", group="镜头", worker=False,
            option_labels={m: gm.label for m, gm in GROUP.models.items()},
            help="按 AnyCalib 自己的哪个相机模型去拟合（官方的 cam_id）。普通镜头 simple_radial:1（默认），广角 simple_radial:2，"
                 "鱼眼 simple_kb:1 到 :4（项数越多越贴合也越容易过拟合），除法模型 simple_division:1；手机和长焦畸变很小，"
                 "用 simple_pinhole 更稳。交出去的「镜头内参」带着组名 AnyCalib 和这个模型名，「LensDistortion」上"
                 "「镜头内参组」选 AnyCalib、「镜头模型」选同一个就对得上。没列出来的官方模型在 lens.py 里留了档",
        )
        samples: typing.Literal[4, 8, 16] = measured_param(
            "采样帧数", {4: Measured("时间约减半", below=8), 8: Measured("一段镜头 4–10 秒", gb=1.8), 16: Measured("分批送进模型，显存同 8 帧，时间约翻倍", below=8)}, default=8,
            group="镜头", help="在整段里均匀取几帧分别估计，再取中位数。8 帧通常够；画面变化大（有时空镜、有时特写）就多取些")
        checkpoint: typing.Literal["anycalib_gen", "anycalib_pinhole", "anycalib_dist"] = P(
            "anycalib_gen", label="网络权重", group="镜头",
            option_labels={"anycalib_gen": "通用", "anycalib_pinhole": "无畸变画面", "anycalib_dist": "鱼眼/强畸变"},
            help="通用：什么镜头都可以，推荐；无畸变画面：只在普通/长焦镜头上训练，Focal Length 略准；鱼眼/强畸变：GoPro、鱼眼等",
        )
        principal_point: typing.Literal["center", "estimate"] = P(
            "center", label="主点", group="镜头",
            option_labels={"center": "画面中心", "estimate": "估计"},
            help="镜头中心在画面上的位置。实拍素材基本在画面中心，选「画面中心」更稳；裁切过的素材才用「估计」",
        )
        # 上游 intrinsics 中的 Focal Length 与主点均以像素为单位；「Focal Length」「主点 X / Y」输出前按此
        # Filmback 换算为毫米（毫米是 Lab2Shot 的内部标准单位）。「Filmback」输出端口输出该参数的原值
        filmback_mm: float = P(FILMBACK_MM, label="Filmback", unit="mm", gt=0, group="镜头", worker=False,
                               help="上游给的是 Focal Length（px）和像素主点，按这个 Filmback 换算成毫米交出去："
                                    "毫米 = 像素 ÷ 画面宽度 × Filmback。全画幅 36，手机按等效 Focal Length 时保持 36。"
                                    "「Filmback」输出口交的就是这里填的值，直接接「LensDistortion」的同名参数")

    @classmethod
    def known_outputs(cls, params: dict) -> dict[str, dict]:
        """输出的镜头模型在计算前即可由参数确定：「镜头内参」先输出仅含组名和模型名、系数为空的值，
        使下游「LensDistortion」在提交前即可核对两端是否匹配（B-LENS-MODELMISMATCH），无需等待模型运行完成。"""
        gm = GROUP.model(params.get("fit_model") or "")
        if gm is None or gm.table not in LENS_TABLE or not LENS_TABLE[gm.table].params:
            return {}
        return {"lens": value_packet(Path("."), LENS, {"group": GROUP.id, "model": gm.name, "table": gm.table, "params": {}}, said=gm.name).meta}

    @classmethod
    def cook(cls, ctx):
        import json

        import numpy as np
        from lab2shot_worker.files import write_exr

        image = ctx.input("image")
        raw = ctx.run_worker(image, extra={"model": ctx.params["fit_model"]})  # 该参数即官方的 cam_id
        lens = json.loads((raw / "lens.json").read_text(encoding="utf-8"))
        w, h = int(lens["width"]), int(lens["height"])
        frames = image.meta["frames"]
        back = float(ctx.params["filmback_mm"])

        ctx.stage("估镜头")
        # 镜头：Filmback 上的焦距、测量所用的画面、镜头中心（毫米，+y 向上）、以核心公式表表示的畸变（提供的每个模型
        # 在表中均有对应项），标记为估计值，AnyCalib 自身的模型与数值保留在 source 中
        names, values = [str(n) for n in lens["param_names"]], [float(v) for v in lens["params"]]
        distorted = lens_distortion(ctx.params, names, values)
        props = {
            "focal_px": float(lens["focal_px"]), "filmback_mm": back, "width": w, "height": h, "raster": [w, h],
            "pixel_aspect": 1.0, "center_mm": [(float(lens["cx"]) - w / 2) * back / w, -(float(lens["cy"]) - h / 2) * back / w],
            "distortion": distorted,
            "source": {"level": "estimated", "by": "AnyCalib 镜头标定", "model": str(lens["model"]), "params": dict(zip(names, values))},
        }
        picture = {"width": w, "height": h}  # 焦距测量所基于的画面：与目标素材核对
        # 输出数值，每个数值的去向在节点图上可见（此处不将参数烘焙为 ST-map）：
        # Focal Length、Filmback 各占一个端口，其余（模型、系数、主点）打包为一份「镜头内参」。
        # pinhole 没有系数，「镜头内参」输出空包（空结果不属于错误），与 COLMAP 选择无畸变时一致
        model_id = str(ctx.params["fit_model"])  # 组内名称（cam_id），即「镜头内参」中记录的模型名
        out = {
            # 上游输出的是 Focal Length（px）（anycalib_pretrained.py:300-303），按节点的「Filmback」换算为毫米后输出
            "focal": value_packet(ctx.outputs["focal"], FLOAT, focal_px_to_mm(props["focal_px"], w, back), unit="mm", **picture),
            "filmback": value_packet(ctx.outputs["filmback"], FLOAT, back, unit="mm", **picture),
            "lens": (value_packet(ctx.outputs["lens"], LENS, packed_lens(distorted, props["center_mm"], 1.0, group=GROUP.id, name=model_id),
                                  said=model_id, **picture)
                     if LENS_TABLE[str(distorted["model"])].params else Packet(ctx.outputs["lens"], LENS, {"empty": True})),  # 仅公式表上带系数的模型输出内参
        }
        hfov = lens.get("hfov_deg")
        if hfov:
            ctx.say("I-ANYCALIB-LENSFOV", lens=lens_note(lens["focal_px"], w, back), hfov=float(hfov))
        else:
            ctx.say("I-ANYCALIB-LENS", lens=lens_note(lens["focal_px"], w, back))
        return out


NODES = (Calibrate,)
