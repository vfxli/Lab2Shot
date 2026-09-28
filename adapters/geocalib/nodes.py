"""Nodes provided by the GeoCalib extension (Apache-2.0)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (FILMBACK_MM, Official, FLOAT, LENS, VECTOR, LensCalibration, LensParams, P, Port, distorts,
                          distortion, empty_packet, focal_px_to_mm, focal_param, lens_note, packed_lens, plate_lens, value_packet, Cost)

# GeoCalib 自己的三个相机模型 → 核心公式表：lens.py 的 GROUP（也是「LensDistortion」上 GeoCalib 那一组）
from .lens import GROUP

TABLE_MODELS = {name: (gm.table, gm.names[0] if gm.names else "") for name, gm in GROUP.models.items()}


class Calibrate(LensCalibration):
    id = "geocalib.calibrate"
    version = 2  # 「镜头内参」值带组名和公式表 id（nodes/lens.py packed_lens）；更早的缓存没有这两项，不能复用
    strip = {"fit_model": "镜头模型"}  # 视图小控件，和 COLMAP 一样：参数只放这一个，解出的值视图本来就显示
    # 默认每帧都估计重力（隔帧参数）；填了 Focal Length 就固定用它、只估重力方向，放平后地面的残余倾斜明显更小
    main = "gravity"  # what the node is for: its ports are listed by type order, and this one goes first
    # 「RGB」进、Focal Length / Filmback / 镜头内参 出，都由「镜头标定」家族声明
    # （families/lens_calibration.py），和 AnyCalib 一致。重力方向和它的不确定度是上游多给的（AnyCalib 没有），
    # 排在家族那几个前面：它们是这个节点的主结果（`main = "gravity"`），最常单独用的放最上
    outputs = (
        Port("gravity", VECTOR, "重力方向"),
        Port("gravity_error", FLOAT, "重力误差", unit="°"),
    ) + LensCalibration.outputs
    # 「镜头模型」里真的会解出畸变的那几档（家族的 `distorting_models`），从 TABLE_MODELS 算出来：
    # 对应到镜头表上真有系数的模型（distorts()）。径向 k1 = SIMPLE_RADIAL；除法模型 = SIMPLE_DIVISION，
    # 镜头表里有这一档它才算畸变档，没有就当无畸变交（那时「镜头内参」交空）。
    # 这几档之外「镜头内参」口变灰，用户不会接上一根永远是空的线
    distorting_models = tuple(m for m, (table, _) in TABLE_MODELS.items() if distorts(table)) or ("simple_radial",)
    runtime = "geocalib"
    # 上游 GeoCalib.calibrate(img, camera_model=…, priors=…) 交出 camera（Focal Length、主点、畸变）、gravity
    # （重力方向）、covariance 和各种 uncertainty（extractor.py:117-127）。
    official = Official(
        cite="third_party/geocalib/repo/geocalib/extractor.py:72-127",
        takes={"image": "img"},
        gives={"gravity": "gravity", "gravity_error": "uncertainty", "focal": "camera", "lens": "camera"},
        ours={"filmback": "filmback_mm"},  # 节点自己的「Filmback」参数原样带给下游
        note="「重力误差」是上游的 gravity_uncertainty（extractor.py:126 那一行把所有带 uncertainty 的键"
             "都交出来，它本身算在 siclib/models/optimization/lm_optimizer.py:417）。"
             "「Focal Length」是 camera 对象上的 f，「镜头内参」是它的 camera_model 和 dist（extractor.py:117）打成一份。"
             "**单位**：上游的 camera.f 是**像素**，我们的「Focal Length」口交的是**毫米**，用节点上的「Filmback」"
             "参数换算（毫米 = 像素 ÷ 画面宽度 × Filmback）。毫米是 Lab2Shot 的内部标准单位"
             "。"
             "「Filmback」输出口给的是节点上那个参数的原值——上游没有这一项，这个口只是把它带给下游；"
             "「Filmback」参数本身在**输入**侧也要用：用户填的 Focal Length（mm）要靠它换成像素才能当 prior 送进模型。"
             "上游还给出、目前没有对应输出口的：同一个 uncertainty 字典里还有 roll_uncertainty、"
             "pitch_uncertainty、focal_uncertainty、vfov_uncertainty（lm_optimizer.py:413-418），"
             "我们只取了 gravity_uncertainty；extractor.py:123 还交出 covariance（重力和 Focal Length 的完整协方差矩阵），"
             "以及 :124-125 的稠密 latitude / up field 和它们的 confidence",
    )
    on_node = ("focal_mm", "step")
    # RTX 4090 上量得的显存和速度
    cost = Cost(gpu=True, vram_gb=6.5, seconds_per_frame=0.06)

    class Params(LensParams):
        # 「镜头模型」是要求（上游的 `camera_model`：要它按哪种模型去拟合），「镜头内参」口里带的模型名是结果
        # （它认出来是哪一种），两件事。档位和标签都从 lens.py 的 GROUP 取，
        # 和「LensDistortion」上 GeoCalib 那一组必须一一对上
        fit_model: Literal[tuple(GROUP.models)] = P(  # type: ignore[valid-type]
            "pinhole", label="镜头模型", group="镜头",
            option_labels={m: gm.label for m, gm in GROUP.models.items()},
        )
        step: int = P(1, label="隔帧", ge=1, le=100, group="重力")
        focal_mm: float | None = focal_param()

    @classmethod
    def cook(cls, ctx):
        import json

        import numpy as np

        image = ctx.input("image")
        w, h = image.meta["width"], image.meta["height"]
        used = plate_lens(ctx, image)
        raw = ctx.run_worker(image, extra={"focal_px": used.focal_px}, record=used.record())
        g = np.load(raw / "gravity.npz")
        lens = json.loads((raw / "lens.json").read_text(encoding="utf-8"))

        ctx.stage("写出重力方向和 Focal Length")
        frames = g["frames"].tolist()
        up = g["up"] * np.array([1.0, -1.0, -1.0])  # OpenCV camera -> Lab2Shot camera axes (Y up, Z towards the viewer)
        picture = {"width": w, "height": h}  # what the focal length was measured on: checked against the plate it goes to
        out = {
            "gravity": value_packet(ctx.outputs["gravity"], VECTOR, frames=frames, values=up.tolist()),
            "gravity_error": value_packet(ctx.outputs["gravity_error"], FLOAT, frames=frames,
                                          values=g["uncertainty_deg"].tolist(), unit="°"),
            # 上游给的 Focal Length 是像素，换成毫米再交（用节点上的「Filmback」，LensParams）。
            # 除的是 Focal Length 自己所在的那个宽度（worker 写在 lens.json 里的），不是节点这边的画面宽度：
            # 两者目前相等，但不应依赖这一点（家族的 focal_px_to_mm 一处算法）
            "focal": value_packet(ctx.outputs["focal"], FLOAT,
                                  focal_px_to_mm(lens["focal_px"], lens["width"], ctx.params["filmback_mm"] or FILMBACK_MM),
                                  unit="mm", **picture),
            "filmback": value_packet(ctx.outputs["filmback"], FLOAT,
                                     float(ctx.params["filmback_mm"] or FILMBACK_MM), unit="mm", **picture),
        }
        table, name = TABLE_MODELS.get(str(lens["model"]), ("SIMPLE_PINHOLE", ""))
        k1 = lens.get("k1")
        has_distortion = distorts(table) and k1 is not None
        if "lens" in ctx.wanted:  # 估出了畸变才有镜头内参可交；没估就交空的（空结果不是错误）
            out["lens"] = (value_packet(ctx.outputs["lens"], LENS, packed_lens(distortion(table, {name: float(k1)}), group=GROUP.id, name=str(lens["model"])),
                                        said=str(lens["model"]), **picture)
                           if has_distortion else empty_packet(ctx, "lens"))
        ctx.say("I-GEOCALIB-TILT", roll=float(np.median(g["roll_deg"])), pitch=float(np.median(g["pitch_deg"])),
                count=len(up), error=float(np.median(g["uncertainty_deg"])),
                lens=used.said if lens["source"] == "user" else
                     lens_note(lens["focal_px"], lens["width"], float(ctx.params["filmback_mm"] or FILMBACK_MM)))
        return out


NODES = (Calibrate,)
