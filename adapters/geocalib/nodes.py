"""Nodes provided by the GeoCalib extension (Apache-2.0)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (FILMBACK_MM, Official, FLOAT, Job, LENS, VECTOR, LensCalibration, LensParams, P, Port, distorts,
                          distortion, focal_mm, focal_param, lens_note, packed_lens, plate_lens, value_packet, Cost)

# GeoCalib 自己的三个相机模型 → 核心公式表：lens.py 的 GROUP（也是「LensDistortion」上 GeoCalib 那一组）
from .lens import GROUP

TABLE_MODELS = {name: (gm.table, gm.names[0] if gm.names else "") for name, gm in GROUP.models.items()}


class Calibrate(LensCalibration):
    id = "geocalib.calibrate"
    version = 4  # 4: 给了焦距又选带畸变的模型时照样估畸变系数（不再交空系数）；3: 「镜头模型」选无畸变时也交「镜头内参」（没有系数的镜头），不再是空包
    strip = ("fit_model",)  # 视图小控件，和 COLMAP 一样：参数只放这一个，解出的值视图本来就显示
    # 默认每帧都估计重力（隔帧参数）；填了 Focal Length 就固定用它、只估重力方向，放平后地面的残余倾斜明显更小
    main = "gravity"  # what the node is for: its ports are listed by type order, and this one goes first
    # 「RGB」进、Focal Length / Filmback / 镜头内参 出，都由「镜头标定」家族声明
    # （families/lens_calibration.py），和 AnyCalib 一致。重力方向和它的不确定度是上游多给的（AnyCalib 没有），
    # 排在家族那几个前面：它们是这个节点的主结果（`main = "gravity"`），最常单独用的放最上
    outputs = (
        Port("gravity", VECTOR),
        Port("gravity_error", FLOAT, unit="°"),
    ) + LensCalibration.outputs
    runtime = "geocalib"
    # 上游 GeoCalib.calibrate(img, camera_model=…, priors=…) 交出 camera（Focal Length、主点、畸变）、gravity
    # （重力方向）、covariance 和各种 uncertainty（extractor.py:117-127）。
    official = Official(
        cite="third_party/geocalib/repo/geocalib/extractor.py:72-127",
        takes={"image": "img"},
        gives={"gravity": "gravity", "gravity_error": "uncertainty", "focal": "camera", "lens": "camera"},
        ours={"filmback": "filmback_mm"},  # 节点自己的「Filmback」参数原样带给下游
    )
    on_node = ("focal_mm", "step")
    # RTX 4090 上量得的显存和速度
    cost = Cost(gpu=True, vram_gb=6.5, seconds_per_frame=0.06)

    class Params(LensParams):
        # 「镜头模型」是要求（上游的 `camera_model`：要它按哪种模型去拟合），「镜头内参」口里带的模型名是结果
        # （它认出来是哪一种），两件事。档位和标签都从 lens.py 的 GROUP 取，
        # 和「LensDistortion」上 GeoCalib 那一组必须一一对上
        fit_model: Literal[tuple(GROUP.models)] = P(  # type: ignore[valid-type]
            "pinhole", group="lens",
        )
        step: int = P(1, ge=1, le=100, group="gravity")
        focal_mm: float | None = focal_param()

    @classmethod
    def prepare(cls, ctx) -> Job:
        image = ctx.input("image")
        used = plate_lens(ctx, image)
        return Job(image, extra={"focal_px": used.focal_px}, lens=used)

    @classmethod
    def convert(cls, ctx, raw, job):
        import json

        import numpy as np

        image, used = job.plate, job.lens
        w, h = image.meta["width"], image.meta["height"]
        g = raw.arrays("gravity.npz")
        lens = json.loads(raw.file("lens.json").read_text(encoding="utf-8"))

        ctx.stage("write_outputs")
        frames = g["frames"].tolist()
        up = g["up"] * np.array([1.0, -1.0, -1.0])  # OpenCV camera -> Lab2Shot camera axes (Y up, Z towards the viewer)
        picture = {"width": w, "height": h}  # what the focal length was measured on: checked against the plate it goes to
        out = {
            "gravity": value_packet(ctx.outputs["gravity"], VECTOR, frames=frames, values=up.tolist()),
            "gravity_error": value_packet(ctx.outputs["gravity_error"], FLOAT, frames=frames,
                                          values=g["uncertainty_deg"].tolist(), unit="°"),
            # 上游给的 Focal Length 是像素，换成毫米再交（用节点上的「Filmback」，LensParams）。
            # 除的是 Focal Length 自己所在的那个宽度（worker 写在 lens.json 里的），不是节点这边的画面宽度：
            # 两者目前相等，但不应依赖这一点（data/units.py focal_mm，px 与 mm 换算只此一处）
            "focal": value_packet(ctx.outputs["focal"], FLOAT,
                                  float(focal_mm(lens["focal_px"], ctx.params["filmback_mm"] or FILMBACK_MM, lens["width"])),
                                  unit="mm", **picture),
            "filmback": value_packet(ctx.outputs["filmback"], FLOAT,
                                     float(ctx.params["filmback_mm"] or FILMBACK_MM), unit="mm", **picture),
        }
        table, name = TABLE_MODELS.get(str(lens["model"]), ("SIMPLE_PINHOLE", ""))
        k1 = lens.get("k1")
        if "lens" in ctx.wanted:  # a pinhole is a lens too: one with no coefficients (an identity ST-map downstream)
            coeffs = {name: float(k1)} if distorts(table) and k1 is not None else {}
            out["lens"] = value_packet(ctx.outputs["lens"], LENS, packed_lens(distortion(table, coeffs), group=GROUP.id,
                                       name=str(lens["model"])), said=str(lens["model"]), **picture)
        ctx.say("I-GEOCALIB-TILT", roll=float(np.median(g["roll_deg"])), pitch=float(np.median(g["pitch_deg"])),
                count=len(up), error=float(np.median(g["uncertainty_deg"])),
                lens=used.said if lens["source"] == "user" else
                     lens_note(lens["focal_px"], lens["width"], float(ctx.params["filmback_mm"] or FILMBACK_MM)))
        return out


NODES = (Calibrate,)
