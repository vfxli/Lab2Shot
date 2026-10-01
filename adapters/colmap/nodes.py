"""COLMAP 扩展提供的节点（BSD-3；默认使用 CPU SIFT）。"""

from __future__ import annotations

from typing import Literal


from .extension import OPTION_LICENCES
from lab2shot.sdk import (licence_traits, rgb_port, Official, plate_mask_port, measured_param, Invalid, SolvedLensParams, Msg, Job, WorkerNode, P, Packet, Port,
                          external_camera, FLOAT, LENS, LENS_HELP, LENS_TABLE, PINHOLE_MODELS, lens_note,
                          opencv_points_to_usd, opencv_poses_to_usd, packed_lens, plate_lens, points_packet, solved_camera, unit_cm_param, value_packet,
                          window_of, Cost, Licence, OptionTrait, Param, Measured)

# 「镜头模型」的候选：COLMAP 透视相机模型（third_party/colmap/repo/src/colmap/sensor/models.h
# PERSPECTIVE_CAMERA_MODEL_CASES :215-232）中核心镜头表支持计算的模型（data/lens_models.py COLMAP_MODELS，id、名称、
# 参数名均沿用 COLMAP 官方定义）。选项名使用官方模型名，括号内注明适用的镜头类型；输出口的可用性随所选模型变化。
PINHOLES = PINHOLE_MODELS
# 仅提供在真值素材上验证可解的模型（桶形畸变 TUM RGB-D fr1_desk：OpenCV k1 k2 p1 p2 k3；鱼眼 TUM-VI room1：
# Kannala-Brandt 4 项；45 帧，隔 3 帧）：
#   可用：SIMPLE_PINHOLE、PINHOLE、SIMPLE_RADIAL、RADIAL、OPENCV、SIMPLE_RADIAL_FISHEYE、RADIAL_FISHEYE、OPENCV_FISHEYE；
#   不提供：FULL_OPENCV（12 个系数，两种解算方式、焦距已知未知都发散，只注册 0–13 帧、系数到 ±100）；
#   未验证，暂不提供（镜头表中有公式，可在「LensDistortion」中手动填写）：FOV、THIN_PRISM_FISHEYE、RAD_TAN_THIN_PRISM_FISHEYE、
#   SIMPLE_DIVISION、DIVISION、SIMPLE_FISHEYE、FISHEYE、EUCM。待真值素材验证后开放。
from .lens import GROUP, TESTED  # noqa: F401  # 清单定义于 lens.py：节点提供的选项与「LensDistortion」COLMAP 组的选项一致

OFFERED = tuple(GROUP.models)
FISHEYES = ("SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE")


class CameraSolve(WorkerNode):
    id = "colmap.camera_solve"
    version = 3  # 「镜头模型」选无畸变时也交「镜头内参」（没有系数的镜头），不再是空包
    # 本节点自行解算镜头：输出的相机带有解出的畸变。选择无畸变模型时（模板先去畸变再以默认的 SIMPLE_PINHOLE 解算）
    # 与普通针孔节点相同。具体属于哪种由「镜头模型」决定（pinhole_when；取值见 nodes/applies.py LENSES）
    lens = "solves"
    pinhole_when = Param("fit_model").one_of(*PINHOLES)  # 两种无畸变模型下按普通针孔节点处理
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（9 个镜头）：填真实 Focal Length 或接 AnyCalib，相机轨迹没区别
    #   mask：实测（3–4 个有运动物体的镜头）：相机轨迹没区别
    on_node = ("focal_mm", "mapper", "step")
    # 视图下方的控件只放「镜头模型」参数；解出的 Focal Length、Filmback、镜头内参已在视图中显示
    # （「已知 Focal Length」「Filmback」两个参数未填写时为空，不放入控件）
    strip = {"fit_model": "镜头模型"}
    # 画面必须有视差，固定机位和纯摇镜头会直接报错。本方法为纯几何方法，运动物体须通过「运动物体遮罩」口排除。
    # 尺度任意（默认 1 单位 = 1 米）；Focal Length 已知时与 ViPE 相差 2.8 cm / 0.37°，自行解算 Focal Length 时偏长约 20%（7.7 cm / 2.51°）。
    # 不提供「人物框」输入口：上游只接受遮罩图（--ImageReader.mask_path，黑色处不提取特征）。需要排除人物时，在节点图上连接
    # 「ViTDet 人物框」→「人物框转遮罩」→ 本节点的「运动物体遮罩」口（worker 侧 worker_sdk/lab2shot_worker/recon.py MovingMasks 仅合成布尔图）。
    # 对 COLMAP 而言，遮挡与运动物体遮罩的处理方式相同：黑色区域不提取特征。
    # COLMAP 不属于重建家族（直接继承 WorkerNode），因此输入口在此处声明，不使用 takes_mask
    inputs = (rgb_port(), plate_mask_port("运动物体遮罩"))
    main = "camera"  # 节点的主要输出；输出口按类型顺序排列
    # 解出的镜头以数值输出（输出口与「AnyCalib 镜头标定」一致；ST-map 仅由「LensDistortion」烘焙，在节点图上可见）。
    # Focal Length 和 Filmback 在解出的相机上同样存在，单独输出是为了便于连接到参数。
    # 无畸变的两档也交「镜头内参」（没有系数的镜头）：镜头口的类型不随取值变，下游「LensDistortion」给恒等 ST-map。
    # 输出：相机、点云、Focal Length、Filmback、镜头内参，与上游 cameras.txt 一行的内容对应。
    # Focal Length / Filmback 因常用而单独输出，其余内容（模型 + 系数 + 主点 + 由 fx≠fy 折算的像素比）合并为
    # 「镜头内参」（nodes/lens.py packed_lens）。因此视图下方的控件为 Focal Length、Filmback、镜头内参（显示模型名）三项
    outputs = (Port("camera", "scene.camera", "相机"), Port("points", "scene.points", "点云"),
               # 解出的 Focal Length，单位毫米（与 AnyCalib / GeoCalib 的输出口及单位一致）。
               # 上游 cameras.txt 的 PARAMS 第一项为 Focal Length（px）（doc/format.rst:106），
               # 按节点的「Filmback」换算为毫米；仅做单位换算，不改变官方结果
               Port("focal", FLOAT, "Focal Length", unit="mm",
                    help="COLMAP 解出的 Focal Length，毫米（按节点上的「Filmback」换算）。接「LensDistortion」的「已知 Focal Length」，"
                         "或者接任何要 Focal Length 的解算器"),
               # 「Filmback」原样输出节点参数。该值并非上游计算结果，而是传递给下游，
               # 避免在两个节点上重复填写且不一致（Focal Length 的毫米值按此换算）
               Port("filmback", FLOAT, "Filmback", unit="mm", help="节点上填的「Filmback」原样交出去：Focal Length 是按它换算成毫米的，下游（「LensDistortion」「创建相机」）接这一根就不用再填一遍，两边永远是同一个数"),
               Port("lens", LENS, "镜头内参", help=LENS_HELP + "。「镜头模型」选无畸变的两档时是没有系数的镜头，接「LensDistortion」得到恒等 ST-map"))
    runtime = "colmap"
    # 官方接口的输入输出与解算器一致：
    # COLMAP 官方命令行接受 --image_path（画面）和 --ImageReader.mask_path（遮罩图，黑色处不提取特征；
    # doc/cli.rst:287、292，说明见 src/colmap/controllers/image_reader.h:43-52），
    # 输出稀疏模型的三个文件：cameras.txt（镜头模型和参数）、images.txt（每帧位姿）、points3D.txt（稀疏点云），
    # doc/cli.rst:508-509、doc/format.rst:99-180。
    official = Official(
        cite="third_party/colmap/repo/doc/cli.rst:287-509",
        takes={"image": "--image_path", "mask": "--ImageReader.mask_path"},
        gives={"camera": "images.txt", "points": "points3D.txt", "focal": "cameras.txt", "lens": "cameras.txt"},
        ours={"filmback": "filmback_mm"},  # 节点的「Filmback」参数原样传给下游
        note="遮罩是官方的输入（mask_path，黑色处不提特征）。没有人物框输入口：上游只收遮罩图这一种形式；"
             "要挡人就接「人物框转遮罩」到「运动物体遮罩」口，那一步在节点图上看得见。「镜头内参」（模型 + "
             "畸变系数 + 主点 + 像素比）就是 cameras.txt 里那一行的 MODEL 和 PARAMS（doc/format.rst:106）"
             "打成一份，「相机」是 images.txt 的四元数加位移（doc/format.rst:144）",
    )
    # COLMAP 无法用少于 3 个视图建立模型（worker MIN_REGISTERED，按隔帧后的帧数计），提交前拒绝
    min_frames, min_frames_step = 3, "step"
    # gpu_options：SiftGPU 在 GPU 上运行，任务需等待空闲 GPU
    # vram_gb：「显卡提取特征」开启时使用 SiftGPU（经典特征点算法），显存未单独测量，取保守值（关闭时不使用 GPU，无档位）
    # seconds_per_frame：300 帧隔 2 帧时，提取与匹配约 100 秒、全局建图 118 秒，按参与计算的 300 帧折算
    cost = Cost(vram_gb=2.0, seconds_per_frame=0.8, vram_measured=False, note="只有打开「显卡提取特征」才用显卡；SiftGPU 是经典特征点算法，显存是保守估计")
    licence = Licence(note="COLMAP 是 BSD-3，可以商用；只有打开「显卡提取特征」时用到的 SiftGPU 仅限教育和研究。")
    traits = (
        OptionTrait(Param('sift_gpu').one_of(True), gpu=True),
        *licence_traits(OPTION_LICENCES),
    )

    class Params(SolvedLensParams):
        mapper: Literal["global", "incremental"] = P(
            "global", label="解算方式", group="解算",
            option_labels={"global": "全局", "incremental": "增量"},
        )
        matcher: Literal["sequential", "exhaustive"] = P(
            "sequential", label="匹配方式", group="解算",
            option_labels={"sequential": "相邻帧", "exhaustive": "所有帧两两"},
        )
        # 参数名为「镜头模型」（输出口名为「镜头内参」，二者不重名），对应上游的 `--ImageReader.camera_model`，
        # 指定 COLMAP 解算所用的模型。选项为 COLMAP 模型名，与「LensDistortion」的选项完全一致
        fit_model: Literal[OFFERED] = P(  # type: ignore[valid-type]
            "SIMPLE_PINHOLE", label="镜头模型", group="镜头",
            option_labels={m: LENS_TABLE[m].label for m in OFFERED},
        )
        step: int = P(1, label="隔帧", ge=1, le=10, group="解算")
        # 更高的处理分辨率及其显存占用未经测试，上限取保守值
        resolution: Literal[1000, 1500, 2000] = measured_param(
            "处理分辨率", {1000: Measured(flat=True), 1500: Measured(flat=True), 2000: Measured(flat=True)},
            default=2000, group="解算")
        unit_cm: float = unit_cm_param()
        sift_gpu: bool = P(False, label="显卡提取特征", group="解算")

    @classmethod
    def foresee(cls, params: dict, info) -> list:
        """拒绝「鱼眼模型 + 焦距未知 + 增量解算」的组合：三种鱼眼模型在该组合下无法注册任何帧（针孔下验证过的匹配
        换成鱼眼后，增量建图无法建立初始图像对；全局建图可从 COLMAP 的默认焦距收敛到真值）。在提交前报错，避免计算后才失败。"""
        if params.get("fit_model") in FISHEYES and params.get("mapper") == "incremental" and not params.get("focal_mm"):
            return [Msg("B-COLMAP-FISHEYEINCREMENTAL", model=LENS_TABLE[params["fit_model"]].label)]
        return []

    @classmethod
    def prepare(cls, ctx) -> Job:
        image = ctx.input("image")
        used_frames = len(image.meta["frames"][::ctx.params["step"]])
        # 全部帧两两匹配为 O(n²) 对，帧数较多时计算极慢或耗尽内存（风险来自「匹配方式」与「帧数」的组合，而非单个参数）；
        # 提示中注明的适用范围为几十帧
        if ctx.params["matcher"] == "exhaustive" and used_frames > 300:
            raise Invalid(Msg("E-COLMAP-EXHAUSTIVE", frames=used_frames))
        # 遮罩（值 > 0.5 表示运动物体）区域内的特征被忽略
        used = plate_lens(ctx, image)
        return Job(image, extra={"focal_px": used.focal_px}, inputs=ctx.input_files("mask"), lens=used)

    @classmethod
    def convert(cls, ctx, raw, job):
        import json

        import numpy as np

        image, used = job.plate, job.lens
        frames = image.meta["frames"]
        w, h = image.meta["width"], image.meta["height"]
        ctx.stage("写出相机和点云")
        cams = json.loads(raw.file("cameras.json").read_text(encoding="utf-8"))
        scale = float(ctx.params["unit_cm"])
        solved = {int(f): opencv_poses_to_usd(np.asarray(m, np.float64), scale) for f, m in cams["poses"].items()}
        got = external_camera(cams["model"], cams["params"])  # 经由统一的镜头模型表
        if got is None:
            raise Invalid(Msg("E-COLMAP-MODEL", model=cams["model"]))
        # COLMAP 按「处理分辨率」处理传入的全部像素，包括 overscan 区域（去畸变后的画面位于画布上）。
        # 其结果以自身像素为单位，因此按其所见画布缩放，再将主点移到描述相机所用的画面坐标系上
        window = window_of(image)
        (canvas_w, _), (left, top) = window.canvas, window.offset
        px = canvas_w / float(cams.get("width") or canvas_w)  # COLMAP 像素 -> 画面像素
        focal = got["fx"] * px  # 像素焦距不受裁切影响，仅受缩放影响
        dist = got["distortion"]
        lens = {} if not LENS_TABLE[dist["model"]].params else {"distortion": dist, "raster": [canvas_w, window.canvas[1]],
                                                  "source": {"level": "solved", "by": "COLMAP"}}
        info = {"extension": "colmap", "camera_model": cams["model"], "reprojection_error_px": cams["mean_reprojection_error_px"],
                "lens": used.said}
        camera_out = solved_camera(ctx, image, list(solved), focal, list(solved.values()), info=info, filmback_mm=used.filmback_mm,
                                   fy_px=got["fy"] * px, principal_px=(got["cx"] * px - left, got["cy"] * px - top), lens=lens)

        pts = raw.arrays("points.npz")
        xyz = opencv_points_to_usd(pts["xyz"], scale).astype(np.float32)  # OpenCV 世界 → 本项目的 Y 向上、cm
        rgb = pts["rgb"].astype(np.float32) / 255.0
        # 稀疏点云是静态的：只有一组点，没有时间采样
        points = points_packet(ctx.outputs["points"], frames, "colmap_points", [xyz], [rgb], scale="relative",
                               width_cm=max(scale * 0.02, 0.5), info=info, width=w, height=h)
        ctx.say("I-COLMAP-SOLVED", solved=len(solved), frames=len(frames), error=float(cams["mean_reprojection_error_px"]),
                lens=lens_note(focal, w, used.filmback_mm), points=len(xyz))
        # 解出的镜头以数值输出：Focal Length、Filmback 各占一个输出口，其余合并为「镜头内参」。
        # 主点由像素换算为相对画面中心的毫米偏移（+y 向上，公式见 lens_models.py 模块说明）；fx≠fy 的模型
        # （PINHOLE、OPENCV 等）折算为像素比 fy/fx（镜头表中已有该项）。两种无畸变模型没有系数：交没有系数的镜头
        cx_mm = (got["cx"] * px - w / 2) * used.filmback_mm / w
        cy_mm = -(got["cy"] * px - h / 2) * used.filmback_mm / w
        picture = {"width": w, "height": h}
        values = {
            # 解出的 Focal Length，单位毫米（Focal Length（px）÷ 画面宽度 × Filmback）
            "focal": value_packet(ctx.outputs["focal"], FLOAT, float(focal) / w * used.filmback_mm, unit="mm", **picture),
            "filmback": value_packet(ctx.outputs["filmback"], FLOAT, float(used.filmback_mm), unit="mm", **picture),
            "lens": value_packet(ctx.outputs["lens"], LENS, packed_lens(dist, (cx_mm, cy_mm), float(got["fy"]) / float(got["fx"]), group="colmap", name=str(dist["model"])),
                                 said=str(dist["model"]), **picture),
        }
        return {"camera": camera_out, "points": points, **{k: v for k, v in values.items() if k in ctx.wanted}}


NODES = (CameraSolve,)
