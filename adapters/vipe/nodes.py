"""NVIDIA ViPE 扩展提供的节点：相机解算，以及基于解算结果的深度计算。

两个节点串联：先解算相机，再基于解算结果计算深度。共用声明集中在基类 `ViPE` 中。

拆分依据：上游本身即如此划分。`vipe/pipeline/default.py` 的 `run()` 先运行 SLAM，
得到 `SLAMOutput`（轨迹 + 内参 + `slam_map` 三维点），再交给 post 阶段计算深度
（`third_party/vipe/repo/vipe/pipeline/default.py:88-98`）；
设置 `pipeline.post.depth_align_model=null` 时只输出位姿、不输出深度
（`third_party/vipe/repo/docs/usage.md:63-64`，原文为 "Run pose-only output without depth post-processing"）。

因此节点间传递的是「相机 + 点云」，而非仅有相机：深度计算除轨迹外还需要 SLAM 解出的三维点，
`AdaptiveDepthProcessor` 逐帧将其投影到画面上作为深度提示
（`third_party/vipe/repo/vipe/pipeline/processors.py:279` 读取 `slam_map`，`:297` 和 `:320` 调用 `project_map`）。
由 worker 负责将这两项还原为上游所需的 `SLAMMap` 结构。

「物体分割」同样经连线传递：GroundingDINO + SAM + DeAOT 是解算阶段的处理器
（`third_party/vipe/repo/vipe/pipeline/default.py:70-78`，位于 `_add_init_processors`），
两个预设都在 SLAM 之前运行，用于将移动的人、车排除在解算之外。深度计算使用同一份遮罩
（`third_party/vipe/repo/vipe/pipeline/processors.py:326` 和 `:341-343`，均为 `if frame.mask is not None`）。
因此遮罩由解算节点输出并传给深度节点，深度节点不重新运行这三个模型：
重新运行既属重复计算，两次分割出的物体也未必一致。
"""

from __future__ import annotations

from typing import Literal


from lab2shot.sdk import (rgb_port, CameraSamples, Official, Invalid, MissingFrames, Msg, NodeParams, usd_points_to_m,
                          Job, LensParams, WorkerNode, open_scene, P, scene_points,
                          Port, SameShot, camera_port, depth_maps, empty_packet, frame_maps, opencv_points_to_usd,
                          opencv_poses_to_usd, plate_lens, points_packet, send_camera, solved_camera, window_of, Cost, Licence, NONCOMMERCIAL)


SKY = "sky"  # 上游 VideoFrame.SKY_PROMPT（streams/base.py:64）：唯一不属于运动物体的实例
PATTERN = "instance_{}.npz"  # worker 写出实例编号的文件名，每帧一个文件


class ViPE(WorkerNode):
    """两个 ViPE 节点的共用声明：相同的运行环境、针孔相机前提、许可证说明和缺帧处理（跳过）。

    Focal Length 只有一个来源：「ViPE 相机解算」没有相机输入口，Focal Length 作为参数经 `extra` 传递；
    「ViPE 深度图」必须接入相机，Focal Length 取自该相机，由 `send_camera` 写入 camera_in.npz，
    节点上不提供「已知 Focal Length」参数。上游 post 阶段读取的也是 `slam_output.intrinsics`，
    不单独接受 Focal Length（输入与上游解算器保持一致）。

    该类没有 `id`，不是节点（也不在 `NODES` 中），仅用于集中共用声明。
    """

    lens = "pinhole"  # 将画面视为无畸变镜头，要求输入已去畸变的画面
    runtime = "vipe"
    missing_frames = MissingFrames.SKIP


class CameraSolve(ViPE):
    id = "vipe.camera_solve"
    version = 4  # 4: warns when SLAM did not converge (too few keyframes) or the focal is not settled yet; 3: an undistorted canvas is solved without its overscan (the official input has no black borders); 2: preserve solved fx/fy/principal point rather than replacing them with one averaged focal
    on_node = ("focal_mm", "mode")
    main = "camera"  # 输出口按类型顺序排列，主输出口标明节点的主要产出
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（12 个镜头）：填真实 Focal Length，相机轨迹和深度没区别；接 AnyCalib 估的 Focal Length 4 个镜头变差
    # 相机须有位移；固定机位或纯摇镜头下自动 Focal Length 不可靠（sh010 解为 1052 px，实际约 588 px）；
    # 自动识别画面中运动的人和物体并在解算时排除；尺度接近米制；填写真实 Focal Length 对相机轨迹无影响（12 个镜头）。
    # 「点云」为整段共用一套：slam_map 中的所有点位于同一 SLAM 世界坐标系（slam/interface.py:27-38 dense_disp_xyz，
    # 按关键帧分块仅记录来源帧），而非每帧位于各自相机空间。
    # ViPE 自行检测运动的人和物体（GroundingDINO + SAM），因此不提供遮罩 / 人物框输入
    inputs = (rgb_port(),)
    outputs = (Port("camera", "scene.camera"),
               Port("points", "scene.points", may_be_empty=True),
               Port("objects", "image.1", may_be_empty=True))
    # 输入输出与上游解算器一一对应：
    # 本节点运行官方的 pose_only / pose_only_long 两个预设，它们只运行到 SLAM 为止
    # （vipe/pipeline/pose_only.py:34-41：跳过 depth 的 post 处理和写盘），
    # 因此输出的三项即 SLAM 阶段的产物：
    #   位姿 + 内参（slam_output.trajectory / .intrinsics）、slam_map（三维点），
    #   以及 SLAM 之前 init 阶段分割出的运动物体实例（TrackAnythingProcessor，default.py:70-78）。
    official = Official(
        # 三处引用分别说明：帧上包含的数据（streams/base.py）、SLAM 的输出（slam/interface.py）、
        # 两条流水线的终止位置（pose_only.py）。校验会在这些行中查找下列每个符号
        cite=["third_party/vipe/repo/vipe/streams/base.py:46-76",
              "third_party/vipe/repo/vipe/slam/interface.py:186-197",
              "third_party/vipe/repo/vipe/pipeline/pose_only.py:34-41"],
        takes={"image": "rgb"},
        gives={"camera": "pose", "points": "slam_map", "objects": "instance"},
    )
    # vram_gb：在 RTX 4090 上以默认「标准」模式测得
    cost = Cost(gpu=True, vram_gb=9.6, seconds_per_frame=0.093)

    class Params(LensParams):
        mode: Literal["pose_only", "pose_only_long"] = P("pose_only", group="solve")

    @classmethod
    def prepare(cls, ctx) -> Job:
        image = ctx.input("image")
        used = plate_lens(ctx, image)
        return Job(image, extra={"focal_px": used.focal_px, "crop": _solve_box(window_of(image))}, lens=used)

    @classmethod
    def convert(cls, ctx, raw, job):
        image, used = job.plate, job.lens
        out = {"camera": _camera(ctx, raw, image, used)}
        if "points" in ctx.wanted:
            out["points"] = _slam_points(ctx, raw, image)
        out.update(_instances(ctx, raw, image))
        return out


class Depth(ViPE):
    """ViPE 的 post 阶段：基于解算好的相机及其三维点计算逐帧深度图。"""

    id = "vipe.depth"
    version = 3  # 3: SLAM points sent in the camera's own world (they were mirrored and never used); 2: full camera calibration is restored for depth alignment
    on_node = ("model",)
    # 深度位于接入相机的世界坐标系中；运动物体取自接入的「物体分割」（不接入也可计算）
    main = "depth"
    inputs = (
        rgb_port(),
        camera_port(optional=False),
        Port("points", "scene.points", expects=(SameShot("image"),)),
        Port("objects", "image.1", optional=True, expects=(SameShot("image"),)),
    )
    outputs = (Port("depth", "image.1", means=("scale",)),)
    # 输入输出与上游解算器一一对应：
    # 本节点运行 `DefaultAnnotationPipeline._add_post_processors`（default.py:88-98），
    # 其输入为「帧 + slam_output」，该步骤实际读取 slam_output 中的三项：
    #   trajectory（每帧位姿）、intrinsics（内参）、slam_map（三维点），
    # 对应上面三个输入口。深度处理器另读取同一帧的 frame.mask（processors.py:326、:341-343），
    # 即 init 阶段分割出的运动物体，因此「物体分割」为第四个（可选）输入口。
    official = Official(
        # 三处引用分别说明：post 阶段的组装（default.py）、深度处理器读取的数据（processors.py）、
        # 输出在帧上的名称（streams/base.py）
        cite=["third_party/vipe/repo/vipe/pipeline/default.py:88-98",
              "third_party/vipe/repo/vipe/pipeline/processors.py:274-345",
              "third_party/vipe/repo/vipe/streams/base.py:46-76"],
        takes={"image": "rgb", "camera": "slam_output.get_view_trajectory",
               "points": "slam_output.slam_map", "objects": "frame.mask"},
        gives={"depth": "metric_depth"},
    )
    # vram_gb：在 RTX 4090 上以默认「标准」深度模型测得
    cost = Cost(gpu=True, vram_gb=9.7, seconds_per_frame=0.15)
    # both of its modes run non-commercial depth models; 「ViPE 相机解算」's modes use none (the extension is 可商用)
    licence = Licence(NONCOMMERCIAL, note=True)

    class Params(NodeParams):
        # 不提供「已知 Focal Length」「Filmback」：本节点必须接入相机，内参取自该相机
        # （上游 post 阶段读取 slam_output.intrinsics，`default.py:90-92`）。
        # 若另设 Focal Length 参数，同一个值将有两个来源进入 worker，优先级不明确
        model: Literal["default", "dav3"] = P("default", group="depth")

    @classmethod
    def prepare(cls, ctx) -> Job:
        image = ctx.input("image")
        camera, points, objects = ctx.input("camera"), ctx.input("points"), ctx.input("objects")
        frames = image.meta["frames"]
        # Focal Length 仅取自接入的相机，节点上没有其他来源
        focal = CameraSamples.from_packet(camera, frames).focal_px(image.meta["width"])
        sent = {"camera": send_camera(ctx, camera, frames, focal),
                "points": _send_points(ctx, points, image)}
        _same_world(sent["camera"], sent["points"])
        extra = None
        if objects is not None and not objects.meta.get("empty"):
            sent.update(ctx.input_files("objects"))
            extra = {"sky": _sky_ids(objects)}
        else:  # 可选输入未接入或为空：主结果照常计算，并在节点上说明未使用的输入
            ctx.say("N-VIPE-NOOBJECTS")
        return Job(image, inputs=sent, extra=extra or {})

    @classmethod
    def convert(cls, ctx, raw, job):
        return {"depth": depth_maps(ctx, raw, job.plate, "depth_{}.npz")}


def _slam_points(ctx, raw, image):
    """SLAM 建立的三维点 -> 「点云」，每个关键帧一份。

    上游的 `SLAMMap` 按关键帧分块存储（`third_party/vipe/repo/vipe/slam/interface.py:27-38`：
    `dense_disp_xyz` 为所有点的拼接，`dense_disp_packinfo` 记录各块的起止，
    `dense_disp_frame_inds` 记录各块所属的帧）。写成每个关键帧一份的点云可完整保留这三项，
    「ViPE 深度图」读回时能还原为上游所需的结构。若合并为单一静态点云，关键帧归属将丢失，
    而 `project_map` 每帧只取邻近 ±3 个关键帧的点（`processors.py:141-143`），无法还原则偏离官方流程。
    """
    import numpy as np

    d = raw.arrays("slam_points.npz")
    counts = np.asarray(d["counts"], np.int64)
    if not counts.size or int(counts.sum()) == 0:  # 长镜头方式不建地图：空点云输出空包
        return empty_packet(ctx, "points")
    # worker 输出的是流内序号（即上游 dense_disp_frame_inds，processors.py:141 按其 searchsorted），
    # 此处换算回镜头的原始帧号，因为点云包的时间采样以帧号为准
    shot = list(image.meta["frames"])
    keyframes = [shot[min(int(i), len(shot) - 1)] for i in np.asarray(d["frames"], np.int64)]
    xyz = opencv_points_to_usd(np.asarray(d["xyz"], np.float64))
    rgb = np.asarray(d["rgb"], np.float32)
    ends = np.cumsum(counts)
    starts = ends - counts
    return points_packet(ctx.outputs["points"], keyframes, "vipe_slam_points",
                         [xyz[s:e] for s, e in zip(starts, ends)], [rgb[s:e] for s, e in zip(starts, ends)],
                         scale="metric", points_from="native")  # the method's own points (geometry.py's word), not from a depth map


def _same_world(camera_npz, points_npz):
    """点和相机必须在同一个世界里：每个关键帧的点大多应在它自己那台相机前方。上游对不上时不报错，只是悄悄
    退回不带 SLAM 提示的深度（Minimum UV score 0），所以在这里先查：多数点在相机背后就停下说清楚，不静默降级。"""
    import numpy as np

    cam, pts = np.load(camera_npz), np.load(points_npz)
    c2w = cam["cam_to_world"] if "cam_to_world" in cam.files else None
    if c2w is None:
        return
    ahead = total = 0
    start = 0
    for count, frame in zip(pts["counts"], pts["frames"]):
        xyz = pts["xyz"][start:start + count]
        start += count
        m = c2w[int(frame)]
        z = ((xyz - m[:3, 3]) @ m[:3, :3])[:, 2]  # OpenCV 相机轴：+Z 朝前
        ahead += int((z > 0).sum())
        total += len(z)
    if total and ahead / total < 0.5:
        raise Invalid(Msg("E-VIPE-POINTSBEHIND", share=ahead / total))


def _send_points(ctx, points, image):
    """「点云」输入 -> worker 读取的 points_in.npz（与 `send_camera` 送出的相机同一个世界坐标，米，按关键帧分块）。

    `send_camera` 只把相机自身的轴换成 OpenCV，世界仍是本项目的 Y 轴向上；点必须在同一个世界里，
    否则上游投影时点全在相机背后（Minimum UV score 0），静默退回不带 SLAM 提示的深度。
    分块依据包内的帧号（每个关键帧一个时间采样，由 `_slam_points` 写入）。
    """
    import numpy as np
    from pxr import Usd

    shot = list(image.meta["frames"])
    stage = open_scene([points])
    blocks, rgbs, keyframes = [], [], []
    for f in points.meta["frames"]:
        clouds = [(p, c) for _, p, _, c in scene_points(stage, Usd.TimeCode(f))]
        if not clouds:
            continue
        xyz = np.concatenate([p for p, _ in clouds])
        colors = [c for _, c in clouds if c is not None]
        blocks.append(usd_points_to_m(xyz))  # 与相机同一个世界（Y 轴向上），换成米（换算统一定义于 data/units.py）
        rgbs.append(np.concatenate(colors) if len(colors) == len(clouds) else np.zeros((len(xyz), 3), np.float32))
        # 上游按流内序号索引关键帧（processors.py:141 的 searchsorted 使用流内序号），而非原始帧号
        keyframes.append(shot.index(int(f)) if int(f) in shot else len(shot) - 1)
    if not blocks:
        raise Invalid(Msg("E-VIPE-NOPOINTS"))
    path = ctx.work / "points_in.npz"
    np.savez(path, xyz=np.concatenate(blocks), rgb=np.concatenate(rgbs).astype(np.float32),
             counts=np.asarray([len(b) for b in blocks], np.int64), frames=np.asarray(keyframes, np.int64))
    return path


def _sky_ids(objects) -> list[int]:
    """返回「物体分割」类别表中表示天空的编号。

    上游计算 `frame.sky_mask` 时在 `instance_phrases` 中查找 `SKY_PROMPT`
    （`third_party/vipe/repo/vipe/streams/base.py:263-267`），深度对齐时据此排除天空
    （`third_party/vipe/repo/vipe/pipeline/processors.py:341-343`）。
    编号图本身经输入口传递（`ctx.input_files`，与其他节点向 worker 传图的方式相同）；
    天空编号列表经参数传递。"""
    names = objects.meta.get("classes") or []
    return [int(c["index"]) for c in names if str(c.get("name", "")).split("_")[0].strip().lower() == SKY]


def _instances(ctx, raw, image):
    """官方的运动物体实例遮罩及其词表 -> 「物体分割」。

    ViPE 在解算前分割并遮挡移动的人、车、动物（GroundingDINO + SAM + DeAOT），官方
    `save_artifacts` 将实例遮罩和词表保存为文件（vipe/utils/io.py:359-380）。worker 将其写入
    raw/instance_<帧>.npz（uint8 编号图）和 result.json 的 instance_phrases（编号 -> 词）。

    编号图的格式与「SAM 3 视频分割」相同：编号即像素值，
    名称写入「类别表」，写多层 EXR 时对应 Cryptomatte 的 person_01。
    需要遮罩（例如仅保留运动物体）时，在节点图上接入「分割转遮罩」`mask_from_segments`。
    该节点是独立的显式工具，并非上游产物，因此不在本节点上设输出口（节点输出与上游一一对应）。"""
    import re

    import numpy as np

    if "objects" not in ctx.wanted:
        return {}
    # 没有任何实例遮罩（画面中未发现运动物体）时输出空包，空结果不是错误。
    # 未写入任何帧的 ExrWriter 无法生成数据包，因此先检查再决定处理路径
    if not any(raw.path(PATTERN.format(f)).exists() for f in image.meta["frames"]):
        return {"objects": empty_packet(ctx, "objects")}
    phrases = {int(k): str(v) for k, v in (raw.result().get("instance_phrases") or {}).items() if int(k) > 0}

    def name(word: str, oid: int) -> str:  # Cryptomatte manifest 名称，与「SAM 3 视频分割」的写法一致，如 person_01
        slug = re.sub(r"\s+", "_", (word or "object").strip()) or "object"
        return f"{slug}_{oid:02d}"

    classes = [{"index": i, "name": name(phrases[i], i)} for i in sorted(phrases)]
    maps = {"objects": ("image.1", lambda d: d["instance"].astype(np.float32),
                        {"value_range": (0.0, float(max(phrases, default=1))), "classes": classes})}
    return frame_maps(ctx, raw, image, maps, PATTERN, "write_objects")


def _solve_box(window) -> list[int] | None:
    """去畸变画布带扩边时，交给 ViPE 解算的那一块（画布坐标 x, y, w, h）；没有扩边为 None（整幅）。

    官方输入是一段普通视频，没有黑边；扩边里镜头照不到的黑角静止、无纹理，GeoCalib 和 SLAM 都会看到
    （实测 R04 60 帧自动焦距：裁掉 519 px，带黑边 506 / 529 px，真值 517）。取画面框里以画布中心为中心的最大矩形：
    画布中心就是主点（data/lens_models.py fit_canvas：去畸变画布按镜头中心居中），ViPE 假定主点在画面中心，
    这样裁完它仍在中心；对称扩边时这一块就是画面框本身。"""
    if not window.has_overscan:
        return None
    (cw, ch), (ox, oy) = window.canvas, window.offset
    mx, my = cw / 2, ch / 2
    hx = int(min(mx - ox, ox + window.width - mx))
    hy = int(min(my - oy, oy + window.height - my))
    if hx < 8 or hy < 8:  # the plate frame does not hold the canvas centre: nothing sensible to cut
        return None
    return [int(round(mx - hx)), int(round(my - hy)), 2 * hx, 2 * hy]


def _camera(ctx, raw, image, used):
    """ViPE 的相机（OpenCV，世界坐标系为第一帧相机，米）-> 「相机」输出，使用所用镜头的 Filmback。"""
    cam = raw.arrays("camera.npz")
    source = raw.result().get("focal_source")
    left, top, _, _ = window_of(image).overscan
    return solved_camera(ctx, image, cam["frames"], float(cam["fx"]), opencv_poses_to_usd(cam["cam_to_world"]),
                         fy_px=float(cam["fy"]), principal_px=(float(cam["cx"]) - left, float(cam["cy"]) - top),
                         filmback_mm=used.filmback_mm, info={"extension": "vipe", "focal_source": source, "lens": used.said})


NODES = (CameraSolve, Depth)
