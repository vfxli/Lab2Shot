"""深度与相机家族：从画面计算深度及其观察相机。逐帧计算与一次处理整段是同一类工作的两个档位。

家族形态：输入 `image.3`（画面），输出深度图 + 相机。其余端口（天空遮罩、法线图、点云、距离图、射线场、
运动物体遮罩、规范坐标、面部遮罩）是各节点按上游能力自行声明的可选端口，不构成家族的划分依据；
「点云」是两档共用的派生端口（`kit/maps.py family_points`）。端口顺序由 `nodes/base.py __init_subclass__`
中的 `in_port_order` 按数据类型统一排列，与声明顺序无关。

逐帧还是整段是各节点自身的性质，不作为家族划分依据：同一个项目可以在两档各提供一个节点（Depth Anything 3）。

命名为「深度与相机」而非 reconstruction / geometry：后两者是算法术语；Nuke 和 Houdini 中也没有
「画面 → 深度 + 相机」的现成节点名可供借用。

两档的区别只在于上游返回相机的方式：

| 档 | 上游计算方式 | 相机来源 | 成员 |
|---|---|---|---|
| `PerFrameDepthCamera` | 逐帧独立计算 | 每帧的内参（`frame_<n>.npz` 的 `intrinsics`）；相机位于原点且静止 | MoGe、UniDepth、UniK3D、Depth Anything 3、FaceAnything |
| `WholeShotDepthCamera` | 一次处理整段 | 整段联合解算的轨迹（`cameras.npz` 的 `K` + `cam_to_world`） | CUT3R、Depth Anything 3、LingBot-Map、MapAnything、MonST3R、Pi3、VGGT |
"""

from __future__ import annotations

import numpy as np

from ...data.packet import Packet
from ...data.payloads import SIGNED, ExrWriter, window_of
from ...data.units import M_TO_CM
from ...errors import Invalid
from ...messages import Msg
from ..applies import Cost
from ..base import NodeParams, P, Port
from ..kit.cameras import plate_lens, sent_fov_x_deg, solved_camera
from ..kit.confidence import ConfidenceWriter
from ..kit.maps import family_points, native_points_of, points_params, turn_to_camera
from ..kit.ports import plate_mask_port
from ..lens import LensParams, takes_lens
from .base import Job, MissingFrames, RawOutput, WorkerNode


class DepthCamera(WorkerNode):
    """两档共用的形态：输入画面，输出深度图及其观察相机。

    端口：输入 `image`（image.3）；输出 `depth`（image.1，厘米）、`camera`（scene.camera）以及家族自带的 `points`。
    声明了 `sky_map`（其 worker 写入 npz 的数组名）的节点还会得到「天空遮罩」输出：
    只需在适配器中写一行声明，无需代码。

    `points` 是家族提供的便利输出：所有项目使用相同的反投影，仅在有连线时计算，
    且只保留在声明了 `native_points` 的节点上（nodes/base.py __init_subclass__）。

    选择哪一档只取决于一个问题：上游是逐帧运行（`PerFrameDepthCamera`），
    还是一次处理整段（`WholeShotDepthCamera`）。"""

    lens = "pinhole"  # 将画面视为无畸变镜头拍摄：声明需要去畸变的画面
    inputs = (Port("image", "image.3", "RGB"),)
    # 没有「相机」输入端口：成员的上游均不接受逐帧外参（UniDepth 只接受内参 `Pinhole(K=intrinsics)`；UniK3D 的
    # decoder 自行输出内参；Depth Anything 3 使用其自身输出的内外参）。仅有内参不构成相机输入，内参由「Focal Length」
    # 「Filmback」两个普通参数提供。上游不具备的端口不自行添加。
    outputs = (Port("depth", "image.1", "深度图", means=("scale",)), Port("camera", "scene.camera", "相机"),
               Port("points", "scene.points", "点云", made_from=("depth", "camera", "confidence")))
    cost = Cost(gpu=True)
    # 模型额外计算了天空概率图时，在节点上声明 `sky_map = "sky"`（其 worker 在 npz 中使用的数组名），
    # 家族即自动增加「天空遮罩」端口，做法与 `native_points` / `keypoints` / `smpl_body` 相同：
    # 上游输出几种数据就有几个端口，未声明的不增加。
    sky_map: str = ""

    def __init_subclass__(cls, **kw):
        # 该端口定义在家族上：声明了 sky_map 的节点自动增加「天空遮罩」，适配器中只需一行声明
        if cls.sky_map and not any(p.name == "sky" for p in cls.outputs):
            cls.outputs = (*cls.outputs, Port("sky", "image.1", "天空遮罩",
                                              help="模型自己判的天空概率（0–1，越大越像天空）。"
                                                   "它也是深度图有效位的来源：天空那块没有距离可言"))
        super().__init_subclass__(**kw)

    # ---------------------------------------------------------------- 两档共用的步骤

    @classmethod
    def sky_writer(cls, ctx, image) -> ExrWriter | None:
        """官方另外输出天空概率时（Depth Anything 3）：原样输出（而非二值化后的有效位）。

        `None` 表示该节点没有这张图，或没有下游连接（仅在需要时写出）。"""
        if not (cls.sky_map and "sky" in ctx.wanted):
            return None
        return ExrWriter(ctx.outputs["sky"], 1, validity=True, window=window_of(image))

    @classmethod
    def add_sky(cls, sky: ExrWriter | None, ctx, frame: int, d, shape) -> None:
        """该帧的天空概率（有效位全为 True：整张图均有值）。worker 未写出时立即报错，不静默缺少该图。"""
        if sky is None:
            return
        if cls.sky_map not in d:
            raise Invalid(Msg("E-FAMILY-NOSKY", node=ctx.label))
        sky.add(frame, d[cls.sky_map], np.ones(shape, bool))

    @classmethod
    def add_points(cls, ctx, out: dict[str, Packet], raw: RawOutput, image, camera: Packet) -> None:
        """家族的点云输出：声明了 `native_points` 时使用模型自身的三维点，否则由深度 + 相机反投影。

        `camera` 单独传入，是因为不输出相机的节点（`solves_camera=False`）的相机不在 `out` 中。"""
        if "points" in ctx.wanted:
            out["points"] = family_points(ctx, out["depth"], camera, image, confidence=out.get("confidence"),
                                          native=native_points_of(cls, raw))


class PerFrameDepthCamera(DepthCamera):
    """逐帧档：上游逐帧计算，每帧返回各自的深度和内参；相机为位于原点的固定机位。

    基于单张图像的逐帧几何：深度（+ 法线，+ 置信度）及其观察相机。镜头取节点的 Focal Length，否则取模型估计值；
    Params 不是 LensParams 的节点不接受镜头参数（由模型自行估计，如 FaceAnything）。

    原始数据约定：raw/frame_<n>.npz 包含 depth [H,W]（米）、mask [H,W]（有值的像素）、intrinsics [3,3]，可选
    normal [H,W,3]（OpenCV 相机坐标）和 confidence [H,W]（模型自身的置信度，含义见 `confidence`）；raw/result.json
    中模型单位为相对值时 "metric" 为 false（按米的方式乘以 100 写出）。缺少任一帧时失败。

    输出深度（厘米；无效像素在 alpha 中标记）、法线（GL 相机坐标）、置信度，以及深度的观察相机：位于原点的
    固定机位，镜头取所用镜头或模型焦距的中位数。世界位置和点云由「深度转世界位置」/「深度转点云」生成。
    Job.notes：无。"""

    main = "depth"
    streams = True  # 逐帧写出 EXR：每帧写完即为最终字节，可边算边看
    on_node = ("focal_mm", "model")
    # 上游是否输出内参。默认为 True：UniDepth、MoGe、Depth Anything 3 均返回 `intrinsics`。
    # UniK3D 声明为 False：其返回值中没有 intrinsics（只有 confidence / distance / depth / points / rays），
    # 该相机由 worker 将其 `rays` 以最小二乘拟合为针孔模型得到，不属于上游输出。
    # 该相机本就位于原点且静止，不输出也不影响点云数值：家族内部仍用它进行反投影和放置。
    solves_camera: bool = True
    version = 5  # 5：声明了 sky_map 的节点增加一个「天空遮罩」端口
    # 4：点云输出的去飞点改为依照上游实现（深度跳变且法线折角），结果改变，旧缓存作废
    # 3：声明了 native_points 的节点输出模型自身的三维点；点云增加来源说明
    # 2：模型置信度作为独立输出（复用模型的原始结果）

    def __init_subclass__(cls, **kw):
        # 上游不输出内参时不应存在「相机」输出端口（见上方 solves_camera 声明）
        if not cls.solves_camera:
            cls.outputs = tuple(Port(p.name, p.type, p.label, made_from=("depth", "confidence")) if p.name == "points" else p
                                for p in cls.outputs if p.name != "camera")
        super().__init_subclass__(**kw)

    class Params(LensParams):  # 没有相机端口，镜头为两个普通参数
        point_step: int = points_params()["point_step"]
        point_size: float = points_params()["point_size"]

    @classmethod
    def prepare(cls, ctx) -> Job:
        job = Job(ctx.input("image"))
        if not takes_lens(cls):
            return job
        lens = plate_lens(ctx, job.plate)
        return job.with_(extra={"fov_x_deg": sent_fov_x_deg(lens, job.plate)}, lens=lens)

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        from ...data.camera import CameraSamples

        image, lens = job.plate, job.lens
        frames = image.meta["frames"]
        w, h = image.meta["width"], image.meta["height"]
        metric = raw.result().get("metric", True)
        ctx.stage("写出深度图")

        window = window_of(image)  # 发送给 worker 的像素范围（data/windows.py）
        # 深度和相机是家族的核心产物（基于该家族构建的节点也会读回，如 FaceAnything 的点云）；
        # 法线和置信度等附加输出仅在需要时写出。
        maps = {"depth": ExrWriter(ctx.outputs["depth"], 1, validity=True, window=window,
                                   scale="metric" if metric else "relative")}
        if "normal" in ctx.wanted:
            # 法线是模型的计算结果，按原精度写出，不降低位深。
            # 核心自行计算的法线（「深度转法线」「法线空间转换」）写为半精度是 payloads.py 中的另一项取舍，不适用于此处。
            maps["normal"] = ExrWriter(ctx.outputs["normal"], 3, validity=True, value_range=SIGNED, window=window, space="camera")
        sky = cls.sky_writer(ctx, image)
        scores = ConfidenceWriter(ctx, image, cls)
        focal_px = []
        for f, d in raw.frames(ctx, frames):
            mask = d["mask"].astype(bool)
            maps["depth"].add(f, d["depth"] * M_TO_CM, mask)
            if "normal" in maps:
                if "normal" not in d:
                    raise Invalid(Msg("E-FAMILY-NONORMAL", node=ctx.label))
                maps["normal"].add(f, turn_to_camera(d["normal"]), mask)
            cls.add_sky(sky, ctx, f, d, mask.shape)
            scores.add(f, d["confidence"] if "confidence" in d else None)
            focal_px.append(float(d["intrinsics"][0, 0]))

        # 深度的观察相机：一台位于原点、静止、使用该镜头的相机，由本节点自行构造
        # （并非透传：本节点没有相机输入端口）。
        solved = CameraSamples.solved(frames, w, h, lens.focal_px if lens.given else float(np.median(focal_px)),
                                      filmback_mm=lens.filmback_mm, info={"extension": cls.runtime, **lens.info()})
        # 不输出相机的节点（solves_camera=False）仍需该相机进行反投影和放置，只是写入自身的临时目录而不作为输出端口
        if cls.solves_camera:
            cam = solved.write(ctx.outputs["camera"])
        else:  # 不作为输出端口，但家族内部仍需用它反投影和放置：写入本次计算的临时文件夹
            scratch = ctx.work / "_camera"
            scratch.mkdir(parents=True, exist_ok=True)
            cam = solved.write(scratch)
        out = {k: m.packet() for k, m in maps.items()}
        if sky is not None:  # 与深度、法线并列的一张图，输出顺序排在它们之后
            out["sky"] = sky.packet()
        out.update(scores.packet())
        if cls.solves_camera:
            out["camera"] = cam
        cls.add_points(ctx, out, raw, image, cam)
        return out


class WholeShotParams(NodeParams):
    """所有整段节点共用的参数；各节点另外添加其权重及自身的「每段最多帧数」
    （max_frames_param，上限按该模型实测）。"""

    step: int = P(1, label="隔帧", help="每隔几帧算一次，其余帧的相机插值得到（深度图和点云只有算过的那些帧）。长镜头设 2–3 更快、一段能放下更多时间；快速运动的镜头、要每帧深度图时保持 1", ge=1, le=10, group="解算")
    point_step: int = points_params()["point_step"]
    point_size: float = points_params()["point_size"]


class LensWholeShotParams(WholeShotParams, LensParams):
    """可指定焦距的整段方法（MapAnything、MonST3R）：提供 Focal Length 和 Filmback。"""


class WholeShotDepthCamera(DepthCamera):
    """整段档：上游一次处理整段画面，联合解算每帧的相机和深度。

    前馈式多视图重建：由整段画面得到逐帧相机和深度。人物框 / 运动物体遮罩不参与深度和分段拼接
    （模型看到全部像素）。可指定镜头的方法使用 LensWholeShotParams：其 Focal Length 以 focal_px 传给 worker。

    原始数据约定：raw/cameras.npz 包含 frames [F]、K [F,3,3]（像素）、cam_to_world [F,4,4]（OpenCV，世界 = 第一台相机）；
    raw/frame_<n>.npz 包含 depth [H,W]、confidence [H,W]（模型自身的置信度，含义见 `confidence`）、mask [H,W]（保留的
    像素），仅限已重建的帧（隔帧时跳过其余帧）。单位为米或任意单位（此时由节点的「尺度」指定一个单位对应的厘米数）。

    输出相机（逐帧焦距，世界 = 第一台相机，Y 向上；片门取所用镜头；未重建的帧插值得到）、从该相机观察的深度（厘米）
    及置信度，仅限已重建的帧。
    Job.notes："scale"，方法声称为真实距离时为 "metric"，否则为 "relative"（由节点在 prepare 中设置）。"""

    main = "camera"
    on_node = ("step", "max_frames")
    missing_frames = MissingFrames.SKIP
    fact_labels = {"segments": "镜头分段数"}  # worker 将镜头切分的段数（CookContext.fact，convert）
    # 上游实际接受遮罩图时填写（端口标签，如「运动物体遮罩」）：COLMAP 的 `--ImageReader.mask_path`、
    # MonST3R 的 `dynamic_mask_path`。为空表示上游不接受，节点上不显示该端口。
    # 默认没有该端口，由成员声明添加；若默认存在再由成员各自过滤，每个成员都要写过滤代码，新增端口时需多处修改。
    takes_mask: str = ""
    version = 6  # 6：声明了 sky_map 的节点增加一个「天空遮罩」端口
    # 5：点云输出的去飞点改为依照上游实现（深度跳变且法线折角），结果改变，旧缓存作废
    # 4：深度图输出上游原值，不再被「深度边缘 / 运动物体」置为 0（飞点过滤移至「深度转点云」）
    # 3：点云增加来源说明（points_from）
    # 2：模型置信度作为独立输出（复用模型的原始结果）

    def __init_subclass__(cls, **kw):
        # 该端口定义在家族上：声明了 takes_mask 的节点自动增加对应的遮罩输入端口
        if cls.takes_mask and not any(p.name == "mask" for p in cls.inputs):
            cls.inputs = (*cls.inputs, plate_mask_port(cls.takes_mask))
        super().__init_subclass__(**kw)

    @classmethod
    def prepare(cls, ctx) -> Job:
        job = Job(ctx.input("image"), inputs=ctx.input_files("mask"), notes={"scale": "metric"})
        if not takes_lens(cls):
            return job
        lens = plate_lens(ctx, job.plate)
        return job.with_(extra={"focal_px": lens.focal_px}, lens=lens)

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        from ...data.units import opencv_poses_to_usd

        image, lens = job.plate, job.lens
        scale_cm = ctx.params.get("unit_cm", M_TO_CM)
        if raw.path("result.json").exists() and "chunks" in raw.result():  # 分段处理的 worker 会注明分为几段（loops_param）
            ctx.fact("segments", len(raw.result()["chunks"]))
        cams = raw.arrays("cameras.npz")
        solved = [int(f) for f in cams["frames"]]
        poses = opencv_poses_to_usd(np.asarray(cams["cam_to_world"], np.float64), scale_cm)
        K = np.asarray(cams["K"], np.float64)

        ctx.stage("写出深度图")
        depth = ExrWriter(ctx.outputs["depth"], 1, validity=True, window=window_of(image),
                          scale=job.notes["scale"])
        scores = ConfidenceWriter(ctx, image, cls)
        sky = cls.sky_writer(ctx, image)
        for f, d in raw.frames(ctx, image.meta["frames"]):
            # 深度图即上游计算结果，不做加工。有效通道只标记确实无值的像素（非有限值、0 或负值）。
            # `d["mask"]` 不能作为有效通道：它还包含供点云使用的「深度边缘」和「运动物体」两层过滤，
            # `ExrWriter.add` 会把无效像素写为 0，使深度图沿每个物体边缘出现黑带。
            # 此规则影响所有整段重建成员（CUT3R、MonST3R、VGGT、Pi3、MapAnything、LingBot-Map、Depth Anything 3）。
            z = d["depth"].astype(np.float32) * scale_cm
            depth.add(f, z, np.isfinite(z) & (z > 0))
            scores.add(f, d["confidence"] if "confidence" in d else None)
            cls.add_sky(sky, ctx, f, d, z.shape)
        info = {"extension": cls.runtime, "units": "cm" if scale_cm == M_TO_CM else f"{scale_cm} cm per unit", **lens.info()}
        out = {"camera": solved_camera(ctx, image, solved, K[:, 0, 0], poses, info=info, filmback_mm=lens.filmback_mm,
                                       fy_px=K[:, 1, 1], principal_px=K[:, :2, 2]),  # 保留完整内参
               "depth": depth.packet(), **scores.packet(), **({"sky": sky.packet()} if sky is not None else {})}
        cls.add_points(ctx, out, raw, image, out["camera"])
        return out
