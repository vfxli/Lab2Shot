"""Pixel3DMM 扩展提供的节点（CC BY-NC 4.0，仅限研究用途）。"""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, 
    NORMALIZE,
    CameraLensParams,
    Cost,
    Licence,
    Measured,
    OptionTrait,
    Param,
    Port,
    WorldHumans,
    camera_normals,
    camera_port,
    curves_packet,
    frame_maps,
    measured_param,
)

# 「精度」三档：迭代次数（逐帧 iters + 联合 global_iters）与提前结束阈值同时变化。不接受任意数值：
# 迭代次数可使单次计算从十几分钟增至一小时，因此仅提供三档经过验证的取值
QUALITY_LABELS = {"fast": "快", "standard": "标准", "fine": "精细"}


class Face(WorldHumans):
    id = "pixel3dmm.face"
    # 引用官方追踪器逐帧保存的内容（Tracker.save_checkpoint）：FLAME 的 exp / shape / eyes / eyelids /
    # jaw / neck / R / t，相机（R_base、t_base、fl、pp）以及 FLAME 计算出的顶点。
    official = Official(
        cite=("third_party/pixel3dmm/repo/src/pixel3dmm/tracking/tracker.py:290-332",
              "third_party/pixel3dmm/repo/scripts/network_inference.py:55-160"),
        takes={"image": "Image.open"},
        gives={"character": "shape", "expressions": "exp", "camera": "cam_params", "normal": "normals", "uv": "uv_map"},
        note="① **官方的整套 FLAME 参数就是「人物」这个口**，没有另立类型、也没有另加口（SMPL / SMPL-X / MANO / FLAME / MHR "
             "这类参数化人体就是「蒙皮 + 权重 + 骨架动画」，装成「蒙皮角色」，不另立数据类型）。"
             "逐项对上（worker.py head_in_camera）：`shape`（300 个，整段一个）变成这张脸的静止网格和静止骨架"
             "（npz rest_vertices / rest_joints），配 FLAME 自己的蒙皮权重（npz skin_weights）；"
             "`R` 和 `t`（头每帧的旋转和位移，连同相机的 R_base / t_base）变成根关节每帧的旋转和位移"
             "（npz local_rotations[:, 0] 和 transl）；`neck` 变成脖子关节（local_rotations[:, 1]）、"
             "`jaw` 变成下巴关节（[:, 2]）、`eyes` 变成左右眼球两个关节（[:, 3:5]）；`exp`（100 个）和 "
             "`eyelids`（2 个）变成「人物」上的 102 条 blendShape（npz blendshapes / blendshape_names / "
             "blendshape_weights），同一份数字另外走「表情曲线」这个口。`joint_transforms` 是上游把这些关节"
             "旋转按骨架层级乘起来的结果，我们这边由同一批 local_rotations 算出来（npz joints），不是另一份数据。"
             "**一个参数都没丢。**"
             "② 「图像」输入口是官方的：`scripts/network_inference.py:121 img` "
             "（还有 tracker.py:586 images）。"
             "③ 「法线图」「UV 坐标图」两个输出口同样是官方的："
             "`scripts/network_inference.py:146-154 output['uv_map'] / output['normals']`。"
             "④ 官方的相机只有 fl / pp（内参）加一台基准位姿 R_base / t_base，整段共用一台。",
    )
    on_node = ("focal_mm", "quality")
    # 适用于面部足够大的特写，每一帧都须有人脸；整段联合解算，最多 1000 帧；
    # 相机为方法自行解出的一台固定相机（Focal Length + 镜头中心），头部相对其运动
    inputs = (Port("image", "image.3", "RGB"), camera_port())
    # 第二阶段每次联合计算 16 帧，少于 16 帧时上游会崩溃，因此在提交前拦截（nodes/expects.py FrameCount）。
    # 1000 帧上限由 tracker.py 固定，worker 启动时即按此拒绝（「一次最多 N 帧」提示的一键修正
    # 是用「FrameHold」选取一帧，不适用于此情形，因此不声明 most_frames）
    min_frames = 16
    outputs = WorldHumans.outputs + (
        Port("expressions", "curves", "表情曲线"),
        # 官方提供的两张屏幕空间预测：法线图转换到相机空间，与其他项目约定一致；
        # UV 图为 FLAME 自身的 UV 展开，与核心「规范坐标转 UV」的输出类型和术语相同
        Port("normal", "image.3", "法线图", means=("space",)),
        Port("uv", "image.2", "UV 坐标图", means=("projection",)),
    )
    runtime = "pixel3dmm"
    # 本方法自行解算相机：上游的 cam_params 包含 Focal Length、主点和整段共用的相机位姿
    # （third_party/pixel3dmm/repo/src/pixel3dmm/tracking/tracker.py 的 fl / pp / R_base / t_base），
    # 属于官方输出，因此保留「相机」输出口（families/humans.py solves_camera）。
    solves_camera = True
    camera_to_worker = None  # 拟合在自身裁切区域内进行，只向 worker 传递镜头的 Focal Length
    default_focal_mm = None  # 未提供时由 Pixel3DMM 自行解算 Focal Length
    # 在 RTX 4090 / 5090 上以 113 帧 772×855 测得。显存峰值出现在「面部分割」步骤
    # （facer / FaRL），与镜头长度和精度档无关，因此为定值；秒/帧按 113 帧的测量折算，
    # 其中包含大量与帧数无关的固定开销（五个进程分别加载模型、torch.compile、整段联合优化）
    cost = Cost(gpu=True, vram_gb=19.8, seconds_per_frame=6.2)
    traits = (
        OptionTrait(Param("quality").one_of("fast"), seconds_per_frame=3.7),
        OptionTrait(Param("quality").one_of("standard"), seconds_per_frame=6.2),
        OptionTrait(Param("quality").one_of("fine"), seconds_per_frame=11.4),
    )
    licence = Licence(note="Pixel3DMM 代码和权重 CC BY-NC 4.0；FLAME 面部模型和 MICA 身份网络同样只限非商用科研。")

    class Params(CameraLensParams):
        quality: Literal["fast", "standard", "fine"] = measured_param(
            "质量", {"fast": Measured("3.7 秒/帧", flat=True), "standard": Measured("6.2 秒/帧", flat=True),
                    "fine": Measured("11.4 秒/帧", flat=True)},
            default="standard", group="拟合", option_labels=QUALITY_LABELS,
            help="拟合迭代多少步：快 = 逐帧 100 步 + 联合 1500 步，标准 = 200 + 5000（官方默认），"
                 "精细 = 400 + 10000。越精细贴得越紧、抖得越少，时间成倍增长，显存不变（峰值在分割那一步）")

    @classmethod
    def prepare(cls, ctx):
        """家族的任务，另附是否接入了相机：接入相机的镜头中心为画面中心，而本方法的镜头中心为裁切区域中心，
        两者相距较远时由 worker 提示。"""
        job = super().prepare(ctx)
        return job.with_(extra={"has_camera": ctx.input("camera") is not None})

    @classmethod
    def convert(cls, ctx, raw, job):
        """世界人体家族的输出，另加 102 条表情曲线和先验网络给出的两张屏幕空间图。

        两张图以裁切尺寸返回。整段使用同一个裁切矩形（见 worker 的 result.json），因此在此贴回画面画布，
        其余区域标记为「没有值」。
        """
        import numpy as np

        image = job.plate
        out = super().convert(ctx, raw, job)
        d = raw.arrays("person_01.npz")
        frames, own = image.meta["frames"], [int(f) for f in d["frames"]]
        weights = d["blendshape_weights"]
        values = np.stack([np.interp(frames, own, weights[:, k]) for k in range(weights.shape[1])], 1)
        out["expressions"] = curves_packet(ctx.outputs["expressions"], frames,
                                           [str(n) for n in d["blendshape_names"]], values, extension=cls.runtime)

        ymin, ymax, xmin, xmax = (int(v) for v in raw.result()["crop"])
        width, height = image.meta["width"], image.meta["height"]

        def onto_plate(read):
            """将裁切尺寸的图贴到画面画布上，并生成对应的「哪里有值」遮罩。"""
            def placed(d):
                got = read(d)
                values, valid = got if isinstance(got, tuple) else (got, None)
                values = np.asarray(values, np.float32)
                canvas = np.zeros((height, width, values.shape[-1]), np.float32)
                canvas[ymin:ymax, xmin:xmax] = values
                mask = np.zeros((height, width), np.float32)
                mask[ymin:ymax, xmin:xmax] = 1.0 if valid is None else np.asarray(valid, np.float32)
                return canvas, mask
            return placed

        normal_kind, normal_read, normal_opts, normal_resample = camera_normals("normal", "valid")
        maps = {
            "normal": (normal_kind, onto_plate(normal_read), normal_opts, normal_resample),
            # FLAME 官方的 UV 展开，并非按几何投影得到（区别于核心「规范坐标转 UV」的三种方式），因此 projection 为「未知」
            "uv": ("image.2", onto_plate(lambda d: (d["uv"], d["valid"])), {"projection": "unknown"}),
        }
        out |= frame_maps(ctx, raw, image, maps, stage="写出法线图和 UV 坐标图")
        return out


NODES = (Face,)
