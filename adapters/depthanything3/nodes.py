"""Nodes provided by the Depth Anything 3 extension (code Apache-2.0; Base / Metric weights Apache-2.0, Giant / Large CC-BY-NC-4.0)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, Confidence, Invalid, PerFrameDepthCamera, Msg, P, WholeShotDepthCamera,
                          WholeShotParams, loops_param, max_frames_param, resolution_param, unit_cm_param, Cost,
                          Licence, OptionTrait, Param, Measured)

LICENSE = "代码 Apache-2.0；Base 和 Metric-Large 权重 Apache-2.0 可以商用，Giant 和 Large 1.1 权重 CC-BY-NC-4.0 只能研究用。"
ANYVIEW = {"da3nested-giant-large-1.1": "Giant", "da3-large-1.1": "Large", "da3-base": "Base"}


class Geometry(PerFrameDepthCamera):
    id = "depthanything3.geometry"
    # 上游的 Prediction 里有一张天空概率图（`specs.py:38 sky`）：声明出来，家族就多一个「天空遮罩」口
    sky_map = "sky"
    # 每帧单独估计：不填 Focal Length 时逐帧 Focal Length 跳动很大、深度明显闪动；深度由 Metric-Large 换算成米；
    # Metric 模型必须有 Focal Length（真实 Focal Length 的 AbsRel 0.105，AnyCalib 估的 0.192）
    runtime = "depthanything3"
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # 数值来自 RTX 4090，默认 Metric-Large
    # DA3 的网络 forward(x, extrinsics=None, intrinsics=None)（da3.py:100-107）交出 depth / sky /
    # conf / extrinsics / intrinsics（specs.py:35-45 的 Prediction）。
    official = Official(
        cite="third_party/depthanything3/repo/src/depth_anything_3/model/da3.py:100-155",
        takes={"image": "x"},
        gives={"depth": "depth", "sky": "sky", "camera": "intrinsics"},
        note="x 就是画面（da3.py:114「x: Input images (B, N, 3, H, W)」）。"
             "逐帧模式下我们只用它的内参（intrinsics），不用它预测的外参。上游那两个可选的 extrinsics / "
             "intrinsics 输入只在**两个都给**时进网络（da3.py forward：有 extrinsics 才编相机 token）；"
             "逐帧没有外参，所以「已知 Focal Length」参数（worker 收到的 fov_x_deg）**不进网络**：Giant / "
             "Large 上它顶掉预测的内参、用于随后的公制缩放和深度对齐（worker.py run_nested），Metric-Large "
             "上它定尺度（metric = focal × output / 300）",
    )
    cost = Cost(gpu=True, vram_gb=2.6, seconds_per_frame=0.03)
    licence = Licence(note=LICENSE)
    traits = (
        OptionTrait(Param('model').one_of('da3nested-giant-large-1.1', 'da3-large-1.1'), noncommercial=True),
    )

    class Params(PerFrameDepthCamera.Params):  # 家族的 Params：镜头（没有模型自己的点云：点云间隔 / 点的大小随「点云」口去掉）
        model: Literal["da3metric-large", "da3nested-giant-large-1.1", "da3-large-1.1", "da3-base"] = P(
            "da3metric-large", label="模型", group="几何",
            option_labels={"da3metric-large": "Metric-Large", **ANYVIEW},
            # Metric-Large 自己不估 Focal Length：没有 Focal Length 它算不出真实距离（prepare 会拒绝）。声明出来，网页就在
            # **提交前**把这一档变灰、写清原因（B-COOK-OPTION），不用发到服务器再报 E-DEPTHANYTHING3-NOFOCAL
            # 「已知 Focal Length」是这个节点自己的参数（常驻输入口 `param:focal_mm`，填的和接线的都算它有值）；
            # 深度家族没有「相机」输入口，条件里不要引用它
            option_applies={"da3metric-large": Param("focal_mm").set()},
        )
        resolution: Literal[252, 378, 504] = resolution_param(
            {252: Measured(below=504), 378: Measured(below=504), 504: Measured(gb=12.8)},
            default=504)
        ray_pose: bool = P(False, label="光线求 Focal Length", group="几何")

    @classmethod
    def prepare(cls, ctx):
        job = super().prepare(ctx)
        if ctx.params["model"] == "da3metric-large" and not job.lens.given:
            raise Invalid(Msg("E-DEPTHANYTHING3-NOFOCAL"))
        return job


class Reconstruct(WholeShotDepthCamera):
    id = "depthanything3.reconstruct"
    sky_map = "sky"  # 官方 Prediction 里的天空概率图（specs.py sky）：交出去，不只当有效位用
    # DA3 的 forward 只吃 x / extrinsics / intrinsics（da3.py:100-107），没有任何遮罩，所以不声明 takes_mask。
    # 想只重建画面的一部分，在送进去之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个「RGB」口
    # 7: 没有遮罩输入，有效位只看天空和置信度、运动像素照样保留；版本号进缓存键，结果变了就加一
    version = 7
    # 归在「深度与法线」而不是「三维重建」：上游多视图只交 depth / conf / extrinsics / intrinsics
    # （specs.py:35-45 的 Prediction 没有点图字段），点云是它的导出工具用深度反投影的（utils/export/glb.py:210），
    # 不算原生的场景点云
    # 移动、固定都可，固定机位分段拼接时会漂；深度用 Metric-Large 换算成米；每段最多 110 帧
    # （Giant 124 帧要 19.4 GB）；模型看的是整幅画面，运动物体也在里面
    runtime = "depthanything3"
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # 数值来自 RTX 4090，默认 Large 权重，124 帧一次
    # DA3 的网络 forward(x, extrinsics=None, intrinsics=None)（da3.py:100-107）交出 depth / sky /
    # conf / extrinsics / intrinsics（specs.py:35-45 的 Prediction）。
    official = Official(
        cite="third_party/depthanything3/repo/src/depth_anything_3/model/da3.py:100-155",
        takes={"image": "x"},
        gives={"depth": "depth", "sky": "sky", "camera": "intrinsics"},
        note="「相机」是 intrinsics 加 extrinsics（da3.py:115-116 / specs.py:40-41）。没有遮罩或人物框输入口："
             "DA3 的 forward 只吃 x / extrinsics / intrinsics，遮罩进不了模型；想局部重建，"
             "在送进去之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ "
             "这个节点的「RGB」口。DA3 不出三维点图（Prediction 里只有 depth / sky / conf / extrinsics / "
             "intrinsics），所以这个节点没有「点云」口",
    )
    cost = Cost(gpu=True, vram_gb=11.6, seconds_per_frame=0.056)
    licence = Licence(note=LICENSE)
    traits = (
        OptionTrait(Param('model').one_of('da3nested-giant-large-1.1', 'da3-large-1.1'), noncommercial=True),
    )

    class Params(WholeShotParams):
        model: Literal["da3nested-giant-large-1.1", "da3-large-1.1", "da3-base"] = P(
            "da3-large-1.1", label="模型", group="解算", option_labels=ANYVIEW,
        )
        resolution: Literal[252, 378, 504] = resolution_param(
            {252: Measured(below=504), 378: Measured(below=504), 504: Measured(gb=19.4)}, default=504)
        # 24 GB 显卡、504 分辨率下 Giant 124 帧占 19.4 GB：按最吃显存的 Giant 统一封顶，Large / Base 没有单独测过，沿用这个更保守的上限
        max_frames: Literal[32, 64, 110] = max_frames_param(
            {32: Measured(below=110), 64: Measured(below=110), 110: Measured(gb=19.4)},
            default=110)
        loops: bool = loops_param()
        unit_cm: float = unit_cm_param()


NODES = (Geometry, Reconstruct)
