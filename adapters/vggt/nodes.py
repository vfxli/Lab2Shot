"""Nodes provided by the VGGT extension (code: VGGT License; weights CC-BY-NC-4.0 or VGGT License)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, Confidence, P, WholeShotDepthCamera, WholeShotParams, loops_param, max_frames_param,
                          resolution_param, unit_cm_param, Cost, Licence, OptionTrait, Param, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "vggt.reconstruct"
    # 官方的输入等于解算器的输入：VGGT.forward 只吃 images。
    # 家族的默认口里还带着「运动物体遮罩」「人物框」，那是给真的吃遮罩的 MonST3R 留的
    # （上游 dynamic_mask_path），这里按官方的口筛掉。想只重建画面的一部分，在送进去之前把其余部分涂黑：
    # 人物检测 → 人物框转遮罩 → 图像相乘 → 这个「图像」口，图上一眼看得见
    native_points = "points"  # worker 交出 VGGT 自己的 world_points（放在每帧的相机空间里）
    # 结果变了就加一，让旧缓存作废。7：不再抹掉运动像素，raw 里多了官方的 points
    version = 7
    # docs.md：有视差的镜头和静止机位都可用（固定机位位移只有场景深度的 0.3%）；尺度是自己归一化的任意单位；
    # 模型看的是整幅画面，运动物体也在里面；每段帧数安全上限 130（按竖画面测得）
    runtime = "vggt"
    # 官方的输入等于解算器的输入、输出等于输出：
    # VGGT.forward(images, query_points=None) 交出 pose_enc / depth / depth_conf / world_points /
    # world_points_conf（+ 给了 query_points 时的 track / vis / conf），vggt.py:29-52。
    official = Official(
        cite="third_party/vggt/repo/vggt/models/vggt.py:29-52",
        takes={"image": "images"},
        gives={"depth": "depth", "camera": "pose_enc", "points": "world_points"},
        note="「点云」是官方点头出的世界点图 world_points（vggt.py:45）：原来 worker 把这个头整个关掉"
             "（enable_point=False），官方算得出的一份结果直接丢了，现在补上。"
             "worker 把它放回每一帧的相机空间再交出来（无损，只是换坐标系），由家族按相机摆回世界。"
             "「运动物体遮罩」「人物框」两个输入口是我们自己加的：上游 forward 只有 images 和 query_points"
             "两个入参（vggt.py:29），那两张图一个字节都没进模型，只在算完之后把运动像素从有效位和分段拼接里"
             "去掉。已删除，想局部重建改走「人物检测 → 人物框转遮罩 → 图像相乘」。"
             "还没有的口：track / vis（vggt.py:50-51），它们要 query_points，这个节点没有那个输入",
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # vram_gb: RTX 4090 上测得（docs.md），默认原版权重
    cost = Cost(gpu=True, vram_gb=13.0, seconds_per_frame=0.2)
    licence = Licence(note="代码是 VGGT License（可商用，禁止军事用途）；原版权重 CC-BY-NC-4.0 只能研究用，商用版权重（需申请）可以商用。")
    traits = (
        OptionTrait(Param('model').one_of('original'), noncommercial=True),
    )

    class Params(WholeShotParams):
        model: Literal["original", "commercial"] = P(
            "original", label="模型", group="解算",
            option_labels={"original": "原版", "commercial": "商用版"},
            help="原版：论文发布的权重，只能研究用；商用版：Meta 另外训练的可商用权重，要先在 Hugging Face 申请，通过后重新安装扩展包",
        )
        resolution: Literal[280, 392, 518] = resolution_param(
            {280: Measured("比实测的一档省", below=518), 392: Measured("比实测的一档省", below=518), 518: Measured("训练尺寸：1080×1920 横幅 150 帧分两段 45 秒", gb=20.0)}, default=518, note="；518 是模型的训练尺寸，最稳")
        # 一次性把整段看完（不是流式），24G 显卡上：竖幅约 130 帧、横幅约 230 帧都在 20 GB 内；按竖幅的更紧上限统一封顶
        max_frames: Literal[32, 64, 130] = max_frames_param(
            {32: Measured("比实测的一档省", below=130), 64: Measured("比实测的一档省", below=130), 130: Measured("竖幅实测（横幅显存更少）", gb=20.0)},
            default=130)
        loops: bool = loops_param()
        unit_cm: float = unit_cm_param("VGGT 的尺度要手动缩放：先按米（100）放，和 ViPE 或实测距离对比后再调")

    @classmethod
    def prepare(cls, ctx):
        return super().prepare(ctx).with_(notes={"scale": "relative"})  # the model's units are arbitrary


NODES = (Reconstruct,)
