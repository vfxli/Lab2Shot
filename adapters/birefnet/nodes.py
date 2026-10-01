"""BiRefNet 扩展包提供的节点（代码与权重均为 MIT）。"""

from __future__ import annotations

from typing import Literal


from lab2shot.sdk import (rgb_port, Official, MatteNode, WorkerNode, NodeParams, P, Port, fp16_param, Cost, Licence, Job,
                          Measured, MissingFrames, frame_maps, foreground_entry, measured_param, plate_mask_port)


class Matte(MatteNode):
    id = "birefnet.matte"
    # 上游 inference.py：inputs（一批画面）-> scaled_preds（经 sigmoid 的 alpha）
    official = Official(
        cite="third_party/birefnet/repo/inference.py:33-36",
        takes={"image": "inputs"},
        gives={"alpha": "scaled_preds"},
        note="上游只吃画面，没有任何提示（遮罩、框、点）；模型自己判断主体。",
    )
    on_node = ("model",)
    # 逐帧独立计算，没有帧间约束，边缘可能出现轻微的呼吸闪烁；默认 BiRefNet_HR-matting（2048），
    # 头发、运动模糊处有半透明过渡（1024 的 BiRefNet-matting 同样，速度快约 3 倍）
    # 输入「RGB」、输出「Alpha」、SKIP 缺帧处理以及经 matte() 写出，均由抠像家族的 `MatteNode` 实现
    runtime = "birefnet"
    # 显存按「模型」分档实测（measured_param，见 Params.model 旁的表）；这里的 vram_gb 是最轻的一档（1024 模型），
    # 调度器取它和所选模型实测值中的大者（nodes/applies.py setting_vram），所以 1024 模型不会被 2048 / 2304 的数挡住。
    # seconds_per_frame：默认 BiRefNet_HR-matting，4K 素材整段实测 0.46 秒/帧（网络本身 0.44 秒）；1024 模型约 0.15 秒
    cost = Cost(gpu=True, vram_gb=8.5, seconds_per_frame=0.46)
    licence = Licence(note="代码和权重都是 MIT；训练数据里有仅限研究的数据集，严格的商业交付前建议做一次法务确认。")

    class Params(NodeParams):
        # 每个模型的显存都是实测（RTX 4090；整卡峰值减空闲基线，nvidia-smi 每 20 毫秒采样；用 worker 自己的
        # load_model / input_size / Matte，每种单独一个进程；括号里是 torch 保留峰值，只作参考）。
        # worker 的自适应批大小：第一批 1 帧量出每帧显存，再按 min(8 GB, 空闲 − 2 GB) 定批，所以 2048 / 2304 的模型
        # 永远一帧一批，1024 模型最多一批 2 帧。每档取 worker 实际会走到的批大小下、半精度 / 全精度中较大的那个，向上取到 0.1：
        #   BiRefNet-matting  1024×1024，2 帧：半精度 7.60 GB（7.1），全精度 8.41 GB（8.0）              → 8.5
        #   BiRefNet          1024×1024，2 帧：半精度 7.53 GB（7.1），全精度 8.44 GB（8.0）              → 8.5
        #   BiRefNet_HR-matting 2048×2048，1 帧：半精度 15.04 GB（14.6），全精度 15.41 GB（15.0）        → 15.5
        #   BiRefNet_HR       2048×2048，1 帧：半精度 15.03 GB（14.6），全精度 15.41 GB（15.0）          → 15.5
        #   BiRefNet_dynamic  最坏是方形画面 2304×2304，1 帧：半精度 18.27 GB（17.8），全精度 18.64 GB（18.2）→ 18.7
        #     （16:9 的 4K：3840×2160 → 2304×1280 为 11.03 GB，4096×2160 → 2304×1216 为 10.28 / 10.60 GB）
        # 原先一刀切的 11.5 是按面积推算的，比默认模型实际要的还少约 4 GB。
        model: Literal["matting", "general", "hr_matting", "hr", "dynamic"] = measured_param(
            "模型",
            {"matting": Measured(gb=8.5), "general": Measured(gb=8.5), "hr_matting": Measured(gb=15.5),
             "hr": Measured(gb=15.5), "dynamic": Measured(gb=18.7)},
            default="hr_matting", group="抠像",
            # 官方权重名 + 官方模型卡写的训练尺寸（BiRefNet_dynamic 为 256x256 ~ 2304x2304 任意尺寸）
            option_labels={
                "matting": "BiRefNet-matting（1024×1024）",
                "general": "BiRefNet（1024×1024）",
                "hr_matting": "BiRefNet_HR-matting（2048×2048）",
                "hr": "BiRefNet_HR（2048×2048）",
                "dynamic": "BiRefNet_dynamic（256–2304 任意尺寸）",
            },
        )
        # 处理尺寸由模型决定（官方模型卡的训练尺寸）：BiRefNet / BiRefNet-matting 按方形 1024×1024、
        # BiRefNet_HR / BiRefNet_HR-matting 按方形 2048×2048 压扁计算；BiRefNet_dynamic 按原画面分辨率算
        # （官方模型卡说不要缩放），超出训练范围 256–2304 才等比缩进范围。所以没有「处理分辨率」参数。
        fp16: bool = fp16_param("抠像")  # 半精度用 bf16，和上游 config.py 取的一样


class Foreground(WorkerNode):
    """官方 refine_foreground（FB blur fusion，官方致谢 PhotoRoom fast-foreground-estimation）：
    从「图 + Alpha」反解出去掉背景色污染的前景色。不 refine alpha，估计的是前景颜色。"""
    id = "birefnet.foreground"
    finishes = True  # it cleans the colour of a matte made upstream: a matting card's project is the matting's
    # CPU 与 GPU 的偶数核补边相差一个像素，结果版本需区分两种实现，避免混用前景缓存。
    version = 2
    # 官方 image_proc.py:80 refine_foreground，官方两个教程（单图 / 视频）都用它配着抠像展示；
    # worker 调的是它的 CPU 路径 FB_blur_fusion_foreground_estimator_cpu_2（image_proc.py，cv2.blur；不占显卡，见下方 cost）
    official = Official(
        cite="third_party/birefnet/repo/image_proc.py:74-103",
        takes={"image": "image", "mask": "mask"},
        gives={"foreground": "estimated_foreground"},
        # 「未修正前景」这一路是节点自身的输入（原图 × alpha），不是官方函数的输出；它不对应上游任何名字，
        # 所以登记在 ours 而不是 gives。给合成师在 Nuke 里算修正量（前景 − 未修正前景）用
        ours={"plate": "image × mask（上游 refine_foreground 收到的原图，预乘 alpha）"},
        note="两趟均值模糊（第一趟 r，第二趟固定 6）从邻域统计反解 alpha 混合公式 image ≈ a·FG + (1-a)·B，"
             "估计边缘半透明像素本该有的纯前景色。alpha=1 的内部原样不动，只有边缘带被换算。"
             "不是模型，纯后处理，无权重。",
    )
    on_node = ("radius",)
    runtime = "birefnet"
    streams = True  # 每帧独立、写完即最终字节：可边算边看
    missing_frames = MissingFrames.SKIP  # 没有 Alpha 的帧跳过（接线时 SameShot 已检查覆盖）
    # 任何来源的 alpha 都行（BiRefNet、MatAnyone、SAM 3、手工 roto），不只本扩展的抠像
    inputs = (rgb_port(), plate_mask_port("Alpha", optional=False, every_frame=True))
    outputs = (Port("foreground", "image.4", "前景", alpha=True,
                    help="去掉背景色之后的前景色（预乘 alpha 的 RGBA）：头发丝、半透明边缘上原来混进来的"
                         "背景色和绿幕溢色，官方 refine_foreground 反解时已经去掉。在 Nuke 里拿它当前景合成，"
                         "边上不再带着旧背景的颜色；接「序列图输出设置」写成 EXR"),
               Port("plate", "image.4", "未修正前景", alpha=True,
                    help="原图乘 alpha（预乘 RGBA），没有经过边缘解混合，和「前景」走同一条写出和色彩转换。"
                         "两路都写成 EXR 读进 Nuke，「前景」减「未修正前景」（Merge minus）就是修正量，"
                         "只在 alpha 0–1 的边缘带不为 0：自己的 plate × alpha 加上 k × 修正量，k 从 0 到 1 "
                         "控制用多少 AI 的颜色，实心区域一个字节都不碰"))
    # 在 CPU 上算（官方 CPU 版 FB_blur_fusion_foreground_estimator_cpu_2，cv2.blur），不占显卡：
    # r=90 实测（Ryzen 9 9950X3D，只计算法本身）1080p 约 0.095 秒/帧、4K 约 0.45 秒/帧（真实任务里 worker 记的是
    # 0.13 / 0.68 秒/帧：同时在读 PNG、写结果），在「1080p ≤ 0.2 秒、
    # 4K ≤ 0.8 秒就改用 CPU」的线内（GPU 版 1080p 0.020 秒、4K 0.114 秒，为两次均值模糊占一整张卡不值得）。
    # seconds_per_frame 和抠像节点同一个口径：4K 素材真实任务里 worker 记的每帧秒数
    cost = Cost(gpu=False, seconds_per_frame=0.68)
    licence = Licence(note="代码 MIT（同 BiRefNet 本体），纯数学后处理，无权重、无训练数据问题。")

    class Params(NodeParams):
        # 官方默认 90；高分辨率素材边缘带更宽（按像素计），需要更大的半径
        radius: int = P(90, label="模糊半径", unit="px", ge=10, le=400, group="前景")

    @classmethod
    def prepare(cls, ctx) -> Job:
        return Job(ctx.input("image"), inputs=ctx.input_files("mask"))

    @classmethod
    def convert(cls, ctx, raw, job):
        # worker 的 raw/frame_<n>.npz：foreground、plate [H,W,3] 预乘 sRGB、alpha [H,W]，两路都按家族约定写成预乘 RGBA EXR
        # （同一个 foreground_entry：同样的写出、同样的色彩空间标记，下游输出节点对两路做同样的转换）；只写有人要的口
        return frame_maps(ctx, raw, job.plate, {"foreground": foreground_entry(), "plate": foreground_entry("plate")},
                          stage="写出前景")


NODES = (Matte, Foreground)
