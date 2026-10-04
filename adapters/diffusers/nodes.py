"""Nodes provided by the Diffusers extension. First model: Qwen-Image 2.1 (Apache-2.0 code, Qwen Research License
weights: research / evaluation use only, commercial use needs a separate licence from Qwen).

One node, one pipeline, three ways of running it (chosen by the mode parameter): text-to-image, single-image
editing and multi-reference composition.

Every node needs the generative capability on top of its licence (the extension declares generative = True): an
account the administrator has not given it to does not see these nodes at all (nodes/tags.py may).
"""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (DiffusionImage, DiffusionImageParams, Official, Port, Cost, Licence,
                          EITHER, Measured, Param, measured_param)

# 官方默认的生成分辨率按用法不同：文生图示例是原生 2K（2048）；编辑与多参考的示例不传 output_resolution，
# 用管线默认 1024（pipeline_qwenimage21.py:528）
OFFICIAL_RESOLUTION = {"text_to_image": 2048, "edit_image": 1024, "reference_images": 1024}

# 上游 QwenImage21Pipeline.__call__ 的真实签名（pinned diffusers 检出）：三种用法一条管线——文生图不传 image、
# 单图编辑传一张、多参考图传列表（官方 README：至多 10 张）。所有签名行（505-529）与调用路径上的尺寸推导
# （598-627）都在引文里。
SIGNATURE = ("third_party/diffusers/repo/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py:505-529",
             "third_party/diffusers/repo/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py:598-627")
TAKES = {"prompt": "prompt", "negative": "negative_prompt", "steps": "num_inference_steps",
         "true_cfg": "true_cfg_scale", "resolution": "output_resolution"}


class QwenImage21(DiffusionImage):
    """What the Qwen-Image 2.1 node shares: the licence, the cost and the official contract of one pipeline."""

    runtime = "diffusers"
    # 一条 QwenImage21Pipeline 承载三种用法；显存与耗时按 docs.md 的实测记录（RTX 5090，官方
    # enable_model_cpu_offload——全量 bf16 32.4GB 装不进 32GB 卡，offload 是两卡上的唯一路径；offload 峰值以
    # transformer 权重为主，与卡无关，4090 同样适用）
    cost = Cost(gpu=True, vram_gb=19.0, measured_on="RTX 5090 32 GB (cpu offload)",
                whole=True,
                ram_gb=40)
    licence = Licence(note=True)
    official = Official(
        cite=SIGNATURE,
        takes=TAKES,
        gives={"image": "images"},
    )

    class Params(DiffusionImageParams):
        # 同家族的「生成分辨率」，另随「模式」取官方默认（derive）：换模式时一并设为该用法的官方值
        resolution: Literal[512, 768, 1024, 1536, 2048] = measured_param(
            {512: Measured(below=2048), 768: Measured(below=2048), 1024: Measured(below=2048),
             1536: Measured(below=2048), 2048: Measured(gb=19.0)},
            default=2048, group="generation", words="family.image_generation.resolution", derived_from=("mode",))

    @classmethod
    def derive(cls, params: dict) -> dict:
        """「模式」改了：生成分辨率取该用法的官方默认（文生图 2048，编辑 / 多参考 1024）。"""
        return {"resolution": OFFICIAL_RESOLUTION.get(params.get("mode"), 2048)}


class Generate(QwenImage21):
    """QwenImage generate: one pipeline, one node, three ways by mode (text-to-image with no picture, image editing
    with one image, multi-reference with an image plus a references list, at most 10 upstream). The output is always
    one RGBA still (the model's native 4-channel VAE; a transparent background asked for the official way)."""

    id = "diffusers.generate"
    version = 2  # 2：多参考时输出比例跟主体（显式传主体尺寸）
    official = Official(
        cite=SIGNATURE,
        takes={"image": "image", "references": "image", **TAKES},
        gives={"image": "images"},
    )
    # 一图进一图出：序列在提交前拒绝并提示插「FrameHold」（光照探针同款）；通道跟随（RGBA 可编辑）
    inputs = (
        Port("image", "image", optional=True, data=False, alpha=True,
             applies=Param("mode").one_of("edit_image", "reference_images")),
        Port("references", "image[]", optional=True, data=EITHER,
             applies=Param("mode").one_of("reference_images")),
    )
    most_frames = 1


NODES = (Generate,)
