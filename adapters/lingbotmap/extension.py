"""Robbyant LingBot-Map: streaming feed-forward reconstruction (per-frame camera,
depth and point cloud) for very long shots."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

LINGBOT_URL = "https://github.com/Robbyant/lingbot-map.git"
LINGBOT_COMMIT = "849e690bb086103637e44b1e91878d9d43a8bf0c"  # main

# The official ModelScope copy (listed in upstream's README next to Hugging Face robbyant/lingbot-map
# @ 204754b72bb24f561f8d7e7e1e4e4cd9e809adf9): the same files (identical sha256), and many times faster
# than Hugging Face from mainland China. Pinned per file to the commit that added it.
MODELSCOPE = "https://www.modelscope.cn/models/Robbyant/lingbot-map/resolve"
SKYSEG_REPO = "JianyuanWang/skyseg"
SKYSEG_REVISION = "3ba8c6df1d9ba9ff26f637c7ba9568ac11a9aa7f"


class LingBotMap(Extension):
    name = "lingbotmap"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "LingBot-Map"
    summary = "面向流式三维重建的前馈三维基础模型，用几何上下文 Transformer 统一坐标锚定、稠密几何线索和长程漂移校正"
    homepage = "https://github.com/Robbyant/lingbot-map"
    source = GitSource(url=LINGBOT_URL, commit=LINGBOT_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="Apache-2.0（代码和权重）；部分代码文件来自 VGGT（Meta VGGT License）",
        url="https://github.com/Robbyant/lingbot-map/blob/main/LICENSE.txt",
        summary=(
            "代码 Apache-2.0，可商用。权重 lingbot-map-long / lingbot-map（Hugging Face robbyant/lingbot-map）"
            "按 Apache-2.0 发布（模型卡写明 Apache-2.0，没有填 license 字段）；网络从 DINOv2-Large（Apache-2.0）"
            "初始化后自行训练，没有用 VGGT 的非商用权重。注意：仓库里 geometry / pose_enc / rotation / 预测头 / load_fn "
            "等文件带 Meta 版权头，来自 VGGT 代码，原许可是 Meta VGGT License（允许商用，但须遵守其可接受使用政策，"
            "禁止军事、武器、关键基础设施等用途）。天空分割模型 skyseg.onnx 为 MIT。"
            "训练数据包含 ScanNet、Matterport3D、HM3D、Waymo 等研究许可数据集，权重本身按 Apache-2.0 发布"
        ),
    )
    env = EnvSpec(
        python="3.12",
        # Upstream's recommended build (README: torch 2.8.0 + CUDA 12.8).
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
    )
    weights = (
        Weight(
            key="lingbot-map-long",
            kind="url",
            source=f"{MODELSCOPE}/d25265c3d6060a598426893ab196cb29a2a1cc7c/lingbot-map-long.pt",
            dest="lingbot-map-long.pt",
            sha256="832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409",
            note="LingBot-Map long：长镜头 / 大场景更稳，默认（Apache-2.0，4.6 GB）",
        ),
        Weight(
            key="lingbot-map",
            kind="url",
            source=f"{MODELSCOPE}/0dae3b2f80ce0b0dd739d5b05b7b15f32e55ea78/lingbot-map.pt",
            dest="lingbot-map.pt",
            sha256="ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72",
            note="LingBot-Map 平衡版：论文和评测用的权重（Apache-2.0，4.6 GB）",
            option=("model", "balanced"),
        ),
        hf_file(
            SKYSEG_REPO, SKYSEG_REVISION, "skyseg.onnx",
            key="skyseg",
            dest="skyseg.onnx",
            sha256="ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39",
            note="天空分割 ONNX，去掉天空的点（MIT，176 MB）",
            option=("mask_sky", True),
        ),
    )


EXTENSION = LingBotMap()
