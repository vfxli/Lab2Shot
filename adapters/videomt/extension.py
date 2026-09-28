"""VidEoMT (Your ViT is Secretly Also a Video Segmentation Model, CVPR 2026): video panoptic segmentation with class and
instance ids that stay the same through the shot."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

# main @ 2026-09-04 (README update after the LVMT / PMT releases; the VidEoMT code is unchanged since the release).
VIDEOMT = GitSource(url="https://github.com/tue-mps/videomt.git", commit="f351ccbfb2096c71bf3bce6241c8eb257c5cffeb")
WEIGHTS_REPO, WEIGHTS_REVISION = "tue-mps/VidEoMT", "8c953a359f03041e5cdb30b385709ddaae91293e"
VIPSEG = "vipseg_vit_large_55.2.pth"


class VidEoMT(Extension):
    name = "videomt"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "VidEoMT"
    summary = "只有编码器的在线视频分割模型，建在普通的 ViT 上，空间和时间的推理都在编码器里完成"
    homepage = "https://github.com/tue-mps/videomt"
    source = VIDEOMT
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="MIT（训练数据仅限非商用研究）",
        url="https://github.com/tue-mps/videomt/blob/main/LICENSE",
        summary=(
            "代码和权重都是 MIT；但这份全景分割权重只在 VIPSeg 数据集上训练，VIPSeg 的许可写明“只用于非商业研究”，"
            "所以按非商用对待，商用前要法务确认。骨干网络 DINOv2 是 Apache-2.0。"
            "仓库里有几个文件带 NVIDIA Source Code License-NC（来自 MinVIS 的训练和评测框架），Lab2Shot 不运行它们，"
            "只用 MIT 的网络部分"
        ),
    )
    # torch 的 cu128 build 带 sm_120（Blackwell）的 kernel（cu126 的没有，也没打 PTX），所以 Ada 和 Blackwell 都能跑；
    # 环境装在 .venv-ada-blackwell，调度只把任务派到这两种架构的卡上
    env_archs = ("sm_89", "sm_120")  # Ada and Blackwell: third_party/videomt/.venv-ada-blackwell
    env = EnvSpec(
        python="3.12",
        # Pure PyTorch: detectron2 (upstream's training framework) is not needed to run the network.
        # 2.8.0+cu128 (not 2.7.x): the version several other extensions in this tree pin, so the
        # install reuses their cached wheels instead of a fresh multi-GB download.
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
    )
    weights = (
        hf_file(WEIGHTS_REPO, WEIGHTS_REVISION, VIPSEG, key="vipseg_vit_large",
                sha256="7b9374bbaf46d25e0cddf796ce3d1852da0d6ecf744b496325969ec6c60322ef",
                note="VidEoMT-L VIPSeg 全景分割（1.3 GB，MIT；训练数据 VIPSeg 仅限非商用研究）"),
    )


EXTENSION = VidEoMT()
