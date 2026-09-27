"""TAPIP3D (Zhang et al., 2025): long-term 3D point tracking in a camera-stabilised 3D feature cloud. It takes the
plate with a depth and a camera per frame (any depth node, any camera node of Lab2Shot) and follows points in that
camera's world and scale, through occlusions."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, CUDA_13_2_TOOLKIT, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

TAPIP3D_URL = "https://github.com/zbw001/TAPIP3D.git"
TAPIP3D_COMMIT = "4cb7e69a1687f67d56ec3e506768f51f2c581b46"  # train / eval code released

HF_REPO = "zbww/tapip3d"
HF_REVISION = "08730a588204f258f7a86530df15b2a6426e5f5b"
MODEL_SHA256 = "3a9514d526559838e6158360af2b857da596d064b06a94fae7fd3b85134b2b1e"  # tapip3d_final.pth, 309 MB


class TAPIP3D(Extension):
    name = "tapip3d"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "TAPIP3D"
    summary = "单目 RGB 和 RGB-D 视频里的长时前馈三维点跟踪，用一团持久的世界坐标特征云抵掉相机自身的运动"
    homepage = "https://tapip3d.github.io/"
    source = GitSource(url=TAPIP3D_URL, commit=TAPIP3D_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="Apache-2.0 + MIT 权重",
        url="https://github.com/zbw001/TAPIP3D/blob/main/LICENSE",
        summary=(
            "代码 Apache-2.0，权重（Hugging Face zbww/tapip3d，模型卡写明 MIT）可以商用。只用仓库里的跟踪模型和它自带的 pointops2"
            "（MIT）近邻算子；它自己从画面估深度和相机的 MegaSaM 流程不装：深度和相机由 Lab2Shot 的节点接进来"
        ),
    )
    import_repo = ""  # its `models`, `utils`, `datasets` and `third_party` packages from the pinned repo
    env = EnvSpec(
        python="3.12",
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
        cuda_toolkit=CUDA_13_2_TOOLKIT,  # pointops2's kernels need headers that compile against glibc >= 2.43
        build="build.py",  # pointops2's CUDA ops from the pinned repo
        pickled_checkpoints=True,  # tapip3d_final.pth holds its config next to the weights (pinned, sha256-checked)
    )
    weights = (
        hf_file(HF_REPO, HF_REVISION, "tapip3d_final.pth", key="tapip3d", sha256=MODEL_SHA256,
                note="TAPIP3D（MIT，309 MB）"),
    )


EXTENSION = TAPIP3D()
