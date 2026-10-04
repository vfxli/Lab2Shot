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
    homepage = "https://tapip3d.github.io/"
    source = GitSource(url=TAPIP3D_URL, commit=TAPIP3D_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/zbw001/TAPIP3D/blob/main/LICENSE",
    )
    generative = False
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
        hf_file(HF_REPO, HF_REVISION, "tapip3d_final.pth", key="tapip3d", sha256=MODEL_SHA256),
    )


EXTENSION = TAPIP3D()
