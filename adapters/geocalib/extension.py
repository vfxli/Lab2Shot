"""GeoCalib (ETH Zurich CVG / Microsoft, ECCV 2024): single-image calibration by a network plus a geometric
optimisation: which way is up (gravity: roll and pitch) and the lens (focal, optional radial distortion)."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, downloads

GEOCALIB_URL = "https://github.com/cvg/GeoCalib.git"
GEOCALIB_COMMIT = "97b8968e7798a66bf04fcf791fb535624241bda7"  # main

# The authors' checkpoints (GitHub release v1.0), the files the package itself would fetch through torch.hub at run
# time: fetched here at install instead, and checked.
RELEASE = "https://github.com/cvg/GeoCalib/releases/download/v1.0"
CHECKPOINTS = {
    "pinhole": downloads.GEOCALIB_PINHOLE.sha256,  # ViPE uses it too: pinned in extensions/downloads.py
    "distorted": "13cc505928e3ff4eb26c00bff73861ab2b11b804a546323456cf5462e1f8f447",
}


from .lens import GROUP


class GeoCalib(Extension):
    name = "geocalib"
    lens_groups = (GROUP,)  # its own camera model names -> the core lens formula table (lens.py)
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "GeoCalib"
    summary = "单图标定算法，只从一张画面估计相机内参和重力方向；把几何优化和深度学习结合起来"
    homepage = "https://github.com/cvg/GeoCalib"
    source = GitSource(url=GEOCALIB_URL, commit=GEOCALIB_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="Apache-2.0（代码；权重随仓库发布）",
        url="https://github.com/cvg/GeoCalib/blob/main/LICENSE",
        summary=(
            "代码 Apache-2.0，可商用。两个权重是同一仓库的 GitHub 发布文件，没有另写许可证，按仓库的 Apache-2.0 理解；"
            "训练数据 OpenPano 来自 PolyHaven（CC0）、HDRMAPS 和 Laval 室内 HDR 数据集（该数据集本身只许非商用），"
            "作者没有说明权重的商用是否受训练数据限制"
        ),
    )
    # Upstream: torch, torchvision, opencv-python, kornia (matplotlib only for its plots). Same torch as AnyCalib.
    env = EnvSpec(
        python="3.12",
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
    )
    import_repo = ""  # the pinned repo's geocalib package, imported from repo/ (not pip-installed)
    weights = tuple(
        Weight(
            key=f"geocalib-{name}",
            kind="url",
            source=f"{RELEASE}/geocalib-{name}.tar",
            dest=f"geocalib-{name}.tar",
            sha256=sha256,
            note=f"GeoCalib {'普通镜头' if name == 'pinhole' else '畸变镜头'}权重（116 MB）",
        )
        for name, sha256 in CHECKPOINTS.items()
    )


EXTENSION = GeoCalib()
