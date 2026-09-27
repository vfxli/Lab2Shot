"""AnyCalib: single-image lens calibration (focal, principal point, distortion)
for many camera models, combined over a few frames of a shot -> lens.json + ST-maps.
"""

from __future__ import annotations


from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

ANYCALIB_URL = "https://github.com/javrtg/AnyCalib.git"
ANYCALIB_COMMIT = "027a8497d893f4b2596f23d6324c05e4b81064ed"  # matches the v1.0.0 weights

# Same files as the GitHub release v1.0.0, from Hugging Face at a pinned revision
# (checked against the LFS sha256 after download).
HF_REVISION = "660ab9b7924ccca12d1e353585594e34f1251341"
CHECKPOINTS = {
    "anycalib_gen": "65d375de971456d0bcc2c8dcdeb711e4f2ac892366b8ae7dd4351868a9d4d940",
    "anycalib_pinhole": "e73b174563bcb90dc9a0e348ee64fbb898901d2a1786623e62fc0e3b176a1a38",
    "anycalib_dist": "89090e4b43e78b3ba16fefdb305f500047b870fb93da4d498ffa935ee27e8ee5",
}


from .lens import GROUP


class AnyCalib(Extension):
    name = "anycalib"
    lens_groups = (GROUP,)  # AnyCalib's own model names -> core formula table (lens.py)
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "AnyCalib"
    summary = "用任选的一种镜头模型，从一张透视 / 编辑过 / 带畸变的画面做相机标定"
    homepage = "https://github.com/javrtg/AnyCalib"
    source = GitSource(url=ANYCALIB_URL, commit=ANYCALIB_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="Apache-2.0（代码和权重）",
        url="https://github.com/javrtg/AnyCalib/blob/main/LICENSE",
        summary=(
            "代码和权重均为 Apache-2.0，可商用（README 与 Hugging Face 模型卡均声明）。"
            "骨干网络为 DINOv2 ViT-L 结构（Apache-2.0），权重已包含在 AnyCalib 检查点里，不另外下载。"
            "训练数据含 Laval 室内 HDR 数据集，作者说明已获其许可以宽松许可证发布权重"
        ),
    )
    env = EnvSpec(
        python="3.12",
        # Upstream only needs "torch"; 2.8.0 cu128 runs on both the 4090 (sm_89) and the 5090 (sm_120).
        torch=("torch==2.8.0",),
        torch_backend="cu128",
    )
    weights = tuple(
        hf_file(
            "javrtg/AnyCalib", HF_REVISION, f"{name}.pt",
            key=name.replace("_", "-"),
            dest=f"{name}.pt",
            note=f"AnyCalib {name}（DINOv2 ViT-L + DPT 解码器，1.28 GB，Apache-2.0）",
            sha256=sha256,
        )
        for name, sha256 in CHECKPOINTS.items()
    )

    def worker_env(self) -> dict[str, str]:
        return {
            "XFORMERS_DISABLED": "1",  # plain PyTorch attention, identical on every machine
        }


EXTENSION = AnyCalib()
