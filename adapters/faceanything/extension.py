"""FaceAnything: per-pixel face depth + canonical facial coordinates from plates.

Research release only: code and weights are CC BY-NC 4.0 (non-commercial).
One node, faceanything.solve (adapters/faceanything/nodes.py).
"""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

FACEANYTHING_URL = "https://github.com/kocasariumut/FaceAnything.git"
FACEANYTHING_COMMIT = "68e9a82524d327eb2e35daecb4f0442ab1f2bc09"  # ECCV 2026 release

# Robust Video Matting (the background matte FaceAnything's own pipeline uses,
# normally fetched through torch.hub at run time): code at a pinned commit and
# the ResNet-50 weights from its v1.0.0 release, so the worker runs offline.
RVM_COMMIT = "53d74c6826735f01f4406b5ca9075eee27bec094"
RVM_DIR = f"rvm/RobustVideoMatting-{RVM_COMMIT}"  # inside weights/, as extracted from GitHub's archive
RVM_CHECKPOINT = "torch/hub/checkpoints/rvm_resnet50.pth"  # where torch.hub itself would cache it


class FaceAnything(Extension):
    name = "faceanything"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "FaceAnything"
    summary = "统一的前馈模型，从任意画面序列做高保真的 4D 面部重建和稠密跟踪；做法是逐像素预测规范面部坐标"
    homepage = "https://kocasariumut.github.io/FaceAnything/"
    source = GitSource(url=FACEANYTHING_URL, commit=FACEANYTHING_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        uses=("FLAME",),
        name="CC BY-NC 4.0（非商用）",
        url="https://github.com/kocasariumut/FaceAnything/blob/main/LICENSE",
        summary=(
            "非商用：代码和权重均为 CC BY-NC 4.0，不能用于商业制作（含商业项目的镜头）。"
            "模型主干 Depth-Anything-3 DA3-GIANT 同为 CC BY-NC 4.0，训练数据来自 FLAME 模型拟合"
            "（FLAME 许可证同样只允许非商业科研），商用授权需分别联系作者。"
            "背景遮罩用 Robust Video Matting（代码和权重为 GPL-3.0：可商用，但修改后分发需同样以 GPL-3.0 开源）。"
            "注意：原仓库 pyproject.toml 写着 Apache-2.0，以仓库 LICENSE 文件和 Hugging Face 权重页的 CC BY-NC 4.0 为准"
        ),
    )
    import_repo = "src"  # imported as `faceanything` / `depth_anything_3` from repo/src
    # Blackwell（sm_120）：xformers 0.0.35 没有自带的 CUDA 核心，走 torch 的 scaled_dot_product_attention，
    # torch 支持的架构它都支持（更早版本的预编译核心没有 sm_120，见 requirements.txt）；它要求 torch>=2.10。
    env_archs = ("sm_89", "sm_120")  # Ada and Blackwell: third_party/faceanything/.venv-ada-blackwell
    env = EnvSpec(
        # torch 2.10.0（CUDA 12.8）满足 xformers 0.0.35 的 torch>=2.10；torchvision 0.25.0 是与它配对的版本。
        python="3.11",
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
    )
    weights = (
        Weight(
            key="faceanything",
            kind="hf",
            source="UmutKocasari/FaceAnything",
            revision="e8ef3d6fd1d5049b801cfced29421b8611aaabef",
            dest="faceanything",
            files=("checkpoint.pt",),
            note="FaceAnything 模型（DA3-GIANT 主干 + 规范坐标头，约 15 GB；CC BY-NC 4.0）",
        ),
        Weight(
            key="rvm-code",
            sha256="e688d5add3b7329c867f451f9f1a33805fd7d7869b9383c49d1e0be242225b67",
            kind="zip",
            source=f"https://github.com/PeterL1n/RobustVideoMatting/archive/{RVM_COMMIT}.zip",
            dest="rvm",
            note="Robust Video Matting 网络代码（GPL-3.0）",
        ),
        Weight(
            key="rvm-resnet50",
            sha256="c191a807251164c073dce5fa408e7a816070d539b882b2a3150330a9fec112ce",
            kind="url",
            source="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth",
            dest=RVM_CHECKPOINT,
            note="Robust Video Matting ResNet-50 背景遮罩权重（GPL-3.0）",
        ),
    )

    def worker_env(self) -> dict[str, str]:
        weights = self.paths.weights
        return {
            "FACEANYTHING_RVM_DIR": str(weights / RVM_DIR),
            "HF_HOME": str(weights / "hf"),
        }


EXTENSION = FaceAnything()
