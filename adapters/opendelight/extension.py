"""OpenDelight: remove the lighting from face photos / frames -> 基础色 (base colour, upstream says albedo) + face matte.

Research release: code GPL-3.0, weights without a license (trained on FaceOLAT,
academic use only, plus private data) -> non-commercial. The worker runs as its
own process in its own environment, so the GPL does not reach Lab2Shot.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from lab2shot.sdk import RESEARCH, EnvSpec, Extension, GitSource, InstallError, LicenseInfo, Msg, Weight

OPENDELIGHT_URL = "https://github.com/yxuhan/OpenDelight.git"
OPENDELIGHT_COMMIT = "431ad736371dcfbf2ce29cfa84b04512df4ea107"  # SIGGRAPH 2026 release

# ibug face detector / landmarks (MIT). Their setup.py only works as an editable
# install, so the worker imports them straight from these pinned source archives.
IBUG_FD_COMMIT = "db2a4e8eae8c9c53385ff0773e9db08f03cf21ad"
IBUG_FA_COMMIT = "9cf5494e443f26d567972f3f50f6212d65b76c01"
# RetinaFace ResNet-50 is a git-lfs object in face_detection (the archive only has the
# pointer): fetched from GitHub's LFS media URL and checked against the pointer's oid.
RETINAFACE_R50 = "ibug/Resnet50_Final.pth"
RETINAFACE_R50_SHA256 = "6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d"

# Upstream's Google Drive archive (doc/TEST.md): opendelight/{base_delight_network,unet_enhancer}.pth
OPENDELIGHT_TAR = "opendelight.tar"
OPENDELIGHT_DIR = "opendelight"
OPENDELIGHT_FILES = ("base_delight_network.pth", "unet_enhancer.pth")

DAVID_ONNX = "david/foreground-segmentation-model-vitl16_384.onnx"
FARL_LAPA = "facer/face_parsing.farl.lapa.main_ema_136500_jit191.pt"


class OpenDelight(Extension):
    name = "opendelight"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "OpenDelight"
    summary = "完全开源、面向面部外观采集的高性能去光照先验"
    homepage = "https://yxuhan.github.io/OpenDelight/"
    source = GitSource(url=OPENDELIGHT_URL, commit=OPENDELIGHT_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,
        name="GPL-3.0 代码 · 权重非商用",
        url="https://github.com/yxuhan/OpenDelight/blob/main/LICENSE",
        summary=(
            "代码 GPL-3.0；权重没有声明许可证，训练数据为 FaceOLAT（仅限学术研究）和作者的私有数据，"
            "只能用于研究和评估，作者授予商用许可前不能用于商业制作（联系 hanyx22@mails.tsinghua.edu.cn）。"
            "辅助模型：DAViD 抠像（MIT）、ibug 面部检测/关键点（MIT）、FaRL 面部分割（MIT，但训练数据 LaPa 仅限非商用）；"
            "MAE 初始化权重（CC BY-NC 4.0）推理时用不到，不下载。"
            "以独立进程运行，GPL 不影响 Lab2Shot 本身"
        ),
    )
    # torch 本身支持 sm_120（Blackwell）；onnxruntime-gpu 1.22.0 的预编译 CUDA provider 只到 sm_90，
    # 所以 requirements.txt 里的 onnxruntime-gpu 是更新的版本。环境装在 .venv-ada-blackwell
    # （见 Extension.env_archs），不碰旧的 .venv。
    env_archs = ("sm_89", "sm_120")  # Ada and Blackwell: third_party/opendelight/.venv-ada-blackwell
    env = EnvSpec(
        python="3.11",
        # Upstream pins torch 2.3.1 cu121; 2.8.0 cu128 runs the same code and brings the
        # cuDNN 9 that onnxruntime-gpu needs. No compiled extensions.
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
    )
    weights = (
        Weight(
            key="opendelight",
            sha256="a0f0690e0ad27cdd65f73b5f14abeac4dc536577178cb9ac97217094a8bc50bb",
            kind="url",
            source="https://drive.usercontent.google.com/download?id=1bCIKOGNlKcGgObg5AeErUHkRTMuv0HHZ&export=download&confirm=t",
            dest=OPENDELIGHT_TAR,
            note="OpenDelight网络 + UNet 增强网络（无许可证声明，按非商用对待；安装后处理时解包）",
        ),
        Weight(
            key="david-foreground",
            sha256="ff003d3c0dff61a19af4d3721716c4e7eeba3830e70e2b6abc53d6ecde8683bf",
            kind="url",
            source="https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/foreground-segmentation-model-vitl16_384.onnx",
            dest=DAVID_ONNX,
            note="DAViD 软前景抠像 ViT-L ONNX（MIT）",
        ),
        Weight(
            key="farl-lapa-448",
            sha256="f5a874906795ef89fadd7cf3b5b218ed8550fa9dbb383b7c0f95726c3a352914",
            kind="url",
            source="https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt",
            dest=FARL_LAPA,
            note="FaRL 面部分割 LaPa 448（facer，MIT；训练数据 LaPa 仅限非商用）",
        ),
        Weight(
            key="ibug-face-detection",
            sha256="957c45e8ecc443f828b4a12ae1142b09cbdb2457e090c64061f46637e5031384",
            kind="zip",
            source=f"https://github.com/hhj1897/face_detection/archive/{IBUG_FD_COMMIT}.zip",
            dest="ibug/face_detection",
            note="ibug RetinaFace 面部检测代码（MIT）",
        ),
        Weight(
            key="ibug-retinaface-r50",
            kind="url",
            source=(
                f"https://media.githubusercontent.com/media/hhj1897/face_detection/{IBUG_FD_COMMIT}"
                "/ibug/face_detection/retina_face/weights/Resnet50_Final.pth"
            ),
            dest=RETINAFACE_R50,
            note="ibug RetinaFace ResNet-50 权重（git-lfs 文件，MIT）",
            sha256=RETINAFACE_R50_SHA256,  # the real file, not a git-lfs pointer or an error page
        ),
        Weight(
            key="ibug-face-alignment",
            sha256="5ab91f5faa927763a9d19eb074dc17683a6b82c79a1e595b86b9f8ab2ac7d4ed",
            kind="zip",
            source=f"https://github.com/hhj1897/face_alignment/archive/{IBUG_FA_COMMIT}.zip",
            dest="ibug/face_alignment",
            note="ibug FAN 68 点面部关键点代码和权重（MIT）",
        ),
    )

    def post_install(self, run, paths) -> None:
        weights = paths.weights
        # Unpack the Google Drive tar (the installer only unpacks zip archives).
        out = weights / OPENDELIGHT_DIR
        with tarfile.open(weights / OPENDELIGHT_TAR) as tar:
            members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
            for name in OPENDELIGHT_FILES:
                member = members.get(name)
                if member is None:
                    raise InstallError(Msg("E-OPENDELIGHT-TARMISSING", archive=OPENDELIGHT_TAR, name=name))
                target = out / name
                if target.exists() and target.stat().st_size == member.size:
                    continue
                out.mkdir(parents=True, exist_ok=True)
                part = target.with_name(name + ".part")
                with tar.extractfile(member) as src, part.open("wb") as dst:
                    while chunk := src.read(1 << 22):
                        dst.write(chunk)
                part.replace(target)


EXTENSION = OpenDelight()
