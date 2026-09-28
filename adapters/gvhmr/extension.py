"""GVHMR (ZJU 3DV, SIGGRAPH Asia 2024): world-grounded human motion from video
via gravity-view coordinates. SMPL-X bodies in a gravity-aligned (Y-up) world,
camera rotation from upstream's SimpleVO (or locked off, or an input camera).

Needs the SMPL-X model file, which the user downloads after registering
(lab2shot.extensions.manual).
"""

from __future__ import annotations

from lab2shot.sdk import CUDA_13_2_TOOLKIT, NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file, body_model_weight


GVHMR_URL = "https://github.com/zju3dv/GVHMR.git"
GVHMR_COMMIT = "6ec3ca39336c50492c0fae65fba2fb831fc7d866"  # main (parallel SimpleVO)

# Upstream's Google Drive folder (docs/INSTALL.md) hits "quota exceeded" for the big
# files, so they come from a Hugging Face mirror of exactly that folder, pinned and
# sha256-checked (the gvhmr checkpoint is byte-identical to the Drive original).
# Laid out as the code expects (inputs/checkpoints/...): the worker makes weights/
# the project root for these paths.
MIRROR = "camenduru/GVHMR"
MIRROR_REVISION = "21b32d5389e2e59c0737d4c4095bbc0b8c23f66b"
MIRROR_FILES = {
    # key: (path in the mirror, sha256, note). WHAM's download script ships the same
    # hmr2a / ViTPose files (same bytes): the wham extension pins them from this mirror too.
    # With no 人物框 wired, upstream's own YOLOv8x tracker (hmr4d/utils/preproc/tracker.py) finds
    # the people, so its checkpoint is needed.
    "yolo": ("yolo/yolov8x.pt",
             "c4d5a3f000d771762f03fc8b57ebd0aae324aeaefdd6e68492a9c4470f2d1e8b", "YOLOv8x 人物检测和跟踪（Ultralytics AGPL-3.0），131 MB"),
    "gvhmr": ("gvhmr/gvhmr_siga24_release.ckpt",
              "4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91", "GVHMR 主网络（非商用），156 MB"),
    "hmr2a": ("hmr2/epoch=10-step=25000.ckpt",
              "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e", "HMR2.0a 图像特征（4D-Humans，MIT），2.7 GB"),
    "vitpose-h": ("vitpose/vitpose-h-multi-coco.pth",
                  "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc", "ViTPose-H 2D 关键点（Apache-2.0），2.5 GB"),
}


class GVHMR(Extension):
    name = "gvhmr"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "GVHMR"
    summary = "从单目视频恢复落在世界里的人体动作，用重力-视角坐标解决世界坐标系定义不唯一的问题"
    homepage = "https://zju3dv.github.io/gvhmr/"
    source = GitSource(url=GVHMR_URL, commit=GVHMR_COMMIT)
    # pytorch3d v0.7.9, built from source without Pulsar (build_pytorch3d.py). Checked out by the installer like the repo
    # (mirror, retry, pinned commit) and handed to the script as LAB2SHOT_EXTRA_PYTORCH3D — it never fetches on its own
    extra_sources = {"pytorch3d": GitSource(url="https://github.com/facebookresearch/pytorch3d.git", commit="33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba")}
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="GVHMR 非商用许可 + SMPL-X 非商用",
        url="https://github.com/zju3dv/GVHMR/blob/main/LICENSE",
        summary=(
            "非商用。GVHMR 代码和权重：浙江大学许可，只允许教育、科研和非营利用途，基于它的修改必须开源且禁止商用"
            "（商用需联系 xwzhou@zju.edu.cn）。运行必须用 SMPL-X 人体模型（SMPLX_NEUTRAL.npz），需要在 smpl-x.is.tue.mpg.de "
            "注册后自己下载，仅限非商用科研，禁止再分发。其余权重：HMR2.0a（4D-Humans，MIT，训练数据含非商用数据集）、"
            "ViTPose-H（Apache-2.0，多数据集训练）、YOLOv8x（Ultralytics AGPL-3.0，上游自己的人物检测和跟踪）。"
            "依赖 pytorch3d（BSD）、pycolmap（BSD）、smplx 代码（MPI 非商用许可）、ultralytics（AGPL-3.0）"
        ),
    )
    import_repo = ""
    # 上游 requirements.txt 钉的 torch 2.3.0+cu121 只编到 sm_50-90、没打 PTX，在 Blackwell（sm_120）显卡上
    # 一启动就崩；所以 torch 用带 sm_120 的版本，pytorch3d（Meta 没有对应新 torch/CUDA 组合的预编译 wheel）
    # 从源码编译，TORCH_CUDA_ARCH_LIST 取设置「编译目标架构」与下面 env_archs 两边都有的（installer/envbuild.py
    # target_archs，经 lab2shot_worker.build.cuda_build_env 传给编译脚本），与编译这台机器插的卡无关。
    # 装在 .venv-ada-blackwell。
    env_archs = ("sm_89", "sm_120")  # Ada and Blackwell: third_party/gvhmr/.venv-ada-blackwell
    env = EnvSpec(
        python="3.10",
        # torch 2.9.0+cu130: the machine's CUDA ([build] cuda_home, 12.9 here) cannot
        # compile pytorch3d against this glibc ("error: exception specification" in
        # bits/mathcalls.h — the glibc >= 2.43 issue CUDA_13_2_TOOLKIT's docstring names), so the
        # pip CUDA 13.2 toolkit is used instead (as tram/vipe/wham do for their own CUDA kernels),
        # and torch's CUDA major (13) has to match the toolkit's: cu130 rather than cu128
        # (still sm_89 + sm_120; Blackwell support starts at cu128 and cu130 carries it too).
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu130",
        cuda_toolkit=CUDA_13_2_TOOLKIT,
        # Not EnvSpec.compiled: built for two architectures at once, pytorch3d's Pulsar
        # renderer does not link, and only a build script can remove Pulsar from the
        # sources first (see build_pytorch3d.py).
        build="build_pytorch3d.py",
        # 没接「人物框」时 worker 调上游自己的 YOLOv8x 跟踪器（hmr4d/utils/preproc/tracker.py Tracker），
        # 它经 ultralytics 8.2.42 的 `torch.load(file, map_location="cpu")`（ultralytics/nn/tasks.py:775）
        # 加载 yolov8x.pt，没有显式传 weights_only；torch 2.6 起该参数默认 True，会抛
        # 「Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel」。yolov8x.pt 钉死了 revision、
        # 校验过 sha256，所以和 WHAM 一样声明为 pickle 检查点。这一条不进 env_fingerprint，不触发重建环境。
        pickled_checkpoints=True,  # yolov8x.pt（ultralytics 的 torch.load 不传 weights_only）
    )
    weights = tuple(
        hf_file(MIRROR, MIRROR_REVISION, path, key=key, dest=f"inputs/checkpoints/{path}", note=note, sha256=sha256)
        for key, (path, sha256, note) in MIRROR_FILES.items()
    ) + (body_model_weight("smplx"),)


EXTENSION = GVHMR()
