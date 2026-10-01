"""WHAM (CVPR 2024, Shin et al.): world-grounded human motion from video. SMPL
bodies in a gravity-aligned (Y-up) world with foot-contact-aware trajectory
refinement; camera rotation from DPVO visual odometry (or locked off, or an
input camera).

Needs the SMPL model file, which the user downloads after registering
(lab2shot.extensions.manual).
"""

from __future__ import annotations

from lab2shot.sdk import (
    downloads,
    CUDA_13_2_TOOLKIT,
    RESEARCH,
    EnvSpec,
    Extension,
    GitSource,
    LicenseInfo,
    Weight,
    hf_file,
    body_model_weight,
)

WHAM_URL = "https://github.com/yohanshin/WHAM.git"
WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"  # 2024-04-18 (last commit)
VITPOSE = downloads.VITPOSE  # WHAM's third-party/ViTPose submodule

# fetch_demo_data.sh, laid out as the code expects relative to its working directory
# (the worker runs in weights/). WHAM's own files come from its Google Drive; the
# hmr2a / ViTPose / DPVO files are the same bytes as GVHMR's and come from the
# pinned Hugging Face mirror the gvhmr extension uses too (Drive often says "quota exceeded"; the installer keeps
# one copy of identical files).
# The node has no 人物框 input: upstream's own DetectionModel (lib/models/preproc/detector.py)
# detects the people with YOLOv8x, so its checkpoint is needed (fetch_demo_data.sh:43 pulls it
# from Drive; the same bytes are in the pinned mirror below).
_DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
DRIVE_FILES = {
    # key: (Drive file id, dest under weights/, sha256, note)
    "wham": ("19qkI-a6xuwob9_RFNSPWf1yWErwVVlks", "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar",
             "91d250d2d298b00f200aa39df36253b55ca434188c2934d8e91e5e0777fb67fd", "WHAM 主网络（demo 默认的 BEDLAM+3DPW 版），503 MB"),
    # joint regressors / mean pose derived from SMPL (redistributed by WHAM's authors)
    "smpl-aux": ("1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ", "_downloads/body_models.tar.gz",
                 "533a8b6b05ba3bc3d35c1dcaec757a4164a669625116a16d30d7642f3b200c8c", "SMPL 关节回归矩阵等辅助文件，1 MB"),
}
MIRROR, MIRROR_REVISION = downloads.GVHMR_MIRROR, downloads.GVHMR_MIRROR_REVISION
MIRRORED = {
    # key: (path in the mirror, dest under weights/, sha256, note)
    "hmr2a": (downloads.HMR2A.filename, "checkpoints/hmr2a.ckpt", downloads.HMR2A.sha256,
              "HMR2.0a 图像特征（4D-Humans，MIT），2.7 GB"),
    "vitpose-h": (downloads.VITPOSE_H.filename, "checkpoints/vitpose-h-multi-coco.pth", downloads.VITPOSE_H.sha256,
                  "ViTPose-H 2D 关键点（Apache-2.0），2.5 GB"),
    "dpvo": ("dpvo/dpvo.pth", "checkpoints/dpvo.pth",
             "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12", "DPVO 视觉里程计（MIT），14 MB"),
    "yolo": (downloads.YOLOV8X.filename, "checkpoints/yolov8x.pt", downloads.YOLOV8X.sha256,
             "YOLOv8x 人物检测（Ultralytics AGPL-3.0），131 MB"),
}

class WHAM(Extension):
    name = "wham"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "WHAM"
    summary = ("把二维关键点序列结合动捕数据和画面特征抬到三维，再用 SLAM 给的相机角速度估计全局轨迹，"
               "并按脚接触修正")
    homepage = "https://wham.is.tue.mpg.de/"
    source = GitSource(url=WHAM_URL, commit=WHAM_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,  # its weights are research only: stricter than 非商用 (nodes/tags.py)
        uses=("SMPL", "AMASS", "BEDLAM", "3DPW"),
        name="MIT（代码）+ SMPL 非商用 + 权重仅限研究",
        url="https://github.com/yohanshin/WHAM/blob/main/LICENSE",
        summary=(
            "仅限研究。WHAM 代码 MIT；但运行必须用 SMPL 人体模型（SMPL_NEUTRAL），需要在 smpl.is.tue.mpg.de 注册后自己下载，"
            "仅限非商用科研，禁止再分发；随 WHAM 下载的关节回归矩阵等辅助文件由 SMPL 派生，同样按 SMPL 许可。"
            "WHAM 权重作者没有单独写许可，训练数据含 AMASS、BEDLAM、3DPW 等仅限研究的数据集，只按研究用途使用。"
            "其余：DPVO 代码和权重 MIT，HMR2.0a（4D-Humans）MIT，ViTPose-H 和 mmcv / mmpose Apache-2.0，"
            "YOLOv8x 权重和 ultralytics 代码 AGPL-3.0（上游自己的人物检测），"
            "smplx 代码（MPI 非商用许可），chumpy MIT，torch-scatter MIT"
        ),
    )
    env = EnvSpec(
        python="3.10",  # chumpy (reads the SMPL pickle) still calls inspect.getargspec, gone in 3.11
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu130",
        cuda_toolkit=CUDA_13_2_TOOLKIT,  # DPVO's CUDA kernels need headers that compile against glibc >= 2.43
        # Old setup.py packages: built with the environment's setuptools<81, without
        # build isolation and without their pins. Pure Python: nothing is compiled here.
        compiled=(
            "mmcv==1.3.9",  # "lite" mmcv without compiled ops, as upstream pins it
            downloads.pip_git("mmpose", VITPOSE),
            downloads.pip_git("chumpy", downloads.CHUMPY),
        ),
        compiled_cuda=False,
        build="build_dpvo.py",  # DPVO from the submodule, with Eigen 3.4.0
        pickled_checkpoints=True,  # mmcv / WHAM checkpoints
    )
    submodules = ("third-party/DPVO", "third-party/ViTPose")
    extra_sources = {"eigen-3.4.0": GitSource(url="https://gitlab.com/libeigen/eigen.git", commit="3147391d946bb4b6c68edd901f2add6ac1f31f8c")}  # tag 3.4.0, what DPVO's setup.py expects
    weights = tuple(
        Weight(key=key, kind="url", source=_DRIVE.format(drive_id), dest=dest, note=note, sha256=sha256)
        for key, (drive_id, dest, sha256, note) in DRIVE_FILES.items()
    ) + tuple(
        hf_file(MIRROR, MIRROR_REVISION, path, key=key, dest=dest, note=note, sha256=sha256)
        for key, (path, dest, sha256, note) in MIRRORED.items()
    ) + (body_model_weight("smpl"),)

    def worker_env(self) -> dict[str, str]:
        return {"PYTHONPATH": str(self.paths.repo)}


EXTENSION = WHAM()
