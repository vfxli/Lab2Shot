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
VITPOSE = GitSource("https://github.com/ViTAE-Transformer/ViTPose.git", "d5216452796c90c6bc29f5c5ec0bdba94366768a")  # WHAM's third-party/ViTPose submodule
CHUMPY = GitSource("https://github.com/mattloper/chumpy.git", "580566eafc9ac68b2614b64d6f7aaa84eebb70da")

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
    # key: (Drive file id, dest under weights/, sha256); each one's note: extension.wham.weight.<key>.note
    "wham": ("19qkI-a6xuwob9_RFNSPWf1yWErwVVlks", "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar",
             "91d250d2d298b00f200aa39df36253b55ca434188c2934d8e91e5e0777fb67fd"),
    # joint regressors / mean pose derived from SMPL (redistributed by WHAM's authors)
    "smpl-aux": ("1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ", "_downloads/body_models.tar.gz",
                 "533a8b6b05ba3bc3d35c1dcaec757a4164a669625116a16d30d7642f3b200c8c"),
}
MIRROR, MIRROR_REVISION = "camenduru/GVHMR", "21b32d5389e2e59c0737d4c4095bbc0b8c23f66b"
MIRRORED = {
    # key: (path in the mirror, dest under weights/, sha256); each one's note: extension.wham.weight.<key>.note
    "hmr2a": ("hmr2/epoch=10-step=25000.ckpt", "checkpoints/hmr2a.ckpt",
              "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e"),
    "vitpose-h": ("vitpose/vitpose-h-multi-coco.pth", "checkpoints/vitpose-h-multi-coco.pth",
                  "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"),
    "dpvo": ("dpvo/dpvo.pth", "checkpoints/dpvo.pth",
             "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12"),
    "yolo": ("yolo/yolov8x.pt", "checkpoints/yolov8x.pt",
             "c4d5a3f000d771762f03fc8b57ebd0aae324aeaefdd6e68492a9c4470f2d1e8b"),
}

class WHAM(Extension):
    name = "wham"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "WHAM"
    homepage = "https://wham.is.tue.mpg.de/"
    source = GitSource(url=WHAM_URL, commit=WHAM_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,  # its weights are research only: stricter than non-commercial (nodes/tags.py)
        uses=("SMPL", "AMASS", "BEDLAM", "3DPW"),
        url="https://github.com/yohanshin/WHAM/blob/main/LICENSE",
    )
    generative = False
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
            downloads.pip_git("chumpy", CHUMPY),
        ),
        compiled_cuda=False,
        build="build_dpvo.py",  # DPVO from the submodule, with Eigen 3.4.0
        pickled_checkpoints=True,  # mmcv / WHAM checkpoints
    )
    submodules = ("third-party/DPVO", "third-party/ViTPose")
    extra_sources = {"eigen-3.4.0": GitSource(url="https://gitlab.com/libeigen/eigen.git", commit="3147391d946bb4b6c68edd901f2add6ac1f31f8c")}  # tag 3.4.0, what DPVO's setup.py expects
    weights = tuple(
        Weight(key=key, kind="url", source=_DRIVE.format(drive_id), dest=dest, sha256=sha256)
        for key, (drive_id, dest, sha256) in DRIVE_FILES.items()
    ) + tuple(
        hf_file(MIRROR, MIRROR_REVISION, path, key=key, dest=dest, sha256=sha256)
        for key, (path, dest, sha256) in MIRRORED.items()
    ) + (body_model_weight("smpl"),)

    def worker_env(self) -> dict[str, str]:
        return {"PYTHONPATH": str(self.paths.repo)}


EXTENSION = WHAM()
