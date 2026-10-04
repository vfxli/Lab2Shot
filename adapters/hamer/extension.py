"""HaMeR (Hand Mesh Recovery): 3D hand meshes (MANO) from single frames.

Upstream pipeline, run per frame: upstream's own ViTDet-H detector finds the people
(demo.py --body_detector vitdet) -> ViTPose+-H whole-body keypoints give each person's
left and right hand box -> HaMeR (ViT-H) regresses MANO parameters and a camera for
each hand crop. MANO itself (MANO_RIGHT.pkl) needs a registration: the user
downloads it by hand (lab2shot.extensions.manual).
"""

from __future__ import annotations

from lab2shot.sdk import RESEARCH, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file, body_model_weight, downloads

HAMER_URL = "https://github.com/geopavlakos/hamer.git"
HAMER_COMMIT = "3a01849f4148352e9260b69bf28b65d1671a4905"  # main (README update; code as of 2024-10)
DETECTRON2 = GitSource("https://github.com/facebookresearch/detectron2.git", "a1ce2f956a1d2212ad672e3c47d53405c2fe4312")
CHUMPY = GitSource("https://github.com/mattloper/chumpy.git", "580566eafc9ac68b2614b64d6f7aaa84eebb70da")
# ViTDet-H, the people detector (SAM 3D Body and TRAM pin the same file)
VITDET_H = downloads.Download(
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl",
    "8601bc52000c8a87960f3db6a9672596c5e06ce33bc30a3b8f96a96efe42ae60")

# The repo's third-party/ViTPose submodule at this commit (the installer does not
# fetch submodules). Its package (an mmpose 0.x fork) is built into the environment
# from this commit; the same commit's source archive supplies the model configs.
VITPOSE = GitSource("https://github.com/ViTAE-Transformer/ViTPose.git", "d5216452796c90c6bc29f5c5ec0bdba94366768a")  # WHAM pins the same
VITPOSE_DIR = f"vitpose/ViTPose-{VITPOSE.commit}"  # inside weights/
VITPOSE_ZIP_SHA256 = "e70a504460c016c032c30943f08644bae5f0d477833ac66e16e0bfdc8781e1be"  # GitHub source archive

# fetch_demo_data.sh downloads one 6 GB tar (hamer_demo_data.tar.gz, Google Drive /
# utexas.edu). The authors' own Hugging Face Space serves the same files one by one
# (identical sha256 to the tar's files), so nothing has to be unpacked:
# pinned Space revision, sha256 = the files' LFS ids.
# The Space also contains MANO_RIGHT.pkl: deliberately NOT downloaded (the MANO
# licence forbids redistribution; the user gets MANO from its own site).
SPACE, SPACE_REVISION = "spaces/geopavlakos/HaMeR", "aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a"
DEMO_FILES = {
    # key: (path inside the Space = path under weights/, sha256); each note: extension.hamer.weight.<key>.note
    "hamer-ckpt": ("_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
                   "e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53"),
    # the tar's copy differs only in the two MANO paths (the worker sets both itself)
    "hamer-config": ("_DATA/hamer_ckpts/model_config.yaml",
                     "2217f6bcf27ae4c074e2a33a3409d822bbfd02931373deedb9f71287adeabe16"),
    "mano-mean-params": ("_DATA/data/mano_mean_params.npz",
                         "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988"),
    "vitpose-huge-wholebody": ("_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth",
                               "b0555e1e2392e6a2be2d9265368f344d70ccbfd656ad480aa5c1de2e604519c9"),
}

class HaMeR(Extension):
    name = "hamer"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "HaMeR"
    homepage = "https://geopavlakos.github.io/hamer/"
    source = GitSource(url=HAMER_URL, commit=HAMER_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,  # its weights are research only: stricter than 非商用 (nodes/tags.py)
        uses=("MANO",),
        url="https://github.com/geopavlakos/hamer/blob/main/LICENSE.md",
    )
    generative = False
    import_repo = ""
    env = EnvSpec(
        python="3.10",  # chumpy (reads the MANO pickle) still calls inspect.getargspec, gone in 3.11
        # Same torch as the sam_3d_body extension (upstream Docker: 2.2 / cu118; the code runs unchanged).
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
        # Old setup.py packages: built with the environment's setuptools<81 (they
        # import pkg_resources), without build isolation and without their pins.
        compiled=(
            # the person detector (ViTDet-H, load_detector); same commit as the sam_3d_body extension.
            # Its own CUDA kernels are unused (ROIAlign / NMS run through torchvision), see compiled_cuda
            downloads.pip_git("detectron2", DETECTRON2),
            "mmcv==1.3.9",  # "lite" mmcv without compiled ops, as upstream pins it
            downloads.pip_git("mmpose", VITPOSE),
            downloads.pip_git("chumpy", CHUMPY),
        ),
        compiled_cuda=False,  # C++ ops only (detectron2), the rest pure Python: nothing needs the CUDA toolkit
    )
    weights = (
        *(hf_file(SPACE, SPACE_REVISION, path, key=key, dest=path, sha256=sha)
          for key, (path, sha) in DEMO_FILES.items()),
        # 上游自己的人物检测器（demo.py --body_detector vitdet）。demo.py 写的是这个网址，
        # detectron2 会在运行时自己下载；这里在安装时先下好放在本地，worker 直接指到这个文件。
        VITDET_H.weight(key="vitdet", dest="vitdet/model_final_f05665.pkl"),
        Weight(key="vitpose-configs", kind="zip", source=downloads.archive_zip(VITPOSE), dest="vitpose",
               sha256=VITPOSE_ZIP_SHA256),
        body_model_weight("mano"),
    )

    def worker_env(self) -> dict[str, str]:
        return {
            "HAMER_VITPOSE_DIR": str(self.paths.weights / VITPOSE_DIR),
            # pyrender is imported by hamer.utils (never used for rendering here)
            "PYOPENGL_PLATFORM": "egl",
        }


EXTENSION = HaMeR()
