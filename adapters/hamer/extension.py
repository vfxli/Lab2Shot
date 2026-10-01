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

# The repo's third-party/ViTPose submodule at this commit (the installer does not
# fetch submodules). Its package (an mmpose 0.x fork) is built into the environment
# from this commit; the same commit's source archive supplies the model configs.
VITPOSE = downloads.VITPOSE  # WHAM pins the same
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
    # key: (path inside the Space = path under weights/, sha256, note)
    "hamer-ckpt": ("_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
                   "e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53", "HaMeR ViT-H 模型（2.7 GB，仅限研究）"),
    # the tar's copy differs only in the two MANO paths (the worker sets both itself)
    "hamer-config": ("_DATA/hamer_ckpts/model_config.yaml",
                     "2217f6bcf27ae4c074e2a33a3409d822bbfd02931373deedb9f71287adeabe16", "HaMeR 模型配置"),
    "mano-mean-params": ("_DATA/data/mano_mean_params.npz",
                         "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988", "MANO 平均姿态（HaMeR 自带）"),
    "vitpose-huge-wholebody": ("_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth",
                               "b0555e1e2392e6a2be2d9265368f344d70ccbfd656ad480aa5c1de2e604519c9",
                               "ViTPose+-H 全身关键点，用来找手（3.8 GB）"),
}

class HaMeR(Extension):
    name = "hamer"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "HaMeR"
    summary = "从单目画面恢复手的三维形态；HaMeR 是全 Transformer 架构，比以往的做法明显更准更稳"
    homepage = "https://geopavlakos.github.io/hamer/"
    source = GitSource(url=HAMER_URL, commit=HAMER_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,  # its weights are research only: stricter than 非商用 (nodes/tags.py)
        uses=("MANO",),
        name="MIT（代码）+ MANO 非商用 + 权重仅限研究",
        url="https://github.com/geopavlakos/hamer/blob/main/LICENSE.md",
        summary=(
            "仅限研究。HaMeR 代码 MIT；但运行必须用 MANO 手部模型（MANO_RIGHT.pkl），MANO 许可仅限非商用科研/教学/艺术项目，"
            "需要在 mano.is.tue.mpg.de 注册后自己下载，禁止再分发。HaMeR 权重作者没有单独写许可，"
            "训练数据含 FreiHAND、InterHand2.6M（CC-BY-NC）等非商用数据集，只按研究用途使用。"
            "找人用的 ViTDet-H（detectron2）代码和权重 Apache-2.0；"
            "找手用的 ViTPose+-H 全身关键点：代码 Apache-2.0，权重随仓库发布（MAE 预训练，MAE 权重为 CC-BY-NC-4.0）；"
            "依赖 mmcv 1.3.9、mmpose（ViTPose 分支）Apache-2.0，"
            "smplx 代码（MPI 非商用许可，只用来读取 MANO），chumpy MIT。不用 nvdiffrast"
        ),
    )
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
            downloads.pip_git("detectron2", downloads.DETECTRON2),
            "mmcv==1.3.9",  # "lite" mmcv without compiled ops, as upstream pins it
            downloads.pip_git("mmpose", VITPOSE),
            downloads.pip_git("chumpy", downloads.CHUMPY),
        ),
        compiled_cuda=False,  # C++ ops only (detectron2), the rest pure Python: nothing needs the CUDA toolkit
    )
    weights = (
        *(hf_file(SPACE, SPACE_REVISION, path, key=key, dest=path, sha256=sha, note=note)
          for key, (path, sha, note) in DEMO_FILES.items()),
        # 上游自己的人物检测器（demo.py --body_detector vitdet）。demo.py 写的是这个网址，
        # detectron2 会在运行时自己下载；这里在安装时先下好放在本地，worker 直接指到这个文件。
        # 同一个文件 SAM 3D Body 和 TRAM 也在用（登记在 extensions/downloads.py）。
        downloads.VITDET_H.weight(key="vitdet", dest="vitdet/model_final_f05665.pkl", note="人物检测 ViTDet-H（Apache-2.0，2.8 GB）"),
        Weight(key="vitpose-configs", kind="zip", source=downloads.archive_zip(VITPOSE), dest="vitpose",
               sha256=VITPOSE_ZIP_SHA256, note="ViTPose 源码（只用其中的模型配置文件，Apache-2.0）"),
        body_model_weight("mano"),
    )

    def worker_env(self) -> dict[str, str]:
        return {
            "HAMER_VITPOSE_DIR": str(self.paths.weights / VITPOSE_DIR),
            # pyrender is imported by hamer.utils (never used for rendering here)
            "PYOPENGL_PLATFORM": "egl",
        }


EXTENSION = HaMeR()
