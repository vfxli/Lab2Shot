"""Pixel3DMM (ICLR 2026, Giebenhain et al., TUM): a FLAME face tracker driven by two
screen-space priors.

Per frame a ViT predicts a surface-normal map and a canonical-face (UV) map; the
tracker then fits FLAME to the whole shot in two stages — frame by frame
(`iters`), then all frames jointly (`global_iters`) with one camera for the shot
(`global_camera`), an optimised focal length and temporal smoothness on
expression, jaw, neck, head rotation and translation.

Preprocessing is three more repositories, pinned here instead of upstream's
install_preprocessing_pipeline.sh (which clones over SSH and copies its
replacement files into them): PIPNet (face box + 98 landmarks), MICA (identity
from one arcface embedding) and facer / FaRL (face parsing). None of the four
checkouts is modified: the composed tree upstream's env_paths expects is built
from symlinks by build_p3dmm.py.

Needs the FLAME face model, which the user downloads after registering
(lab2shot.extensions.manual).
"""

from __future__ import annotations

from lab2shot.sdk import (
    CUDA_13_2_TOOLKIT,
    NONCOMMERCIAL,
    EnvSpec,
    Extension,
    GitSource,
    LicenseInfo,
    Weight,
    body_model_weight,
)

P3DMM_URL = "https://github.com/SimonGiebenhain/pixel3dmm.git"
P3DMM_COMMIT = "fcd1fa973c7715b02a8948dfc679dff53cf85924"  # latest (README)

# The three preprocessing repositories, at the commits this extension was built against.
FACER = GitSource(url="https://github.com/FacePerceiver/facer.git",
                  commit="ddd35c76ff840174b8a5403ad1c1255e37b8782b")
MICA = GitSource(url="https://github.com/Zielon/MICA.git",
                 commit="af22e7a5810d474bc28a1433db533723d6bd2b07")
PIPNET = GitSource(url="https://github.com/jhb86253817/PIPNet.git",
                   commit="b9eab58816437403a34aa5bc3adeafe5081fd36b")
# pytorch3d is compiled from source by build_p3dmm.py (knn_points, load_obj, Meshes), not pip-installed:
# its point renderer "pulsar" no longer links with a CUDA 13 compiler and is dropped from the build.
PYTORCH3D = GitSource(url="https://github.com/facebookresearch/pytorch3d.git",
                      commit="33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba")  # tag v0.7.9

_DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

# key: (url, dest under weights/, sha256, note)
FILES = {
    "uv": (_DRIVE.format("1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY"), "uv.ckpt",
           "dff9d73feec47914b704759f57ebffb8c58d2aef550b426013fe31eae21707b8",
           "Pixel3DMM 规范面部坐标（UV）预测网络，2.1 GB（作者 Google Drive，CC BY-NC 4.0）"),
    "normals": (_DRIVE.format("1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD"), "normals.ckpt",
                "e856799d55db54c7537c8ee3c5a4938c13cc0b24082ce7e4e7f35f0d0f0e28da",
                "Pixel3DMM 法线预测网络，1.4 GB（作者 Google Drive，CC BY-NC 4.0）"),
    "mica": (_DRIVE.format("1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt"), "mica/mica.tar",
             "4542a467d9e8f7521474a1d00eac89552bebef0b331b72bf7fbd6f065ff64d7b", "MICA 身份网络（马普所，仅限非商用科研），479 MB"),
    "pipnet-wflw": (_DRIVE.format("1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C"),
                    "pipnet/epoch59.pth",
                    "44002daaf3187e6e4b99e0fd9a52f60ffa05e06ec5df3721ace950fd2f2cc120", "PIPNet WFLW 98 点关键点（ResNet-18，MIT），47 MB"),
    # facer / FaRL and the torchvision backbone go straight into the offline torch hub cache
    # (TORCH_HOME = weights/torch), where facer's download_jit and torchvision look for them.
    "farl-celebm": ("https://github.com/FacePerceiver/facer/releases/download/models-v1/"
                    "face_parsing.farl.celebm.main_ema_181500_jit.pt",
                    "torch/hub/checkpoints/face_parsing.farl.celebm.main_ema_181500_jit.pt",
                    "bbc1f0e9f68c80eb83a0b23f33850d1e10f2ec1eda96884112d111c2c1f15c79",
                    "FaRL CelebAMask-HQ 面部分割（微软，MIT）"),
    "retinaface-mobilenet": ("https://github.com/elliottzheng/face-detection/releases/download/0.0.1/"
                             "mobilenet0.25_Final.pth", "torch/hub/checkpoints/mobilenet0.25_Final.pth",
                             "2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220",
                             "RetinaFace MobileNet-0.25 人脸检测（facer 用，MIT）"),
    # PIPNet builds its landmark network on torchvision's ImageNet ResNet-18 (pretrained=True) before loading
    # its own checkpoint over it: the file torchvision asks for (ResNet18_Weights.IMAGENET1K_V1) must be there
    "resnet18-imagenet": ("https://download.pytorch.org/models/resnet18-f37072fd.pth",
                          "torch/hub/checkpoints/resnet18-f37072fd.pth",
                          "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
                          "ImageNet ResNet-18（torchvision，BSD-3-Clause），PIPNet 建网络用"),
}
# insightface's antelopev2 (MICA's face detector) unzips into the environment's own HOME
ANTELOPE = (_DRIVE.format("16PWKI_RjjbE4_kqpElG-YFqe8FpXjads"), "insightface/models/antelopev2",
            "7353a5fdca5a90e11d2792e0236032b2fe42adc1ea23eaef5cf8c8b57e7e9393",
            "InsightFace antelopev2 人脸检测 / arcface（MICA 用，非商用研究），344 MB")


class Pixel3DMM(Extension):
    name = "pixel3dmm"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Pixel3DMM"
    summary = ("一组高度泛化的视觉 transformer，逐像素预测几何线索，用来约束三维可变形面部模型（3DMM）的优化，"
               "从单张画面解出三维人脸")
    homepage = "https://simongiebenhain.github.io/pixel3dmm/"
    source = GitSource(url=P3DMM_URL, commit=P3DMM_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="CC BY-NC 4.0（代码和权重）+ FLAME 非商用 + MICA 非商用",
        url="https://github.com/SimonGiebenhain/pixel3dmm/blob/main/LICENSE",
        summary=(
            "非商用。Pixel3DMM 代码和两个预测网络权重是 CC BY-NC 4.0：只能用于研究和评估，不能用于商业制作。"
            "运行必须用 FLAME 面部模型（FLAME 2020 generic_model.pkl，可选 FLAME 2023），"
            "需要在 flame.is.tue.mpg.de 注册后自己下载，仅限非商用科研、禁止再分发。"
            "身份先验用马普所 MICA（专有代码，只许有许可的非商用使用；权重同样非商用），"
            "它的人脸检测用 InsightFace antelopev2（仅限非商用研究）。"
            "光栅化用 NVIDIA nvdiffrast（NVIDIA Source Code License，非商用研究）。"
            "其余：PIPNet 关键点 MIT，facer / FaRL 面部分割 MIT，pytorch3d BSD-3-Clause，chumpy MIT"
        ),
    )
    extra_sources = {"facer": FACER, "MICA": MICA, "PIPNet": PIPNET, "pytorch3d": PYTORCH3D}
    import_repo = None  # the worker imports from the composed code base, not from repo/ (worker_env PYTHONPATH)
    worker_modules = ("codebase.py", "steps.py")
    env = EnvSpec(
        python="3.10",  # chumpy (reads the FLAME pickle) still calls inspect.getargspec, gone in 3.11
        # Same torch as TRAM: the newest that has sm_120 (RTX 5090) kernels, and on this machine only the
        # pip CUDA 13.2 headers compile against glibc >= 2.43 (pytorch3d and nvdiffrast both compile).
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu130",
        cuda_toolkit=CUDA_13_2_TOOLKIT,
        compiled=(
            "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae",
            "chumpy @ git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da",
        ),
        build="build_p3dmm.py",
        build_files=("codebase.py",),
        pickled_checkpoints=True,  # uv.ckpt / normals.ckpt / mica.tar are pickled Lightning checkpoints
    )
    weights = tuple(
        Weight(key=key, kind="url", source=url, dest=dest, sha256=sha, note=note)
        for key, (url, dest, sha, note) in FILES.items()
    ) + (
        Weight(key="antelopev2", kind="zip", source=ANTELOPE[0], dest=ANTELOPE[1], sha256=ANTELOPE[2], note=ANTELOPE[3]),
        body_model_weight("flame"),
    )

    def worker_env(self) -> dict[str, str]:
        root, cache = self.paths.root, self.paths.root / "cache"
        return {
            # the tree upstream's env_paths expects, built from symlinks by build_p3dmm.py
            "PIXEL3DMM_CODE_BASE": str(root / "codebase"),
            "PYTHONPATH": str(root / "codebase" / "src"),
            # its own home: insightface (MICA's detector) only ever looks in ~/.insightface, and
            # nothing this worker runs may write into the person's home directory
            "HOME": str(cache / "home"),
            "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),  # nothing should compile at cook time; if it tries, not in $HOME
            # the composed tree symlinks each source file, so Python would write its __pycache__ next to the real
            # file — inside the pinned checkouts. 原始仓库永远不改，连 .pyc 也不留
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",  # upstream imports pyplot; there is no display
            "WANDB_MODE": "offline",  # it imports wandb but never calls init
            "WANDB_DISABLED": "true",
            "OPENCV_IO_ENABLE_OPENEXR": "1",  # its tracker sets this at import
        }


EXTENSION = Pixel3DMM()
