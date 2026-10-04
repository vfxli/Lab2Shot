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
    downloads,
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
# its point renderer "pulsar" does not link with a CUDA 13 compiler and is dropped from the build.
PYTORCH3D = GitSource("https://github.com/facebookresearch/pytorch3d.git", "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba")  # tag v0.7.9, the same as gvhmr
CHUMPY = GitSource("https://github.com/mattloper/chumpy.git", "580566eafc9ac68b2614b64d6f7aaa84eebb70da")

_DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

# key: (url, dest under weights/, sha256); each one's note: extension.pixel3dmm.weight.<key>.note
FILES = {
    "uv": (_DRIVE.format("1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY"), "uv.ckpt",
           "dff9d73feec47914b704759f57ebffb8c58d2aef550b426013fe31eae21707b8"),
    "normals": (_DRIVE.format("1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD"), "normals.ckpt",
                "e856799d55db54c7537c8ee3c5a4938c13cc0b24082ce7e4e7f35f0d0f0e28da"),
    "mica": (_DRIVE.format("1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt"), "mica/mica.tar",
             "4542a467d9e8f7521474a1d00eac89552bebef0b331b72bf7fbd6f065ff64d7b"),
    "pipnet-wflw": (_DRIVE.format("1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C"),
                    "pipnet/epoch59.pth",
                    "44002daaf3187e6e4b99e0fd9a52f60ffa05e06ec5df3721ace950fd2f2cc120"),
    # facer / FaRL and the torchvision backbone go straight into the offline torch hub cache
    # (TORCH_HOME = weights/torch), where facer's download_jit and torchvision look for them.
    "farl-celebm": ("https://github.com/FacePerceiver/facer/releases/download/models-v1/"
                    "face_parsing.farl.celebm.main_ema_181500_jit.pt",
                    "torch/hub/checkpoints/face_parsing.farl.celebm.main_ema_181500_jit.pt",
                    "bbc1f0e9f68c80eb83a0b23f33850d1e10f2ec1eda96884112d111c2c1f15c79"),
    "retinaface-mobilenet": ("https://github.com/elliottzheng/face-detection/releases/download/0.0.1/"
                             "mobilenet0.25_Final.pth", "torch/hub/checkpoints/mobilenet0.25_Final.pth",
                             "2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220"),
    # PIPNet builds its landmark network on torchvision's ImageNet ResNet-18 (pretrained=True) before loading
    # its own checkpoint over it: the file torchvision asks for (ResNet18_Weights.IMAGENET1K_V1) must be there
    "resnet18-imagenet": ("https://download.pytorch.org/models/resnet18-f37072fd.pth",
                          "torch/hub/checkpoints/resnet18-f37072fd.pth",
                          "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
}
# insightface's antelopev2 (MICA's face detector) unzips into the environment's own HOME
ANTELOPE = (_DRIVE.format("16PWKI_RjjbE4_kqpElG-YFqe8FpXjads"), "insightface/models/antelopev2",
            "7353a5fdca5a90e11d2792e0236032b2fe42adc1ea23eaef5cf8c8b57e7e9393")


class Pixel3DMM(Extension):
    name = "pixel3dmm"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Pixel3DMM"
    homepage = "https://simongiebenhain.github.io/pixel3dmm/"
    source = GitSource(url=P3DMM_URL, commit=P3DMM_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        uses=("FLAME",),
        url="https://github.com/SimonGiebenhain/pixel3dmm/blob/main/LICENSE",
    )
    generative = False
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
            downloads.pip_git("chumpy", CHUMPY),
        ),
        build="build_p3dmm.py",
        build_files=("codebase.py",),
        pickled_checkpoints=True,  # uv.ckpt / normals.ckpt / mica.tar are pickled Lightning checkpoints
    )
    weights = tuple(
        Weight(key=key, kind="url", source=url, dest=dest, sha256=sha)
        for key, (url, dest, sha) in FILES.items()
    ) + (
        Weight(key="antelopev2", kind="zip", source=ANTELOPE[0], dest=ANTELOPE[1], sha256=ANTELOPE[2]),
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
            # file — inside the pinned checkouts, which are never written to, not even a .pyc
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",  # upstream imports pyplot; there is no display
            "WANDB_MODE": "offline",  # it imports wandb but never calls init
            "WANDB_DISABLED": "true",
            "OPENCV_IO_ENABLE_OPENEXR": "1",  # its tracker sets this at import
        }


EXTENSION = Pixel3DMM()
