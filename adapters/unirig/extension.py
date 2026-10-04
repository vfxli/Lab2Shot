"""UniRig (SIGGRAPH 2026 / TOG, Zhang et al., Tsinghua + Tripo): automatic rigging of a 3D model.

Two stages, both in this one environment:

1. 骨架: a GPT-style transformer reads the mesh (a Michelangelo shape encoder over 65 536 sampled points)
   and writes the skeleton out as a token sequence (its Skeleton Tree Tokenization), so the hierarchy it
   produces is always a valid tree. Autoregressive: the same mesh with another 随机种子 gives another skeleton.
2. 权重: a point transformer (Pointcept PTv3) over the mesh plus bone-point cross attention predicts every
   vertex's weight for every bone of that skeleton.

Upstream drives both stages through `run.py` (a Lightning `predict`) and reads and writes the mesh with Blender
(`bpy`, `src/data/extract.py`, `src/inference/merge.py`). Lab2Shot does neither: the mesh comes out of Lab2Shot's own USD
packet and the skin goes back into it, so the original UVs, normals, subsets and materials survive
(adapters/unirig/worker.py). Nothing in the pinned checkout is modified: the tree `run.py` reads is composed in each
job's own folder, `src` a symlink to the checkout and `configs` a copy with this adapter's overrides (codebase.py),
and this adapter's `runner.py` is upstream's own predict path without the `bpy` import that `run.py` carries for its
command-line mode.
"""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

UNIRIG_URL = "https://github.com/VAST-AI-Research/UniRig.git"
UNIRIG_COMMIT = "6793c6640ff01c8fb389f3993434124bb43d2933"  # latest commit on main when pinned

WEIGHTS_REPO = "VAST-AI/UniRig"
WEIGHTS_REV = "36842e2b5947e9e60f89275b83208c8e74071c63"
SKELETON_CKPT = "skeleton/articulation-xl_quantization_256/model.ckpt"  # 1.4 GB
SKIN_CKPT = "skin/articulation-xl/model.ckpt"  # 4.6 GB

# The skeleton model builds its transformer from facebook/opt-350m's network shape (AutoConfig.from_pretrained in
# src/model/unirig_ar.py); the weights come from the checkpoint, never from OPT. Only config.json is needed, and the
# worker points transformers at this folder, so nothing reaches for the Hub at cook time.
OPT_REV = "08ab08cc4b72ff5593870b5d527cf4230323703c"


class UniRig(Extension):
    name = "unirig"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "UniRig"
    homepage = "https://zjp-shadow.github.io/works/UniRig/"
    source = GitSource(url=UNIRIG_URL, commit=UNIRIG_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/VAST-AI-Research/UniRig/blob/main/LICENSE",
    )
    generative = False
    import_repo = None  # worker.py puts the composed tree (codebase.compose) on sys.path itself, not the checkout
    worker_modules = ("codebase.py", "runner.py")
    env = EnvSpec(
        python="3.11",  # upstream's own version; the only one with prebuilt flash_attn and torch_scatter wheels here
        # The newest pair that has ready-made torch_scatter (data.pyg.org pt28cu128) and flash_attn (v2.8.3
        # cu12torch2.8) wheels, and whose cu128 kernels cover sm_89 (RTX 4090) and sm_120 (RTX 5090).
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
        pickled_checkpoints=True,  # the two model.ckpt are pickled Lightning checkpoints
        imports=("spconv.pytorch", "torch_scatter", "torch_cluster", "flash_attn", "open3d", "trimesh", "fast_simplification"),
    )
    weights = (
        Weight(key="unirig", kind="hf", source=WEIGHTS_REPO, revision=WEIGHTS_REV, dest="",
               files=(SKELETON_CKPT, SKIN_CKPT)),
        hf_file("facebook/opt-350m", OPT_REV, "config.json", key="opt-350m/config.json", dest="opt-350m/config.json"),
    )

    def worker_env(self) -> dict[str, str]:
        cache = self.paths.root / "cache"
        return {
            # not read by worker.py, which composes its tree in each job's own folder (codebase.compose)
            "UNIRIG_CODE_BASE": str(self.paths.root / "codebase"),
            # upstream writes its __pycache__ next to the source file, and the composed tree's `src` is a symlink
            # into the pinned checkout, which must stay unmodified, including .pyc files
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
            "WANDB_MODE": "offline",
            "WANDB_DISABLED": "true",
            "XDG_CACHE_HOME": str(cache),
        }


EXTENSION = UniRig()
