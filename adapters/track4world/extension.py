"""Track4World (TencentARC, ECCV 2026): feed-forward dense 4D tracking. Every pixel of a reference frame followed through
the shot in 3D (world-centric scene flow), with the cameras and depth of a Depth Anything 3 backbone. Research only:
its licence forbids commercial and production use."""

from __future__ import annotations

from lab2shot.sdk import RESEARCH, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

TRACK4WORLD_URL = "https://github.com/TencentARC/Track4World.git"
TRACK4WORLD_COMMIT = "fbd59ffadf2de9fccba5ea017e13af239075fbe9"

HF_REPO = "TencentARC/Track4World"
HF_REVISION = "93ae34410efd05d1b7abfdb0f749bc93f7655404"
MODEL_SHA256 = "6c589c0f41632a2329920087dcd8cbd6465d4198d574f0d4186eeada78c492c7"  # track4world_da3.pth, 5.5 GB

# The Depth Anything 3 backbone it is built on (its metric branch keeps these weights): the files the Depth Anything 3
# extension pins, stored once for both.
DA3_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
DA3_FILES = {"config.json": "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468",
             "model.safetensors": "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"}

# The authors' fork of utils3d (unproject with use_ray), which Track4World imports. The repository is the package itself
# (no setup.py; upstream clones it next to its code): checked out as vendor/utils3d, vendor/ goes on the worker's path.
UTILS3D = GitSource(url="https://github.com/jiah-cloud/utils3d.git", commit="2072c024c73f7c0f83e0da23eef5f2d9ac575249")

LICENSE_NOTE = "仅限研究，许可证禁止用于生产"


class Track4World(Extension):
    name = "track4world"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Track4World"
    summary = ("从单目视频以全局前馈的方式估计任意两帧之间每个像素的稠密三维场景流；" + LICENSE_NOTE)
    homepage = "https://github.com/TencentARC/Track4World"
    source = GitSource(url=TRACK4WORLD_URL, commit=TRACK4WORLD_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,
        name="Track4World License（仅限学术研究，禁止商用和生产）",
        url="https://github.com/TencentARC/Track4World/blob/main/LICENSE.txt",
        summary=(
            LICENSE_NOTE + "：许可证原文要求只用于学术目的，“在任何情况下都不得用于商业或生产用途”，范围包括推理代码和权重。"
            "用到的 Depth Anything 3 骨干（DA3NESTED-GIANT-LARGE-1.1）权重是 CC BY-NC 4.0 非商用；随仓库的 AllTracker 部分是 MIT。"
            "只能拿来做研究和测试，不能进任何项目的交付"
        ),
    )
    import_repo = ""  # its `track4world` package from the pinned repo
    env = EnvSpec(
        python="3.12",
        # Pure PyTorch (xformers and gsplat are optional upstream and not installed): the same torch build as other
        # extensions (shared uv cache).
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
        pickled_checkpoints=True,  # track4world_da3.pth (pinned, sha256-checked)
    )
    extra_sources = {"vendor/utils3d": UTILS3D}
    weights = (
        hf_file(HF_REPO, HF_REVISION, "track4world_da3.pth", key="track4world_da3", sha256=MODEL_SHA256,
                note="Track4World（Depth Anything 3 骨干，5.5 GB；" + LICENSE_NOTE + "）"),
        *(hf_file(DA3_REPO, DA3_REVISION, name, key=f"da3nested/{name}", dest=f"da3nested-giant-large-1.1/{name}", sha256=sha,
                  note="Depth Anything 3 骨干 DA3NESTED-GIANT-LARGE-1.1（CC BY-NC 4.0，6.8 GB）" if name != "config.json"
                  else "Depth Anything 3 骨干的网络配置")
          for name, sha in DA3_FILES.items()),
    )

    def worker_env(self) -> dict[str, str]:
        return {"LAB2SHOT_VENDOR_DIR": str(self.paths.root / "vendor")}


EXTENSION = Track4World()
