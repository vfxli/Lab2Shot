"""AllTracker (Harley et al., ICCV 2025): dense long-range point tracking. The flow from one query frame to every
other frame of the shot, for every pixel, with visibility and confidence."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

ALLTRACKER_URL = "https://github.com/aharley/alltracker.git"
ALLTRACKER_COMMIT = "e7553135e7b361590dbccd10e2b274b024f41cd6"  # master

HF_REPO = "aharley/alltracker"
HF_REVISION = "c8ba31225828cab9ec512d0627f0f826e641a877"
MODEL_SHA256 = "ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d"  # alltracker.pth, 66 MB


class AllTracker(Extension):
    name = "alltracker"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "AllTracker"
    summary = "一个点跟踪模型，比同类更快更准，同时在高分辨率上给出稠密（全像素）的结果"
    homepage = "https://alltracker.github.io/"
    source = GitSource(url=ALLTRACKER_URL, commit=ALLTRACKER_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="MIT",
        url="https://github.com/aharley/alltracker/blob/master/LICENSE",
        summary="代码 MIT，权重（Hugging Face aharley/alltracker，模型卡写明 MIT）同样 MIT：可以商用、修改和再分发，保留版权声明即可",
    )
    import_repo = ""  # AllTracker's `nets` and `utils` packages from the pinned repo
    env = EnvSpec(
        python="3.12",
        # Pure PyTorch: the same torch build as other extensions (shared uv cache).
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
        pickled_checkpoints=True,  # alltracker.pth holds {"model": state_dict, ...} (pinned, sha256-checked)
    )
    weights = (
        hf_file(HF_REPO, HF_REVISION, "alltracker.pth", key="alltracker", sha256=MODEL_SHA256,
                note="AllTracker（MIT，66 MB；Kubric + 多个真实 / 合成数据集训练的完整模型）"),
    )


EXTENSION = AllTracker()
