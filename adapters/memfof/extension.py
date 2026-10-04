"""MEMFOF (MSU Graphics & Media Lab, ICCV 2025): memory-efficient multi-frame optical flow at Full HD. Three frames
in, the middle frame's backward and forward flow out in one pass."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

MEMFOF_URL = "https://github.com/msu-video-group/memfof.git"
MEMFOF_COMMIT = "a51de9fc59c6fe20ba08e079372c7b583d58a712"  # dev (the default branch)

# The authors' own weights (not PTLFlow's re-trained ones, which are CC BY-NC-SA). Tartan-T-TSKH: the model the
# authors recommend for real footage (trained on TartanAir, then Things, then Sintel + KITTI + HD1K + Spring).
HF_REPO = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"
HF_REVISION = "6c6c9aa3ad64f93aee8efbc2f7a6e4535814ee96"
MODEL_SHA256 = "5e1388c0d6de309855959cc4d449dbfc57d6cd6e3a8af440365ab30e34367b78"  # model.safetensors, 303 MB


class Memfof(Extension):
    name = "memfof"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "MEMFOF"
    homepage = "https://msu-video-group.github.io/memfof"
    source = GitSource(url=MEMFOF_URL, commit=MEMFOF_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/msu-video-group/memfof/blob/dev/LICENSE",
    )
    generative = False
    import_repo = ""  # the `memfof` package from the pinned repo (its setup only adds huggingface / safetensors)
    env = EnvSpec(
        python="3.12",
        # Pure PyTorch (no compiled ops): the same torch build as other extensions (shared uv cache).
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
    )
    weights = (
        hf_file(HF_REPO, HF_REVISION, "model.safetensors", key="memfof_tskh", sha256=MODEL_SHA256),
        hf_file(HF_REPO, HF_REVISION, "config.json", key="memfof_tskh_config"),
    )


EXTENSION = Memfof()
