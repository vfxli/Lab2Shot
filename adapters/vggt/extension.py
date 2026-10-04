"""Meta VGGT: feed-forward multi-view reconstruction (cameras + depth + points of a
whole shot in one network pass), used to cross-check camera solves."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

from .vggt_models import MODELS

VGGT_URL = "https://github.com/facebookresearch/vggt.git"
# main @ 2026-05-18: includes the 2026-05-15 aggregator fix that keeps only the
# layers the heads read (2-3x more frames in the same GPU memory).
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"


class Vggt(Extension):
    name = "vggt"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Meta VGGT"
    homepage = "https://github.com/facebookresearch/vggt"
    source = GitSource(url=VGGT_URL, commit=VGGT_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt",
    )
    generative = False
    import_repo = ""
    worker_modules = ("vggt_models.py",)
    env = EnvSpec(
        python="3.12",
        # Upstream pins torch 2.3.1; the model is plain PyTorch (SDPA attention, no
        # compiled ops) and runs unchanged on 2.10. cu128 wheels include sm_89.
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
    )
    weights = (
        hf_file(
            *MODELS["original"][:3],
            key="VGGT-1B",
            sha256=MODELS["original"][4],
        ),
        Weight(
            key="VGGT-1B-Commercial",
            kind="hf",
            source=MODELS["commercial"][0],
            revision=MODELS["commercial"][1],
            dest="VGGT-1B-Commercial",
            files=("model.safetensors", "LICENSE", "README.md", "config.json"),
            gated=True,
            option=("model", "commercial"),
        ),
    )


EXTENSION = Vggt()
