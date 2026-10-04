"""Pi3 (π³, Shanghai AI Lab): feed-forward multi-view reconstruction (cameras + depth +
points of a whole shot in one network pass), used to cross-check camera solves.
Same environment as the VGGT extension (its own copy of the pins and requirements); the worker uses the SDK driver
(lab2shot_worker.feedforward)."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

from .pi3_models import MODELS

PI3_URL = "https://github.com/yyfz/Pi3.git"
# main: Pi3 + Pi3X (conv head, metric scale, depth-normal edge filter).
PI3_COMMIT = "9fa3ddb3f8d53041f8b2738df404f62223bbaa7b"


class Pi3(Extension):
    name = "pi3"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Pi3 (π³)"
    homepage = "https://github.com/yyfz/Pi3"
    source = GitSource(url=PI3_URL, commit=PI3_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        url="https://github.com/yyfz/Pi3#-license",
    )
    generative = False
    # Same environment as VGGT, pins and requirements.txt copied (both are plain PyTorch;
    # upstream pins torch 2.5.1; pi3's plyfile / gradio extras are not needed).
    import_repo = ""
    worker_modules = ("pi3_models.py",)
    env = EnvSpec(python="3.12", torch=("torch==2.10.0", "torchvision==0.25.0"), torch_backend="cu128")
    weights = tuple(
        hf_file(
            repo, rev, filename,
            key=repo.split("/")[1],
            sha256=sha,
        )
        for repo, rev, filename, _size, sha in MODELS.values()
    )


EXTENSION = Pi3()
