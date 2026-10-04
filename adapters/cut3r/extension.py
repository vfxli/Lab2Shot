"""CUT3R (Continuous 3D Perception Model, UC Berkeley / Google DeepMind): a recurrent
network that reads a video frame by frame and keeps a 3D state, giving each frame's
camera, focal length and metric depth, also with people moving in the shot.
Both this worker and MonST3R's use the worker SDK's reconstruction helpers (lab2shot_worker.recon) and this
adapter's DUSt3R model input (dust3r_input.py; MonST3R requires this extension for it)."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

CUT3R_URL = "https://github.com/CUT3R/CUT3R.git"
# main @ 2025-08-27 (last commit; inference code unchanged since the 2025-03 release).
CUT3R_COMMIT = "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf"

# 512 px (long side, any aspect ratio), DPT head, trained on sequences of 4-64 views.
# The authors publish it on Google Drive only (README "Download Checkpoints"); the
# sha256 is of the downloaded file (3 173 761 006 bytes).
CHECKPOINT_ID = "1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD"
CHECKPOINT = "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"


class Cut3r(Extension):
    name = "cut3r"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "CUT3R"
    homepage = "https://cut3r.github.io/"
    source = GitSource(url=CUT3R_URL, commit=CUT3R_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        url="https://github.com/CUT3R/CUT3R/blob/main/LICENSE",
    )
    generative = False
    # Upstream: Python 3.11 + PyTorch for CUDA 12.1. The models are plain PyTorch (the
    # optional cuRoPE kernel is replaced by the same formula in PyTorch, see worker.py),
    # so the same torch as VGGT / Pi3 is used (cu128 wheels include sm_89).
    # Same environment as monst3r.
    env = EnvSpec(
        python="3.12",
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
    )
    worker_modules = ("dust3r_input.py",)
    weights = (
        Weight(
            key="cut3r_512_dpt_4_64",
            kind="url",
            source=f"https://drive.usercontent.google.com/download?id={CHECKPOINT_ID}&export=download&confirm=t",
            dest=CHECKPOINT,
            sha256=CHECKPOINT_SHA256,
        ),
    )


EXTENSION = Cut3r()
