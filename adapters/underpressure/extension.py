"""UnderPressure (InterDigital R&D France, SCA 2022): a small 1D convolutional network that estimates the vertical
ground reaction forces (vGRFs) under both feet from motion alone, derives reliable foot-contact labels from them, and
cleans footskate with an optimisation-based IK that keeps the contacts consistent with those forces.

Nothing is downloaded: the trained network is `pretrained.tar` inside the pinned repository (4 MB), and the reference
skeleton the motion is retargeted onto comes from the repository's own `footskate_samples/0.pt`. The environment is
torch alone — the whole project is numpy-free torch and the Python standard library (panda3d is only for its viewer,
which Lab2Shot does not use).
"""

from __future__ import annotations

from lab2shot.sdk import EnvSpec, Extension, GitSource, LicenseInfo, RESEARCH

UNDERPRESSURE_URL = "https://github.com/InterDigitalInc/UnderPressure.git"
UNDERPRESSURE_COMMIT = "e7ab73e7466444f262ec1c55f2f26afe51168796"  # 2023-05-14, the repository's latest


class UnderPressure(Extension):
    name = "underpressure"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "UnderPressure"
    homepage = "https://github.com/InterDigitalInc/UnderPressure"
    source = GitSource(url=UNDERPRESSURE_URL, commit=UNDERPRESSURE_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,
        url="https://github.com/InterDigitalInc/UnderPressure/blob/main/LICENCE.txt",
    )
    generative = False
    import_repo = ""  # the repository's modules (anim, data, models, footskate, util) are imported from its own folder
    env = EnvSpec(
        python="3.11",
        torch=("torch==2.8.0",),
        torch_backend="cu128",  # RTX 4090 and RTX 5090 (sm_120) both run this build
        imports=("torch",),
    )
    weights = ()  # the trained network ships inside the repository (repo/pretrained.tar, 4 MB)


EXTENSION = UnderPressure()
