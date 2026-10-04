"""MatAnyone 2 (S-Lab, NTU): temporally stable video matting of people, guided by
a mask on one frame, propagated through the shot with a memory of past frames."""

from __future__ import annotations


from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

MATANYONE2_URL = "https://github.com/pq-yang/MatAnyone2.git"
MATANYONE2_COMMIT = "0079197acd6d16a741f71558809c06c586c579e0"  # main

# The checkpoint the official inference script downloads (GitHub release v1.0.0).
CHECKPOINT = "matanyone2.pth"
CHECKPOINT_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
CHECKPOINT_SHA256 = "5e9821e4087231427376b437c85bb6e072b41e582314f06fd524f75bc4af5914"


class MatAnyone(Extension):
    name = "matanyone"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "MatAnyone 2"
    homepage = "https://github.com/pq-yang/MatAnyone2"
    source = GitSource(url=MATANYONE2_URL, commit=MATANYONE2_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        url="https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE.txt",
    )
    generative = False
    import_repo = ""
    env = EnvSpec(
        python="3.12",
        # Upstream's uv.lock: torch 2.10.0 from the cu128 index (includes sm_89). No compiled ops.
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
    )
    weights = (
        Weight(
            key="matanyone2",
            kind="url",
            source=CHECKPOINT_URL,
            dest=CHECKPOINT,
            sha256=CHECKPOINT_SHA256,
        ),
    )


EXTENSION = MatAnyone()
