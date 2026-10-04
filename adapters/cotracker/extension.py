"""Meta CoTracker3: long-range 2D point tracking of many points jointly
(offline: whole windows at once, both directions; online: sliding windows)."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file


COTRACKER_URL = "https://github.com/facebookresearch/co-tracker.git"
COTRACKER_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"  # main (CoTracker3 release + fixes)

HF_REPO = "facebook/cotracker3"
HF_REVISION = "bf55ea50d4390e1820a267f131cd6587240fb2c5"
# mode -> (file, bytes, LFS sha256). The "scaled" models (trained on Kubric and then
# on ~100k real videos with pseudo-labels) are what upstream's torch.hub entries load.
CHECKPOINTS = {
    "offline": ("scaled_offline.pth", 101890938, "2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"),
    "online": ("scaled_online.pth", 101695610, "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"),
}


class CoTracker(Extension):
    name = "cotracker"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "CoTracker3"
    homepage = "https://cotracker3.github.io/"
    source = GitSource(url=COTRACKER_URL, commit=COTRACKER_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        url="https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md",
    )
    generative = False
    env = EnvSpec(
        python="3.12",
        # Pure PyTorch (no compiled ops); same build as tapnext (shared uv cache).
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
    )
    weights = tuple(
        hf_file(
            HF_REPO, HF_REVISION, filename,
            key=f"cotracker3_{mode}",
            dest=filename,
            sha256=sha256,
        )
        for mode, (filename, size, sha256) in CHECKPOINTS.items()
    )


EXTENSION = CoTracker()
