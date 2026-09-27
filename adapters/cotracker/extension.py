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
    summary = "基于 Transformer 的快速模型，能跟住视频里的任意一个点，把光流的一些好处带进点跟踪"
    homepage = "https://cotracker3.github.io/"
    source = GitSource(url=COTRACKER_URL, commit=COTRACKER_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="CC-BY-NC-4.0（非商用）",
        url="https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md",
        summary=(
            "非商用：代码（LICENSE.md）和 CoTracker3 权重（Hugging Face facebook/cotracker3 模型卡）都是 "
            "CC-BY-NC 4.0，只能用于研究等非商业用途，使用时需署名 Meta。"
            "仓库里少量代码来自 PIPs（MIT）、TAP-Vid 和 LocoTrack（Apache-2.0）。"
            "只依赖 torch / torchvision，没有其他隐藏的非商用依赖"
        ),
    )
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
            note=f"CoTracker3 {mode}（CC-BY-NC 4.0，{size / 1e6:.0f} MB）",
            sha256=sha256,
        )
        for mode, (filename, size, sha256) in CHECKPOINTS.items()
    )


EXTENSION = CoTracker()
