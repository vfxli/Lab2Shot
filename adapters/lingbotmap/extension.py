"""Robbyant LingBot-Map: streaming feed-forward reconstruction (per-frame camera
and depth) for very long shots."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

LINGBOT_URL = "https://github.com/Robbyant/lingbot-map.git"
LINGBOT_COMMIT = "849e690bb086103637e44b1e91878d9d43a8bf0c"  # main

# The official ModelScope copy (listed in upstream's README next to Hugging Face robbyant/lingbot-map
# @ 204754b72bb24f561f8d7e7e1e4e4cd9e809adf9): the same files (identical sha256), and many times faster
# than Hugging Face from mainland China. Pinned per file to the commit that added it.
MODELSCOPE = "https://www.modelscope.cn/models/Robbyant/lingbot-map/resolve"
SKYSEG_REPO = "JianyuanWang/skyseg"
SKYSEG_REVISION = "3ba8c6df1d9ba9ff26f637c7ba9568ac11a9aa7f"


class LingBotMap(Extension):
    name = "lingbotmap"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "LingBot-Map"
    homepage = "https://github.com/Robbyant/lingbot-map"
    source = GitSource(url=LINGBOT_URL, commit=LINGBOT_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/Robbyant/lingbot-map/blob/main/LICENSE.txt",
    )
    generative = False
    env = EnvSpec(
        python="3.12",
        # Upstream's recommended build (README: torch 2.8.0 + CUDA 12.8).
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
    )
    weights = (
        Weight(
            key="lingbot-map-long",
            kind="url",
            source=f"{MODELSCOPE}/d25265c3d6060a598426893ab196cb29a2a1cc7c/lingbot-map-long.pt",
            dest="lingbot-map-long.pt",
            sha256="832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409",
        ),
        Weight(
            key="lingbot-map",
            kind="url",
            source=f"{MODELSCOPE}/0dae3b2f80ce0b0dd739d5b05b7b15f32e55ea78/lingbot-map.pt",
            dest="lingbot-map.pt",
            sha256="ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72",
            option=("model", "balanced"),
        ),
        hf_file(
            SKYSEG_REPO, SKYSEG_REVISION, "skyseg.onnx",
            key="skyseg",
            dest="skyseg.onnx",
            sha256="ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39",
            option=("mask_sky", True),
        ),
    )


EXTENSION = LingBotMap()
