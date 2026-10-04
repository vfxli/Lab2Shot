"""Meta MapAnything: feed-forward multi-view metric reconstruction (per-frame
intrinsics, camera poses, depth, confidence) from the frames of one shot."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file, downloads

MAPANYTHING_URL = "https://github.com/facebookresearch/map-anything.git"
MAPANYTHING_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"  # v1.1.4 (+ #165)

# The two checkpoints (same architecture, same config.json; different training data):
# 「模型」 value -> (Hugging Face repo, revision, LFS sha256 of model.safetensors, licence). The one table: the node's
# 「模型」 options, which of them is non-commercial, and the folder and licence the worker is given all come from it.
MODELS = {
    "main": (
        "facebook/map-anything", "a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa",
        "981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0", "CC-BY-NC-4.0",
    ),
    "apache": (
        "facebook/map-anything-apache", "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a",
        "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201", "Apache-2.0",
    ),
}
OPTION_LICENCES = {"model": {k: NONCOMMERCIAL for k, (*_, licence) in MODELS.items() if "-NC-" in licence}}

# The DINOv2 ViT-g backbone is built by torch.hub.load("facebookresearch/dinov2", ...),
# which would fetch the code from GitHub at run time. The worker builds it from this
# pinned copy instead (architecture only: the weights are inside model.safetensors).
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CODE = downloads.Download(f"https://github.com/facebookresearch/dinov2/archive/{DINOV2_COMMIT}.zip",
                                 "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48", kind="zip")


class MapAnything(Extension):
    name = "mapanything"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Meta MapAnything"
    homepage = "https://github.com/facebookresearch/map-anything"
    source = GitSource(url=MAPANYTHING_URL, commit=MAPANYTHING_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/facebookresearch/map-anything/blob/main/LICENSE",
    )
    generative = False
    env = EnvSpec(
        python="3.12",
        # Upstream pins no torch; 2.9.0 / CUDA 13.0, the same pair as adapters/vipe (torchaudio,
        # which UniCeption declares, is last released for torch 2.11).
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu130",
    )
    weights = (
        *(
            hf_file(
                repo, revision, filename,
                key=f"{key}-{filename.split('.')[0]}",
                dest=f"{repo}/{filename}",
                # model.safetensors: the LFS object's sha256; config.json is pinned by the revision
                sha256=sha if filename == "model.safetensors" else "",
            )
            for key, (repo, revision, sha, licence) in MODELS.items()
            for filename in ("config.json", "model.safetensors")
        ),
        DINOV2_CODE.weight(key="dinov2-code", dest="dinov2"),
    )

    def worker_env(self) -> dict[str, str]:
        return {"MAPANYTHING_DINOV2_CODE": str(self.paths.weights / "dinov2" / f"dinov2-{DINOV2_COMMIT}")}


EXTENSION = MapAnything()
