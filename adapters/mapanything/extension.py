"""Meta MapAnything: feed-forward multi-view metric reconstruction (per-frame
intrinsics, camera poses, depth, confidence) from the frames of one shot."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

MAPANYTHING_URL = "https://github.com/facebookresearch/map-anything.git"
MAPANYTHING_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"  # v1.1.4 (+ #165)

# The two checkpoints (same architecture, same config.json; different training data):
# weights key -> (Hugging Face repo, revision, LFS sha256 of model.safetensors, licence)
MODELS = {
    "apache": (
        "facebook/map-anything-apache", "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a",
        "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201", "Apache-2.0",
    ),
    "main": (
        "facebook/map-anything", "a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa",
        "981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0", "CC-BY-NC-4.0",
    ),
}

# The DINOv2 ViT-g backbone is built by torch.hub.load("facebookresearch/dinov2", ...),
# which would fetch the code from GitHub at run time. The worker builds it from this
# pinned copy instead (architecture only: the weights are inside model.safetensors).
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_ZIP_SHA256 = "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48"


class MapAnything(Extension):
    name = "mapanything"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Meta MapAnything"
    summary = "通用真实尺度三维重建的开源研究框架；端到端训练的 Transformer 按画面、标定、位姿或深度图回归出场景的三维几何"
    homepage = "https://github.com/facebookresearch/map-anything"
    source = GitSource(url=MAPANYTHING_URL, commit=MAPANYTHING_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="Apache-2.0（代码、apache 权重）/ CC-BY-NC-4.0（main 权重，非商用）",
        url="https://github.com/facebookresearch/map-anything/blob/main/LICENSE",
        summary=(
            "代码 Apache-2.0，可商用。两套权重：map-anything-apache 为 Apache-2.0，可商用；"
            "map-anything（main，精度更高）为 CC-BY-NC-4.0，仅限非商用（研究可用，需署名）。"
            "依赖 UniCeption 为 BSD-3-Clause；DINOv2 主干网络代码为 Apache-2.0"
            "（其仓库里另有 Cell-DINO / XRay-DINO 非商用代码和权重，本扩展不导入也不下载）"
        ),
    )
    env = EnvSpec(
        python="3.12",
        # Upstream pins no torch; 2.9.0 / CUDA 13.0 as ViPE on this machine (torchaudio,
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
                note=f"{repo}（{licence}{'，非商用' if 'NC' in licence else ''}）",
            )
            for key, (repo, revision, sha, licence) in MODELS.items()
            for filename in ("config.json", "model.safetensors")
        ),
        Weight(
            key="dinov2-code",
            kind="zip",
            source=f"https://github.com/facebookresearch/dinov2/archive/{DINOV2_COMMIT}.zip",
            dest="dinov2",
            sha256=DINOV2_ZIP_SHA256,
            note="DINOv2 主干网络结构代码（Apache-2.0，只用结构，不下载 DINOv2 权重）",
        ),
    )

    def worker_env(self) -> dict[str, str]:
        return {"MAPANYTHING_DINOV2_CODE": str(self.paths.weights / "dinov2" / f"dinov2-{DINOV2_COMMIT}")}


EXTENSION = MapAnything()
