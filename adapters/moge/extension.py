"""Microsoft MoGe: per-frame monocular geometry (metric point map, depth, normals,
valid mask, camera intrinsics) from single frames."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

MOGE_URL = "https://github.com/microsoft/MoGe.git"
MOGE_COMMIT = "74fbce054ebed49800de42d0ad0e83495065719a"  # "V3" (MoGe-3 release)

# Hugging Face repo -> pinned revision. Every model is a single model.pt, stored as
# weights/<repo id>/model.pt; the worker loads exactly that file (offline).
# All MIT (model cards), DINOv2 backbone (Apache-2.0).
MODELS = {
    # repo: (pinned revision, model.pt LFS sha256)
    "Ruicheng/moge-3-vitl": ("184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be", "9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925"),  # 370M params, 1.5 GB
    "Ruicheng/moge-3-vitg": ("6ef26c5a4b4148dab5ccaacdb08b72dc66380475", "ce7c15417e9105c2ace7b4272e2cc69e36940921211eb7fa05d4d0bb03f0a00c"),  # 1.25B params, 5.0 GB
}


class MoGe(Extension):
    name = "moge"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Microsoft MoGe"
    summary = "从单目开放域画面恢复三维几何，包括真实尺度的点图、深度图、法线图和相机视场角"
    homepage = "https://github.com/microsoft/MoGe"
    source = GitSource(url=MOGE_URL, commit=MOGE_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="MIT",
        url="https://github.com/microsoft/MoGe/blob/main/LICENSE",
        summary=(
            "代码和 MoGe-3 权重（moge-3-vitl / moge-3-vitg）均为 MIT，可商用；主干网络 DINOv2 为 Apache-2.0。"
            "依赖 utils3d-moge、FlexGEMM 也是 MIT。utils3d-moge 里有可选的 nvdiffrast（NVIDIA 非商用）光栅化模块，"
            "MoGe 推理用不到，本扩展不安装它"
        ),
    )
    env = EnvSpec(
        python="3.12",
        # Upstream's uv.lock at the pinned commit: torch 2.13.0 built for CUDA 13.0
        # (brings Triton 3.7, which FlexGEMM's sparse convolutions run on).
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
    )
    weights = tuple(
        hf_file(
            repo, rev, "model.pt",
            key=repo.split("/")[1],
            dest=f"{repo}/model.pt",
            sha256=sha256,
            note=f"{repo}（MIT）",
        )
        for repo, (rev, sha256) in MODELS.items()
    )

    def worker_env(self) -> dict[str, str]:
        return {
            # MoGe-3's refiner runs on Triton (FlexGEMM): its autotune results stay inside the extension.
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(self.paths.root / "cache" / "flex_gemm_autotune.json"),
        }


EXTENSION = MoGe()
