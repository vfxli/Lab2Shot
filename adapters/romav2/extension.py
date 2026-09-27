"""RoMa v2 (Edstedt et al., 2025): dense feature matching between any two images. A warp from every pixel of one
image into the other, with how sure it is that the pixel is seen in both."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

ROMAV2_URL = "https://github.com/Parskatt/romav2.git"
ROMAV2_COMMIT = "95c9968145c8906b7b59383258e9f73b02853d89"  # v2.0.1

# The DINOv3 ViT-L/16 network code RoMa v2 builds its descriptor from. Upstream fetches it through torch.hub at run
# time (network); it is pinned here to the commit romav2 names (features.py) and loaded locally. Its weights come
# inside romav2.0.1.pt (the descriptor is part of the model's state dict), so no DINOv3 checkpoint is downloaded.
DINOV3 = GitSource(url="https://github.com/facebookresearch/dinov3.git", commit="adc254450203739c8149213a7a69d8d905b4fcfa")

WEIGHTS_URL = "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt"
WEIGHTS_SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"  # 1.1 GB (GitHub release: no revision, the installer checks the sha256)


class RomaV2(Extension):
    name = "romav2"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "RoMa v2"
    summary = "稠密特征匹配：估计同一个三维场景的两张画面之间的全部对应关系"
    homepage = "https://github.com/Parskatt/romav2"
    source = GitSource(url=ROMAV2_URL, commit=ROMAV2_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="MIT + DINOv3 License",
        url="https://github.com/Parskatt/romav2/blob/main/LICENSE",
        summary=(
            "RoMa v2 的代码和权重是 MIT。权重里包含 DINOv3 ViT-L 骨干，受 Meta 的 DINOv3 License 约束：允许商用、修改和再分发"
            "（再分发要附许可证原文，发表论文要注明用了 DINOv3），禁止军事、武器、核等用途和受制裁方使用。可商用"
        ),
    )
    import_repo = "src"  # the `romav2` package from repo/src (not pip-installed: its fused-local-corr kernel pins torch 2.11)
    env = EnvSpec(
        python="3.12",
        # Pure PyTorch: the same torch build as other extensions (shared uv cache). Upstream's optional fused local
        # correlation kernel (fused-local-corr, a wheel for torch 2.11 only) is not installed: romav2 then computes
        # the same correlation with plain torch ops (local_correlation.native_torch_local_corr).
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
        pickled_checkpoints=True,  # romav2.0.1.pt (pinned, sha256-checked)
    )
    extra_sources = {"dinov3": DINOV3}
    weights = (
        Weight(key="romav2", kind="url", source=WEIGHTS_URL, dest="romav2.0.1.pt", sha256=WEIGHTS_SHA256,
               note="RoMa v2.0.1（1.1 GB，MIT；含 DINOv3 ViT-L 骨干，DINOv3 License）"),
    )

    def worker_env(self) -> dict[str, str]:
        return {"LAB2SHOT_DINOV3_DIR": str(self.paths.root / "dinov3")}


EXTENSION = RomaV2()
