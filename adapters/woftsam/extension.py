"""WOFTSAM (Serych & Matas, ECCV 2026): long-term planar tracking. A plane given by its four corners on one frame is
followed through the shot as a full homography per frame: weighted RAFT optical flow (WOFT) for precision, SAM 2 mask
contours (SAM-H) to find the plane again after an occlusion or a failure."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file, downloads

WOFTSAM_URL = "https://github.com/serycjon/WOFTSAM.git"
WOFTSAM_COMMIT = "5131bdc27d4de1be03d9a40ac8a99eef73dfab0c"  # 2026-08-25 (MPOT-3K evaluation, ECCV 2026)

# The authors' SAM 2 fork (one commit on upstream: SAM 2's memory is not updated on frames where the plane is not
# there, so it finds the plane again after an occlusion). Imported from this checkout, not pip-installed: its CUDA
# connected-components op (only for filling small holes in masks) is not built.
SAM2 = GitSource(url="https://github.com/serycjon/sam2.git", commit="903a6cdb054d1b2fba3636d8aa673574d37cfe77")

# SAM-H tells the plane's four corners apart after it was lost with DINOv2 ViT-S/14 (registers) features, loaded by
# torch.hub.load("facebookresearch/dinov2", ...): the code from this pinned archive (the one MapAnything pins: stored
# once), the weights as the hub caches them (TORCH_HOME/hub/checkpoints).
DINOV2_COMMIT = downloads.DINOV2_COMMIT
DINOV2_WEIGHTS = "dinov2_vits14_reg4_pretrain.pth"
DINOV2_WEIGHTS_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"  # 88 MB

SAM2_REPO = "facebook/sam2.1-hiera-tiny"
SAM2_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
SAM2_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"  # sam2.1_hiera_tiny.pt, 156 MB


class WOFTSAM(Extension):
    name = "woftsam"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "WOFTSAM"
    summary = "分割引导的单应估计，用于长时平面跟踪；在 POT-210 和 PlanarTrack 两个基准上达到当前最好成绩"
    homepage = "https://cmp.felk.cvut.cz/~serycjon/WOFTSAM/"
    source = GitSource(url=WOFTSAM_URL, commit=WOFTSAM_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="CC BY-NC-SA 4.0（非商用）",
        url="https://github.com/serycjon/WOFTSAM/blob/main/LICENSE",
        summary=(
            "非商用：WOFTSAM 的代码和它的加权 RAFT 权重是 CC BY-NC-SA 4.0，只能用于研究等非商业用途，改动后要以同样的许可发布、"
            "署名作者（Serych、Matas）。仓库里的 RAFT 代码是 BSD-3。另外用到的 SAM 2.1 tiny 权重和 SAM 2 代码是 Apache-2.0，"
            "DINOv2 ViT-S/14 代码和权重是 Apache-2.0"
        ),
    )
    import_repo = "src"  # the `flatsam` package from repo/src (its configs and weights are found next to it)
    env = EnvSpec(
        python="3.12",
        # Pure PyTorch (no compiled ops): the same torch build as other extensions (shared uv cache).
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
        pickled_checkpoints=True,  # the weighted-RAFT and SAM 2.1 checkpoints (pinned, sha256-checked)
    )
    extra_sources = {"sam2": SAM2}
    weights = (
        hf_file(SAM2_REPO, SAM2_REVISION, "sam2.1_hiera_tiny.pt", key="sam2.1_hiera_tiny", sha256=SAM2_SHA256,
                note="SAM 2.1 Hiera tiny（Apache-2.0，156 MB）：跟着平面的遮罩，丢了以后重新找回"),
        downloads.DINOV2_CODE.weight(key="dinov2-code", dest="dinov2", note="DINOv2 网络结构代码（Apache-2.0）"),
        Weight(key="dinov2_vits14_reg", kind="url",
               source=f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/{DINOV2_WEIGHTS}",
               dest=f"torch/hub/checkpoints/{DINOV2_WEIGHTS}", sha256=DINOV2_WEIGHTS_SHA256,
               note="DINOv2 ViT-S/14 带 register（Apache-2.0，88 MB）：平面丢了以后分清四个角谁是谁"),
    )

    def worker_env(self) -> dict[str, str]:
        return {"LAB2SHOT_SAM2_DIR": str(self.paths.root / "sam2"),
                "LAB2SHOT_DINOV2_DIR": str(self.paths.weights / "dinov2" / f"dinov2-{DINOV2_COMMIT}")}


EXTENSION = WOFTSAM()
