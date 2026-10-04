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
# torch.hub.load("facebookresearch/dinov2", ...): the code from this pinned archive (MapAnything pins the same: stored
# once), the weights as the hub caches them (TORCH_HOME/hub/checkpoints).
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CODE = downloads.Download(f"https://github.com/facebookresearch/dinov2/archive/{DINOV2_COMMIT}.zip",
                                 "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48", kind="zip")
DINOV2_WEIGHTS = "dinov2_vits14_reg4_pretrain.pth"
DINOV2_WEIGHTS_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"  # 88 MB

SAM2_REPO = "facebook/sam2.1-hiera-tiny"
SAM2_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
SAM2_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"  # sam2.1_hiera_tiny.pt, 156 MB


class WOFTSAM(Extension):
    name = "woftsam"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "WOFTSAM"
    homepage = "https://cmp.felk.cvut.cz/~serycjon/WOFTSAM/"
    source = GitSource(url=WOFTSAM_URL, commit=WOFTSAM_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        url="https://github.com/serycjon/WOFTSAM/blob/main/LICENSE",
    )
    generative = False
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
        hf_file(SAM2_REPO, SAM2_REVISION, "sam2.1_hiera_tiny.pt", key="sam2.1_hiera_tiny", sha256=SAM2_SHA256),
        DINOV2_CODE.weight(key="dinov2-code", dest="dinov2"),
        Weight(key="dinov2_vits14_reg", kind="url",
               source=f"https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/{DINOV2_WEIGHTS}",
               dest=f"torch/hub/checkpoints/{DINOV2_WEIGHTS}", sha256=DINOV2_WEIGHTS_SHA256),
    )

    def worker_env(self) -> dict[str, str]:
        return {"LAB2SHOT_SAM2_DIR": str(self.paths.root / "sam2"),
                "LAB2SHOT_DINOV2_DIR": str(self.paths.weights / "dinov2" / f"dinov2-{DINOV2_COMMIT}")}


EXTENSION = WOFTSAM()
