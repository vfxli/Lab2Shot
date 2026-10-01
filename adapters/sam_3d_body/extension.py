"""SAM 3D Body: full-body human motion from plates -> a USD skinned character in camera space, plus the ViTDet
people detector (人物框)."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file, downloads

# DINOv3 network code for the backbone. Upstream fetches the moving main branch through
# torch.hub at run time (network, unpinned); it is pinned and local instead.
DINOV3 = GitSource(url="https://github.com/facebookresearch/dinov3.git", commit="6876159a11b4df116f30f667f8c9888617df0751")

MODEL_REPO, MODEL_REVISION = "facebook/sam-3d-body-dinov3", "11aaa346c7204874a1cbafe3d39a979080b2c55a"
MOGE_REPO, MOGE_REVISION = "Ruicheng/moge-2-vitl-normal", "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"
# The model and its rig, and the MoGe-2 focal estimate. The one table: fast_sam_3d_body imports it (and DINOV3).
MODEL_WEIGHTS = (
    hf_file(MODEL_REPO, MODEL_REVISION, "model.ckpt", key="sam-3d-body-dinov3", dest="sam-3d-body-dinov3/model.ckpt",
            sha256="b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf", gated=True,
            note="SAM 3D Body 权重（2.1 GB，SAM License）；需要在 Hugging Face 页面申请权限"),
    hf_file(MODEL_REPO, MODEL_REVISION, "assets/mhr_model.pt", key="mhr-model", dest="sam-3d-body-dinov3/assets/mhr_model.pt",
            sha256="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc", gated=True,
            note="MHR 人体绑定（0.7 GB，随 SAM 3D Body 权重发布）"),
    hf_file(MODEL_REPO, MODEL_REVISION, "LICENSE", key="sam-license", dest="sam-3d-body-dinov3/LICENSE", gated=True,
            note="SAM License 原文（使用和分发权重时要附带）"),
    hf_file(MOGE_REPO, MOGE_REVISION, "model.pt", key="moge-2", dest="moge-2-vitl-normal/model.pt",
            sha256="280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01", note="Focal Length 估计 MoGe-2（MIT，1.3 GB）"),
)


class Sam3DBody(Extension):
    name = "sam_3d_body"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "SAM 3D Body"
    summary = "从单张画面做全身三维人体网格恢复；基于 MHR 参数化网格表示估计身体、脚和手的姿态"
    homepage = "https://github.com/facebookresearch/sam-3d-body"
    source = GitSource(
        url="https://github.com/facebookresearch/sam-3d-body.git",
        commit="b5c765a0d89d789985e186d396315e7590887b94",
    )
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="SAM License",
        url="https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE",
        summary="允许商用、修改和再分发（需附带许可证）；发表论文需注明使用了 SAM 3D Body",
    )
    import_repo = ""
    worker_modules = ("sam3dbody.py",)  # the solve and MHR rig, Fast SAM 3D Body's worker imports it too
    env = EnvSpec(
        python="3.11",
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
        compiled=(
            downloads.pip_git("detectron2", downloads.DETECTRON2),
        ),
        # The ViTDet detector runs ROIAlign / NMS through torchvision on the GPU;
        # detectron2's own CUDA kernels (deformable conv, rotated boxes) are unused.
        # Building without them means no CUDA toolkit is needed to install.
        compiled_cuda=False,
    )
    extra_sources = {"dinov3": DINOV3}
    weights = MODEL_WEIGHTS + (
        hf_file(MODEL_REPO, MODEL_REVISION, "model_config.yaml", key="sam-3d-body-config", dest="sam-3d-body-dinov3/model_config.yaml",
                gated=True, note="SAM 3D Body 网络配置"),
        downloads.VITDET_H.weight(key="vitdet", dest="vitdet/model_final_f05665.pkl", note="人物检测 ViTDet-H（Apache-2.0，2.8 GB）"),
    )

    def worker_env(self) -> dict[str, str]:
        return {
            "LAB2SHOT_DINOV3_DIR": str(self.paths.root / "dinov3"),
            # Any value disables the optional pymomentum path: use the TorchScript MHR shipped with the weights.
            "MOMENTUM_ENABLED": "0",
        }


EXTENSION = Sam3DBody()
