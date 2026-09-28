"""Fast SAM 3D Body: SAM 3D Body's own weights run through a faster inference
path (all people and both hands of a frame in one batch, pruned decoder passes).
It builds on the sam_3d_body extension (requires): the same node (its nodes subclass SAM 3D Body 全身动作) and the same
solve and MHR rig in the worker (adapters/sam_3d_body/sam3dbody.py). The weights are pinned here as well."""

from __future__ import annotations

import shutil

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

# SAM 3D Body's: the DINOv3 network code of the backbone, the model and its rig, the MoGe-2 focal estimate (the same
# files as the sam_3d_body extension's: the installer keeps one copy of identical files).
DINOV3 = GitSource(url="https://github.com/facebookresearch/dinov3.git", commit="6876159a11b4df116f30f667f8c9888617df0751")
MODEL_REPO, MODEL_REVISION = "facebook/sam-3d-body-dinov3", "11aaa346c7204874a1cbafe3d39a979080b2c55a"
MOGE_REPO, MOGE_REVISION = "Ruicheng/moge-2-vitl-normal", "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"
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

YOLO_WEIGHT = "yolo/yolo11m-pose.pt"
YOLO_SHA256 = "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2"


class FastSam3DBody(Extension):
    name = "fast_sam_3d_body"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Fast SAM 3D Body"
    summary = "一个免训练的加速框架，把 SAM 3D Body 的推理路径重排，达到交互级速度"
    homepage = "https://github.com/yangtiming/Fast-SAM-3D-Body"
    source = GitSource(
        url="https://github.com/yangtiming/Fast-SAM-3D-Body.git",
        commit="808b53c7d9c26a7e511d31144f1e5efb058e15c9",
    )
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="MIT + SAM License + DINOv3 License + AGPL-3.0（YOLO11-Pose）",
        url="https://github.com/yangtiming/Fast-SAM-3D-Body/blob/main/LICENSE",
        summary=(
            "提速代码 MIT；但它是在 Meta SAM 3D Body 代码上改的，原有部分和权重（与 SAM 3D Body 扩展包同一份）仍按 SAM License"
            "（可商用、可修改再分发，需附许可证，发表论文需注明，禁军事用途）；骨干网络代码 DINOv3 License（同类条款）；"
            "MoGe-2 Focal Length 估计 MIT。找手腕用的 Ultralytics YOLO11-Pose（代码和权重）是 AGPL-3.0："
            "自己内部使用无妨，修改后对外提供服务或分发需按 AGPL 开源，商用闭源要买 Ultralytics 企业许可"
        ),
    )
    import_repo = ""
    requires = ("sam_3d_body",)  # its node and its worker's solve (sam3dbody.py) are SAM 3D Body's
    env = EnvSpec(
        python="3.11",
        # Same torch as the sam_3d_body extension (upstream tested 2.5.1; the code runs unchanged).
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
    )
    extra_sources = {"dinov3": DINOV3}
    weights = MODEL_WEIGHTS + (
        Weight(
            key="yolo11m-pose",
            kind="url",
            source="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt",
            dest=YOLO_WEIGHT,
            note="YOLO11-Pose 手腕关键点（Ultralytics，AGPL-3.0）",
            sha256=YOLO_SHA256,
        ),
    )

    def worker_env(self) -> dict[str, str]:
        root = self.paths.root
        return {
            "LAB2SHOT_DINOV3_DIR": str(root / "dinov3"),
            # Any value disables the optional pymomentum path: use the TorchScript MHR shipped with the weights.
            "MOMENTUM_ENABLED": "0",
            # YOLO_CONFIG_DIR / YOLO_OFFLINE come from Extension.base_env() (every ultralytics worker gets them).
            # Ultralytics compares `str(os.getenv("YOLO_OFFLINE","")).lower() != "true"` (ultralytics/utils/__init__.py:516):
            # the value must be "True"; "1" would silently do nothing.
        }

    def post_install(self, run, paths) -> None:
        # Upstream's model config for these weights (fp16 backbone instead of bf16): a file of its own, never
        # written through a shared weight file.
        config = paths.weights / "sam-3d-body-dinov3" / "model_config.yaml"
        config.unlink(missing_ok=True)
        shutil.copyfile(paths.repo / "checkpoints" / "sam-3d-body-dinov3" / "model_config.yaml", config)


EXTENSION = FastSam3DBody()
