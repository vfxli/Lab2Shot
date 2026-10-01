"""Fast SAM 3D Body: SAM 3D Body's own weights run through a faster inference
path (all people and both hands of a frame in one batch, pruned decoder passes).
It builds on the sam_3d_body extension (requires): the same node (its nodes subclass SAM 3D Body 全身动作) and the same
solve and MHR rig in the worker (adapters/sam_3d_body/sam3dbody.py), and its weights (MODEL_WEIGHTS, imported)."""

from __future__ import annotations

import shutil

from adapters.sam_3d_body.extension import DINOV3, MODEL_WEIGHTS  # SAM 3D Body's weights: one table (requires)
from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

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
        tag=COMMERCIAL,  # AGPL (YOLO11-Pose) allows commercial use with its own duty to publish: said, not a class
        name="MIT + SAM License + DINOv3 License + AGPL-3.0（YOLO11-Pose）",
        url="https://github.com/yangtiming/Fast-SAM-3D-Body/blob/main/LICENSE",
        summary=(
            "可商用。提速代码 MIT；但它是在 Meta SAM 3D Body 代码上改的，原有部分和权重（与 SAM 3D Body 扩展包同一份）仍按 SAM License"
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
