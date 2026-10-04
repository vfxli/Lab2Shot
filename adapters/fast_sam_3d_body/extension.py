"""Fast SAM 3D Body: SAM 3D Body's own weights run through a faster inference
path (all people and both hands of a frame in one batch, pruned decoder passes).
It builds on the sam_3d_body extension (requires): the same node (its nodes subclass sam_3d_body.solve) and the same
solve and MHR rig in the worker (adapters/sam_3d_body/sam3dbody.py), and its weights (MODEL_WEIGHTS, imported)."""

from __future__ import annotations

import shutil

from adapters.sam_3d_body.extension import DINOV3, MODEL_WEIGHTS  # SAM 3D Body's weights: one table (requires)
from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

YOLO_WEIGHT = "yolo/yolo11m-pose.pt"
YOLO_SHA256 = "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2"

# TensorRT (upstream's TensorRT path, setup_env.sh step 8: the backbone engine, trt_engines.py).
# Its 3.1 GB library wheel is only on NVIDIA's index (PyPI has a stub that downloads it at build time in one piece, which
# fails on a slow or dropping line): it comes through the installer's resumable, sha256-checked download as a file of
# its own, then post_install puts it into the environment with the PyPI bindings (requirements.txt) and the `tensorrt`
# import package. 10.13 runs on Ada (RTX 4090) and Blackwell (RTX 5090).
TENSORRT_VERSION = "10.13.3.9"
TENSORRT_LIBS = f"tensorrt_cu12_libs-{TENSORRT_VERSION}-py2.py3-none-manylinux_2_28_x86_64.whl"


class FastSam3DBody(Extension):
    name = "fast_sam_3d_body"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Fast SAM 3D Body"
    homepage = "https://github.com/yangtiming/Fast-SAM-3D-Body"
    source = GitSource(
        url="https://github.com/yangtiming/Fast-SAM-3D-Body.git",
        commit="808b53c7d9c26a7e511d31144f1e5efb058e15c9",
    )
    license = LicenseInfo(
        tag=COMMERCIAL,  # AGPL (YOLO11-Pose) allows commercial use with its own duty to publish: said, not a class
        url="https://github.com/yangtiming/Fast-SAM-3D-Body/blob/main/LICENSE",
    )
    generative = False
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
            sha256=YOLO_SHA256,
        ),
        Weight(
            key="tensorrt-libs",
            kind="url",
            source=f"https://pypi.nvidia.com/tensorrt-cu12-libs/{TENSORRT_LIBS}",
            dest=f"tensorrt/{TENSORRT_LIBS}",
            sha256="bf2008eca911411aa93b852825ea992de33396451ea11713a6eb97c411d3b2e9",
        ),
    )

    def worker_env(self) -> dict[str, str]:
        root = self.paths.root
        return {
            "LAB2SHOT_DINOV3_DIR": str(root / "dinov3"),
            # Any value disables the optional pymomentum path: use the TorchScript MHR shipped with the weights.
            "MOMENTUM_ENABLED": "0",
            # upstream's TensorRT engines, built per GPU on first use (trt_engines.py): the extension's own cache
            "LAB2SHOT_FAST_SAM_3D_BODY_TRT_DIR": str(root / "cache" / "tensorrt"),
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
        # TensorRT's libraries (downloaded as a weight, see TENSORRT_LIBS) and its `tensorrt` import package (a small
        # PyPI source package whose only dependencies are the libraries and the bindings, both already here)
        run(["uv", "pip", "install", "--python", str(paths.python), "--no-deps",
             str(paths.weights / "tensorrt" / TENSORRT_LIBS), f"tensorrt-cu12=={TENSORRT_VERSION}"])


EXTENSION = FastSam3DBody()
