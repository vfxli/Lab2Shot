"""TensorRT engines for the Fast SAM 3D Body worker: upstream's own TensorRT path (README "TensorRT Acceleration",
build_tensorrt.sh, run_demo.sh at the pinned commit), built per GPU on first use and cached in the extension's own
folder (<extension>/cache/tensorrt/<GPU>-sm<cc>-trt<version>/).

One of upstream's three engines is used: the DINOv3 backbone (convert_backbone_tensorrt.py: ONNX export of
get_intermediate_layers, FP16 engine, dynamic batch), loaded through upstream's own TRTDinov3Backbone
(sam_3d_body/models/backbones/dinov3_tensorrt.py), the switch upstream's Dinov3Backbone makes for USE_TRT_BACKBONE=1.
On an RTX 5090, 8 frames, 3 people: pose 0.24 -> 0.19 s a frame; bodies within 1 mm of PyTorch, fingers within 2 cm.
Not used:
  - the YOLO11-Pose engine (convert_yolo_pose_trt.py): measured on the RTX 5090 it detected no faster (0.04 s a frame
    either way), took 6 minutes to build, and its FP16 wrists moved the hand crops (fingers up to 4 cm off PyTorch's);
  - the MoGe encoder engine (convert_moge_encoder_trt.py, FOV_TRT): it is built for MoGe-2 ViT-S at a fixed 512 input,
    while this extension measures the focal length with MoGe-2 ViT-L at full resolution (sam_3d_body's choice);
    swapping it would change the focal length, and it runs only a few times per shot.

Missing engines are built by this file run as a separate process (`python trt_engines.py <kind> ...`, the worker's own
environment): a crash or out-of-memory while building ends that process, not the job. Whatever cannot be built or
loaded leaves that part on PyTorch, with a warning (W-FASTSAM3DBODY-NOTRT); a failed build is remembered for that
GPU / TensorRT version (a file `<kind>.failed` beside the engines; delete it to try again), so a job does not spend
minutes on a build that failed before. LAB2SHOT_FAST_SAM_3D_BODY_TRT=0 in the service's environment turns TensorRT off.
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BACKBONE = "backbone"
FILES = {BACKBONE: "backbone_dinov3_fp16.engine"}
BUILD_TIMEOUT_S = 30 * 60
# The backbone takes a frame's people and their hand crops in one batch (PARALLEL_DECODERS): at most
# family.PEOPLE_BATCH people x (body + two hands). Larger batches run in slices of this size.
MAX_BATCH = 24
OPT_BATCH = 3  # one person: body + two hands
MAX_REL_ERROR = 0.05  # the engine's backbone features against PyTorch's on the same input, when it is loaded


def enabled() -> bool:
    return os.environ.get("LAB2SHOT_FAST_SAM_3D_BODY_TRT", "1") != "0"


def tensorrt_version() -> str:
    """The installed TensorRT, "" when it is not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return ""
    for dist in ("tensorrt-cu12", "tensorrt", "tensorrt-cu13"):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    return ""


def engine_dir(root: Path) -> Path:
    """One folder per GPU model and TensorRT version: an engine runs only on the GPU and TensorRT that built it."""
    import torch

    name = re.sub(r"[^A-Za-z0-9]+", "_", torch.cuda.get_device_name()).strip("_")
    major, minor = torch.cuda.get_device_capability()
    return root / f"{name}-sm{major}{minor}-trt{tensorrt_version()}"


def engines(run, root: Path, weights: Path, kinds: tuple[str, ...]) -> dict[str, Path]:
    """kind -> engine file for each of `kinds` that is ready on this GPU, building the missing ones first; the others
    (TensorRT off, not installed, or the build failed) are left out and said once (W-FASTSAM3DBODY-NOTRT)."""
    from lab2shot_worker import say

    if not enabled():
        return {}
    if not tensorrt_version():
        say("W-FASTSAM3DBODY-NOTRT", reason="TensorRT is not installed in this extension's environment")
        return {}
    folder = engine_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    ready: dict[str, Path] = {}
    failed: list[str] = []
    with open(folder / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one build per GPU model at a time (another worker may be building it)
        for stale in folder.glob(".build-*"):  # left by a build that was cut off (killed, power lost): builds hold the lock
            shutil.rmtree(stale, ignore_errors=True)
        for kind in kinds:
            engine = folder / FILES[kind]
            marker = folder / f"{kind}.failed"
            if not engine.is_file() and not marker.is_file():
                run.stage("build_engine", part=kind)
                build(kind, engine, weights, folder)
            if engine.is_file():
                ready[kind] = engine
            else:
                failed.append(f"{kind}: {marker.read_text(errors='replace').strip()[-300:] if marker.is_file() else '?'}")
    if failed:
        say("W-FASTSAM3DBODY-NOTRT", reason="; ".join(failed) + f" (to try again, delete the .failed file in {folder})")
    return ready


def build(kind: str, engine: Path, weights: Path, folder: Path) -> None:
    """Build one engine in a process of its own; on failure write <kind>.failed with the reason."""
    log = folder / f"{kind}.build.log"
    env = dict(os.environ)
    # the builder loads plain PyTorch models: no torch.compile, no engine of its own
    env.update(USE_COMPILE="0", USE_COMPILE_BACKBONE="0", DECODER_COMPILE="0", USE_TRT_BACKBONE="0")
    command = [sys.executable, str(Path(__file__).resolve()), kind, str(weights), str(engine)]
    started = time.time()
    try:
        with open(log, "w") as out:
            proc = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                  timeout=BUILD_TIMEOUT_S)
        reason = "" if proc.returncode == 0 and engine.is_file() else f"builder exited with {proc.returncode}"
    except subprocess.TimeoutExpired:
        reason = f"no engine after {BUILD_TIMEOUT_S // 60} minutes"
    if reason:
        engine.unlink(missing_ok=True)
        tail = log.read_text(errors="replace").strip().splitlines()[-3:] if log.is_file() else []
        (folder / f"{kind}.failed").write_text(" | ".join([reason, *tail]) + "\n")
    print(f"[tensorrt] {kind}: {'built' if not reason else 'failed'} in {time.time() - started:.0f}s ({engine})", flush=True)


# --------------------------------------------------------------------------- loading (in the worker)


def give_up(engine: Path, kind: str, exc: BaseException) -> None:
    """An engine that was built but does not load or does not agree with PyTorch: said, and marked failed (as a failed
    build) so later jobs go straight to PyTorch instead of loading and checking it again."""
    import gc

    from lab2shot_worker import say

    reason = f"{type(exc).__name__}: {exc}"[:300]
    (engine.parent / f"{kind}.failed").write_text(f"loading the engine failed | {reason}\n")
    gc.collect()
    say("W-FASTSAM3DBODY-NOTRT", reason=f"{kind}: {reason} (to try again, delete {engine.parent / (kind + '.failed')})")


def attach_backbone(model, engine: Path, image_size: int) -> None:
    """Swap the PyTorch DINOv3 encoder of a loaded model for the engine, as upstream's Dinov3Backbone does for
    USE_TRT_BACKBONE=1 (its forward then goes to _trt_backbone). Checked against PyTorch on one input first; raises
    (leaving the model untouched) when the engine does not load or does not agree."""
    import torch
    from sam_3d_body.models.backbones.dinov3_tensorrt import create_tensorrt_backbone

    backbone = model.backbone
    trt = Sliced(create_tensorrt_backbone(engine_path=str(engine), name=backbone.name,
                                          image_size=(image_size, image_size)))
    device = next(backbone.encoder.parameters()).device
    # a batch of more than one: an export that froze the batch size would show here
    x = torch.randn(OPT_BATCH, 3, image_size, image_size, device=device, generator=torch.Generator(device).manual_seed(0))
    with torch.no_grad():
        ref = backbone(x.type(model.backbone_dtype)).float()
        got = trt(x.type(model.backbone_dtype)).float()
        torch.cuda.synchronize()
    error = float((got - ref).norm() / ref.norm().clamp_min(1e-6))
    if not torch.isfinite(got).all() or got.shape != ref.shape or error > MAX_REL_ERROR:
        raise RuntimeError(f"engine output differs from PyTorch: shape {tuple(got.shape)} vs {tuple(ref.shape)}, "
                           f"relative error {error:.4f}")
    backbone._trt_backbone = trt
    backbone._use_tensorrt = True
    del backbone.encoder  # its ~0.8 B parameters are in the engine now
    torch.cuda.empty_cache()
    print(f"[tensorrt] backbone engine in use ({engine}), relative error against PyTorch {error:.5f}", flush=True)


class Sliced:
    """Upstream's TRTDinov3Backbone for any batch size: batches above the engine's MAX_BATCH run in slices."""

    def __init__(self, engine):
        self.engine = engine

    def __call__(self, x, extra_embed=None):
        import torch

        if x.shape[0] <= MAX_BATCH:
            return self.engine(x, extra_embed)
        return torch.cat([self.engine(part, extra_embed) for part in x.split(MAX_BATCH)])

    def get_layer_depth(self, *args, **kwargs):
        return self.engine.get_layer_depth(*args, **kwargs)


# --------------------------------------------------------------------------- building (a process of its own)


def build_backbone(weights: Path, engine: Path) -> None:
    """Upstream's convert_backbone_tensorrt.py steps 1 and 2 (ONNX export, FP16 engine), its paths pointed at this
    extension's weights and engine folder instead of the repository's checkpoints/."""
    import convert_backbone_tensorrt as upstream
    from lab2shot_worker import local_hub

    local_hub({"dinov3": os.environ["LAB2SHOT_DINOV3_DIR"]}, "Fast SAM 3D Body")
    work = engine.parent / f".build-{os.getpid()}"
    work.mkdir()
    try:
        upstream.CHECKPOINT_DIR = str(weights / "sam-3d-body-dinov3")
        upstream.TRT_OUTPUT_DIR = str(work)
        upstream.ONNX_PATH = str(work / "backbone_dinov3.onnx")
        upstream.TRT_PATH = str(work / FILES[BACKBONE])
        size = int(os.environ.get("IMG_SIZE", "512"))
        upstream.IMAGE_SIZE = (size, size)
        backbone = upstream.load_backbone()
        upstream.step1_export_onnx(backbone, [1, OPT_BATCH, MAX_BATCH])
        del backbone
        import torch

        torch.cuda.empty_cache()
        if not upstream.step2_convert_tensorrt([1, OPT_BATCH, MAX_BATCH]):
            raise SystemExit("TensorRT engine build failed")
        os.replace(upstream.TRT_PATH, engine)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    kind, weights_dir, engine_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    {BACKBONE: build_backbone}[kind](weights_dir, engine_path)
