"""Fast SAM 3D Body worker. Runs inside third_party/fast_sam_3d_body/.venv with
the Fast SAM 3D Body repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>

Same job and raw output as the sam_3d_body worker's solve node (the solve, rig and
converter are shared: sam3dbody.py of the sam_3d_body extension, which this one requires): the given person
boxes, or without them this repo's own YOLO11-Pose detector (as upstream's demo detects people itself) -> one
focal length for the shot -> Fast SAM 3D Body per frame, the people of a frame in batches, hand crops placed from
the YOLO11-Pose wrists so body and hands run through the backbone together -> per person: lock shape, smooth,
re-evaluate the MHR rig -> raw npz files.

One YOLO11-Pose pass gives both the boxes and the wrists: the detector is loaded once and the frames are read once.
"""

from __future__ import annotations

import os
from pathlib import Path

# Upstream's optimized settings (run_demo.sh at the pinned commit), except torch.compile, which is off by default:
# compiling (USE_COMPILE / USE_COMPILE_BACKBONE / DECODER_COMPILE, reduce-overhead CUDA graphs) costs one to two
# minutes in every new worker process and once more for each new number of people in a batch, and saves about
# 0.02 s a frame (docs.md), so only shots of thousands of frames gain from it. To turn it on, start the service with
# LAB2SHOT_FAST_SAM_3D_BODY_COMPILE=1 in its environment (workers inherit it); the compiled kernels are then cached on
# disk in the extension's own TORCHINDUCTOR_CACHE_DIR (Extension.base_env). TensorRT: upstream's backbone engine,
# built per GPU on first use and cached in the extension's folder (trt_engines.py; when it cannot be built or loaded
# the backbone stays on PyTorch, with a warning). Read by the repo at import time; the environment may override
# any of them.
COMPILE = "1" if os.environ.get("LAB2SHOT_FAST_SAM_3D_BODY_COMPILE", "0") == "1" else "0"
FAST_SETTINGS = {
    "IMG_SIZE": "512",  # upstream: smaller is faster but loses accuracy
    "LAYER_DTYPE": "fp32",  # decoder precision; upstream says multi-person needs fp32
    "GPU_HAND_PREP": "1",  # hand crops cut on the GPU
    "PARALLEL_DECODERS": "1",  # body + hand crops through the backbone in one batch
    "SKIP_KEYPOINT_PROMPT": "1",  # no second keypoint-prompted decoder pass
    "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
    "BODY_INTERM_PRED_LAYERS": "0,1,2",  # pruned intermediate predictions
    "HAND_INTERM_PRED_LAYERS": "0,1",
    "MHR_NO_CORRECTIVES": "1",  # no pose correctives in the network's MHR head (the exported rig has none either)
    "MHR_USE_CUDA_GRAPH": "0",
    "USE_COMPILE": COMPILE,
    "USE_COMPILE_BACKBONE": COMPILE,
    "DECODER_COMPILE": COMPILE,
    "COMPILE_MODE": "reduce-overhead",
    "DEBUG_NAN": "0",
    "INTERM_TIMING": "0",
}
for _key, _value in FAST_SETTINGS.items():
    os.environ.setdefault(_key, _value)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lab2shot_worker import (MemoryBound, limit_gpu_memory, local_hub, offload, quiet, read_frame, resident,  # noqa: E402
                             say, serve)
from lab2shot_worker import tracking  # noqa: E402
from lab2shot_worker.run import Run  # noqa: E402

# the sam_3d_body adapter's module (this extension requires sam_3d_body: its folder is on the worker's path)
import sam3dbody as family  # noqa: E402
import trt_engines  # noqa: E402  (this adapter's folder: the worker script's own)

YOLO_THRESH = 0.5  # YOLO11-Pose person confidence
MATCH_IOU = 0.3  # a given person box takes the wrists of the YOLO person it overlaps most
NO_KEYPOINTS = np.zeros((17, 3), np.float32)
WRISTS = [9, 10]  # COCO left / right wrist
# Upstream puts a hand crop on the body centre when a wrist is below this confidence,
# which gives made-up fingers; such frames take upstream's other path instead (hand
# crops from the body decoder, as in SAM 3D Body: correct, a little slower).
MIN_WRIST_CONF = 0.3


@resident
def load_fov_model(path: Path, device: torch.device):
    from tools.build_fov_estimator import load_moge

    with quiet():
        return load_moge(device, path=str(path), half=False).eval()


@resident
def load_yolo(path: Path, device: torch.device):
    from tools.build_detector import HumanDetector

    with quiet():
        return HumanDetector(name="yolo_pose", device=device, model=str(path))


@resident
def load_body(weights: Path, device: torch.device):
    """Fast SAM 3D Body and its config (the estimator around them is made per job: it keeps the last image's
    results, and its detector is the job's FramePeople)."""
    from sam_3d_body import load_sam_3d_body

    with quiet():
        return load_sam_3d_body(str(weights / "model.ckpt"), device=device, mhr_path=str(weights / "assets" / "mhr_model.pt"))


@resident(movable=False)  # the TensorRT engine's memory is not PyTorch's: leaving the GPU frees it
def load_body_engine(weights: Path, device: torch.device, engine: Path):
    """load_body with upstream's TensorRT backbone in place of the PyTorch DINOv3 encoder (trt_engines.attach_backbone).
    Raises when the engine does not load or does not agree with PyTorch."""
    model, cfg = load_body.__wrapped__(weights, device)
    trt_engines.attach_backbone(model, engine, int(os.environ["IMG_SIZE"]))
    return model, cfg


def load_any_body(run: Run, device, engine: Path | None):
    """The model with the TensorRT backbone when there is an engine, else (or when it fails to load) PyTorch."""
    weights = run.job.weights_dir / "sam-3d-body-dinov3"
    if engine is not None:
        try:
            return run.model("load_model", load_body_engine, weights, device, engine,
                             stage_params={"model": "Fast SAM 3D Body (TensorRT)"})
        except Exception as exc:  # noqa: BLE001 - any engine problem: the PyTorch path, said
            trt_engines.give_up(engine, trt_engines.BACKBONE, exc)
            torch.cuda.empty_cache()
    return run.model("load_model", load_body, weights, device, stage_params={"model": "Fast SAM 3D Body"})


def measure_focal(job, device, loaded: list):
    """rgb -> focal px: MoGe-2 ViT-L, as the sam_3d_body worker (through this repo's own loader; appended to
    `loaded`)."""
    from tools.build_fov_estimator import run_moge

    model = load_fov_model(job.weights_dir / "moge-2-vitl-normal" / "model.pt", device)
    loaded.append(model)

    def measure(rgb):
        with quiet():
            return float(run_moge(model, rgb, device)[0, 0, 0])

    return measure


def detect_people(run: Run, device) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """frame -> (boxes [N,4], COCO keypoints [N,17,3]) from YOLO11-Pose: who is in each frame and where their
    wrists are.

    The people when no boxes input is connected (a connected one goes straight to upstream's process_one_image
    bboxes; then this runs only for the wrists, see node_solve). This repo's own detector, not SAM 3D Body's ViTDet:
      - this repo's demo uses YOLO11 as its default person detector, `yolo_pose` being one of its three choices
        (third_party/fast_sam_3d_body/repo/demo_human.py:623-629, `--detector` default "yolo",
        choices ["vitdet", "yolo", "yolo_pose"]), and its hand crops take the same YOLO-Pose wrists
        (run_publisher.py:305 `hand_box_source="yolo_pose"`, run_yolo_pose in build_detector.py:27-31);
      - `yolo_pose` rather than `yolo`: one detection gives both boxes and wrists, so the detector is loaded once
        and the frames are read once; the wrists infer() needs are this pass's;
      - it is the only detector whose weights this extension installs (third_party/fast_sam_3d_body/weights/
        yolo/yolo11m-pose.pt; ViTDet's 2.7 GB model_final_f05665.pkl is in the sam_3d_body extension's weights).
    """
    # PyTorch, not upstream's YOLO-Pose TensorRT engine: on an RTX 5090 the engine detected no faster (0.04 s a frame
    # either way, reading the frame dominates), took 6 minutes to build, and its FP16 wrists moved the hand crops
    # (fingers up to 4 cm apart from PyTorch's; the backbone engine alone: under 2 cm, bodies under 1 mm)
    detector = run.model("load_model", load_yolo, run.job.weights_dir / "yolo" / "yolo11m-pose.pt", device,
                         stage_params={"model": "YOLO11-Pose"})
    run.stage("detect_people")
    found = {}
    for n, (frame, path) in run.each(run.job.frames, ""):
        with quiet():
            det = detector.run_human_detection(read_frame(path, order="bgr"), bbox_thr=YOLO_THRESH, default_to_full_image=False)
        found[frame] = (np.asarray(det["boxes"], np.float64).reshape(-1, 4), np.asarray(det["keypoints"], np.float32).reshape(-1, 17, 3))
    return found


def wrists_for(boxes: np.ndarray, found: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, bool]:
    """Keypoints of the YOLO person each box overlaps most, and whether every box got both wrists."""
    out = np.stack([NO_KEYPOINTS] * len(boxes))
    if found is not None and len(found[0]):
        scores = tracking.iou(boxes, found[0])
        for i, j in enumerate(scores.argmax(1)):
            if scores[i, j] >= MATCH_IOU:
                out[i] = found[1][j]
    return out, bool((out[:, WRISTS, 2] > MIN_WRIST_CONF).all())


class FramePeople:
    """Stands in for the estimator's detector: hands it this frame's chosen people and their wrists,
    which is what switches upstream to its fast path (body + hand crops in one backbone batch)."""

    def __init__(self):
        self.boxes = np.zeros((0, 4))
        self.keypoints = np.zeros((0, 17, 3), np.float32)

    def run_human_detection(self, img, **kwargs):
        return {"boxes": self.boxes, "keypoints": self.keypoints}


def load_estimator(run: Run, device, batch_sizes: list[int], engine: Path | None = None):
    # the batch sizes the estimator warms the compiled graphs up for (read when it is made, every job): this job's
    os.environ["COMPILE_WARMUP_BATCH_SIZES"] = ",".join(map(str, batch_sizes))
    from sam_3d_body import SAM3DBodyEstimator

    model, cfg = load_any_body(run, device, engine)
    people = FramePeople()
    with quiet():
        estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=people)
    return model, estimator, people


def infer(run: Run, estimator, people: FramePeople, selected, wrists, cam_int, full: bool) -> dict[int, dict[int, dict]]:
    """person id -> frame -> Fast SAM 3D Body output; all people of a frame in one batch."""
    run.stage("estimate_pose")
    per_person: dict[int, dict[int, dict]] = {pid: {} for pid, _ in selected}
    serial = 0
    batch = [family.PEOPLE_BATCH]  # people per batch; once lowered for memory it stays lowered (else every frame fails once first)
    # the picture goes in from memory, read once per frame and shared by its batches (RGB, as upstream's real-time
    # path hands it over: run_publisher.py:300-306), and upstream's per-call empty_cache is skipped (keep_gpu_cache)
    with family.keep_gpu_cache():
        for n, (frame, path) in run.each(run.job.frames, ""):
            serial += infer_frame(run, estimator, people, selected, wrists, cam_int, full, per_person, batch, frame, path)
    if serial:
        say("N-FASTSAM3DBODY-SLOWHANDS", frames=int(serial))
    return per_person


def infer_frame(run: Run, estimator, people: FramePeople, selected, wrists, cam_int, full: bool, per_person, batch,
                frame, path) -> int:
    """One frame of infer(); how many of its batches took the slower hand path."""
    present = [(pid, t.boxes[frame]) for pid, t in selected if frame in t.boxes]
    if not present:
        return 0
    image = read_frame(path)  # RGB: process_one_image takes an array as RGB

    # a frame's people in batches, every one solved (sam3dbody.py batches); a batch that does not fit the GPU
    # redoes the frame one size smaller. Each batch hands its boxes and wrists to FramePeople (standing in for
    # upstream's detector), so upstream takes its fast path for that batch
    def estimate(size, frame=frame, image=image, present=present):
        batch[0] = size
        pairs, slow = [], 0
        for chunk in family.batches(present, size):
            people.boxes = np.stack([b for _, b in chunk]).astype(np.float32)
            people.keypoints, wrists_seen = wrists_for(people.boxes, wrists.get(frame))
            parallel = full and wrists_seen
            slow += full and not wrists_seen
            with quiet():
                outs = estimator.process_one_image(
                    image,
                    cam_int=cam_int(frame) if callable(cam_int) else cam_int,
                    inference_type="full" if full else "body",
                    hand_box_source="yolo_pose" if parallel else "body_decoder",
                )
            pairs.extend(zip([pid for pid, _ in chunk], outs))
        return pairs, slow

    pairs, slow = run.fit(MemoryBound.batch(family.BATCH_STEPS), estimate, batch[0])
    for pid, out in pairs:
        per_person[pid][frame] = out
    return slow


def node_solve(run: Run, device) -> None:
    """Node fast_sam_3d_body.solve: same inputs, parameters and raw output as sam_3d_body.solve; without
    person boxes the people come from this repo's own YOLO11-Pose (detect_people)."""
    job = run.job
    full = bool(job.params["hand_refine"])  # False: body decoder only (the wrists are then not used)
    given = bool(job.inputs.get("boxes"))
    # given 「人物框」 are solved as given (upstream's process_one_image takes bboxes); YOLO-Pose then runs only for
    # the wrists (hand refinement). Without them it detects, as upstream's demo does: boxes and wrists in one pass
    detect = full or not given
    # upstream's TensorRT engines this job uses, built first when missing (before any model of the job is loaded)
    engines = trt_engines.engines(run, Path(os.environ["LAB2SHOT_FAST_SAM_3D_BODY_TRT_DIR"]), job.weights_dir,
                                  (trt_engines.BACKBONE,))
    found = detect_people(run, device) if detect else {}
    selected = family.given_people(job, "N-FASTSAM3DBODY-NOBODYCHOSEN") if given \
        else family.people_tracks({frame: boxes for frame, (boxes, _) in found.items()})
    fov = []
    cam_int, focal_px, focal_source = family.shot_camera(job, lambda: measure_focal(job, device, fov))
    for model in fov:
        offload(model)  # off the GPU while Fast SAM 3D Body works
    wrists = found if full else {}
    # a batch is at most family.PEOPLE_BATCH people (infer() splits a frame's people the same way): the warm-up batch
    # sizes, read only when USE_COMPILE=1 is set in the environment
    counts = sorted({min(sum(frame in t.boxes for _, t in selected), family.PEOPLE_BATCH) for frame, _ in job.frames} - {0})
    model, estimator, people = load_estimator(run, device, counts or [1], engines.get(trt_engines.BACKBONE))
    per_person = infer(run, estimator, people, selected, wrists, cam_int, full)
    head = model.head_pose
    compiled = getattr(head, "_compiled", False)
    head._compiled = False  # rig evaluation: plain MHR (batches of 64 frames, not the per-frame compiled graph)
    try:
        family.write_people(job, per_person, head, focal_px, focal_source)
    finally:
        head._compiled = compiled  # the model stays loaded for the next job's inference


def main(job_path: str) -> None:
    nodes = {"fast_sam_3d_body.solve": node_solve}
    run = Run.start(job_path, tuple(nodes), "Fast SAM 3D Body")
    # an allocation beyond the free memory fails instead of spilling into RAM (WSL/Windows): before the focal estimator
    # too, which shot_camera loads on its own (run.model would set the cap only at the first model it loads)
    run.gpu_cap_mb = limit_gpu_memory()
    local_hub({"dinov3": os.environ["LAB2SHOT_DINOV3_DIR"]}, "Fast SAM 3D Body")  # the pinned local DINOv3 code
    nodes[run.node](run, torch.device("cuda"))


if __name__ == "__main__":
    serve(main)
