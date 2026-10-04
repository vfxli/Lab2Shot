"""SAM 3D Body worker. Runs inside third_party/sam_3d_body/.venv with the
original repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>

Node ViTDet 人物框 (sam_3d_body.detect_people): ViTDet on every frame -> IoU tracks -> raw/boxes.json.
Node SAM 3D Body 全身动作 (sam_3d_body.solve): the wired 人物框, or without them the same ViTDet pass inside the
solve (upstream detects the people itself: demo.py builds a HumanDetector and hands process_one_image no boxes at
all) -> one focal length for the shot, or a wired per-frame one -> SAM 3D Body per frame with that focal -> per
person: lock body shape, smooth the motion, re-evaluate the MHR rig (sam3dbody.py) -> raw npz files.

Both nodes use the same detect_people() and the same @resident load_detector(), so ViTDet is loaded only once
(when both nodes run in one process, the second receives the same model instance).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import MemoryBound, limit_gpu_memory, local_hub, offload, quiet, read_frame, resident, serve
from lab2shot_worker import tracking
from lab2shot_worker.run import Run

import sam3dbody as family  # this adapter's own module (Extension.worker_modules)

# Default of upstream demo.py `--bbox_thresh` (third_party/sam_3d_body/repo/demo.py:177-182), used when solve
# detects people itself. The 「检测阈值」 parameter of the 「ViTDet 人物框」 node has the same default.
DEMO_BBOX_THRESH = 0.8


@resident
def load_fov_estimator(path: Path, device: torch.device):
    from tools.build_fov_estimator import FOVEstimator

    return FOVEstimator(name="moge2", device=device, path=str(path))


@resident
def load_detector(path: Path, device: torch.device):
    from tools.build_detector import HumanDetector

    return HumanDetector(name="vitdet", device=device, path=str(path))


@resident
def load_body(weights: Path, device: torch.device):
    """SAM 3D Body and its config (the estimator around them is made per job: it keeps the last image's results)."""
    from sam_3d_body import load_sam_3d_body

    return load_sam_3d_body(str(weights / "model.ckpt"), device=device, mhr_path=str(weights / "assets" / "mhr_model.pt"))


def measure_focal(job, device, loaded: list):
    """rgb -> focal px: upstream's MoGe-2 field-of-view estimator (appended to `loaded`)."""
    fov = load_fov_estimator(job.weights_dir / "moge-2-vitl-normal" / "model.pt", device)
    loaded.append(fov)
    return lambda rgb: float(fov.get_cam_intrinsics(rgb)[0, 0, 0])


def detect_people(run: Run, detector, thresh: float) -> dict[int, np.ndarray]:
    run.stage("detect_people")
    detections = {}
    for n, (frame, path) in run.each(run.job.frames, ""):
        boxes = detector.run_human_detection(
            read_frame(path, order="bgr"), det_cat_id=0, bbox_thr=thresh, nms_thr=0.3, default_to_full_image=False
        )
        detections[frame] = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return detections


def infer(run: Run, estimator, selected, cam_int, inference_type: str) -> dict[int, dict[int, dict]]:
    """person id -> frame -> SAM 3D Body output. `cam_int`: a [1,3,3] tensor or frame -> tensor."""
    run.stage("estimate_pose")
    per_person: dict[int, dict[int, dict]] = {pid: {} for pid, _ in selected}
    batch = [family.PEOPLE_BATCH]  # people per batch; once lowered for lack of memory it stays lowered (so later frames do not fail first)
    with family.keep_gpu_cache():
        for n, (frame, path) in run.each(run.job.frames, ""):
            infer_frame(run, estimator, selected, cam_int, inference_type, per_person, batch, frame, path)
    return per_person


def infer_frame(run: Run, estimator, selected, cam_int, inference_type: str, per_person, batch, frame, path) -> None:
    """One frame of infer(). The picture is read once (RGB, as upstream's real-time path hands it over in memory,
    run_publisher.py:300-306) and shared by every batch of the frame, instead of process_one_image reading the file
    again for each batch."""
    present = [(pid, t.boxes[frame]) for pid, t in selected if frame in t.boxes]
    if present:
        image = read_frame(path)  # RGB: process_one_image takes an array as RGB (estimator:102-104)

        # every person in the frame is solved, in batches (sam3dbody.py batches); a batch that does not fit
        # in GPU memory is retried on this frame with the next smaller size
        def solve(size, frame=frame, image=image, present=present):
            batch[0] = size
            pairs = []
            for chunk in family.batches(present, size):
                with quiet():  # its per-call prints ("make sure the input image is in RGB format")
                    outs = estimator.process_one_image(
                        image,
                        bboxes=np.stack([b for _, b in chunk]),
                        cam_int=cam_int(frame) if callable(cam_int) else cam_int,
                        inference_type=inference_type,
                    )
                pairs.extend(zip([pid for pid, _ in chunk], outs))
            return pairs
        for pid, out in run.fit(MemoryBound.batch(family.BATCH_STEPS), solve, batch[0]):
            per_person[pid][frame] = out


def node_detect_people(run: Run, device) -> None:
    """Node ViTDet 人物框: every person, one id each, a box per frame -> raw/boxes.json."""
    job = run.job
    detector = run.model("load_model", load_detector, job.weights_dir / "vitdet", device, stage_params={"model": "ViTDet"})
    tracks = tracking.track_boxes(detect_people(run, detector, job.params["threshold"]))
    people = [
        {"id": i, "prominence": t.prominence, "boxes": {str(f): [round(float(v), 2) for v in b] for f, b in sorted(t.boxes.items())}}
        for i, t in enumerate(tracks, start=1)
    ]
    (job.raw_dir / "boxes.json").write_text(json.dumps({"people": people}), encoding="utf-8")
    run.finish([f for f, _ in job.frames], kind="boxes", people=len(people))


def node_solve(run: Run, device) -> None:
    """Node SAM 3D Body 全身动作: plate in, human motion out. 「人物框」 is an optional input passed directly to the
    upstream function (process_one_image accepts bboxes): when wired, people are solved from those boxes and no
    detection runs. Otherwise this follows the upstream entry point demo.py, which takes `--image_folder` and builds
    its own HumanDetector (vitdet by default); it uses the same detect_people() and load_detector() as the
    「ViTDet 人物框」 node (same @resident cache key, so the detector is not loaded twice).
    No camera input or output: only a Focal Length (the node's 「已知 Focal Length」 parameter, or a per-frame wired value);
    results stay in camera space."""
    from sam_3d_body import SAM3DBodyEstimator

    job = run.job
    if job.inputs.get("boxes"):
        # 人物框 wired: solve from those boxes without detection (upstream process_one_image accepts bboxes)
        selected = family.given_people(job, "N-SAM3DBODY-NOBODYCHOSEN")
    else:
        detector = run.model("load_model", load_detector, job.weights_dir / "vitdet", device, stage_params={"model": "ViTDet"})
        detections = detect_people(run, detector, DEMO_BBOX_THRESH)
        offload(detector)  # off the GPU after detection, leaving the memory to SAM 3D Body
        selected = family.people_tracks(detections)
    fov = []
    cam_int, focal_px, focal_source = family.shot_camera(job, lambda: measure_focal(job, device, fov))
    for model in fov:
        offload(model)  # off the GPU while SAM 3D Body works
    model, cfg = run.model("load_model", load_body, job.weights_dir / "sam-3d-body-dinov3", device,
                           stage_params={"model": "SAM 3D Body"})
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=None)
    per_person = infer(run, estimator, selected, cam_int, "full" if job.params["hand_refine"] else "body")
    family.write_people(job, per_person, model.head_pose, focal_px, focal_source)


def main(job_path: str) -> None:
    nodes = {"sam_3d_body.detect_people": node_detect_people, "sam_3d_body.solve": node_solve}
    run = Run.start(job_path, tuple(nodes), "SAM 3D Body")
    # an allocation beyond the free memory fails instead of spilling into RAM (WSL/Windows): before the focal estimator
    # too, which shot_camera loads on its own (run.model would set the cap only at the first model it loads)
    run.gpu_cap_mb = limit_gpu_memory()
    # upstream builds the backbone from torch.hub's moving dinov3 main branch: the pinned local copy instead
    local_hub({"dinov3": os.environ["LAB2SHOT_DINOV3_DIR"]}, "SAM 3D Body")
    nodes[run.node](run, torch.device("cuda"))


if __name__ == "__main__":
    serve(main)
