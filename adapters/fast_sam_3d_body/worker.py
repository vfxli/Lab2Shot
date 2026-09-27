"""Fast SAM 3D Body worker. Runs inside third_party/fast_sam_3d_body/.venv with
the Fast SAM 3D Body repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>

Same job and raw output as the sam_3d_body worker's solve node (the solve, rig and
converter are shared: sam3dbody.py of the sam_3d_body extension, which this one requires): this repo's own
YOLO11-Pose detector finds the people (upstream detects them itself; boxes are not an input) -> one focal length
for the shot -> Fast SAM 3D Body per frame, every person of the frame in one batch, hand crops placed from the
same YOLO11-Pose pass's wrists so body and hands run through the backbone together -> per person: lock shape,
smooth, re-evaluate the MHR rig -> raw npz files.

一次 YOLO11-Pose 就同时给出框和手腕，所以检测器只加载一次、画面只过一遍。
"""

from __future__ import annotations

import os
from pathlib import Path

# Upstream's optimized settings (run_demo.sh at the pinned commit), minus TensorRT
# (engines are built per GPU; not installed) and minus torch.compile: on the 4090
# compiling saved ~0.02 s per frame (0.168 -> 0.145-0.153 s) but cost 60-90 s per
# run even with a warm cache, so it only pays off beyond ~3000 frames. Read by
# the repo at import time; the environment may override any of them
# (USE_COMPILE=1 USE_COMPILE_BACKBONE=1 restores upstream's compiled path).
FAST_SETTINGS = {
    "IMG_SIZE": "512",  # upstream: smaller is faster but loses accuracy
    "LAYER_DTYPE": "fp32",  # decoder precision; upstream says multi-person needs fp32
    "GPU_HAND_PREP": "1",  # hand crops cut on the GPU
    "PARALLEL_DECODERS": "1",  # body + hand crops through the backbone in one batch
    "SKIP_KEYPOINT_PROMPT": "1",  # no second keypoint-prompted decoder pass
    "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
    "BODY_INTERM_PRED_LAYERS": "0,1,2",  # pruned intermediate predictions
    "HAND_INTERM_PRED_LAYERS": "0,1",
    "MHR_NO_CORRECTIVES": "1",  # only inside the network; the exported rig keeps correctives
    "MHR_USE_CUDA_GRAPH": "0",
    "USE_COMPILE": "0",
    "USE_COMPILE_BACKBONE": "0",
    "DECODER_COMPILE": "1",  # only with USE_COMPILE=1
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
    """frame -> (boxes [N,4], COCO keypoints [N,17,3]) from YOLO11-Pose: 这一帧有哪些人，他们的手腕在哪。

    这是没接「人物框」时的路（「人物框」是可选输入：上游 process_one_image 收 bboxes，接了就按框解，
    这里只在还要手腕时跑——见 node_solve）。用的是这个仓库自己的检测器，不是 SAM 3D Body 那套 ViTDet：
      - 这个仓库自己的 demo 把 YOLO11 当默认人物检测器，`yolo_pose` 是它列出来的三档之一
        （third_party/fast_sam_3d_body/repo/demo_human.py:623-629，`--detector` default "yolo"，
        choices ["vitdet", "yolo", "yolo_pose"]），手部裁切也走同一份 YOLO-Pose 的手腕
        （run_publisher.py:305 `hand_box_source="yolo_pose"`，build_detector.py:27-31 的 run_yolo_pose）；
      - 选 `yolo_pose` 而不是 `yolo`：一次检测同时给出框和手腕，检测器只加载一次、画面只过一遍，
        后面 infer() 要的手腕就是这一趟的结果；
      - 装好的权重也只有它（third_party/fast_sam_3d_body/weights/ 下是 yolo/yolo11m-pose.pt，
        没有 ViTDet 那 2.7 GB 的 model_final_f05665.pkl，那份在 sam_3d_body 扩展的权重目录里）。
    """
    detector = run.model("YOLO11-Pose 模型", load_yolo, run.job.weights_dir / "yolo" / "yolo11m-pose.pt", device)
    run.stage("检测人物")
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


def load_estimator(run: Run, device, batch_sizes: list[int]):
    # only read with USE_COMPILE=1 (compile warm-up); a process kept loaded keeps its first job's
    os.environ.setdefault("COMPILE_WARMUP_BATCH_SIZES", ",".join(map(str, batch_sizes)))
    from sam_3d_body import SAM3DBodyEstimator

    model, cfg = run.model("Fast SAM 3D Body 模型", load_body, run.job.weights_dir / "sam-3d-body-dinov3", device)
    people = FramePeople()
    with quiet():
        estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg, human_detector=people)
    return model, estimator, people


def infer(run: Run, estimator, people: FramePeople, selected, wrists, cam_int, full: bool) -> dict[int, dict[int, dict]]:
    """person id -> frame -> Fast SAM 3D Body output; all people of a frame in one batch."""
    run.stage("估计人体姿态")
    per_person: dict[int, dict[int, dict]] = {pid: {} for pid, _ in selected}
    serial = 0
    batch = [family.PEOPLE_BATCH]  # 一批几个人；显存不够降过一档就一直用降过的那档（不然每一帧都先撞一次）
    for n, (frame, path) in run.each(run.job.frames, ""):
        present = [(pid, t.boxes[frame]) for pid, t in selected if frame in t.boxes]
        if present:
            # 一帧里的人分批解，每个人都解（sam3dbody.py batches）：一批装不下显存就换小一档重来这一帧。
            # 每一批把这批人的框和手腕交给 FramePeople（它顶替上游的检测器），上游就按这一批走快路
            def estimate(size, frame=frame, path=path, present=present):
                batch[0] = size
                pairs, slow = [], 0
                for chunk in family.batches(present, size):
                    people.boxes = np.stack([b for _, b in chunk]).astype(np.float32)
                    people.keypoints, wrists_seen = wrists_for(people.boxes, wrists.get(frame))
                    parallel = full and wrists_seen
                    slow += full and not wrists_seen
                    with quiet():
                        outs = estimator.process_one_image(
                            str(path),
                            cam_int=cam_int(frame) if callable(cam_int) else cam_int,
                            inference_type="full" if full else "body",
                            hand_box_source="yolo_pose" if parallel else "body_decoder",
                        )
                    pairs.extend(zip([pid for pid, _ in chunk], outs))
                return pairs, slow

            pairs, slow = run.fit(MemoryBound.batch(family.BATCH_STEPS), estimate, batch[0])
            serial += slow
            for pid, out in pairs:
                per_person[pid][frame] = out
    if serial:
        say("N-FASTSAM3DBODY-SLOWHANDS", frames=int(serial))
    return per_person


def node_solve(run: Run, device) -> None:
    """Node Fast SAM 3D Body 全身动作: same inputs, parameters and raw output as sam_3d_body.solve —
    画面进去，人物动作出来，没接框时由这个仓库自己的 YOLO11-Pose 检出来（见 detect_people）。"""
    job = run.job
    full = bool(job.params["hand_refine"])  # False: body decoder only (the wrists are then not used)
    given = bool(job.inputs.get("boxes"))
    # 接了「人物框」就按框解（上游 process_one_image 收 bboxes）；YOLO-Pose 只在还要手腕（手部精修）时跑。
    # 没接就照官方 demo 的路自己检：一趟同时拿到框和手腕
    found = detect_people(run, device) if (full or not given) else {}
    selected = family.given_people(job, "N-FASTSAM3DBODY-NOBODYCHOSEN") if given \
        else family.people_tracks({frame: boxes for frame, (boxes, _) in found.items()})
    fov = []
    cam_int, focal_px, focal_source = family.shot_camera(job, lambda: measure_focal(job, device, fov))
    for model in fov:
        offload(model)  # off the GPU while Fast SAM 3D Body works
    wrists = found if full else {}
    # a batch is at most family.PEOPLE_BATCH people (infer() splits a frame's people the same way): keeps this bounded
    # too, in case USE_COMPILE is ever turned back on (env override) and actually reads these batch sizes
    counts = sorted({min(sum(frame in t.boxes for _, t in selected), family.PEOPLE_BATCH) for frame, _ in job.frames} - {0})
    model, estimator, people = load_estimator(run, device, counts or [1])
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
