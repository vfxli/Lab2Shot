"""Two-stage Transformer worker: the in-betweening contract (lab2shot_worker.rig_motion) on the LaFAN1 skeleton.
Runs inside third_party/two_stage_transformer/.venv with the pinned repo's packages/ on PYTHONPATH; never imports
Lab2Shot core.

    python worker.py <job.json>        (node "two_stage_transformer.inbetween")

The rig's keys are turned onto LaFAN1's skeleton (joints, bone offsets and a standing reference pose from the LaFAN1
dataset the extension installs), then each gap between two keys is one call of upstream's full method (Context
Transformer, then Detail Transformer: train/detail_model.evaluate) at 30 fps, set up as its eval script does: 10
context frames ending on the first key, the transition, the next key as the target and one frame after it. The context
of a later gap is what the earlier gaps produced; the first gap's context is the straight path from the first key
towards the second, continued backwards (the model needs motion leading into the key, an animator gives only the key).
The frame after a target is one step along the straight path to the key after it (or on past the last key).
Upstream takes at most 65 frames per call: two keys at most 54 frames (1.8 s) apart.

Parameters (job["params"]):
    post_process   bool   upstream's curve offset (anim_post_process) on the Detail Transformer's output
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import fail, progress, resident, serve
from lab2shot_worker import rig_motion as ib
from lab2shot_worker.run import Run
from lab2shot_shared import motion as mo

NODE = "two_stage_transformer.inbetween"
EXT = "two_stage_transformer"
MODEL_FPS = 30  # LaFAN1: the model's own frame rate
CONTEXT = 10  # context frames before a transition (upstream's train.context_len)
MAX_WINDOW = 65  # upstream's max_seq_len: context + transition + target + one frame after it
MAX_GAP = MAX_WINDOW - CONTEXT - 1  # frames from one key to the next
REFERENCE = ("walk2_subject4.bvh", 6873)  # a LaFAN1 frame where the actor stands still, arms down, looking ahead
MODELS = {"context": ("lafan1_context_model", "train_stats_context.pkl"),
          "detail": ("lafan1_detail_model", "train_stats_detail.pkl")}


@resident
def load_skeleton(bvh: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """LaFAN1's joints (names, parents), bone offsets [J,3] (cm) and the reference pose (joint-to-world [J,4,4])."""
    from motion_inbetween.data import bvh as bvh_io
    from motion_inbetween.data import utils_np

    anim = bvh_io.load_bvh(str(bvh), start=REFERENCE[1], end=REFERENCE[1])
    rot, pos = utils_np.fk(anim.rotations, anim.positions, anim.parents)
    rest = np.repeat(np.eye(4)[None], len(anim.names), 0)
    rest[:, :3, :3], rest[:, :3, 3] = rot[0], pos[0]
    return list(anim.names), np.asarray(anim.parents), np.asarray(anim.offsets, np.float64), rest


@resident
def load_models(weights: Path, configs: Path, device: torch.device):
    """Upstream's Context and Detail Transformers (the repo's configs, the released checkpoints) and the training
    statistics they were normalised with."""
    from motion_inbetween.model import ContextTransformer, DetailTransformer

    out = {}
    for kind, cls in (("context", ContextTransformer), ("detail", DetailTransformer)):
        name, stats_file = MODELS[kind]
        config = json.loads((configs / f"{name}.json").read_text(encoding="utf-8"))
        folder = weights / "experiments" / f"{name}_release"
        model = cls(config["model"]).to(device).eval()
        model.load_state_dict(torch.load(folder / f"checkpoint_{name}_release.pth", map_location=device, weights_only=False)["model"])
        with open(folder / stats_file, "rb") as fh:
            stats = pickle.load(fh)
        out[kind] = (model, config, stats)
    return out


def tensor(x, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def fill_gap(models, positions: np.ndarray, rotations: np.ndarray, transition: int, post_process: bool,
             device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One upstream call: positions [W,J,3] (root position, the other joints' offsets), local rotations [W,J,3,3] with
    the context, the target and the frame after it filled in (W = CONTEXT + transition + 2). Returns the window's
    local rotations, positions and foot contacts [W,4] (the transition frames are the model's)."""
    from motion_inbetween.data import utils_torch as data_utils
    from motion_inbetween.train import context_model as ctx_mdl
    from motion_inbetween.train import detail_model as det_mdl

    (ctx, ctx_config, ctx_stats), (det, det_config, det_stats) = models["context"], models["detail"]
    window = CONTEXT + transition + 2
    target = CONTEXT + transition
    pos, rot, pos_offset, rot_offset = data_utils.to_start_centered_data(
        tensor(positions[None], device), tensor(rotations[None], device), CONTEXT, return_offset=True)
    pos_new, rot_new, contacts = det_mdl.evaluate(
        det, ctx, pos, rot, torch.zeros((1, window, 4), device=device), slice(CONTEXT, target), det_config["indices"],
        tensor(ctx_stats["mean"], device), tensor(ctx_stats["std"], device),
        tensor(det_stats["state"]["mean"], device), tensor(det_stats["state"]["std"], device),
        det_mdl.get_attention_mask(window, target, device), ctx_mdl.get_attention_mask(window, CONTEXT, target, device),
        post_process)
    pos_new, rot_new = data_utils.reverse_root_pos_rot_offset(pos_new, rot_new, pos_offset, rot_offset)
    return rot_new[0].cpu().numpy().astype(np.float64), pos_new[0].cpu().numpy().astype(np.float64), contacts[0].cpu().numpy()


def main(job_path: str) -> None:
    # the model runs on the CPU too (measured fine at these sizes): no CUDA check, and no GPU bookkeeping without one
    run = Run.start(job_path, NODE, "Two-stage Transformer", gpu=torch.cuda.is_available())
    job = run.job
    weights = job.weights_dir
    bvh = weights / "lafan1" / REFERENCE[0]
    run.weights(bvh, *(weights / "experiments" / f"{name}_release" for name, _ in MODELS.values()))
    device = torch.device("cuda" if run.gpu else "cpu")

    run.stage("对齐骨骼")
    motion = ib.read_job(job.inputs["motion"])
    names, parents, offsets, rest = load_skeleton(bvh)
    try:
        retarget = motion.retarget(mo.Skeleton(names, parents, rest))
    except ValueError as exc:
        fail("E-TWOSTAGETRANSFORMER-SKELETON", reason=str(exc))
    world, root = retarget.to_model(motion.poses)
    keys = motion.model_frames(MODEL_FPS)
    gaps = np.diff(keys)
    if (gaps < 1).any():
        k = int(np.argmax(gaps < 1))
        fail("E-TWOSTAGETRANSFORMER-KEYSTOOCLOSE", first=int(motion.keys[k]), second=int(motion.keys[k + 1]))
    if (gaps > MAX_GAP).any():
        k = int(np.argmax(gaps > MAX_GAP))
        fail("E-TWOSTAGETRANSFORMER-KEYGAP", first=int(motion.keys[k]), second=int(motion.keys[k + 1]), seconds=float(gaps[k] / MODEL_FPS),
             max_seconds=MAX_GAP / MODEL_FPS, max_frames=MAX_GAP)

    # the timeline at 30 fps, CONTEXT frames before the first key: local rotations (quaternions) and the root
    total = int(keys[-1]) + 1
    quat = np.zeros((CONTEXT + total, len(names), 4))
    pos = np.zeros((CONTEXT + total, 3))
    contacts = np.zeros((total, 4), np.float32)
    key_quat = mo.continuous(mo.matrix_to_quat(mo.local_from_world(world, parents)))
    back = -np.arange(CONTEXT, 0, -1) / gaps[0]  # the first key's context: its path towards the second key, backwards
    quat[:CONTEXT] = mo.slerp(key_quat[0][None], key_quat[1][None], back[:, None])
    pos[:CONTEXT] = root[0] + (root[1] - root[0]) * back[:, None]
    quat[CONTEXT + keys], pos[CONTEXT + keys] = key_quat, root

    models = run.model("Two-stage Transformer 模型", load_models, weights, job.repo_dir / "configs", device)
    run.stage("Two-stage Transformer 动作补帧")
    post = bool(job.params["post_process"])
    for i in range(len(keys) - 1):
        s, e = int(keys[i]), int(keys[i + 1])
        if e - s > 1:
            if i + 2 < len(keys):  # one step towards the key after the target
                u = 1.0 / (keys[i + 2] - e)
                after_q, after_p = mo.slerp(key_quat[i + 1], key_quat[i + 2], u), root[i + 1] + (root[i + 2] - root[i + 1]) * u
            else:  # past the last key, as it was approached
                u = 1.0 + 1.0 / (e - s)
                after_q, after_p = mo.slerp(key_quat[i], key_quat[i + 1], u), root[i] + (root[i + 1] - root[i]) * u
            window = np.arange(s + 1, CONTEXT + e + 1)  # timeline rows of model frames s - CONTEXT + 1 ... e
            q = np.concatenate([quat[window], after_q[None]])
            p = np.concatenate([pos[window], after_p[None]])
            q[CONTEXT:CONTEXT + e - s - 1] = [1.0, 0.0, 0.0, 0.0]  # upstream's placeholders for the frames to fill
            p[CONTEXT:CONTEXT + e - s - 1] = 0.0
            positions = np.repeat(offsets[None], len(q), 0)
            positions[:, 0] = p
            rot, filled, contact = fill_gap(models, positions, mo.quat_to_matrix(q), e - s - 1, post, device)
            gap = slice(CONTEXT, CONTEXT + e - s - 1)
            quat[CONTEXT + s + 1:CONTEXT + e] = mo.matrix_to_quat(rot[gap])
            pos[CONTEXT + s + 1:CONTEXT + e] = filled[gap, 0]
            contacts[s + 1:e] = contact[gap]
        progress(i + 1, len(keys) - 1, "补帧")
    for k in keys:  # a key's contact: as the frames next to it
        contacts[k] = contacts[max(k - 1, 0)] if k > 0 else contacts[min(k + 1, total - 1)]

    rotations = mo.world_from_local(mo.quat_to_matrix(quat[CONTEXT:]), parents)
    ib.write_result(run, retarget, MODEL_FPS, keys, rotations, pos[CONTEXT:], contacts, method="Two-stage Transformer",
                    skeleton="LaFAN1, 22 joints", reference=f"{REFERENCE[0]} frame {REFERENCE[1]}",
                    post_process=post)


if __name__ == "__main__":
    serve(main)
