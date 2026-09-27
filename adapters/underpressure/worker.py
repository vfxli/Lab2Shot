"""UnderPressure worker: the rig-and-model contract (lab2shot_worker.rig_motion) with UnderPressure's foot-contact
network and its footskate cleanup. Runs inside third_party/underpressure/.venv with the pinned repo on PYTHONPATH;
never imports Lab2Shot core.

    python worker.py <job.json>        (node "underpressure.footskate")

The rig comes in as joint-to-world transforms on its own frames (cm, Y up, the shot's frame rate). This worker:

  1. retargets it onto UnderPressure's own 23-joint skeleton (the contract's Retarget: rest poses aligned, leg
     lengths scaled); SKELETON below is that skeleton's rest pose, read once from the repository's own
     footskate_samples (all twelve share it exactly);
  2. resamples to the rate the network was trained at, and only that one (feeding 24 frames a second straight in
     reads as a near-static pose and labels almost everything a contact: measured 69 % agreement with the
     resampled result, a silently wrong result rather than an error);
  3. converts cm / Y up to the project's own metres / Z up, builds the global unit quaternions (w, x, y, z) its FK
     expects, and runs the network for the per-cell vertical ground reaction forces and the foot contacts;
  4. runs upstream's `footskate.Cleaner` (its own optimisation-based IK, at its published weights) and sends the
     cleaned motion back on the contract's timeline, with the contacts as the contract's [T,4];
  5. writes the network's own per-cell vertical ground reaction forces (demo.py:22 `model.vGRFs(positions)`,
     [T, 2 feet, 16 insole cells], in body weights) to raw/vgrfs.npz, already resampled onto the frames the node
     sent, for the node's 「足底力」 output.

No 「脚滑」 curve is output: upstream has no such quantity (demo.py produces only contacts and vGRFs).

Runs on the CPU, not the GPU (measured: a 500-frame cleanup takes 2.4 s on the CPU and 2.7 s on an RTX 5090, and the
detection of 12 000 frames takes 0.17 s on the CPU; the model is four convolutions). Six threads: the same cleanup takes
2.4 s with six and 22 s with the machine's thirty-two.

Parameters: contact_margin (how many frames a contact is widened by on each side).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")  # 9x faster than letting torch take all 32 (measured)

import numpy as np
import torch

from lab2shot_worker import fail, say, serve
from lab2shot_worker import rig_motion as rm
from lab2shot_worker.run import Run
from lab2shot_shared import motion as mo
from lab2shot_shared.units import M_TO_CM

NODE = "underpressure.footskate"
MODEL_FPS = 100.0  # data.FRAMERATE: the only rate this network was trained at
ITERATIONS = 100  # footskate.Cleaner's published setting (demo.py). Not a parameter: measured over the twelve
# shipped samples, 50 / 100 / 200 iterations reduce sliding by 30 / 31 / 27 % on average with no useful ordering;
# the cleanup's own weights pull the pose back towards the original, so more iterations do not clean more (docs.md).
WEIGHTS = dict(qweight=1e-3, tweight=1e2, cweight=1e-5, fweight=5e-5)  # demo.py's, all published
BODY_WEIGHT_RANGE = (0.6, 1.5)  # a clip's mean total vGRF, in body weights: outside this the input is not human-scale
GROUND_TOLERANCE_M = 0.02  # how far off the floor a take may sit before the node says it put it down first

# UnderPressure's own rest skeleton (data.TOPOLOGY order, metres, Z up, global offsets from the pelvis). It is the
# subject of the repository's footskate_samples, and all twelve of them carry bit-identical copies of it; the
# database it was captured with is not needed and is never downloaded. Stature implied: about 1.70 m.
SKELETON_M = np.array([
    (0.0, 0.0, 0.0),                                 # pelvis
    (-0.011099, 0.0, 0.099887073),                   # spine_1
    (-0.011099, 0.0, 0.211157978),                   # spine_2
    (-0.011026, 0.0, 0.312735021),                   # spine_3
    (-0.010953, 0.0, 0.414197981),                   # spine_4
    (-0.010953, 0.0, 0.555684030),                   # neck
    (-0.011110, 0.0, 0.650126994),                   # head
    (-0.010953, -0.030161001, 0.493465006),          # right_clavicle
    (-0.010953, -0.169781998, 0.493465006),          # right_shoulder
    (-0.010953, -0.465777010, 0.493465006),          # right_elbow
    (-0.010915, -0.708707988, 0.493465006),          # right_wrist
    (-0.010953, 0.030161001, 0.493465006),           # left_clavicle
    (-0.010953, 0.169781998, 0.493465006),           # left_shoulder
    (-0.010953, 0.465777010, 0.493465006),           # left_elbow
    (-0.010915, 0.708707988, 0.493465006),           # left_wrist
    (0.000019, -0.081832998, -0.000167966),          # right_hip
    (0.000036, -0.081832998, -0.428551972),          # right_knee
    (0.000065, -0.081832998, -0.846672952),          # right_ankle
    (0.168822005, -0.081832998, -0.921157956),       # right_foot (the ball of the foot)
    (0.000021, 0.081832998, -0.000189960),           # left_hip
    (0.000038, 0.081832998, -0.428574979),           # left_knee
    (0.000067, 0.081832998, -0.846694946),           # left_ankle
    (0.168824002, 0.081832998, -0.921180964),        # left_foot
], np.float64)

# cm / Y up (Lab2Shot) <-> m / Z up (UnderPressure). One rotation, its own inverse up to the sign of the swap.
Y_TO_Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # (x, y, z)_Yup -> (x, -z, y)_Zup
Z_TO_Y = Y_TO_Z.T


def _on_path(folder: Path) -> None:
    """The pinned repository goes on sys.path: its modules import each other by bare name (import anim, data)."""
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def load_model(repo: Path):
    """The contact network (four convolutions) with its published weights, from the repository's own pretrained.tar."""
    import models  # noqa: PLC0415 (the pinned repo, on sys.path)

    state = torch.load(repo / "pretrained.tar", map_location="cpu")["model"]
    return models.DeepNetwork(state_dict=state).eval()


def model_skeleton() -> mo.Skeleton:
    """UnderPressure's skeleton as the contract's retargeting sees it: cm, Y up, rest rotations the identity."""
    from data import TOPOLOGY  # noqa: PLC0415 (the pinned repo, on sys.path)

    names = list(TOPOLOGY.joints())
    parents = np.array([TOPOLOGY.parent(j, index=True) for j in range(len(names))], np.int64)
    rest = np.repeat(np.eye(4)[None], len(names), 0)
    rest[:, :3, 3] = SKELETON_M @ Z_TO_Y.T * M_TO_CM
    return mo.Skeleton(names, parents, rest)


def to_underpressure(world: np.ndarray, root: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The model's world rotations [T,23,3,3] and root position [T,3] (cm, Y up) as what its FK reads:
    global unit quaternions (w, x, y, z), the rest skeleton, and the pelvis trajectory (metres, Z up)."""
    turn = Y_TO_Z @ np.asarray(world, np.float64) @ Z_TO_Y  # the same rotation, expressed in the project's axes
    quats = mo.matrix_to_quat(turn)  # (w, x, y, z), which is util.SU2's own layout
    skeleton = torch.as_tensor(SKELETON_M, dtype=torch.float32)[None]
    trajectory = torch.as_tensor(np.asarray(root, np.float64) @ Y_TO_Z.T / M_TO_CM, dtype=torch.float32)[:, None]
    return torch.as_tensor(quats, dtype=torch.float32), skeleton, trajectory


def from_underpressure(angles: torch.Tensor, trajectory: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Back the other way: (world rotations [T,23,3,3], root position [T,3]) in cm, Y up."""
    turn = mo.quat_to_matrix(angles.detach().cpu().numpy().astype(np.float64))
    world = Z_TO_Y @ turn @ Y_TO_Z
    root = trajectory.detach().cpu().numpy().astype(np.float64)[:, 0] @ Z_TO_Y.T * M_TO_CM
    return world, root


def ground_offset(positions: torch.Tensor) -> float:
    """How far the take sits off the floor, in metres: the lowest the balls of the feet ever get.

    The cleanup pins a contact to z = 0 (footskate.Cleaner's contact_locations: the target's height is zero),
    so a take whose floor is elsewhere is pulled there. Measured on real mocap sitting 24 cm low, the whole
    body moved by up to 35 cm and the feet slid more afterwards, not less. Upstream's own captures rest with the
    toe joint at 0 (±6 mm), so this puts the take where the model expects it, and the node shifts it back
    afterwards: the delivered animation stands exactly where the artist's did."""
    from data import TOPOLOGY  # noqa: PLC0415

    toes = [TOPOLOGY.index(n) for n in ("left_foot", "right_foot")]
    return float(positions[:, toes, 2].min())


def write_vgrfs(job, vgrfs: torch.Tensor, timeline: np.ndarray, at: np.ndarray) -> None:
    """The network's own vGRFs (demo.py:22 `model.vGRFs(positions)`): [T, 2 (left, right), 16 insole cells], a share
    of body weight per cell, on the model's timeline. Written onto the frames the node sent (the node only has to
    read it and name the curves), so the node never has to know the model's frame rate."""
    flat = np.asarray(vgrfs.detach().cpu().numpy(), np.float64).reshape(len(timeline), -1)
    job.raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(job.raw_dir / "vgrfs.npz", values=mo.resample_values(timeline, flat, at).astype(np.float32))


def contacts_four(contacts: torch.Tensor) -> np.ndarray:
    """data.Contacts.from_forces gives [T, 2 (front, back), 2 (left, right)]; the contract wants
    [T,4] = left heel, left toe, right heel, right toe (front = the ball of the foot, back = the ankle)."""
    c = contacts.detach().cpu().numpy().astype(np.float32)
    return np.stack([c[:, 1, 0], c[:, 0, 0], c[:, 1, 1], c[:, 0, 1]], axis=1)


def segments(on: np.ndarray) -> int:
    return int(np.count_nonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])) > 0))


def cleanup(run: Run) -> None:
    job = run.job
    _on_path(job.repo_dir)
    import data as up_data  # noqa: PLC0415
    from footskate import Cleaner  # noqa: PLC0415

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    motion = rm.read_job(job.inputs["motion"])
    run.stage("对齐骨骼")
    try:
        retarget = motion.retarget(model_skeleton())
    except ValueError as exc:
        fail("E-UNDERPRESSURE-SKELETON", reason=str(exc))
    rotations, root = retarget.to_model(motion.poses)

    # the network's own frame rate, whatever the shot's is
    at = motion.model_times(MODEL_FPS)
    t = np.arange(int(np.floor(at[-1])) + 1, dtype=np.float64)
    world = mo.resample_rotations(at, rotations, t)
    root_t = mo.resample_values(at, root, t)
    if abs(motion.fps - MODEL_FPS) > 1e-6:
        say("N-UNDERPRESSURE-RESAMPLED", fps=motion.fps, frames=len(t))

    model = run.model("UnderPressure 模型", load_model, job.repo_dir)

    run.stage("估计脚底支撑力")
    angles, skeleton, trajectory = to_underpressure(world, root_t)
    from anim import FK  # noqa: PLC0415

    positions = FK(angles, skeleton, trajectory, up_data.TOPOLOGY)
    floor = ground_offset(positions)
    if abs(floor) > GROUND_TOLERANCE_M:
        say("N-UNDERPRESSURE-GROUNDED", offset=floor * M_TO_CM)
        trajectory = trajectory.clone()
        trajectory[..., 2] -= floor
        positions = FK(angles, skeleton, trajectory, up_data.TOPOLOGY)
    with torch.no_grad():
        vgrfs = model.vGRFs(positions.unsqueeze(0)).squeeze(0)
    # the network's output is a fraction of body weight per insole cell; over a whole take the total must average
    # about one body weight. Far off means the skeleton that reached it is not human-scale (a wrong joint mapping,
    # a rig in the wrong units), and the forces and therefore the contacts would be silently meaningless.
    weight = float(vgrfs.sum(dim=(-1, -2)).mean())
    if not BODY_WEIGHT_RANGE[0] <= weight <= BODY_WEIGHT_RANGE[1]:
        say("W-UNDERPRESSURE-WEIGHT", weight=weight, port="character")
    contacts = contacts_four(up_data.Contacts.from_forces(vgrfs))
    write_vgrfs(job, vgrfs, t, at)  # upstream vGRFs, read by the node's 「足底力」 output
    frames_on = contacts.max(axis=1) > 0.5
    if not frames_on.any():
        say("N-UNDERPRESSURE-NOCONTACT", port="character")
        rm.write_result(run, retarget, MODEL_FPS, at, world, root_t, contacts, method="UnderPressure",
                        cleaned=False, body_weight=round(weight, 3))
        return
    say("I-UNDERPRESSURE-CONTACT", frames=int(frames_on.sum()), total=len(t),
        left=segments(contacts[:, 0] > 0.5), right=segments(contacts[:, 2] > 0.5))

    run.stage("解接触约束，去掉脚滑")
    # Cleaner wraps the trajectory it is handed in a torch.nn.Parameter and lets Adam step it: on the CPU that is
    # the caller's own tensor, so it must get copies (measured: 3.45 cm of drift into the input after 20 steps).
    cleaner = Cleaner(model, iterations=ITERATIONS, margin=int(job.params["contact_margin"]), device="cpu", **WEIGHTS)
    clean_angles, _, clean_trajectory = cleaner(angles.clone(), skeleton.clone(), trajectory.clone())
    clean_trajectory = clean_trajectory.clone()
    clean_trajectory[..., 2] += floor  # back where the artist's take stood
    clean_world, clean_root = from_underpressure(clean_angles, clean_trajectory)

    moved = float(np.linalg.norm(clean_root - root_t, axis=1).mean())
    rm.write_result(run, retarget, MODEL_FPS, at, clean_world, clean_root, contacts, method="UnderPressure",
                    cleaned=True, iterations=ITERATIONS, margin=int(job.params["contact_margin"]),
                    contact_frames=int(frames_on.sum()), body_weight=round(weight, 3), root_moved_cm=round(moved, 2),
                    ground_offset_cm=round(floor * M_TO_CM, 1),
                    model_frames=len(t))


def main(job_path: str) -> None:
    cleanup(Run.start(job_path, NODE, "UnderPressure", gpu=False))  # CPU, not GPU (the module doc says why)


if __name__ == "__main__":
    serve(main)
