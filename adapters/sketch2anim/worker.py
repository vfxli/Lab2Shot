"""Sketch2Anim worker: a stick figure drawn on a picture becomes a 22-joint body animation.

Runs inside third_party/sketch2anim/.venv with the pinned repository on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>        (node "sketch2anim.motion")

Input (job.inputs["sketch"], written by the node): the joints the artist placed, in image pixels, one row per
drawn figure, plus the model frame of each. All handling of the model's own coordinate system is in this file:

  * 18 drawn joints -> 22: upstream zeroes spine2, spine3 and the two shoulders and rebuilds them from the
    drawn ones (`utils.py convert_kps_joint`, lines 4-16). Upstream does this on the 3D joints before projecting; here
    it is done on the drawn 2D joints. The projection is a rotation followed by dropping z, both linear, and the four
    rebuilt joints are linear combinations of drawn ones, so both orders give the same result.
  * pixels -> metres: the model works in metres with y up and the feet on y = 0 (HumanML3D). The dataset's mean
    pose (datasets/humanml_spatial_norm/Mean_raw.npy) gives the height of a standing body, so the first drawn
    figure's height in pixels is mapped to one body height; its lowest foot defines the ground and its pelvis
    x = 0 (upstream centres the motion the same way, demo_kp_traj_2d.py lines 175-177). Every later figure uses the
    same scale, so the figures' hips form a trajectory.
  * normalisation: identical to how the training data was built (`mld/data/humanml/dataset.py` lines 602-614): only
    x and y are normalised by Mean_raw / Std_raw, the third channel stays 1, and everything not drawn is zero (the
    model's masks use that zero to identify conditioned entries). The released demo normalises all three channels;
    the training code is followed here because it matches what the weights were trained on.
  * the sketch's camera: the 2D sketch is the body seen through an orthographic camera turned by
    Rx(angle_x)·Ry(angle_y) (`utils.py rotate_pose` / `project2D`). Training sampled angle_x in 0..30 and angle_y in
    -45..45 (dataset.py 469-473), so the node only offers angles inside that range.

What goes out (raw/motion.npz): the model's 22 joints turned into a skeleton with rotations by upstream's own IK
(`visualization/joints2bvh.py Joint2BVHConvertor`, the template skeleton it ships), resampled to the frame rate the
node asked for, in centimetres. The node turns that into a USD skeletal animation.

Parameters (job["params"]): prompt, seed, steps, text_guidance, control, foot_lock, plus what the node adds:
angle_x, angle_y (the sketch view), fps and out_frames (the frame range). The length in model frames (20 fps)
comes with the sketch.
"""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import numpy as np

from lab2shot_worker import fail, progress, resident, save_npz, serve
from lab2shot_worker.run import Run
from lab2shot_shared import motion as mo
from lab2shot_shared.units import M_TO_CM
from model_spec import MAX_MODEL_FRAMES as MAX_FRAMES, MODEL_FPS  # nodes.py plans with the same

NODE = "sketch2anim.motion"
EXT = "sketch2anim"
# the 22 body joints the model works in are the first 22 SMPL joints; these four are not drawn, upstream rebuilds
# them from the drawn ones (utils.py convert_kps_joint)
DERIVED = (6, 9, 16, 17)
DRAWN = tuple(j for j in range(22) if j not in DERIVED)  # 18, the order nodes/handles.py FIGURE_JOINTS declares
ROOT, HEAD, FEET = 0, 15, (10, 11)  # pelvis, head, the two toe joints


def _numpy_shim() -> None:
    """upstream visualization/Animation.py imports numpy.core.umath_tests, removed from numpy years ago; its one use
    is matrix_multiply, which is np.matmul. Registered here so the pinned repository is not touched."""
    if "numpy.core.umath_tests" in sys.modules:
        return
    shim = types.ModuleType("numpy.core.umath_tests")
    shim.matrix_multiply = lambda a, b: np.matmul(a, b)
    sys.modules["numpy.core.umath_tests"] = shim


def _enter_repo(repo: Path) -> None:
    """The repository reads datasets/humanml3d/*.npy by relative path, so it runs from its own folder. It is
    appended to sys.path, never prepended: it has a top-level `datasets` folder that would otherwise shadow the
    package of the same name that sentence-transformers imports."""
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    _numpy_shim()


@resident
def load_model(repo: Path, weights: Path, device: str):
    """Upstream's MLD with its ControlNet and pose-aware denoiser, built the way demo_kp_traj_2d.py builds it
    (identify_model_type, lines 100-120) but without the text-to-motion evaluators: they only feed the benchmark
    metrics (mld/models/modeltype/base.py _get_t2m_evaluator) and the generation path never reads them, so the 96
    evaluator tensors in the checkpoint are left where they are and nothing else is missing.

    Steps, text guidance and control strength are not part of the model: MLD reads them at sampling time
    (self.guidance_scale, self.control_scale, self.cfg.model.scheduler.num_inference_timesteps: mld.py:261-322), so
    every job sets them on the kept model (set_sampling) and changing one does not load the model again."""
    import torch
    from omegaconf import OmegaConf

    from mld.config import get_module_config
    from mld.data.humanml.scripts.motion_process import extract_rotations, recover_from_ric
    from mld.models.modeltype.mld import MLD

    mean = np.load("datasets/humanml3d/Mean.npy")
    std = np.load("datasets/humanml3d/Std.npy")
    raw_mean = np.load("datasets/humanml_spatial_norm/Mean_raw.npy")
    raw_std = np.load("datasets/humanml_spatial_norm/Std_raw.npy")

    class Norms:
        """What MLD reads off its data module (mld/models/modeltype/mld.py lines 83-85 and the guidance's
        denorm_spatial). The released code gets these from the HumanML3D loader, which wants the whole dataset just
        to hand over four arrays of numbers; those four arrays ship inside the repository, so this reads them
        straight and nothing else of the dataset is needed."""

        is_mm = False
        njoints = 22

        def feats2joints(self, f):
            return recover_from_ric(f * torch.tensor(std).to(f) + torch.tensor(mean).to(f), 22)

        def norm(self, f):
            return (f - torch.tensor(mean).to(f)) / torch.tensor(std).to(f)

        def feats2rotations(self, f):
            return extract_rotations(f, 22)

        def denorm_spatial(self, h):
            return h * torch.tensor(raw_std).to(h) + torch.tensor(raw_mean).to(h)

    class Model(MLD):
        def _get_t2m_evaluator(self, cfg):  # the benchmark's evaluators: never on the generation path
            pass

    cfg = OmegaConf.load("configs/motionlcm_control.yaml")
    cfg = OmegaConf.merge(cfg, get_module_config(cfg.model, cfg.model.target))
    cfg.model.t5_path = str(weights / "sentence-t5-large")
    cfg.METRIC.TYPE = []  # build no metric: they are the evaluation path, and every one of them wants a dataset
    cfg.DATASET.NFEATS, cfg.DATASET.NJOINTS = 263, 22
    state = torch.load(weights / "checkpoints" / "adapter.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    lcm = "denoiser.time_embedding.cond_proj.weight"
    if lcm in state:
        dim = state[lcm].shape[1]
        cfg.model.denoiser.params.time_cond_proj_dim = dim
        cfg.model.controlnet.params.time_cond_proj_dim = dim
    cfg.model.is_controlnet = "controlnet.controlnet_cond_embedding.0.weight" in state
    model = Model(cfg, Norms()).to(device).eval()
    missing, _ = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("t2m_")]
    if missing:
        fail("E-SKETCH2ANIM-WEIGHTS", count=len(missing), first=missing[0])
    return model, raw_mean.reshape(22, 3), raw_std.reshape(22, 3)


def set_sampling(model, steps: int, guidance: float, control: float) -> None:
    """This job's sampling settings on the kept model, where upstream's __init__ puts them from the config
    (mld.py:63 guidance_scale, :117 control_scale) and where sampling reads the step count (:261-262)."""
    model.cfg.model.guidance_scale, model.guidance_scale = float(guidance), float(guidance)
    model.cfg.model.control_scale = float(control)
    if hasattr(model, "control_scale"):
        model.control_scale = float(control)
    model.cfg.model.scheduler.num_inference_timesteps = int(steps)


def full_pose(drawn: np.ndarray) -> np.ndarray:
    """[K,18,2] drawn joints -> [K,22,2]: upstream's four rebuilt joints (utils.py convert_kps_joint lines 9-15),
    written out on 2D points instead of 3D ones (the projection is linear, so the answer is the same)."""
    out = np.zeros((len(drawn), 22, drawn.shape[-1]), np.float64)
    out[:, list(DRAWN)] = drawn
    out[:, 16] = (out[:, 13] + out[:, 18]) / 2  # left shoulder: between the collar and the elbow
    out[:, 17] = (out[:, 14] + out[:, 19]) / 2  # right shoulder
    delta = (out[:, 12] - out[:, 3]) / 3  # spine2 / spine3: thirds of the way from the waist up to the neck
    out[:, 6] = out[:, 3] + delta
    out[:, 9] = out[:, 3] + delta * 2
    return out


def to_model_world(pixels: np.ndarray, raw_mean: np.ndarray) -> tuple[np.ndarray, dict]:
    """[K,22,2] image pixels (y down) -> the model's 2D world in metres (y up, ground at 0, the first figure's
    pelvis at x = 0). The ruler is the first figure: its head-above-the-feet height is one standing body of the
    dataset's own mean pose, so a figure drawn small or large comes out the same size and only what changes between
    figures (where the hips travel, how high a foot lifts) reaches the model."""
    body = float(raw_mean[HEAD, 1] - raw_mean[list(FEET), 1].mean())  # ~1.46 m from the feet to the head joint
    first = pixels[0]
    tall = float(first[list(FEET), 1].mean() - first[HEAD, 1])  # pixels, y down: feet below the head
    if tall < 4.0:
        fail("E-SKETCH2ANIM-FLATFIGURE", pixels=round(tall, 1))
    scale = body / tall
    ground = float(first[list(FEET), 1].mean())
    out = np.empty_like(pixels)
    out[..., 0] = (pixels[..., 0] - first[0, 0]) * scale
    out[..., 1] = (ground - pixels[..., 1]) * scale
    return out, {"pixels_per_metre": round(1.0 / scale, 1), "body_height_m": round(body, 3)}


def condition(world: np.ndarray, key_frames: np.ndarray, length: int, raw_mean: np.ndarray, raw_std: np.ndarray):
    """The two things the model is conditioned on, built the way the training items were (dataset.py 590-620):

    `pose`  [L,22,3] the whole timeline, zero except on the frames a figure was drawn (the key poses);
    `hint`  [L,22,3] the same shape, holding only the pelvis, on every frame from the first drawn figure to the
            last, the hips path between the figures filled in straight (one figure: no path at all).
    Only x and y are normalised; the third channel stays 1 on the rows that count and 0 on the rest.

    Also returns the requested path as (the frames, the hips [N,2] in metres), so the node can report how far
    the generated body ended up from the line that was drawn; None when only one figure was drawn.
    """
    def norm(j: np.ndarray, joint: int | slice = slice(None)) -> np.ndarray:
        """[...,2] metres -> [...,3] the way the training items were normalised: x and y by the dataset's per-joint
        mean and spread, the third channel left at 1 (that 1 is also what makes a conditioned row non-zero)."""
        out = np.ones((*j.shape[:-1], 3), np.float32)
        out[..., :2] = (j - raw_mean[joint, :2]) / raw_std[joint, :2]
        return out

    pose = np.zeros((length, 22, 3), np.float32)
    mask = np.zeros((length, 1), bool)
    for k, f in enumerate(key_frames):
        pose[f] = norm(world[k])
        mask[f] = True
    hint = np.zeros((length, 22, 3), np.float32)
    if len(key_frames) > 1:
        span = np.arange(int(key_frames[0]), int(key_frames[-1]) + 1)
        path = np.stack([np.interp(span, key_frames, world[:, ROOT, c]) for c in (0, 1)], -1)
        hint[span, ROOT] = norm(path, ROOT)
        for f in key_frames:  # dataset.py 580: the key-pose frames are in the hint as whole poses too
            hint[f] = pose[f]
        return pose, mask, hint, (span, path)
    return pose, mask, hint, None


def generate(model, pose, mask, hint, rotation, length: int, prompt: str, device: str):
    import torch

    def t(a, dtype=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    flat = lambda a: t(a.reshape(1, length, -1))
    batch = {"length": [length], "text": [prompt], "motion_idx": [[int(np.argmax(mask))]],
             "pose": flat(pose), "pose_2d": flat(pose), "hint": flat(hint), "hint_2d": flat(hint),
             "pose_mask": t(mask, torch.bool).unsqueeze(0), "rotation": t(rotation)[None]}
    with torch.no_grad():
        joints, _ = model.forward_gmld_sequence_2d_keypose_traj(batch)
    return joints[0].cpu().numpy().astype(np.float64)


def to_rig(joints: np.ndarray, foot_lock: bool):
    """Upstream's own joints-to-BVH IK (visualization/joints2bvh.py): its template skeleton (22 joints, Mixamo
    names) fitted to the generated joint positions, optionally with its foot-skate clean-up. Returns names,
    parents, bone offsets [J,3] and local rotations [F,J,3,3] plus the root position [F,3], all in metres."""
    from visualization.joints2bvh import Joint2BVHConvertor

    anim, fitted = Joint2BVHConvertor().convert(joints.copy(), None, iterations=10, foot_ik=foot_lock)
    rot = np.asarray(anim.rotations.transforms(), np.float64)  # Quaternions -> [F,J,3,3]
    return (list(anim.names), np.asarray(anim.parents, np.int64), np.asarray(anim.offsets, np.float64),
            rot, np.asarray(anim.positions, np.float64)[:, 0], fitted)


def resample(rot: np.ndarray, root: np.ndarray, fps: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The model's 20 fps result on the shot's own frames: exactly `count` samples at `fps` (lab2shot_shared.motion,
    the same resampling every other node uses): rotations by slerp, the root position linearly. The node sends how
    many frames its start / end range has, and what comes back covers those and no others."""
    if len(rot) == count and abs(fps - MODEL_FPS) < 1e-6:
        return rot, root
    times = np.arange(len(rot)) / MODEL_FPS
    at = np.arange(count) / fps  # frames beyond the model's range hold the last frame (_brackets clamps them)
    return mo.resample_rotations(times, rot, at), mo.resample_values(times, root, at)


def main(job_path: str) -> None:
    import torch

    # the model runs on the CPU too: no CUDA check, and no GPU bookkeeping without one
    run = Run.start(job_path, NODE, "Sketch2Anim", gpu=torch.cuda.is_available())
    job = run.job
    repo, weights = job.repo_dir, job.weights_dir
    run.weights(weights / "checkpoints" / "adapter.ckpt", weights / "sentence-t5-large")
    _enter_repo(repo)

    device = "cuda" if run.gpu else "cpu"
    p = job.params
    sketch = np.load(job.inputs["sketch"])
    length = int(sketch["length"])
    if not 1 <= length <= MAX_FRAMES:
        fail("E-SKETCH2ANIM-LENGTH", frames=length, most=MAX_FRAMES)
    key_frames = np.asarray(sketch["key_frames"], np.int64)
    drawn = np.asarray(sketch["key_xy"], np.float64)
    if len(key_frames) == 0:
        fail("E-SKETCH2ANIM-NOPOSE")

    model, raw_mean, raw_std = run.model("load_model", load_model, repo, weights, device, stage_params={"model": "Sketch2Anim"})
    set_sampling(model, p["steps"], p["text_guidance"], p["control"])
    run.stage("read_sketch")
    world, ruler = to_model_world(full_pose(drawn), raw_mean)
    pose, mask, hint, path = condition(world, key_frames, length, raw_mean, raw_std)
    from utils import rotate_pose  # upstream's sketch camera: Rx(angle_x) then Ry(angle_y), orthographic

    _, rotation = rotate_pose(np.zeros((1, 22, 3)), angle_x=float(p["angle_x"]), angle_y=float(p["angle_y"]))

    run.stage("generate")
    from mld.utils.utils import set_seed

    set_seed(int(p["seed"]))
    t1 = time.time()
    joints = generate(model, pose, mask, hint, rotation, length, str(p["prompt"]), device)
    # two timings are reported: `generate_seconds` is the denoising itself; `seconds` (from Run) also includes the
    # first model load (@resident: later tasks in the same worker process do not load it again)
    made = time.time() - t1
    progress(1, 2, "generate_each")

    run.stage("solve_rotations")
    names, parents, offsets, rot, root, fitted = to_rig(joints, bool(p["foot_lock"]))
    rot, root = resample(rot, root, float(p["fps"]), int(p["out_frames"]))
    # how far the fitted skeleton's joints ended up from the generated points, and how far the generated key pose
    # ended up from what was drawn: both are what the node tells the user, in centimetres
    seen = _projected(joints, rotation)
    ik_cm = float(np.abs(fitted - joints).mean() * M_TO_CM)
    drawn_cm = float(np.abs(seen[key_frames] - world).mean() * M_TO_CM)
    # path deviation: per-frame hip positions compared with the straight line between figures, under the same
    # orthographic camera
    path_cm = float(np.abs(seen[path[0], ROOT] - path[1]).mean() * M_TO_CM) if path is not None else -1.0
    progress(2, 2, "generate_each")

    save_npz(job.raw_dir / "motion.npz", names=np.array(names), parents=parents, offsets=offsets * M_TO_CM,
             rotations=rot, root=root * M_TO_CM)
    run.finish(list(range(len(rot))),  # `frames` stays the count, as the node reads it
               fps=float(p["fps"]), frames=int(len(rot)), joints=len(names), method="Sketch2Anim",
               skeleton="HumanML3D 22 joints", model_fps=MODEL_FPS, model_frames=length,
               keys=[int(f) for f in key_frames], generate_seconds=round(made, 2),
               key_error_cm=round(drawn_cm, 1),
               path_error_cm=round(path_cm, 1), ik_error_cm=round(ik_cm, 2), **ruler)


def _projected(joints: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """The generated body seen through the sketch's camera, so the node can say how far the result is from what was
    drawn (upstream projects the same way: utils.py project2D, and the guidance's project_to_2d)."""
    from utils import convert_kps_joint

    return (convert_kps_joint(joints.copy()) @ rotation.T)[..., :2]


if __name__ == "__main__":
    serve(main)
