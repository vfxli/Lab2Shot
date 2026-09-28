"""The world-humans family (GVHMR, WHAM, TRAM, HaMeR, SMIRK, Pixel3DMM, SAM 3D Body): SMPL-family bodies, hands and
faces placed in a world or a camera, the user's body-model files, and the shared raw contract. Numpy only (what the
SDK declares); each worker runs it inside its own extension environment, never with Lab2Shot core. Node side:
lab2shot/nodes/families/humans.py (WorldHumans). What the SMPL family's skeleton is and how its parameters become a skeletal
animation (both directions) is lab2shot_shared.smpl, one implementation shared by the workers and the core.

The raw contract (write_humans):

    result.json         people: one entry per person, hand or face, {"name": its USD name, "file": its npz, ...};
                        space: "world" or "camera" (see below); world: what the world is, for people
    person_<id>.npz     the rig and the model's own vertices (save_person, or the method's own for MANO / FLAME);
                        a method whose own detector found 2D keypoints on the plate also writes keypoints_2d
                        [F,K,3] (x, y in the plate's pixels, confidence 0..1) and keypoint_names [K]; the node
                        family turns them into its 「2D 关键点」 output, one group per person
                        (WorldHumans.keypoints);
                        solved [S] int64, required: the frames of `frames` that were actually solved (the rest were
                        filled in). A method that fills no gaps writes `solved = frames` (save_person does so).

                        Why it is required and may not be omitted: a person may have only one or two frames actually
                        solved in the whole shot, with the rest filled in;
                        `lab2shot_shared/poses.py interpolate_poses` holds at both ends, so a single solved frame is
                        held over the whole shot, producing a large static figure stuck to the camera. The node side
                        must know how many frames were actually solved to judge whether a person is almost entirely
                        filled in; `frames` is the filled-in result and cannot tell, so only the writer of the npz
                        can state it. Without it the node fails at once (`E-HUMANS-NOSOLVED`,
                        lab2shot/nodes/families/humans.py) instead of silently treating every frame as solved: an
                        optional field would be omitted sooner or later.
    camera.npz          frames, focal_px [F], cam_to_world [F,4,4], optionally principal_px [F,2]
                        (save_camera; the lens centre in the plate's pixels, for a method that fits
                        inside a crop of it)

Conventions of every raw file written here:

    cameras   OpenCV (+X right, +Y down, +Z forward), cam_to_world 4x4, metres
    bodies    SMPL / SMPL-X parameters in the world: v = R(global_orient) (v_rest(betas, pose) - j0) + j0 + transl,
              j0 = the body model's root joint in the rest pose for these betas
    world     space "world": right-handed, metres. Without an input camera: gravity-aligned, +Y up (gravity = -Y), as
              the method defines it. With an input camera: that camera's world.
              space "camera" (HaMeR, SMIRK): each frame's OpenCV camera, the camera.npz fixed at the origin; the
              node places the result through a connected camera frame by frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lab2shot_shared import body_models, smpl
from lab2shot_shared.motion import matrix_to_rotvec, rotvec_to_matrix
# The rigid alignment of the correction step has one implementation shared by the core and the workers
# (lab2shot_shared/poses.py rigid_align); the core's 「相机空间转换」 in its whole-shot mode calls the scaled variant
# beside it, scaled_align (lab2shot/nodes/core/scene.py CameraSpaceConvert).
from lab2shot_shared.poses import rigid_align

from . import fail, link_file, read_frame, save_npz, say, shown
from .recon import InputCamera, interpolate_poses, mean_rotation
from .run import Run

# The 17 body keypoints ViTPose (COCO) finds on the plate before these methods solve anything: GVHMR and WHAM both
# run it, so the names live here once. Left / right are the person's own side, as everywhere else.
COCO17_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle",
)

CAMERA_CONVENTION = "OpenCV camera (+X right, +Y down, +Z forward); cam_to_world 4x4 maps camera to world; metres"
CAMERA_SPACE = ("each frame's OpenCV camera; camera.npz: a fixed camera at the origin (identity cam_to_world), "
                "focal_px per frame, and principal_px where the method solves the lens centre too (else the "
                "image centre)")
BODY_CONVENTION = ("v = R(global_orient) (v_rest - root_rest) + root_rest + transl; vertices / joints are the model's "
                   "own, in the same space")


# --------------------------------------------------------------------------- body models

def body_model_file(model: str, used_by: str, link: Path | None = None) -> Path:
    """The model's file (lab2shot_shared.body_models), also linked at `link` (where upstream reads it) when given;
    otherwise fail() with what to download. (The node is greyed out with the same instructions before it gets here.)"""
    found = body_models.find(model)
    if found is None:
        m = body_models.MODELS[model]
        fail("E-HUMANS-BODYMODEL", project=used_by, model=m.title, file=m.files[0], folder=shown(body_models.ROOT / model))
    if link is not None:
        link_file(link, found)
    return found


def evaluate(model, joints: int, chunk: int = 256, **inputs) -> tuple[np.ndarray, np.ndarray]:
    """A body model layer (smplx / SMPL) over a whole shot, `chunk` frames at a time on the GPU: vertices [F,V,3]
    and its first `joints` joints [F,J,3], float32. Array `inputs` are per frame (global_orient, body_pose, betas,
    transl ...), anything else is passed as it is."""
    import torch

    frames = len(next(v for v in inputs.values() if isinstance(v, np.ndarray)))
    verts, joint_list = [], []
    with torch.no_grad():
        for s in range(0, frames, chunk):
            args = {k: torch.as_tensor(v[s:s + chunk], dtype=torch.float32, device="cuda") if isinstance(v, np.ndarray) else v
                    for k, v in inputs.items()}
            out = model(**args)
            verts.append(out.vertices.cpu().numpy())
            joint_list.append(out.joints[:, :joints].cpu().numpy())
    return np.concatenate(verts).astype(np.float32), np.concatenate(joint_list).astype(np.float32)


# --------------------------------------------------------------------------- parameters and inputs


def plate(run: Run, least: int, why: dict) -> tuple[np.ndarray, int, int]:
    """The lines every worker of this family starts with: frame numbers, whether the shot is long enough, and the
    picture size taken from the first frame.

    The caller states what long enough means: `least` is the minimum frame count, `why` the message explaining why
    that many frames are needed (as returned by lab2shot_worker.reason(), with the code written as a literal at the
    call site).
    """
    job = run.job
    frame_numbers = np.array([f for f, _ in job.frames])
    if len(frame_numbers) < least:
        fail("E-WORKER-TOOFEWFRAMES", least=least, have=len(frame_numbers), why=why)
    first = read_frame(job.frames[0][1])
    return frame_numbers, job.width or first.shape[1], job.height or first.shape[0]


def focal_per_frame(frame_numbers, focal_px: float | None, cam: InputCamera | None, default: float | None,
                    default_label: str) -> tuple[np.ndarray | None, str]:
    """Focal length in pixels for every frame: the input camera's, else the user's, else
    the method's default (None = the method estimates it itself)."""
    n = len(frame_numbers)
    if cam is not None:
        return cam.at(frame_numbers)[0], "camera"
    if focal_px:
        return np.full(n, float(focal_px)), "user"
    return (None if default is None else np.full(n, float(default))), default_label


# --------------------------------------------------------------------------- rotations and transforms


def make_pose(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    r, t = np.asarray(r, np.float64), np.asarray(t, np.float64)
    out = np.zeros((*r.shape[:-2], 4, 4))
    out[..., :3, :3] = r
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    return out


def transform_points(t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """t [4,4] (one transform) or [F,4,4] (one per frame, pts [F,...,3])."""
    t = np.asarray(t, np.float64)
    pts = np.asarray(pts, np.float64)
    if t.ndim == 2:
        return pts @ t[:3, :3].T + t[:3, 3]
    shape = pts.shape
    p = pts.reshape(shape[0], -1, 3)
    return (np.einsum("fij,fnj->fni", t[:, :3, :3], p) + t[:, None, :3, 3]).reshape(shape)


def transform_root(t: np.ndarray, global_orient: np.ndarray, transl: np.ndarray, j0: np.ndarray):
    """Body parameters after moving the whole body by `t` ([4,4] or [F,4,4]); j0: rest root joint [3]."""
    t = np.asarray(t, np.float64)
    r = t[..., :3, :3]
    go = matrix_to_rotvec(r @ rotvec_to_matrix(global_orient)) if t.ndim == 3 else matrix_to_rotvec(np.einsum("ij,fjk->fik", r, rotvec_to_matrix(global_orient)))
    root = np.asarray(transl, np.float64) + j0  # where the root joint sits
    new_root = transform_points(t, root[:, None, :])[:, 0] if t.ndim == 3 else transform_points(t, root)
    return go, new_root - j0


def camera_from_body(r_root_cam, root_cam, r_root_world, root_world) -> np.ndarray:
    """cam_to_world [F,4,4] from the same body seen in the camera and in the world.
    r_*: root joint orientation [F,3,3]; root_*: root joint position [F,3]."""
    r_c2w = np.asarray(r_root_world) @ np.swapaxes(np.asarray(r_root_cam), -1, -2)
    t = np.asarray(root_world) - np.einsum("fij,fj->fi", r_c2w, np.asarray(root_cam))
    return make_pose(r_c2w, t)


def fill_track(frames_all: np.ndarray, known: dict[int, np.ndarray], rot_hint: np.ndarray | None = None) -> np.ndarray:
    """cam_to_world for every frame from the frames that have one (interpolate_poses: positions and rotations
    interpolated, held at the ends); with `rot_hint` ([N,3,3] cam_to_world rotations of all frames in any world,
    e.g. visual odometry) the rotations come from it instead, mapped into this world."""
    out = interpolate_poses(known, frames_all)
    if rot_hint is not None:
        index = {int(f): i for i, f in enumerate(frames_all)}
        ks = sorted(known)
        poses = np.stack([known[k] for k in ks])
        hint_known = np.stack([rot_hint[index[int(k)]] for k in ks])
        a = mean_rotation(poses[:, :3, :3] @ np.swapaxes(hint_known, -1, -2))
        out[:, :3, :3] = a @ rot_hint
        for k in ks:  # keep the measured rotations where there are some
            out[index[int(k)], :3, :3] = known[k][:3, :3]
    return out


def static_pose(c2w: np.ndarray) -> np.ndarray:
    """One camera for a locked-off shot: mean rotation, median position."""
    return make_pose(mean_rotation(c2w[:, :3, :3]), np.median(c2w[:, :3, 3], axis=0))


# --------------------------------------------------------------------------- people


@dataclass
class Person:
    """One solved person. World arrays are in the person's own world until merged."""

    pid: int
    frames: np.ndarray  # [F] frame numbers
    global_orient: np.ndarray  # [F,3] world
    body_pose: np.ndarray  # [F,P]
    betas: np.ndarray  # [B] (one shape for the shot)
    transl: np.ndarray  # [F,3] world
    vertices: np.ndarray  # [F,V,3] world
    joints: np.ndarray  # [F,J,3] world
    j0: np.ndarray  # [3] rest-pose root joint for betas
    # the same body in the camera (OpenCV) frame, for placing it through an input camera
    global_orient_cam: np.ndarray | None = None
    transl_cam: np.ndarray | None = None
    vertices_cam: np.ndarray | None = None
    joints_cam: np.ndarray | None = None
    cam_to_world: np.ndarray | None = None  # [F,4,4] camera implied by body in camera vs world
    # The frames actually solved (a subset of `frames`): methods that fill no gaps leave None and save_person writes
    # `frames`. Methods that fill gaps must set it: the node side uses it to judge whether a person is almost entirely
    # filled in (the contract at the top of the module).
    solved: np.ndarray | None = None
    # what the method's own 2D detector found on the plate, kept as it found it (never the 3D result reprojected):
    # [F,K,3] x, y in the plate's pixels and the detector's confidence 0..1, with a name per keypoint
    keypoints_2d: np.ndarray | None = None
    keypoint_names: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)  # more per-frame parameters, e.g. hand poses

    def move(self, t: np.ndarray) -> None:
        """Move the whole world by t [4,4] (or place it per frame with t [F,4,4])."""
        self.global_orient, self.transl = transform_root(t, self.global_orient, self.transl, self.j0)
        self.vertices = transform_points(t, self.vertices).astype(np.float32)
        self.joints = transform_points(t, self.joints).astype(np.float32)
        if self.cam_to_world is not None:
            self.cam_to_world = (t @ self.cam_to_world) if t.ndim == 2 else t @ self.cam_to_world

    def place_through(self, cam_to_world: np.ndarray) -> None:
        """World = input camera (per frame) x the body in the camera: lines up with the plate exactly."""
        if self.vertices_cam is None:
            raise ValueError("no camera-space body")
        self.global_orient, self.transl = transform_root(cam_to_world, self.global_orient_cam, self.transl_cam, self.j0)
        self.vertices = transform_points(cam_to_world, self.vertices_cam).astype(np.float32)
        self.joints = transform_points(cam_to_world, self.joints_cam).astype(np.float32)
        self.cam_to_world = np.asarray(cam_to_world, np.float64).copy()


def camera_track(reference: Person, frames_all: np.ndarray, rot_hint: np.ndarray | None = None) -> np.ndarray:
    """cam_to_world for every frame: the camera the reference person implies (body in the
    camera vs body in the world), gaps filled by fill_track (positions held / interpolated,
    rotations from `rot_hint`, e.g. the visual odometry)."""
    known = {int(f): reference.cam_to_world[i] for i, f in enumerate(reference.frames)}
    return fill_track(np.asarray(frames_all), known, rot_hint)


def merge_worlds(people: list[Person], frames_all: np.ndarray, track: np.ndarray) -> list[dict]:
    """GVHMR / WHAM solve every person in a world of their own (origin and heading from
    that person). The first person's world is the scene's world and `track` its camera.
    Every other person keeps their own world motion (smooth, grounded) and is moved as a
    whole: rotation = how their implied camera turns against `track`, translation = where
    the plate puts them (their camera-space body seen through `track`), both averaged over
    their frames. Returns per person the residual (how far the moved motion strays from
    where the plate puts it; large means that person should be solved alone: 「人物检测 → 选人 → 人物框转遮罩 →
    图像合成(留下)」 leaves only him in the plate)."""
    index = {int(f): i for i, f in enumerate(frames_all)}
    info = [{"id": people[0].pid, "reference": True}]
    for p in people[1:]:
        cams = track[[index[int(f)] for f in p.frames]]
        r = mean_rotation(cams[:, :3, :3] @ np.swapaxes(p.cam_to_world[:, :3, :3], -1, -2))
        root_cam = np.asarray(p.transl_cam) + p.j0
        target = transform_points(cams, root_cam[:, None, :])[:, 0]  # the root where the plate puts it
        root_world = np.asarray(p.transl) + p.j0
        t = (target - root_world @ r.T).mean(0)
        p.move(make_pose(r, t))
        residual = np.linalg.norm(np.asarray(p.transl) + p.j0 - target, axis=1)
        info.append({"id": p.pid, "residual_rms_m": round(float(np.sqrt(np.mean(residual**2))), 3),
                     "residual_max_m": round(float(residual.max()), 3)})
    return info


def one_world(method: str, people: list[Person], frames_all: np.ndarray, rot_hint: np.ndarray, static: bool,
              cam: InputCamera | None, follow_camera: bool) -> tuple[np.ndarray, dict]:
    """GVHMR / WHAM: every person's own world -> one world and one camera track [N,4,4].

    The first person's world and implied camera (static_pose for a locked-off shot)
    set the scene; the others are merged in (merge_worlds). With an input camera
    the whole scene is moved rigidly onto it and its poses become the camera.
    follow_camera: afterwards every body is placed through the camera frame by frame
    (camera-space body x camera), so it lines up with the plate exactly."""
    n = len(frames_all)
    track = camera_track(people[0], frames_all, rot_hint)
    if static and cam is None:
        track = np.repeat(static_pose(track)[None], n, 0)
    info: dict = {"merge": merge_worlds(people, frames_all, track), "alignment": None,
                  "world": f"{method} gravity-aligned world of person {people[0].pid} (+Y up, gravity -Y, metres)"}
    if cam is not None:
        ext = cam.at(frames_all)[1]
        info["world"] = "input camera's world"
        if not follow_camera:
            t_align, rot_err, pos_err = rigid_align(track, ext)
            for person in people:
                person.move(t_align)
            info["alignment"] = {"camera_rot_rms_deg": round(rot_err, 3), "camera_pos_rms_m": round(pos_err, 4)}
            if pos_err > 0.25:
                say("W-HUMANS-CAMERAOFF", method=method, error_m=float(pos_err))
        track = ext
    if follow_camera:
        index = {int(f): i for i, f in enumerate(frames_all)}
        for person in people:
            person.place_through(track[[index[int(f)] for f in person.frames]])
    return track, info


def lock_shape(betas: np.ndarray, root_rest_of) -> tuple[np.ndarray, np.ndarray]:
    """One body shape for the shot: median betas. Returns (betas [B], transl correction [F,3])
    that keeps the root joint where it was: transl += j0(betas_f) - j0(median)."""
    betas = np.asarray(betas, np.float64)
    if betas.ndim == 1:
        return betas, np.zeros((1, 3))
    med = np.median(betas, axis=0)
    return med, root_rest_of(betas) - root_rest_of(med[None])[0]


# --------------------------------------------------------------------------- output


def save_person(raw: Path, p: Person, model: str, layer) -> dict:
    """raw/person_<id>.npz; returns its write_humans entry. `model`: which body model the method solved ("smpl",
    "smplh", "smplx"; its joints and its pose layout are lab2shot_shared.smpl). `layer`: the smplx-package body
    layer the method evaluated, for the rig: the shaped rest pose, its skin weights and every joint's rotation (so a
    DCC gets a skinned character)."""
    import torch

    body = smpl.body(model)
    joint_names = body.names
    parents = np.asarray(body.parents, np.int32)
    from_layer = layer.parents.detach().cpu().numpy().astype(np.int32)[: len(joint_names)]
    if list(from_layer[1:]) != list(parents[1:]):  # the table and the file the user downloaded must be the same model
        raise ValueError(f"{model}: the body model file's joint tree is not {body.title}'s")
    with torch.no_grad():
        b = torch.as_tensor(np.asarray(p.betas)[None], dtype=layer.shapedirs.dtype, device=layer.shapedirs.device)
        rest_v = layer.v_template + torch.einsum("bl,vcl->bvc", b, layer.shapedirs[..., : b.shape[1]])[0]
        rest_j = (layer.J_regressor @ rest_v)[: len(joint_names)]
    n = len(p.frames)
    # the method's separate pose parts in the model's joint order; a part it does not solve (SMPL-X's jaw and eyes)
    # stays at rest. The hands travel in Person.extra under the model's own parameter names.
    parts = {k: v for k, v in p.extra.items() if k in {name for name, _ in body.layout}}
    local_rotations = rotvec_to_matrix(smpl.pack(body, global_orient=p.global_orient, body_pose=p.body_pose, **parts))
    name = f"person_{p.pid:02d}.npz"
    arrays = {
        "frames": np.asarray(p.frames, np.int64),
        # The frames actually solved (the contract at the top of the module). These methods (GVHMR / WHAM / TRAM)
        # return the span they tracked without filling gaps, so solved equals frames; gap-filling methods set
        # `Person.solved`
        "solved": np.asarray(p.frames if p.solved is None else p.solved, np.int64),
        "body_model": np.array(model),
        "global_orient": np.asarray(p.global_orient, np.float32),
        "body_pose": np.asarray(p.body_pose, np.float32),
        "betas": np.asarray(p.betas, np.float32),
        "transl": np.asarray(p.transl, np.float32),
        "root_rest": np.asarray(p.j0, np.float32),
        "vertices": np.asarray(p.vertices, np.float32),
        "joints": np.asarray(p.joints, np.float32),
        "faces": np.asarray(layer.faces, np.int32),
        "joint_names": np.array(list(joint_names)),
        "parents": parents,
        "rest_vertices": rest_v.cpu().numpy().astype(np.float32),
        "rest_joints": rest_j.cpu().numpy().astype(np.float32),
        "skin_weights": layer.lbs_weights[:, : len(joint_names)].detach().cpu().numpy().astype(np.float32),
        "local_rotations": local_rotations.astype(np.float32),
        **{k: np.asarray(v, np.float32) for k, v in p.extra.items()},
    }
    if p.keypoints_2d is not None:  # the 2D points its own detector found (module docstring)
        kp = np.asarray(p.keypoints_2d, np.float32)
        if kp.shape[:2] != (n, len(p.keypoint_names)):
            raise ValueError(f"keypoints_2d {kp.shape} does not match {n} frames x {len(p.keypoint_names)} names")
        arrays.update(keypoints_2d=kp, keypoint_names=np.array(list(p.keypoint_names)))
    save_npz(raw / name, **arrays)
    return {"name": f"person_{p.pid:02d}", "id": p.pid, "file": name, "frames": [int(f) for f in p.frames]}


def save_camera(raw: Path, frames, focal_px, cam_to_world=None, principal_px=None) -> str:
    """raw/camera.npz: frames, focal_px per frame, cam_to_world [F,4,4] (OpenCV; None = a fixed camera at the
    origin, for results in camera space).

    `principal_px` ((cx, cy) or one per frame, the plate's pixels): only a method that solves the lens centre and
    knows it is not the picture's centre writes it, such as a face tracker that fits inside a crop of the plate
    (Pixel3DMM) does. Left out (None), the node takes the picture's centre."""
    frames = np.asarray(frames, np.int64)
    focal = np.broadcast_to(np.asarray(focal_px, np.float64), frames.shape).copy()
    if cam_to_world is None:
        cam_to_world = np.broadcast_to(np.eye(4), (len(frames), 4, 4))
    extra = {} if principal_px is None else {
        "principal_px": np.broadcast_to(np.asarray(principal_px, np.float64).reshape(-1, 2), (len(frames), 2)).copy()}
    save_npz(raw / "camera.npz", frames=frames, focal_px=focal, cam_to_world=np.asarray(cam_to_world, np.float64), **extra)
    return "camera.npz"


def write_humans(run: Run, frame_numbers, /, people: list[dict], world: str | None, **info) -> None:
    """raw/result.json: `frame_numbers` (the shot's, for the Run's standard fields; a method whose converter reads every
    frame's file by number passes the whole list again as `frames=`); `people` (each {"name": USD name, "file": npz,
    ...}); `world`: what the world is, for people (space "world"), or None for a result in camera space (space
    "camera", module docstring); the method's `info`; the Run's timing and memory fields (Run.finish)."""
    if any(not {"name", "file"} <= set(p) for p in people):
        raise ValueError("every person needs a name and a file")
    space = {"space": "world", "world": world} if world else {"space": "camera", "world": CAMERA_SPACE}
    run.finish([int(f) for f in frame_numbers], kind="world_humans", people=people, **space, units="metres",
               camera_convention=CAMERA_CONVENTION, body_convention=BODY_CONVENTION, **info)
