"""The SAM 3D Body family (SAM 3D Body, Fast SAM 3D Body): MHR rig access through the TorchScript model shipped
with the weights (assets/mhr_model.pt), and the per-person solve both workers share. Runs in their worker
environments (torch, roma): the sam_3d_body worker imports it from this folder, Fast SAM 3D Body's worker from here
too (it requires this extension: Extension.requires puts this folder on its PYTHONPATH).

MHR works in centimeters, Y up; SAM 3D Body's head evaluates it and only then
flips to the OpenCV camera frame. The same head functions are called here and the
unflipped result is kept, which is the GL camera frame (camera looking down -Z).

Skeleton states are [..., 8] = (tx, ty, tz, qx, qy, qz, qw, s) and map a point
as p' = t + R(q) (s p); skinning is plain linear blend skinning against the
template inverse bind pose -- exactly what UsdSkel evaluates.
"""

from __future__ import annotations

import numpy as np
import roma
import torch

from lab2shot_worker import nothing, progress, read_frame, stage
from lab2shot_worker import tracking as pp
from lab2shot_shared.motion import quat_to_matrix
from lab2shot_shared.units import CV_TO_GL, M_TO_CM

CHUNK = 64  # frames per MHR evaluation batch
# 一帧里的人分批喂给模型，每个人都解：sam_3d_body / fast_sam_3d_body 都是把一批人一起送进网络一次算完，
# 显存随人数线性涨。批的大小只关显存：一批装不下就换下一档重来这一帧（lab2shot_worker.fit_memory 的 batch
# 那一档），结果不变，只是慢一点。不按人数丢人：几十个人的镜头按人拆成几十趟整条链重跑（重新检人、重新加载
# 模型）要几个小时，真正解人的时间只是零头。
PEOPLE_BATCH = 8  # 一批最多几个人（RTX 4090 上 8 人一批稳）
BATCH_STEPS = (8, 4, 2, 1)  # 显存不够时依次降到的批大小

def batches(present: list, size: int) -> list[list]:
    """`present`: this frame's [(person_id, box[4]), ...] -> the batches of at most `size` the model sees, biggest boxes
    first (closest / most prominent on screen), the order every batch's outputs come back in."""
    area = lambda pb: float((pb[1][2] - pb[1][0]) * (pb[1][3] - pb[1][1]))  # noqa: E731
    ordered = sorted(present, key=area, reverse=True)
    size = max(1, int(size))
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]

def skel_state_to_matrix(state: np.ndarray) -> np.ndarray:
    """[..., 8] skeleton states -> [..., 4, 4] column-vector matrices."""
    state = np.asarray(state, dtype=np.float64)
    t, q, s = state[..., :3], state[..., 3:7], state[..., 7:8]
    r = quat_to_matrix(q[..., [3, 0, 1, 2]])  # (x, y, z, w) -> the SDK's (w, x, y, z)
    m = np.zeros(state.shape[:-1] + (4, 4))
    m[..., :3, :3] = r * s[..., None]
    m[..., :3, 3] = t
    m[..., 3, 3] = 1.0
    return m

def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def rig_data(head) -> dict:
    """Skeleton, mesh topology, UVs and skin weights of the MHR LOD1 rig."""
    mhr = head.mhr
    character = mhr.character_torch
    lbs = character.linear_blend_skinning
    n_verts = int(lbs.num_vertices)

    bind_world = np.linalg.inv(skel_state_to_matrix(_np(lbs.inverse_bind_pose)))

    # sparse (vertex, joint, weight) triplets -> dense [V, K]
    verts = _np(lbs.vert_indices_flattened).astype(np.int64)
    joints = _np(lbs.skin_indices_flattened).astype(np.int64)
    weights = _np(lbs.skin_weights_flattened).astype(np.float64)
    counts = np.bincount(verts, minlength=n_verts)
    k = int(counts.max())
    order = np.argsort(verts, kind="stable")
    slot = np.arange(len(verts)) - np.repeat(np.cumsum(counts) - counts, counts)
    skin_idx = np.zeros((n_verts, k), np.int32)
    skin_w = np.zeros((n_verts, k), np.float32)
    skin_idx[verts[order], slot] = joints[order]
    skin_w[verts[order], slot] = weights[order]

    mesh = character.mesh
    return {
        "joint_names": np.array(list(mhr.get_joint_names())),
        "parents": _np(character.skeleton.joint_parents).astype(np.int64),
        "bind_world": bind_world,
        "faces": _np(mesh.faces).astype(np.int32),
        "uv": _np(mesh.texcoords).astype(np.float32),
        "uv_faces": _np(mesh.texcoord_faces).astype(np.int32),
        "skin_indices": skin_idx,
        "skin_weights": skin_w,
    }

@torch.no_grad()
def evaluate(head, global_rot, body, hand, scale, shape) -> dict:
    """Per-frame MHR evaluation with SAM 3D Body's parameterization.

    Returns joint_world [F,J,4,4] and vertices [F,V,3] in cm, camera-relative
    orientation but not yet placed (no camera translation).
    """
    device = head.scale_mean.device
    tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    joint_world, vertices = [], []
    for lo in range(0, len(global_rot), CHUNK):
        sl = slice(lo, lo + CHUNK)
        n = len(global_rot[sl])
        shape_t = tensor(shape[sl])
        # 表情全零：官方自己就把 expr_params 置零，权重从来不预测表情
        # （third_party/sam_3d_body/repo/sam_3d_body/models/heads/mhr_head.py:316
        #  `pred_face = pred[:, count : count + self.num_face_comps] * 0`，同一段 :306-307 连下巴也置零；
        #  fast_sam_3d_body 的同名文件 :442、:508、:901 一样）。
        # 所以 estimator 吐的 expr_params 永远是一串 0，送不送进来结果一个字节都不差；「蒙皮角色」上没有表情 blendShape。
        expr = torch.zeros(n, head.num_face_comps, device=device)
        _, model_params = head.mhr_forward(
            global_trans=torch.zeros(n, 3, device=device),
            global_rot=tensor(global_rot[sl]),
            body_pose_params=tensor(body[sl]),
            hand_pose_params=tensor(hand[sl]),
            scale_params=tensor(scale[sl]),
            shape_params=shape_t,
            expr_params=expr,
            return_model_params=True,
        )
        verts, skel_state = head.mhr(shape_t, model_params, expr)
        joint_world.append(skel_state_to_matrix(_np(skel_state)))
        vertices.append(_np(verts).astype(np.float64))
    return {"joint_world": np.concatenate(joint_world), "vertices": np.concatenate(vertices)}

@torch.no_grad()
def rest_vertices(head, shape: np.ndarray) -> np.ndarray:
    """The person's mesh in the template bind pose (identity blend shapes applied), cm."""
    device = head.scale_mean.device
    coeffs = torch.as_tensor(np.asarray(shape)[None], dtype=torch.float32, device=device)
    return _np(head.mhr.character_torch.blend_shape.forward(coeffs))[0]

def smooth_global_rot(euler: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth SAM 3D Body's root rotation through quaternions (no gimbal/wrap artifacts)."""
    rot = roma.euler_to_rotmat("ZYX", torch.as_tensor(euler, dtype=torch.float64))
    q = pp.smooth_quats(roma.rotmat_to_unitquat(rot).numpy(), sigma)
    back = roma.unitquat_to_rotmat(torch.as_tensor(q))
    return roma.rotmat_to_euler("ZYX", back).numpy()

# --------------------------------------------------------------------------- the shared solve

FOV_SAMPLES = 12  # frames used to estimate the shot's focal length

def estimate_focal_px(job, load_measure) -> tuple[float, str]:
    """Shot focal length in pixels: the user's lens (focal_px), or the median of load_measure()(rgb) -> focal px (a
    MoGe-2 estimate; the model is loaded only then) over frames sampled across the shot."""
    focal_px = job.params["focal_px"]
    if focal_px:
        return focal_px, "user"
    stage("估计 Focal Length")
    measure = load_measure()
    frames = job.frames
    picks = np.unique(np.linspace(0, len(frames) - 1, min(FOV_SAMPLES, len(frames))).round().astype(int))
    focals = []
    for n, i in enumerate(picks):
        focals.append(float(measure(read_frame(frames[i][1]))))
        progress(n + 1, len(picks))
    return float(np.median(focals)), "moge2"

def intrinsics(focal_px: float, width: int, height: int) -> torch.Tensor:
    return torch.tensor([[[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]]])

def shot_camera(job, load_measure) -> tuple:
    """(intrinsics: one tensor for the shot or a function of the frame, focal px, where it came from).
    The per-frame focals the node sends as inputs["camera"] when a focal length is wired per frame, else
    estimate_focal_px()."""
    camera_npz = job.inputs.get("camera")
    if camera_npz is None:
        focal_px, source = estimate_focal_px(job, load_measure)
        return intrinsics(focal_px, job.width, job.height), focal_px, source
    cam = np.load(camera_npz)
    focal_of = {int(f): float(v) for f, v in zip(cam["frames"], cam["focal_px"])}
    known = sorted(focal_of)

    def cam_int(frame):  # nearest camera sample for frames the camera does not cover
        focal = focal_of.get(frame) or focal_of[min(known, key=lambda k: abs(k - frame))]
        return intrinsics(focal, job.width, job.height)

    return cam_int, float(np.median(list(focal_of.values()))), "camera"

def given_people(job, code: str) -> list[tuple[int, pp.Track]]:
    """接进来的「人物框」（「ViTDet 人物框」→「选人」）-> [(人物编号, Track), ...]，按输入里的顺序，
    编号就是框里的编号（所以「选人」选了 3 号，解出来的还叫 person_03）。框里一个人都没有：没人可解（`code`）。
    官方函数 process_one_image 本来就收 bboxes，所以框直接交给它。"""
    people = [(pid, pp.Track({f: np.asarray(b, np.float64) for f, b in boxes.items()})) for pid, boxes in job.people().items()]
    if not people:
        nothing(code)
    return people


def people_tracks(detections: dict[int, np.ndarray]) -> list[tuple[int, pp.Track]]:
    """上游自己的检测器逐帧检出来的框 -> [(人物编号, Track), ...]，最显眼的是 1 号
    （和「ViTDet 人物框」节点同一条路：worker.py node_detect_people 也是 track_boxes 之后
    enumerate(..., start=1)，所以两处的编号规则一致）。

    这是没接「人物框」时的路（官方脚本的入口 demo.py:117 收一个画面文件夹，自己建 HumanDetector 检人，
    demo.py:44-49、87-92）。「人物框」是两个解算节点的可选输入（官方函数 process_one_image 收 bboxes），
    接了走上面的 given_people()，这里不跑。

    没接框、又只想算画面里的某一个人，也可以走显式的那条链：
        「ViTDet 人物框」 -> 「选人」 -> 「人物框转遮罩」 -> 「图像合成」（留下）-> 解算器的「RGB」口
    相乘之后画面上只剩那个人，上游自己的检测器自然只会找到他。
    """
    tracks = pp.track_boxes(detections)
    if not tracks:
        nothing("N-WORKER-NOPEOPLE")
    return list(enumerate(tracks, start=1))

def solve_person(outputs: dict[int, dict], head, params: dict, rig: dict) -> dict:
    """Lock shape, smooth, re-evaluate MHR. Returns arrays in cm, camera frame (GL)."""
    frames = sorted(outputs)
    get = lambda key: np.stack([np.asarray(outputs[f][key], dtype=np.float64) for f in frames])  # noqa: E731
    shape, scale = get("shape_params"), get("scale_params")
    body, hand = get("body_pose_params"), get("hand_pose_params")
    global_rot = get("global_rot")  # MHR euler angles (roma "ZYX" order as produced by the head)
    cam_t = get("pred_cam_t")  # OpenCV camera frame, meters

    if params["lock_shape"]:
        shape[:] = np.median(shape, axis=0)
        scale[:] = np.median(scale, axis=0)

    sigma = pp.smoothing_sigma(params["smoothing"])
    if sigma > 0:
        for run in pp.segments(frames):
            idx = [frames.index(f) for f in run]
            body[idx] = pp.smooth_angles(body[idx], sigma)
            hand[idx] = pp.gaussian_smooth(hand[idx], sigma)
            global_rot[idx] = smooth_global_rot(global_rot[idx], sigma)
            cam_t[idx] = pp.gaussian_smooth(cam_t[idx], sigma)
            cam_t[idx, 2] = pp.gaussian_smooth(cam_t[idx, 2:3], 2 * sigma)[:, 0]  # depth is the noisiest

    ev = evaluate(head, global_rot, body, hand, scale, shape)
    offset_cm = (cam_t * CV_TO_GL) * M_TO_CM  # person position in the GL camera frame

    joint_world = ev["joint_world"].copy()
    joint_world[:, :, :3, 3] += offset_cm[:, None, :]
    vertices = ev["vertices"] + offset_cm[:, None, :]

    # Skinning against MHR's own template bind pose reproduces its linear blend
    # skinning exactly (only the small pose-corrective shapes are left out);
    # the person's proportions live in the joint transforms and scales.
    return {
        "frames": np.asarray(frames),
        "joint_world": joint_world.astype(np.float32),
        "vertices": vertices.astype(np.float32),
        "bind_world": rig["bind_world"],
        "rest_vertices": rest_vertices(head, shape[0]).astype(np.float32),
        "faces": rig["faces"],
        "uv": rig["uv"],
        "uv_faces": rig["uv_faces"],
        "parents": rig["parents"],
        "joint_names": rig["joint_names"],
        "skin_indices": rig["skin_indices"],
        "skin_weights": rig["skin_weights"],
        "shape_params": shape[0].astype(np.float32),
        "scale_params": scale[0].astype(np.float32),
        "keypoints_2d": get("pred_keypoints_2d").astype(np.float32),
        "bbox": get("bbox").astype(np.float32),
    }

def write_people(job, per_person: dict[int, dict[int, dict]], head, focal_px: float, focal_source: str) -> None:
    """Every person's solve -> raw/person_<id>.npz, then result.json (the solve node's raw contract)."""
    from lab2shot_worker import save_npz, say, write_result

    stage("锁定体型 · 平滑 · 生成骨骼")
    params = job.params
    rig = rig_data(head)
    out = []
    for n, (pid, outputs) in enumerate(per_person.items()):
        if not outputs:
            continue
        data = solve_person(outputs, head, params, rig)
        name = f"person_{pid:02d}.npz"
        # `solved`：真正解出来的那几帧（契约：lab2shot_worker/world_humans.py 模块开头的 raw 契约）。
        # solve_person 只收有结果的帧，所以这里 `solved` 与 `data["frames"]` 相同
        save_npz(job.raw_dir / name, solved=np.asarray(sorted(int(f) for f in outputs), np.int64), **data)
        missing = len(job.frames) - len(outputs)
        if missing:
            say("N-SAM3DBODY-UNSOLVEDFRAMES", person=int(pid), missing=int(missing), frames=len(job.frames))
        out.append({"id": pid, "file": name, "frames": [int(f) for f in data["frames"]]})
        progress(n + 1, len(per_person))
    write_result(job, people=out, camera={"focal_px": focal_px, "source": focal_source, "width": job.width, "height": job.height})
