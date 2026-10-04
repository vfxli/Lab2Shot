"""HaMeR worker: 3D hand meshes. Runs inside third_party/hamer/.venv with the
pinned repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>

Node hamer.hand_solve, per frame, the upstream demo's pipeline (demo.py):
people (upstream's own ViTDet-H detector, demo.py:44-53 and :76-81, `--body_detector vitdet`)
-> ViTPose+-H whole-body keypoints per person -> a box around each hand's confident
keypoints -> HaMeR on every hand crop (left hands run mirrored, as upstream) -> MANO.

The 人物框 input is optional (the official ViTPoseModel.predict_pose takes the person boxes,
vitpose_model.py:60-72): with one, the hands are looked for in those boxes and upstream's
detector does not run; without one, the official script's own ViTDet-H finds the people
(demo.py:44-53). Upstream runs per image and never links people between frames; the deliverable
here is one file per hand for the whole shot, so the per-frame boxes are linked into tracks by
IoU (lab2shot_worker.tracking, the same code the 「ViTDet 人物框」 node uses).

Output, the body-model family format (lab2shot_worker.world_humans): one file
per hand, raw/person_<track>.npz, track = 2 * person_id + (1 right / 0 left),
stable over the shot; only the frames where that hand was found:

    frames          int  [F]
    body_model      str  "mano"
    side            str  "right" | "left";  person_id int
    global_orient   f32  [F,3]      axis-angle, camera space
    hand_pose       f32  [F,15,3]   axis-angle per joint, full rotations (flat_hand_mean=True, no PCA)
    betas           f32  [10]       one hand shape for the track (median of the frames)
    betas_per_frame f32  [F,10]     HaMeR's own per-frame estimates
    transl          f32  [F,3]      v = R(global_orient) (v_rest(pose) - root_rest) + root_rest + transl
    root_rest       f32  [3]
    vertices        f32  [F,778,3]  metres, camera space of each frame (OpenCV: +X right, +Y down, +Z forward)
    joints          f32  [F,16,3]   the MANO skeleton (joint_names / parents), same space
    keypoints       f32  [F,21,3]   OpenPose hand order (wrist, thumb 1-4, index 1-4, middle, ring, pinky; tips)
    faces           int  [1538,3]   winding of this side (left hands are mirrored meshes)
    joint_names, parents [16]
    rest_vertices   f32  [778,3]    shaped rest pose (zero pose); rest_joints f32 [16,3]
    skin_weights    f32  [778,16]   MANO's LBS weights
    local_rotations f32  [F,16,3,3] each joint relative to its parent; the root's in camera space
    boxes           f32  [F,4]      hand box from the keypoints, xyxy pixels (before rescale_factor)
    keypoints_2d    f32  [F,21,3]   ViTPose hand keypoints: x, y pixels, confidence
    keypoint_names       [21]       what each of them is (HAND_KEYPOINT_NAMES), for the node's 「2D 关键点」
    focal_px        float           full-frame focal length behind transl
    principal_px    [cx, cy]        the picture's centre in the pixels sent (overscan aside), the pinhole's principal point

Left hands: HaMeR runs on the mirrored crop with the right-hand model and mirrors
the mesh back (x -> -x), exactly as upstream. Their rig is that mirror too:
rest pose and joints mirrored, rotations M R M (M = diag(-1, 1, 1)), i.e. axis-angle
(x, -y, -z) = the MANO_LEFT convention. 结果留在相机空间（result.json space="camera"）：
官方只给一个 Focal Length，没有相机，所以这里不写 camera.npz。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from lab2shot_worker import (
    nothing,
    fail,
    offload,
    quiet,
    read_frame,
    resident,
    save_npz,
    serve,
)
from lab2shot_shared import motion as mo
from lab2shot_worker import tracking
from lab2shot_worker import world_humans as wh
from lab2shot_worker.run import Run

# Upstream demo.py values
HAND_KPT_CONF = 0.5  # ViTPose hand keypoint confidence
HAND_MIN_KPTS = 3  # a hand needs more than this many confident keypoints
BATCH = 8  # hands per HaMeR forward pass (demo.py)

VITPOSE_CONFIG = "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py"
MANO_JOINT_NAMES = (
    "wrist", "index1", "index2", "index3", "middle1", "middle2", "middle3", "pinky1", "pinky2", "pinky3",
    "ring1", "ring2", "ring3", "thumb1", "thumb2", "thumb3",
)
# COCO-WholeBody / OpenPose hand order, the order ViTPose gives the 21 hand keypoints in: the wrist, then each
# finger from its base out to its tip. Named like MANO_JOINT_NAMES so an artist reads one set of words.
HAND_KEYPOINT_NAMES = ("wrist", *(f"{finger}{n}" if n < 4 else f"{finger}_tip"
                                  for finger in ("thumb", "index", "middle", "ring", "pinky")
                                  for n in (1, 2, 3, 4)))
MIRROR = np.diag([-1.0, 1.0, 1.0])


# ------------------------------------------------------------------ people (upstream's own detector)


@resident
def load_detector(ckpt: Path):
    """Upstream demo.py:44-53, the `--body_detector vitdet` branch: ViTDet-H through detectron2's LazyConfig,
    box score threshold 0.25 on all three cascade heads. The checkpoint is the local copy of the file
    demo.py names by URL (extension.py, key "vitdet")."""
    import hamer  # noqa: F401  (loads pyrender with EGL before detectron2, as demo.py does)
    from detectron2.config import LazyConfig
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    cfg = LazyConfig.load(str(Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"))
    cfg.train.init_checkpoint = str(ckpt)
    for i in range(3):
        cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    with quiet():
        return DefaultPredictor_Lazy(cfg)


def detect_people(run: Run, detector, frames) -> list[tuple[int, dict[int, np.ndarray]]]:
    """[(person id, {frame: xyxy box})], found by upstream's own detector.

    Per frame this is demo.py:76-81 verbatim: everything ViTDet calls class 0 (person) with a score above 0.5.
    Upstream stops there (it runs on unrelated images); the deliverable is one file per hand for a whole shot, so
    the per-frame boxes are linked into tracks by IoU — the same tracking the 「ViTDet 人物框」 node uses.
    Track 1 is the person with the most screen presence."""
    run.stage("detect_people")
    per_frame: dict[int, np.ndarray] = {}
    for n, (frame, path) in run.each(frames, "detect_people"):
        instances = detector(read_frame(path, order="bgr"))["instances"]
        keep = (instances.pred_classes == 0) & (instances.scores > 0.5)
        per_frame[frame] = instances.pred_boxes.tensor[keep].cpu().numpy().astype(np.float64).reshape(-1, 4)
    return [(pid, track.boxes) for pid, track in enumerate(tracking.track_boxes(per_frame), start=1)]


# ------------------------------------------------------------------ hands


@resident
def load_vitpose(vitpose_dir: Path, weights: Path, device):
    # As in demo.py, HaMeR is imported first: it loads pyrender with EGL. mmpose would otherwise force
    # PYOPENGL_PLATFORM=osmesa, which this machine does not have.
    import hamer.models  # noqa: F401
    from mmpose.apis import init_pose_model

    ckpt = weights / "_DATA" / "vitpose_ckpts" / "vitpose+_huge" / "wholebody.pth"
    with quiet():
        return init_pose_model(str(vitpose_dir / VITPOSE_CONFIG), str(ckpt), device=str(device))


def hands_of_people(pose_model, img_bgr: np.ndarray, people: list[tuple[int, np.ndarray]]) -> list[dict]:
    """ViTPose+ whole-body keypoints per person -> hands: {person, right, box, kpts}. demo.py's rule."""
    from mmpose.apis import inference_top_down_pose_model

    if not people:
        return []
    person_results = [{"bbox": np.append(box, 1.0).astype(np.float32), "person": pid} for pid, box in people]
    with quiet():
        out, _ = inference_top_down_pose_model(pose_model, img_bgr, person_results=person_results,
                                               bbox_thr=None, format="xyxy")
    hands = []
    for res in out:
        kpts = np.asarray(res["keypoints"], np.float32)  # [133,3] COCO-WholeBody: ..., left hand, right hand
        for right, part in ((0, kpts[-42:-21]), (1, kpts[-21:])):
            valid = part[:, 2] > HAND_KPT_CONF
            if valid.sum() > HAND_MIN_KPTS:
                box = np.array([part[valid, 0].min(), part[valid, 1].min(), part[valid, 0].max(), part[valid, 1].max()], np.float32)
                hands.append({"person": int(res["person"]), "right": right, "box": box, "kpts": part})
    return hands


# ------------------------------------------------------------------ HaMeR


@resident
def load_hamer(weights: Path, mano_file: Path, device):
    import torch
    from hamer.configs import get_config
    from hamer.models import HAMER

    ckpt_dir = weights / "_DATA" / "hamer_ckpts"
    cfg = get_config(str(ckpt_dir / "model_config.yaml"), update_cachedir=False)
    cfg.defrost()
    cfg.MANO.MODEL_PATH = str(mano_file.parent)  # smplx looks for MANO_RIGHT.pkl in this folder
    cfg.MANO.MEAN_PARAMS = str(weights / "_DATA" / "data" / "mano_mean_params.npz")
    # load_hamer(): crop shape of the ViT backbone; no backbone init weights (the checkpoint has them)
    if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    cfg.freeze()

    model = HAMER(cfg, init_renderer=False)
    ckpt = ckpt_dir / "checkpoints" / "hamer.ckpt"
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    # MANO comes from the user's own MANO_RIGHT.pkl, never from buffers stored in the checkpoint.
    state = {k: v for k, v in state.items() if not k.startswith("mano.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("mano.")]
    if missing:
        fail("E-WORKER-WEIGHTSMISMATCH", project="HaMeR", extension="hamer", model=ckpt.name,
             missing=len(missing), unexpected=len(unexpected), examples=list(missing[:3]))
    return model.to(device).eval(), cfg


def fast_gaussian(image, sigma, channel_axis=None, preserve_range=False):
    """skimage.filters.gaussian as ViTDetDataset calls it (vitdet_dataset.py:64-71: the whole frame blurred before a
    large hand is cut out, `gaussian(cvimg, sigma=s, channel_axis=2, preserve_range=True)`), done by OpenCV in
    float32: the same kernel (truncate 4.0, radius int(4 s + 0.5)) and the same edge rule ('nearest' =
    BORDER_REPLICATE). skimage runs it in float64 on the CPU, several hundred milliseconds per hand on a 1080p
    frame; the result differs only by float32 rounding."""
    import cv2

    size = 2 * int(4.0 * float(sigma) + 0.5) + 1  # scipy.ndimage: radius int(truncate * sigma + 0.5)
    return cv2.GaussianBlur(np.asarray(image, np.float32), (size, size), float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REPLICATE)


def run_hamer(model, cfg, img_bgr: np.ndarray, hands: list[dict], rescale: float, focal: float, principal, device) -> list[dict]:
    """HaMeR on every hand of a frame -> right-hand-model rotations (of the mirrored crop for left hands),
    betas and the full-frame camera translation, as demo.py computes them."""
    import torch
    from hamer.datasets import vitdet_dataset
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to

    vitdet_dataset.gaussian = fast_gaussian  # the name the dataset blurs with (see fast_gaussian)
    from hamer.utils.renderer import cam_crop_to_full

    boxes = np.stack([h["box"] for h in hands])
    right = np.array([h["right"] for h in hands], np.float32)
    dataset = ViTDetDataset(cfg, img_bgr, boxes, right, rescale_factor=rescale)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0)
    with quiet():  # the dataset prints a line per crop
        batches = list(loader)
    results = []
    for batch in batches:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = (2 * batch["right"] - 1) * pred_cam[:, 1]  # mirrored crop: flip the x offset back
        # cam_crop_to_full 把 img_size / 2 当主点：传 2 × 画面中心（principal_px），主点就是画面中心而不是画布中心
        centre = torch.tensor(principal, dtype=torch.float32, device=pred_cam.device).expand(len(pred_cam), 2) * 2
        cam_t = cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(),
                                 centre, focal).cpu().numpy()
        params = {k: v.float().cpu().numpy() for k, v in out["pred_mano_params"].items()}
        for n in range(len(cam_t)):
            results.append({
                "global_orient": params["global_orient"][n].reshape(3, 3),
                "hand_pose": params["hand_pose"][n].reshape(15, 3, 3),
                "betas": params["betas"][n].reshape(-1),
                "cam_t": cam_t[n],
            })
    return results


# ------------------------------------------------------------------ rig


def hand_track(mano, samples: list[dict], right: bool, device) -> dict:
    """One hand over its frames -> the family arrays: shape locked to the median, MANO re-evaluated, rig."""
    import smplx
    import torch
    from smplx.lbs import blend_shapes, vertices2joints

    rots = np.concatenate([np.stack([s["global_orient"] for s in samples])[:, None], np.stack([s["hand_pose"] for s in samples])], 1)
    betas_frames = np.stack([s["betas"] for s in samples]).astype(np.float32)
    betas = np.median(betas_frames, axis=0).astype(np.float32)
    cam_t = np.stack([s["cam_t"] for s in samples]).astype(np.float32)
    n = len(samples)
    with torch.no_grad():
        t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
        b = t(betas)[None].expand(n, -1)
        # smplx's own MANO forward: vertices and the 16 skeleton joints (HaMeR's subclass adds the tips)
        out = smplx.MANOLayer.forward(mano, global_orient=t(rots[:, :1]), hand_pose=t(rots[:, 1:]), betas=b, pose2rot=False)
        verts, joints = out.vertices, out.joints
        tips = torch.cat([joints, verts[:, mano.extra_joints_idxs]], 1)[:, mano.joint_map]
        rest = mano.v_template[None] + blend_shapes(t(betas)[None], mano.shapedirs)
        rest_joints = vertices2joints(mano.J_regressor, rest)[0]
    arrays = {
        "vertices": verts.cpu().numpy(), "joints": joints.cpu().numpy(), "keypoints": tips.cpu().numpy(),
        "rest_vertices": rest[0].cpu().numpy(), "rest_joints": rest_joints.cpu().numpy(),
    }
    faces = np.asarray(mano.faces, np.int32)
    if not right:  # mirror everything: mesh, skeleton and rotations (M R M)
        arrays = {key: value * np.array([-1.0, 1.0, 1.0], np.float32) for key, value in arrays.items()}
        rots = MIRROR @ rots @ MIRROR
        faces = faces[:, [0, 2, 1]]
    for key in ("vertices", "joints", "keypoints"):
        arrays[key] = arrays[key] + cam_t[:, None, :]
    return {
        **{k: v.astype(np.float32) for k, v in arrays.items()},
        "global_orient": mo.matrix_to_rotvec(rots[:, 0]).astype(np.float32),
        "hand_pose": mo.matrix_to_rotvec(rots[:, 1:]).astype(np.float32),
        "betas": betas,
        "betas_per_frame": betas_frames,
        "transl": cam_t,
        "root_rest": arrays["rest_joints"][0].astype(np.float32),
        "local_rotations": rots.astype(np.float32),
        "faces": faces,
        "skin_weights": mano.lbs_weights.detach().cpu().numpy().astype(np.float32),
        "parents": np.asarray(mano.parents.cpu().numpy(), np.int32),
    }


# ------------------------------------------------------------------ main


def main(job_path: str) -> None:
    import torch

    run = Run.start(job_path, "hamer.hand_solve", "HaMeR")
    job, params = run.job, run.params
    focal = params["focal_px"]  # the node always sends one (a 50 mm lens by default)
    rescale = params["rescale_factor"]
    mano_file = wh.body_model_file("mano", "HaMeR")  # before anything slow
    frames = run.frames()
    weights = Path(job.weights_dir)
    vitpose_dir = Path(os.environ["HAMER_VITPOSE_DIR"])
    device = torch.device("cuda")

    width, height = frames.width, frames.height

    if job.inputs.get("boxes"):
        # 接了「人物框」：按框找手，不自己检人（官方 ViTPoseModel.predict_pose 收人框，nodes.py official 注 ⓪）
        tracks_in = [(int(pid), {int(f): np.asarray(b[:4], np.float64) for f, b in boxes.items()}) for pid, boxes in job.people().items()]
    else:
        detector = run.model("load_model", load_detector, weights / "vitdet" / "model_final_f05665.pkl", stage_params={"model": "ViTDet"})
        tracks_in = detect_people(run, detector, frames.pairs)
        offload(detector)  # 检测完就让出显存：后面是 ViTPose+-H 和 HaMeR

    pose_model = run.model("load_model", load_vitpose, vitpose_dir, weights, device, stage_params={"model": "ViTPose"})
    model, cfg = run.model("load_model", load_hamer, weights, mano_file, device, stage_params={"model": "HaMeR"})

    run.stage("find_hands")
    tracks: dict[tuple[int, int], dict[int, dict]] = {}  # (person, right) -> frame -> sample
    for n, (frame, path) in run.each(frames.pairs, "reconstruct_hands"):
        with run.frame():
            present = [(pid, b[frame]) for pid, b in tracks_in if frame in b]
            img = read_frame(path, order="bgr")
            hands = hands_of_people(pose_model, img, present)
            if hands:
                for hand, res in zip(hands, run_hamer(model, cfg, img, hands, rescale, focal, params["principal_px"], device)):
                    tracks.setdefault((hand["person"], hand["right"]), {})[frame] = {**res, "box": hand["box"], "kpts": hand["kpts"]}
    offload(pose_model)  # off the GPU for the rest of the job

    run.stage("write_results")
    raw = job.raw_dir
    hands = []
    for (pid, right), per_frame in sorted(tracks.items()):
        hand_frames = sorted(per_frame)
        samples = [per_frame[f] for f in hand_frames]
        arrays = hand_track(model.mano, samples, bool(right), device)
        track, side = 2 * pid + right, "right" if right else "left"
        name = f"person_{track:02d}.npz"
        save_npz(
            raw / name,
            frames=np.asarray(hand_frames, np.int64),
            # 真正解出来的那些帧（契约：lab2shot_worker/world_humans.py）。HaMeR 交的就是它自己跟到的
            # 那几帧（`sorted(per_frame)`），中间不补洞，所以和 frames 一样
            solved=np.asarray(hand_frames, np.int64),
            body_model=np.array("mano"),
            side=np.array(side),
            person_id=np.int64(pid),
            joint_names=np.array(MANO_JOINT_NAMES),
            boxes=np.stack([s["box"] for s in samples]).astype(np.float32),
            keypoints_2d=np.stack([s["kpts"] for s in samples]).astype(np.float32),
            keypoint_names=np.array(HAND_KEYPOINT_NAMES),
            focal_px=np.float64(focal),
            **arrays,
        )
        hands.append({"name": f"person_{pid:02d}_{side}_hand", "file": name, "person_id": pid, "side": side,
                      "frames": len(hand_frames)})
    if not hands:
        nothing("N-HAMER-NOHANDS")
    # plate_camera.npz：放手用的那台针孔相机（原点、不动，focal = 上面用的那个，主点在画面中心），节点的「相机」口
    wh.save_camera(raw, frames.numbers, focal, name=wh.PLATE_CAMERA)
    # 不写 camera.npz：没有解出来的相机（官方只有一个 scaled_focal_length，没有解相机），
    # 家族的 convert 也只在 solves_camera 的节点上读它。手就留在相机空间

    wh.write_humans(
        run,
        frames.numbers,
        hands,
        None,
        body_model="mano",
        model="HaMeR (ViT-H, hamer_demo_data) + MANO_RIGHT",
        rig=("v = R(global_orient) (v_rest(pose) - root_rest) + root_rest + transl; local_rotations[:, 0] is the root "
             "in camera space, the others relative to their parent; skinning rest_vertices with skin_weights and "
             "local_rotations gives vertices up to MANO's pose-corrective blend shapes (included in vertices)"),
        mano=("axis-angle, full joint rotations (flat_hand_mean=True, no PCA), betas locked per hand (median); right "
              "hands: MANO_RIGHT; left hands: the exact mirror of the right-hand model (MANO_LEFT convention: "
              "axis-angle (x,-y,-z)), faces with reversed winding"),
        track_id="file person_<2 * person_id + (1 right, 0 left)>.npz, stable for the whole shot",
        focal_px=focal,
        rescale_factor=rescale,
        people_found=len(tracks_in),  # 接了「人物框」是框里的人，否则是上游 ViTDet 找到的人
        width=width,
        height=height,
        frames=frames.numbers,  # the whole list, not the standard [first, last]
    )


if __name__ == "__main__":
    serve(main)
