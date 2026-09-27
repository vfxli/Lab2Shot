"""Google MediaPipe Face Landmarker worker. Runs inside third_party/mediapipe_face/.venv
(pip wheel mediapipe==1.0.0, CPU / TFLite); never imports Lab2Shot core.

    python worker.py <job.json>

job["node"] == "mediapipe_face.face". Params (job["params"]):

    max_faces                int   1      1..8   faces tracked per frame (face slots). > 1: the
                                                 detector runs every frame, no smoothing, a face
                                                 keeps its slot by nearest centre, duplicates dropped
    threshold                float 0.5    0..1   BlazeFace detector score to accept a new face
    min_presence_confidence  float 0.5    0..1   Face Mesh "face present" score; below it the
                                                 face counts as lost and the detector runs again
    min_tracking_confidence  float 0.5    0..1   video mode: overlap (IoU) a new detection needs
                                                 to be taken as an already tracked face
    mode                     str   "video"       "video": frames in order, the face box is carried
                                                 over from the previous frame and (one face only)
                                                 landmarks get MediaPipe's One-Euro smoothing;
                                                 "image": every frame on its own (detector each frame)

Outputs (raw/):

faces.npz, F frames, N = max_faces slots; values are NaN where found is False:
    frames            int32   [F]
    found             bool    [F,N]
    landmarks         float32 [F,N,478,3]  x, y: pixels at the input resolution (x_norm * W,
                                           y_norm * H; 0 = left / top image edge, pixel (i, j)
                                           covers [j, j+1] x [i, i+1]); z = z_norm * W, the same
                                           pixel scale as x: depth relative to the face's centre,
                                           negative = towards the camera (weak perspective,
                                           learned from synthetic data: relative only).
                                           0..467 = canonical mesh vertices, 468..472 = iris of the
                                           subject's right eye (centre first), 473..477 = left iris
    blendshapes       float32 [F,N,52]     0..1 scores, order = blendshape_names
    blendshape_names  str     [52]         MediaPipe names (= ARKit names, see result.json)
    matrices          float32 [F,N,4,4]    MediaPipe facial transformation matrix: canonical face
                                           space (canonical_face.npz, cm) -> MediaPipe metric camera
                                           space: right-handed, cm, camera at the origin looking down
                                           -Z, +X right, +Y up (OpenGL), virtual pinhole camera with
                                           63 deg vertical FOV and the principal point at the image
                                           centre. Rotation + translation (+ uniform scale, ~1).
    matrices_opencv   float32 [F,N,4,4]    diag(1,-1,-1,1) @ matrices: canonical face -> OpenCV
                                           camera (+X right, +Y down, +Z forward), cm, same virtual camera
    virtual_intrinsics float64 [3,3]       that virtual camera in pixels (fy = fx = H/2 / tan(31.5 deg))

The translation assumes the virtual 63 deg lens and an average adult face size
(the canonical face): with the real focal length f (pixels), multiply the
translation (not the rotation) by f / virtual_fx to keep the same image.

canonical_face.npz (from the model bundle's geometry_pipeline_metadata_landmarks,
identical to the repo's canonical_face_model.obj):
    vertices            float32 [468,3]  cm; +X = subject's left, +Y up, +Z out of the face
    uvs                 float32 [468,2]  per vertex, OBJ / USD convention (v up)
    triangles           int32   [898,3]  counter-clockwise seen from the front (normals out)
    procrustes_ids      int32   [33]     landmarks MediaPipe fits the head pose on
    procrustes_weights  float32 [33]
"""

from __future__ import annotations

import os
import struct
import time
import zipfile
from pathlib import Path

import numpy as np

from lab2shot_worker import fail, save_npz, serve
from lab2shot_worker.frame_io import FrameReader
from lab2shot_worker.run import Run

NODE = "mediapipe_face.face"
NUM_LANDMARKS = 478
NUM_BLENDSHAPES = 52
VIRTUAL_VFOV_DEG = 63.0  # face_geometry_from_landmarks_graph.cc (Tasks default environment)
MAX_FACES = 8
READ_THREADS = min(8, os.cpu_count() or 4)
DUPLICATE_OVERLAP = 0.5  # max_faces > 1: boxes sharing more than this of the smaller one = same face
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
METADATA_FILE = "geometry_pipeline_metadata_landmarks.binarypb"

# MediaPipe's 52 categories are "_neutral" plus 51 of ARKit's 52 ARFaceAnchor
# blend shapes with identical names; ARKit's tongueOut is not predicted.
NOT_IN_ARKIT = {"_neutral"}
ARKIT_MISSING = ["tongueOut"]


# ---------------------------------------------------------------------------- canonical face

def _varint(buf: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, i


def _fields(buf: bytes):
    """Minimal protobuf wire-format reader: yields (field number, wire type, value)."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            size, i = _varint(buf, i)
            value, i = buf[i:i + size], i + size
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _floats(wire: int, value) -> list[float]:
    return list(struct.unpack(f"<{len(value) // 4}f", value)) if wire in (2, 5) else []


def _uints(wire: int, value) -> list[int]:
    if wire == 0:
        return [value]
    out, i = [], 0
    while i < len(value):
        v, i = _varint(value, i)
        out.append(v)
    return out


def canonical_face(task: Path) -> dict[str, np.ndarray]:
    """GeometryPipelineMetadata (mediapipe/tasks/cc/vision/face_geometry/proto) from the .task bundle:
    canonical_mesh (1) = Mesh3d {vertex_buffer (3): x y z u v per vertex, index_buffer (4)},
    procrustes_landmark_basis (2) = {landmark_id (1), weight (2)}."""
    with zipfile.ZipFile(task) as bundle:
        blob = bundle.read(METADATA_FILE)
    vertex_buffer: list[float] = []
    index_buffer: list[int] = []
    ids: list[int] = []
    weights: list[float] = []
    for number, _wire, value in _fields(blob):
        if number == 1:
            for n, w, v in _fields(value):
                if n == 3:
                    vertex_buffer += _floats(w, v)
                elif n == 4:
                    index_buffer += _uints(w, v)
        elif number == 2:
            ref = {n: (w, v) for n, w, v in _fields(value)}
            ids.append(ref[1][1])
            weights.append(_floats(*ref[2])[0])
    vertices = np.asarray(vertex_buffer, np.float32).reshape(-1, 5)
    uvs = vertices[:, 3:].copy()
    uvs[:, 1] = 1.0 - uvs[:, 1]  # metadata v runs down the texture; OBJ / USD v runs up
    return {
        "vertices": vertices[:, :3].copy(),
        "uvs": uvs,
        "triangles": np.asarray(index_buffer, np.int32).reshape(-1, 3),
        "procrustes_ids": np.asarray(ids, np.int32),
        "procrustes_weights": np.asarray(weights, np.float32),
    }


# ---------------------------------------------------------------------------- frames

def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection of the 2D boxes around two landmark sets [478,3], relative to
    the smaller box (a duplicate is often a slightly smaller box inside the other)."""
    lo, hi = np.maximum(a[:, :2].min(0), b[:, :2].min(0)), np.minimum(a[:, :2].max(0), b[:, :2].max(0))
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    smaller = min(float(np.prod(np.ptp(p[:, :2], axis=0))) for p in (a, b))
    return inter / max(smaller, 1e-9)


class SlotKeeper:
    """With several faces, MediaPipe returns them in no guaranteed order: keep each
    face in the slot whose last known centre is nearest (greedy, per frame)."""

    def __init__(self, slots: int):
        self.last: list[np.ndarray | None] = [None] * slots

    def unique(self, faces: list) -> tuple[list, int]:
        """Several faces requested: the detector runs on every frame and can hand
        back a face that is already tracked, so the same face shows up twice. Keep
        one face per place (landmark boxes overlapping > DUPLICATE_OVERLAP); the one
        nearest a face seen before wins."""
        known = [p for p in self.last if p is not None]

        def distance(face) -> float:
            centre = face[0][:, :2].mean(0)
            return min((float(np.linalg.norm(centre - p)) for p in known), default=0.0)

        kept: list = []
        for face in sorted(faces, key=distance):
            if all(_overlap(face[0], other[0]) <= DUPLICATE_OVERLAP for other in kept):
                kept.append(face)
        return kept, len(faces) - len(kept)

    def assign(self, centres: list[np.ndarray]) -> list[int]:
        pairs = sorted(
            (float(np.linalg.norm(c - p)), f, s)
            for f, c in enumerate(centres)
            for s, p in enumerate(self.last) if p is not None
        )
        slot_of: dict[int, int] = {}
        for _, f, s in pairs:
            if f not in slot_of and s not in slot_of.values():
                slot_of[f] = s
        # new faces: never-used slots first, then slots whose face was not seen this frame
        free = sorted((s for s in range(len(self.last)) if s not in slot_of.values()),
                      key=lambda s: self.last[s] is not None)
        for f in range(len(centres)):
            if f not in slot_of:
                slot_of[f] = free.pop(0)
        for f, s in slot_of.items():
            self.last[s] = centres[f]
        return [slot_of[f] for f in range(len(centres))]


# ---------------------------------------------------------------------------- main

def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "MediaPipe Face", gpu=False)
    job, p = run.job, run.params
    task = job.weights_dir / "face_landmarker.task"
    run.weights(task, what="模型")

    frames = run.frames()
    raw = job.raw_dir

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    video = p["mode"] == "video"
    fps = job.fps  # the plate's own frame rate (a frame-carrying packet always has one)
    first = frames.numbers[0]
    n, count = p["max_faces"], len(frames)

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task), delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=n,
        min_face_detection_confidence=p["threshold"],
        min_face_presence_confidence=p["min_presence_confidence"],
        min_tracking_confidence=p["min_tracking_confidence"],
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    canonical = canonical_face(task)
    save_npz(raw / "canonical_face.npz", **canonical)
    landmarker = run.model("MediaPipe 面部动作模型", vision.FaceLandmarker.create_from_options, options)

    landmarks = np.full((count, n, NUM_LANDMARKS, 3), np.nan, np.float32)
    blendshapes = np.full((count, n, NUM_BLENDSHAPES), np.nan, np.float32)
    matrices = np.full((count, n, 4, 4), np.nan, np.float32)
    found = np.zeros((count, n), bool)
    names: list[str] | None = None
    slots = SlotKeeper(n)
    size: tuple[int, int] | None = None
    frame_seconds: list[float] = []
    duplicates = 0

    run.stage("逐帧跟踪面部")
    with landmarker, FrameReader(frames.paths, threads=READ_THREADS, ahead=2 * READ_THREADS) as reader:
        # PNG decoding (~50 ms for a 1K frame) is slower than the network (~8 ms):
        # decode a few frames ahead in parallel.
        for i, (frame, _path) in run.each(frames.pairs, "跟踪面部"):
            rgb = reader.get(i)
            h, w = rgb.shape[:2]
            if size is None:
                size = (w, h)
            elif size != (w, h):
                fail("E-MEDIAPIPEFACE-SIZE", frame=frame, width=w, height=h, first_width=size[0], first_height=size[1])
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t = time.time()
            if video:  # timestamps from frame numbers: gaps in the sequence stay gaps in time
                result = landmarker.detect_for_video(image, round((frame - first) * 1000.0 / fps))
            else:
                result = landmarker.detect(image)
            frame_seconds.append(time.time() - t)

            faces = []
            for k, marks in enumerate(result.face_landmarks[:n]):
                pts = np.array([(m.x * w, m.y * h, m.z * w) for m in marks], np.float32)
                cats = result.face_blendshapes[k] if k < len(result.face_blendshapes) else []
                if cats and names is None:
                    names = [c.category_name for c in cats]
                mat = result.facial_transformation_matrixes[k] if k < len(result.facial_transformation_matrixes) else None
                faces.append((pts, [c.score for c in cats], mat))
            if len(faces) > 1:
                faces, dropped = slots.unique(faces)
                duplicates += dropped
            for (pts, scores, mat), s in zip(faces, slots.assign([f[0][:, :2].mean(0) for f in faces])):
                found[i, s] = True
                landmarks[i, s] = pts
                if len(scores) == NUM_BLENDSHAPES:
                    blendshapes[i, s] = scores
                if mat is not None:
                    matrices[i, s] = np.asarray(mat, np.float32).reshape(4, 4)

    w, h = size
    names = names or _default_names()  # no face in the whole shot: the model's fixed order
    fx = (h / 2.0) / np.tan(np.radians(VIRTUAL_VFOV_DEG / 2.0))
    virtual_k = np.array([[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]])
    matrices_cv = np.einsum("ij,fnjk->fnik", OPENGL_TO_OPENCV, matrices).astype(np.float32)

    part = raw / "faces.part.npz"
    save_npz(
        part,
        frames=np.asarray(frames.numbers, np.int32),
        found=found,
        landmarks=landmarks,
        blendshapes=blendshapes,
        blendshape_names=np.asarray(names),
        matrices=matrices,
        matrices_opencv=matrices_cv,
        virtual_intrinsics=virtual_k,
    )
    part.replace(raw / "faces.npz")

    detected = found.any(axis=1)
    rate = float(detected.mean())  # the node says how many frames have no face (W-MEDIAPIPEFACE-MISSING, N-MEDIAPIPEFACE-NOFACE)
    depth = -matrices[..., 2, 3][found]

    run.finish(
        frames.numbers,
        kind="faces",
        node=NODE,
        model="MediaPipe Face Landmarker (face_landmarker.task float16 v1: BlazeFace short range + Face Mesh V2 + Blendshape V2)",
        mediapipe_version=mp.__version__,
        files={
            "faces.npz": "frames, found, landmarks, blendshapes, blendshape_names, matrices, matrices_opencv, virtual_intrinsics",
            "canonical_face.npz": "vertices, uvs, triangles, procrustes_ids, procrustes_weights",
        },
        params=p,
        fps_for_timestamps=fps if video else None,
        smoothing="MediaPipe One-Euro landmark filter (video mode, max_faces = 1 only)" if video and n == 1 else "none",
        width=w,
        height=h,
        frames=frames.numbers,  # the whole list, not the standard [first, last]
        missing="NaN in landmarks / blendshapes / matrices where found is False",
        face_slots=(
            "max_faces > 1: MediaPipe has no face IDs; a face keeps the slot whose last centre is nearest. "
            "The detector runs on every frame and there is no landmark smoothing; results overlapping "
            f"> {DUPLICATE_OVERLAP:g} (intersection / smaller box) are one face (duplicates dropped)"
            if n > 1 else "one slot"
        ),
        duplicates_dropped=duplicates,
        conventions={
            "landmarks": (
                "x = x_norm * width, y = y_norm * height: pixels at the input resolution, origin at the top-left "
                "image corner, pixel (i, j) covers [j, j+1] x [i, i+1]. z = z_norm * width: same pixel scale as x, "
                "relative to the face centre, negative = closer to the camera (MediaPipe: 'the magnitude of z uses "
                "roughly the same scale as x'; learned from synthetic data, relative depth only)"
            ),
            "landmark_ids": "0..467 = canonical mesh vertex ids; 468..472 = subject's right iris (centre, then 4 "
                            "points), 473..477 = subject's left iris",
            "blendshapes": "scores 0..1, order = blendshape_names",
            "matrices": (
                "MediaPipe facial transformation matrix (4x4, column vectors: p_camera = M @ [p_canonical, 1]): "
                "canonical face space (canonical_face.npz, cm) -> MediaPipe metric camera space: right-handed, cm, "
                "camera at the origin looking down -Z, +X right, +Y up (OpenGL convention). Rotation + translation "
                "(+ a uniform scale close to 1). Solved by weighted Procrustes on 33 landmarks against a VIRTUAL "
                f"pinhole camera with {VIRTUAL_VFOV_DEG:g} deg vertical FOV and the principal point at the image "
                "centre, and an average adult face size: for the real lens (focal f in pixels) multiply the "
                "translation by f / virtual_intrinsics[0,0] to keep the same image"
            ),
            "matrices_opencv": "diag(1,-1,-1,1) @ matrices: canonical face -> OpenCV camera space "
                               "(+X right, +Y down, +Z forward), cm, same virtual camera",
            "canonical_face": (
                "468 vertices in cm; +X = the subject's left, +Y up, +Z out of the face (nose tip z = 7.59); "
                "triangles counter-clockwise seen from the front; uvs per vertex with v up (OBJ / USD), "
                "same as mediapipe/modules/face_geometry/data/canonical_face_model.obj in the pinned repo"
            ),
        },
        virtual_camera={"vertical_fov_deg": VIRTUAL_VFOV_DEG, "fx_px": fx, "cx_px": w / 2.0, "cy_px": h / 2.0},
        blendshape_names=names,
        arkit_mapping={name: (None if name in NOT_IN_ARKIT else name) for name in names},
        arkit_not_predicted=ARKIT_MISSING,
        # checked: pasting a closed eye onto the image-right eye (subject's left) raises eyeBlinkLeft
        left_right="Left / Right = the subject's own left / right (as in ARKit); image-right side for a face looking at the camera",
        detection_rate=rate,
        frames_found=int(detected.sum()),
        faces_per_frame_max=int(found.sum(axis=1).max()),
        face_distance_cm_median=float(np.median(depth)) if depth.size else None,
        # the Run's seconds_per_frame is wall clock incl. PNG decoding; this one the network alone
        network_seconds_per_frame=round(float(np.mean(frame_seconds)), 4),
        device="cpu",
    )


def _default_names() -> list[str]:
    """Category order of face_blendshapes.tflite (mediapipe.tasks.python.vision.face_landmarker.Blendshapes)."""
    from mediapipe.tasks.python.vision.face_landmarker import Blendshapes

    def camel(member: str) -> str:
        head, *rest = member.lower().split("_")
        return head + "".join(r.title() for r in rest)

    return ["_neutral" if b.name == "NEUTRAL" else camel(b.name) for b in sorted(Blendshapes, key=int)]


if __name__ == "__main__":
    serve(main)
