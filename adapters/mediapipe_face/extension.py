"""Google MediaPipe Face Landmarker (Tasks API): per frame 478 3D face landmarks,
52 blendshape scores (ARKit names) and a facial transformation matrix
(canonical face -> camera), CPU only."""

from __future__ import annotations


from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

MEDIAPIPE_URL = "https://github.com/google-ai-edge/mediapipe.git"
# Tag v1.0.0 = the pip wheel mediapipe==1.0.0 in requirements.txt. The
# worker runs the wheel; the repo is kept for the licence, the canonical face mesh
# (mediapipe/modules/face_geometry/data/canonical_face_model.obj, with UVs) and
# the source of the conventions documented in worker.py.
MEDIAPIPE_COMMIT = "6d31f1ebc3284db74d211d62bdc4f0a0c29ea120"

# The model bundle: a zip of face_detector.tflite (BlazeFace short range),
# face_landmarks_detector.tflite (Face Mesh V2), face_blendshapes.tflite
# (Blendshape V2) and geometry_pipeline_metadata_landmarks.binarypb (canonical
# face mesh + Procrustes weights). Version 1 is also what ".../latest/" serves.
TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
TASK_FILE = "face_landmarker.task"
TASK_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"  # 3,758,596 bytes


class MediaPipeFace(Extension):
    name = "mediapipe_face"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "MediaPipe Face Landmarker"
    summary = "MediaPipe 面部标志点任务：在画面和视频里检测面部关键点和表情"
    homepage = "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
    source = GitSource(url=MEDIAPIPE_URL, commit=MEDIAPIPE_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="Apache-2.0（代码和模型）",
        url="https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
        summary=(
            "代码（MediaPipe 仓库和 pip 包 mediapipe）为 Apache-2.0，可商用。"
            "face_landmarker.task 里的三个模型按各自模型卡均为 Apache-2.0：BlazeFace 面部检测（近距离版）、"
            "Face Mesh V2（478 点）、Blendshape V2（52 条表情曲线）；标准脸网格也随仓库以 Apache-2.0 发布。"
            "模型卡说明用途为 AR 娱乐，不提供面部识别，不适合做攸关人身安全的判断。"
            "不依赖 torch、nvdiffrast 或 FLAME / SMPL 等需要注册的面部/人体模型"
        ),
    )
    env = EnvSpec(python="3.12")
    weights = (
        Weight(
            key="face-landmarker",
            kind="url",
            source=TASK_URL,
            dest=TASK_FILE,
            note="face_landmarker.task float16 v1（面部检测 + 478 点 + 52 表情，3.6 MB，Apache-2.0）",
            sha256=TASK_SHA256,
        ),
    )


EXTENSION = MediaPipeFace()
