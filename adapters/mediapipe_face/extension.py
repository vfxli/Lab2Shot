"""Google MediaPipe Face Landmarker (Tasks API): per frame 478 3D face landmarks,
52 blendshape scores (ARKit names) and a facial transformation matrix
(canonical face -> camera), CPU only."""

from __future__ import annotations


from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, downloads

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
# SMIRK crops faces with the same file (pinned in its own declaration too)
TASK = downloads.Download(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff")  # 3,758,596 bytes
TASK_FILE = "face_landmarker.task"


class MediaPipeFace(Extension):
    name = "mediapipe_face"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "MediaPipe Face Landmarker"
    homepage = "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
    source = GitSource(url=MEDIAPIPE_URL, commit=MEDIAPIPE_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE",
    )
    generative = False
    env = EnvSpec(python="3.12")
    weights = (
        TASK.weight(key="face-landmarker", dest=TASK_FILE),
    )


EXTENSION = MediaPipeFace()
