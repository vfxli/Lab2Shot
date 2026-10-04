"""SMIRK: 3D facial expressions (FLAME) from single frames.

Per frame: MediaPipe Face Landmarker finds the face and gives the crop (as the
upstream demo does) -> SMIRK's three small encoders regress FLAME shape,
expression, jaw, eyelids, head rotation and an orthographic camera. FLAME itself
(FLAME 2020 generic_model.pkl) needs a registration: the user downloads it by
hand (lab2shot.extensions.manual).
"""

from __future__ import annotations

from lab2shot.sdk import RESEARCH, EnvSpec, Extension, GitSource, LicenseInfo, Weight, body_model_weight, downloads

SMIRK_URL = "https://github.com/georgeretsi/smirk.git"
SMIRK_COMMIT = "c7de404c4389f073906a6db1adabf62efcea3f35"  # latest commit when pinned

# quick_install.sh: the pretrained model from the authors' Google Drive
# (encoder + neural generator; the worker loads only the encoder).
SMIRK_CKPT = "SMIRK_em1.pt"
SMIRK_CKPT_URL = "https://drive.usercontent.google.com/download?id=1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE&export=download&confirm=t"
SMIRK_CKPT_SHA256 = "26b234e3cc31a5de226bcba4321bb5d7343a2ad48234c028b57a1c2f8c2d22d0"  # 140,495,895 bytes

# quick_install.sh also fetches MediaPipe's face_landmarker.task (".../latest/" = version 1): the same file the
# mediapipe_face extension pins (the installer stores identical files once)
TASK = downloads.Download(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff")  # 3,758,596 bytes
CHUMPY = GitSource("https://github.com/mattloper/chumpy.git", "580566eafc9ac68b2614b64d6f7aaa84eebb70da")
TASK_FILE = "face_landmarker.task"

# FLAME 2020 comes from body_model_weight("flame"); the repo already ships the FLAME
# masks and the MediaPipe landmark embedding it uses.


class Smirk(Extension):
    name = "smirk"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "SMIRK"
    homepage = "https://georgeretsi.github.io/smirk/"
    source = GitSource(url=SMIRK_URL, commit=SMIRK_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,  # its weights are research only: stricter than 非商用 (nodes/tags.py)
        uses=("FLAME",),
        url="https://github.com/georgeretsi/smirk/blob/main/LICENSE",
    )
    generative = False
    import_repo = ""
    env = EnvSpec(
        python="3.10",  # chumpy (reads the FLAME pickle) still calls inspect.getargspec, gone in 3.11
        # Same torch as the other extensions (upstream: 2.0.1 / cu117; the encoders run unchanged).
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
        compiled=(downloads.pip_git("chumpy", CHUMPY),),
        compiled_cuda=False,
    )
    weights = (
        Weight(
            key="smirk-em1",
            kind="url",
            source=SMIRK_CKPT_URL,
            dest=SMIRK_CKPT,
            sha256=SMIRK_CKPT_SHA256,
        ),
        TASK.weight(key="face-landmarker", dest=TASK_FILE),
        body_model_weight("flame"),
    )


EXTENSION = Smirk()
