"""SMIRK: 3D facial expressions (FLAME) from single frames.

Per frame: MediaPipe Face Landmarker finds the face and gives the crop (as the
upstream demo does) -> SMIRK's three small encoders regress FLAME shape,
expression, jaw, eyelids, head rotation and an orthographic camera. FLAME itself
(FLAME 2020 generic_model.pkl) needs a registration: the user downloads it by
hand (lab2shot.extensions.manual).
"""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, body_model_weight

SMIRK_URL = "https://github.com/georgeretsi/smirk.git"
SMIRK_COMMIT = "c7de404c4389f073906a6db1adabf62efcea3f35"  # latest commit when pinned

# quick_install.sh: the pretrained model from the authors' Google Drive
# (encoder + neural generator; the worker loads only the encoder).
SMIRK_CKPT = "SMIRK_em1.pt"
SMIRK_CKPT_URL = "https://drive.usercontent.google.com/download?id=1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE&export=download&confirm=t"
SMIRK_CKPT_SHA256 = "26b234e3cc31a5de226bcba4321bb5d7343a2ad48234c028b57a1c2f8c2d22d0"  # 140,495,895 bytes

# quick_install.sh also fetches MediaPipe's face_landmarker.task (".../latest/" =
# version 1): pinned here as in the mediapipe_face extension (the installer keeps one copy of identical files).
TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
TASK_FILE = "face_landmarker.task"
TASK_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"  # 3,758,596 bytes

# FLAME 2020 comes from body_model_weight("flame"); the repo already ships the FLAME
# masks and the MediaPipe landmark embedding it uses.


class Smirk(Extension):
    name = "smirk"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "SMIRK"
    summary = "从单目画面解出三维人脸，面部几何能忠实还原极端、不对称和细微的表情"
    homepage = "https://georgeretsi.github.io/smirk/"
    source = GitSource(url=SMIRK_URL, commit=SMIRK_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="MIT（代码）+ FLAME 2020 非商用 + 权重仅限研究",
        url="https://github.com/georgeretsi/smirk/blob/main/LICENSE",
        summary=(
            "非商用。SMIRK 代码 MIT；但运行必须用 FLAME 2020 面部模型（generic_model.pkl），许可仅限非商用科研，"
            "需要在 flame.is.tue.mpg.de 注册后自己下载，禁止再分发（FLAME 2023 Open 虽是 CC-BY-4.0，但 SMIRK 按 FLAME 2020 训练，不能替换）。"
            "仓库自带的 FLAME 衍生文件（FLAME_masks、landmark_embedding、head_template.obj）同样来自 FLAME 官网。"
            "SMIRK 权重（SMIRK_em1.pt）作者没有单独写许可，训练数据含 LRS3、MEAD、CelebA、FFHQ（均为非商用研究数据），只按研究用途使用。"
            "裁脸用的 MediaPipe Face Landmarker 代码和模型 Apache-2.0。smplx 未使用，不用 pytorch3d / nvdiffrast（渲染部分不需要）"
        ),
    )
    import_repo = ""
    env = EnvSpec(
        python="3.10",  # chumpy (reads the FLAME pickle) still calls inspect.getargspec, gone in 3.11
        # Same torch as the other extensions (upstream: 2.0.1 / cu117; the encoders run unchanged).
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
        compiled=("chumpy @ git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da",),
        compiled_cuda=False,
    )
    weights = (
        Weight(
            key="smirk-em1",
            kind="url",
            source=SMIRK_CKPT_URL,
            dest=SMIRK_CKPT,
            sha256=SMIRK_CKPT_SHA256,
            note="SMIRK 预训练模型（作者 Google Drive，134 MB，仅限研究）",
        ),
        Weight(
            key="face-landmarker",
            kind="url",
            source=TASK_URL,
            dest=TASK_FILE,
            sha256=TASK_SHA256,
            note="MediaPipe face_landmarker.task float16 v1（裁脸用，3.6 MB，Apache-2.0）",
        ),
        body_model_weight("flame"),
    )


EXTENSION = Smirk()
