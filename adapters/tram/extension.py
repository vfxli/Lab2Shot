"""TRAM (ECCV 2024, Wang et al.): global trajectory and motion of 3D humans from
in-the-wild videos. Masked DROID-SLAM (people removed with ViTDet + SAM + DEVA)
gives the camera, ZoeDepth its metric scale, SPEC the gravity direction; VIMO
(ViT-H video transformer) gives each person's SMPL body in the camera, placed in
the world through that camera. There is no camera input: the camera is what TRAM solves.

Needs the SMPL model file, which the user downloads after registering
(lab2shot.extensions.manual).
"""

from __future__ import annotations

from lab2shot.sdk import (
    downloads,
    CUDA_13_2_TOOLKIT,
    RESEARCH,
    EnvSpec,
    Extension,
    GitSource,
    LicenseInfo,
    Weight,
    body_model_weight,
)


TRAM_URL = "https://github.com/yufu-wang/tram.git"
TRAM_COMMIT = "4861c112f3c148201326680a50c9199650da6088"  # 2025-06-08 (README; code of 2025-02: better gravity / floor)
ZOEDEPTH_COMMIT = "d87f17b2f5fdcb174cf4fb115491f4a6c60de152"  # isl-org/ZoeDepth main (torch.hub code)
MIDAS_COMMIT = "454597711a62eabcbf7d1e89f3fb9f569051ac9b"  # isl-org/MiDaS master (ZoeDepth's backbone code)

_DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
# scripts/download_models.sh, laid out as TRAM expects relative to its working
# directory (data/pretrain/...): the worker runs in weights/.
FILES = {
    # key: (url, dest under weights/, sha256, note)
    "vimo": (_DRIVE.format("1fdeUxn_hK4ERGFwuksFpV_-_PHZJuoiW"), "data/pretrain/vimo_checkpoint.pth.tar",
             "197c333255dced98f48b67c0c2f6c630a5d1fe246fa9131f95aa4f15fb4080e0", "VIMO 视频人体网络（TRAM）"),
    "spec": (_DRIVE.format("1t4tO0OM5s8XDvAzPW-5HaOkQuV3dHBdO"), "data/pretrain/camcalib_sa_biased_l2.ckpt",
             "e4480cdd546ff8322978ef76e93c7a70a7d5e82c1390cbdd7a95473ac4595b48", "SPEC 相机俯仰/滚转和视场角（MPI 非商用）"),
    "deva": ("https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth",
             "data/pretrain/DEVA-propagation.pth",
             "5273748214babcbf5bc7aaa23ed4ce5a52f278c974b6c0dafcfbd89fbde25e48", "DEVA 视频分割跟踪（CC BY-NC-SA 4.0）"),
    # DEVA builds its encoders from torchvision's ImageNet ResNets (pretrained=True) before loading its own
    # checkpoint over them: the files must be in the offline torch hub cache (TORCH_HOME = weights/torch).
    "resnet50-imagenet": ("https://download.pytorch.org/models/resnet50-19c8e357.pth", "torch/hub/checkpoints/resnet50-19c8e357.pth",
                          "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097", "ImageNet ResNet-50（torchvision，BSD-3-Clause），DEVA 建网络用"),
    "resnet18-imagenet": ("https://download.pytorch.org/models/resnet18-5c106cde.pth", "torch/hub/checkpoints/resnet18-5c106cde.pth",
                          "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8", "ImageNet ResNet-18（torchvision，BSD-3-Clause），DEVA 建网络用"),
    "sam-vit-h": ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                  "data/pretrain/sam_vit_h_4b8939.pth",
                  "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e", "Segment Anything ViT-H 人物遮罩（Apache-2.0）"),
    "zoedepth-n": ("https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt",
                   "torch/hub/checkpoints/ZoeD_M12_N.pt",
                   "c97f94c4d53c5b788af46c5da0462262aebb37ea116fd70014bcbba93146c33b", "ZoeDepth-N 米制深度定尺度（MIT）"),
    "droid": (downloads.DROID_SLAM.source, "data/pretrain/droid.pth", downloads.DROID_SLAM.sha256, "DROID-SLAM（BSD-3-Clause）"),
    # detectron2 fetches ViTDet through iopath's cache ($FVCORE_CACHE/<url path>): it is put right there
    "vitdet-h": (downloads.VITDET_H.source,
                 "iopath/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl",
                 downloads.VITDET_H.sha256, "ViTDet-H 人物检测（Apache-2.0）"),
}
HUB_CODE = {  # torch.hub code of ZoeDepth and its MiDaS backbone, pinned (loaded with source="local"):
    # key: (GitHub archive of the commit, dest, sha256 of the archive)
    "zoedepth-code": (f"https://github.com/isl-org/ZoeDepth/archive/{ZOEDEPTH_COMMIT}.zip", "hub/ZoeDepth",
                      "3d51517771bccd38f4d213203403935514fa126f4bd11e1c99bdbe229640f871"),
    "midas-code": (f"https://github.com/isl-org/MiDaS/archive/{MIDAS_COMMIT}.zip", "hub/MiDaS",
                   "13c927ef1cc9408b5e0b06e73c50fc6e7d8b4783961b000be0d34270b266bee6"),
}


class TRAM(Extension):
    name = "tram"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "TRAM"
    summary = "两阶段方法，从实拍视频解出人的全局轨迹和动作：SLAM 解相机、用场景背景定尺度，再用 VIMO 回归身体动作"
    homepage = "https://yufu-wang.github.io/tram4d/"
    source = GitSource(url=TRAM_URL, commit=TRAM_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,  # its weights are research only: stricter than 非商用 (nodes/tags.py)
        uses=("SMPL", "BEDLAM", "3DPW", "Human3.6M"),
        name="MIT（代码）+ SMPL 非商用 + SPEC / DEVA 非商用 + 权重仅限研究",
        url="https://github.com/yufu-wang/tram/blob/main/LICENSE",
        summary=(
            "仅限研究。TRAM 代码 MIT，但仓库里的 thirdparty/camcalib（SPEC 相机标定）是马普所专有代码、只许有许可的非商用使用，"
            "其权重 camcalib_sa_biased_l2.ckpt 同样非商用；DEVA 视频跟踪（代码和权重）为 CC BY-NC-SA 4.0 非商用。"
            "运行必须用 SMPL 人体模型（SMPL_NEUTRAL），需要在 smpl.is.tue.mpg.de 注册后自己下载，仅限非商用科研，禁止再分发。"
            "VIMO 权重作者没有单独写许可，训练数据含 BEDLAM、3DPW、Human3.6M 等仅限研究的数据集，只按研究用途使用。"
            "其余：DROID-SLAM（代码和权重）BSD-3-Clause，lietorch BSD-3，ViTDet-H（detectron2）和 Segment Anything ViT-H Apache-2.0，"
            "ZoeDepth / MiDaS MIT，smplx 代码（MPI 非商用许可），chumpy MIT，torch-scatter MIT。"
            "固定机位时不用 DROID-SLAM 和 ZoeDepth 的相机解算部分"
        ),
    )
    import_repo = ""
    env = EnvSpec(
        python="3.10",  # chumpy (reads the SMPL pickle) still calls inspect.getargspec, gone in 3.11
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu130",
        cuda_toolkit=CUDA_13_2_TOOLKIT,  # DROID-SLAM's CUDA kernels need headers that compile against glibc >= 2.43
        compiled=(
            downloads.pip_git("detectron2", downloads.DETECTRON2_TRAM),
            downloads.pip_git("chumpy", downloads.CHUMPY),
        ),
        # ViTDet runs ROIAlign / NMS through torchvision on the GPU; detectron2's own
        # CUDA kernels are unused (DROID-SLAM is compiled with CUDA by build_droid.py).
        compiled_cuda=False,
        build="build_droid.py",
        pickled_checkpoints=True,  # SPEC, DEVA, VIMO checkpoints
    )
    submodules = ("thirdparty/DROID-SLAM/thirdparty/lietorch", "thirdparty/DROID-SLAM/thirdparty/eigen",
                  "thirdparty/Tracking-Anything-with-DEVA")
    weights = tuple(
        Weight(key=key, kind="url", source=url, dest=dest, note=note, sha256=sha256)
        for key, (url, dest, sha256, note) in FILES.items()
    ) + tuple(
        Weight(key=key, kind="zip", source=url, dest=dest, note="torch.hub 代码（固定版本，本地加载）", sha256=sha256)
        for key, (url, dest, sha256) in HUB_CODE.items()
    ) + (body_model_weight("smpl"),)

    def worker_env(self) -> dict[str, str]:
        weights = self.paths.weights
        return {
            "TRAM_ZOEDEPTH_DIR": str(weights / "hub" / "ZoeDepth" / f"ZoeDepth-{ZOEDEPTH_COMMIT}"),
            "TRAM_MIDAS_DIR": str(weights / "hub" / "MiDaS" / f"MiDaS-{MIDAS_COMMIT}"),
            "FVCORE_CACHE": str(weights / "iopath"),
        }


EXTENSION = TRAM()
