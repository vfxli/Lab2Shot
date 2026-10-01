"""NVIDIA ViPE: camera solve (per-frame camera + one focal length) from plates."""

from __future__ import annotations


from lab2shot.sdk import CUDA_13_2_TOOLKIT, COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file, downloads

VIPE_URL = "https://github.com/nv-tlabs/vipe.git"
# First commit with the pose_only / pose_only_long presets (ViPE 1.2.0 + long-sequence SLAM).
VIPE_COMMIT = "8c9f36144e08d8f8c8cf60d20ad70c100d9cff07"

# The ViPE presets that may run. All four share GeoCalib (intrinsics), GroundingDINO +
# SAM ViT-B + DeAOT (masking moving objects) and DROID-SLAM.
#   pose_only / pose_only_long: MoGe-2 ViT-L keyframe depth, camera only. Commercial-use OK.
#   default: UniDepth-V2 ViT-L keyframe depth; per-frame metric depth from UniDepth-V2-L or
#            Prior-Depth-Anything (DAv2-B) on the SLAM map, made temporally stable with
#            Video-Depth-Anything-Small. NON-COMMERCIAL (UniDepth, Depth-Anything-V2-Base).
#   dav3:    DA3METRIC-LARGE keyframe depth; per-frame depth from DA3-GIANT conditioned on the
#            SLAM cameras. NON-COMMERCIAL (DA3-GIANT).
# The other presets (lyra, no_vda, static_vda, wide_angle, panorama) are not provided.
MODES = ("pose_only", "pose_only_long", "default", "dav3")

# Extra checkpoints of the default / dav3 presets, all pinned: weight key ->
# (dest in weights/, Hugging Face repo, revision, file, LFS sha256, licence)
NC_FILES = {
    "unidepth-v2-vitl14-config": (
        "hf/unidepth-v2-vitl14/config.json", downloads.UNIDEPTH_V2_L_CONFIG.repo, downloads.UNIDEPTH_V2_L_CONFIG.revision,
        downloads.UNIDEPTH_V2_L_CONFIG.filename, downloads.UNIDEPTH_V2_L_CONFIG.sha256, "CC-BY-NC-4.0",
    ),
    "unidepth-v2-vitl14": (
        "hf/unidepth-v2-vitl14/model.safetensors", downloads.UNIDEPTH_V2_L.repo, downloads.UNIDEPTH_V2_L.revision,
        downloads.UNIDEPTH_V2_L.filename, downloads.UNIDEPTH_V2_L.sha256, "CC-BY-NC-4.0",
    ),
    "prior-depth-anything-dav2-vitb": (
        "hf/prior-depth-anything/depth_anything_v2_vitb.pth", "Rain729/Prior-Depth-Anything",
        "25b3bfcf85209fb015e3b15259f4df8b6f13bb16", "depth_anything_v2_vitb.pth",
        "0d2b7002e62d39d655571c371333340bd88f67ab95050c03591555aa05645328", "CC-BY-NC-4.0",
    ),
    "prior-depth-anything-vitb": (
        "hf/prior-depth-anything/prior_depth_anything_vitb.pth", "Rain729/Prior-Depth-Anything",
        "25b3bfcf85209fb015e3b15259f4df8b6f13bb16", "prior_depth_anything_vitb.pth",
        "ff747c77af4e409b57fbf0b696c7b742efc4d283b923a12ad590f61c71ddae88", "Apache-2.0",
    ),
    # torch.hub.load_state_dict_from_url finds it in TORCH_HOME/hub/checkpoints by file name.
    "video-depth-anything-small": (
        "torch/hub/checkpoints/video_depth_anything_vits.pth", downloads.VDA_SMALL.repo, downloads.VDA_SMALL.revision,
        downloads.VDA_SMALL.filename, downloads.VDA_SMALL.sha256, "Apache-2.0",
    ),
    "da3metric-large": (
        "hf/DA3METRIC-LARGE/model.safetensors", downloads.DA3METRIC_MODEL.repo, downloads.DA3METRIC_MODEL.revision,
        downloads.DA3METRIC_MODEL.filename, downloads.DA3METRIC_MODEL.sha256, "Apache-2.0",
    ),
    "da3-giant": (
        "hf/DA3-GIANT/model.safetensors", "depth-anything/DA3-GIANT",
        "7cd62ae9315b9dff094d2d300e4ad012640607dd", "model.safetensors",
        "1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001", "CC-BY-NC-4.0",
    ),
}
NC_NOTES = {
    "unidepth-v2-vitl14-config": "UniDepth-V2 ViT-L 配置（default 模式；CC-BY-NC-4.0 非商用）",
    "unidepth-v2-vitl14": "UniDepth-V2 ViT-L 关键帧/逐帧米制深度（default 模式；CC-BY-NC-4.0 非商用）",
    "prior-depth-anything-dav2-vitb": "Prior-Depth-Anything 用的 Depth-Anything-V2-Base（default 模式；原模型 CC-BY-NC-4.0 非商用）",
    "prior-depth-anything-vitb": "Prior-Depth-Anything ViT-B 用 SLAM 点补全深度（default 模式；Apache-2.0）",
    "video-depth-anything-small": "Video-Depth-Anything-Small 深度时序稳定（default 模式；Apache-2.0）",
    "da3metric-large": "Depth-Anything-3 Metric-Large 关键帧米制深度（dav3 模式；Apache-2.0）",
    "da3-giant": "Depth-Anything-3 Giant 多视角逐帧深度（dav3 模式；CC-BY-NC-4.0 非商用）",
}


class ViPE(Extension):
    name = "vipe"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "NVIDIA ViPE"
    summary = ("从未受约束的原始视频估计相机内参、相机运动和稠密的近真实尺度深度图，普通镜头、广角和全景都支持；"
               "深度图那两条流水线非商用")
    homepage = "https://github.com/nv-tlabs/vipe"
    source = GitSource(url=VIPE_URL, commit=VIPE_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,  # the camera solve's modes use only commercial weights; 「ViPE 深度图」 declares its own 非商用
        name="Apache-2.0（pose_only / pose_only_long）；default / dav3 模式含 CC-BY-NC-4.0 非商用模型",
        url="https://github.com/nv-tlabs/vipe/blob/main/LICENSE",
        summary=(
            "可商用（「ViPE 深度图」除外）。代码 Apache-2.0。pose_only / pose_only_long 模式所用权重均可商用"
            "（MoGe-2 MIT、GeoCalib/GroundingDINO/SAM/BERT Apache-2.0、DROID-SLAM/DeAOT BSD-3）。"
            "default 模式非商用：UniDepth-V2 ViT-L（CC-BY-NC-4.0）、Prior-Depth-Anything 内含的 Depth-Anything-V2-Base"
            "（CC-BY-NC-4.0；Prior-Depth-Anything 本身和 Video-Depth-Anything-Small 为 Apache-2.0）。"
            "dav3 模式非商用：Depth-Anything-3 Giant（CC-BY-NC-4.0；DA3METRIC-LARGE 为 Apache-2.0）。"
            "选 default / dav3 的结果（相机和深度）不能用于商业项目。UniK3D 等其他配置不下载也不调用。"
            "注意：运动物体遮罩代码来自 Segment-and-Track-Anything，为 AGPL-3.0（内部使用无妨，修改后对外提供服务需开源）"
        ),
    )
    env = EnvSpec(
        python="3.12",
        # Upstream's tested torch, built for CUDA 13.
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu130",
        cuda_toolkit=CUDA_13_2_TOOLKIT,
        # Built from the pinned commit in uv's cache (repo/ stays untouched):
        # lietorch, DROID-SLAM correlation/BA, GroundingDINO deformable attention, ...
        compiled=(f"nvidia-vipe @ git+{VIPE_URL}@{VIPE_COMMIT}",),
    )
    # Laid out like torch.hub's cache: every worker runs with TORCH_HOME=weights/torch and
    # offline (Extension.base_env), so nothing but these files can ever be loaded.
    weights = (
        downloads.DROID_SLAM.weight(key="droid-slam", dest="torch/hub/droid_slam/droid.pth",
                                    note="DROID-SLAM 光流/BA 网络（BSD-3-Clause）"),
        hf_file(
            "Ruicheng/moge-2-vitl", "39c4d5e957afe587e04eec59dc2bcc3be5ecd968", "model.pt",
            key="moge-2-vitl",
            sha256="3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5",
            dest="torch/hub/moge2/moge-2-vitl.pt",
            note="MoGe-2 ViT-L 关键帧度量深度（MIT）",
        ),
        downloads.GEOCALIB_PINHOLE.weight(key="geocalib-pinhole", dest="torch/hub/geocalib/pinhole.tar",
                                          note="GeoCalib 初始 Focal Length 估计（Apache-2.0）"),
        hf_file(
            "ShilongLiu/GroundingDINO", "a94c9b567a2a374598f05c584e96798a170c56fb", "groundingdino_swint_ogc.pth",
            key="groundingdino-swint",
            sha256="3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
            dest="torch/hub/checkpoints/groundingdino_swint_ogc.pth",
            note="GroundingDINO Swin-T 找运动物体（Apache-2.0）",
        ),
        Weight(
            key="bert-base-uncased",
            kind="hf",
            source="google-bert/bert-base-uncased",
            # pinned to the snapshot installed and self-checked (the download record in weights/bert-base-uncased/.cache)
            revision="86b5e0934494bd15c9632b12f734a8a67f723594",
            dest="bert-base-uncased",
            files=("config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "model.safetensors"),
            note="GroundingDINO 的文本编码器 BERT（Apache-2.0）",
        ),
        Weight(
            key="sam-vit-b",
            sha256="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
            kind="url",
            source="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            dest="torch/hub/sam/sam_vit_b_01ec64.pth",
            note="Segment Anything ViT-B 物体遮罩（Apache-2.0）",
        ),
        Weight(
            key="deaot-r50",
            sha256="7e8a8d83310739bac02817f6bf48b6bbe2bbd7d5325722f1084088eb3aee1e06",
            kind="url",
            source="https://drive.usercontent.google.com/download?id=1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ&export=download&confirm=t",
            dest="torch/hub/aot/R50_DeAOTL_PRE_YTB_DAV.pth",
            note="DeAOT R50 遮罩逐帧跟踪（BSD-3-Clause）",
        ),
    ) + tuple(
        hf_file(
            repo, revision, filename,
            key=key,
            dest=dest,
            note=f"{repo}：{NC_NOTES[key]}",
            sha256=sha256 or "",  # none for small git files (config.json): the pinned revision is the check
        )
        for key, (dest, repo, revision, filename, sha256, _licence) in NC_FILES.items()
    )

    def worker_env(self) -> dict[str, str]:
        weights = self.paths.weights
        return {
            "VIPE_MOGE2_CHECKPOINT": str(weights / "torch" / "hub" / "moge2" / "moge-2-vitl.pt"),
            "VIPE_BERT_DIR": str(weights / "bert-base-uncased"),
            # default / dav3 checkpoints (UniDepth, Prior-Depth-Anything, DA3), by repo name
            "VIPE_HF_DIR": str(weights / "hf"),
        }


EXTENSION = ViPE()
