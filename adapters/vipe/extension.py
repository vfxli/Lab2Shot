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

# DROID-SLAM's and GeoCalib's checkpoints, which every preset loads (TRAM and GeoCalib pin the same files)
DROID_SLAM = downloads.Download(
    "https://drive.usercontent.google.com/download?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh&export=download&confirm=t",
    "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526")
GEOCALIB_PINHOLE = downloads.Download("https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
                                      "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac")

# Extra checkpoints of the default / dav3 presets, all pinned: weight key ->
# (dest in weights/, Hugging Face repo, revision, file, LFS sha256, licence)
NC_FILES = {
    "unidepth-v2-vitl14-config": (
        "hf/unidepth-v2-vitl14/config.json", "lpiccinelli/unidepth-v2-vitl14",
        "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9", "config.json",
        "09eb0ea8de53a6c9a1d428ac98c79847fe2602ea417701261f3f628099a30816", "CC-BY-NC-4.0",
    ),
    "unidepth-v2-vitl14": (
        "hf/unidepth-v2-vitl14/model.safetensors", "lpiccinelli/unidepth-v2-vitl14",
        "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9", "model.safetensors",
        "ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6", "CC-BY-NC-4.0",
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
        "torch/hub/checkpoints/video_depth_anything_vits.pth", "depth-anything/Video-Depth-Anything-Small",
        "256875362cff76724b920335dfb4b29dd611f66e", "video_depth_anything_vits.pth",
        "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609", "Apache-2.0",
    ),
    "da3metric-large": (
        "hf/DA3METRIC-LARGE/model.safetensors", "depth-anything/DA3METRIC-LARGE",
        "4010e39f3634a45bc60553321fb49fb760bd594e", "model.safetensors",
        "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776", "Apache-2.0",
    ),
    "da3-giant": (
        "hf/DA3-GIANT/model.safetensors", "depth-anything/DA3-GIANT",
        "7cd62ae9315b9dff094d2d300e4ad012640607dd", "model.safetensors",
        "1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001", "CC-BY-NC-4.0",
    ),
}
# Each of these weights' note: extension.vipe.weight.<key>.note of the catalogue (i18n/<lang>.toml).


class ViPE(Extension):
    name = "vipe"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "NVIDIA ViPE"
    homepage = "https://github.com/nv-tlabs/vipe"
    source = GitSource(url=VIPE_URL, commit=VIPE_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,  # the camera solve's modes use only commercial weights; vipe.depth declares its own non-commercial
        url="https://github.com/nv-tlabs/vipe/blob/main/LICENSE",
    )
    generative = False
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
        DROID_SLAM.weight(key="droid-slam", dest="torch/hub/droid_slam/droid.pth"),
        hf_file(
            "Ruicheng/moge-2-vitl", "39c4d5e957afe587e04eec59dc2bcc3be5ecd968", "model.pt",
            key="moge-2-vitl",
            sha256="3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5",
            dest="torch/hub/moge2/moge-2-vitl.pt",
        ),
        GEOCALIB_PINHOLE.weight(key="geocalib-pinhole", dest="torch/hub/geocalib/pinhole.tar"),
        hf_file(
            "ShilongLiu/GroundingDINO", "a94c9b567a2a374598f05c584e96798a170c56fb", "groundingdino_swint_ogc.pth",
            key="groundingdino-swint",
            sha256="3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
            dest="torch/hub/checkpoints/groundingdino_swint_ogc.pth",
        ),
        Weight(
            key="bert-base-uncased",
            kind="hf",
            source="google-bert/bert-base-uncased",
            # pinned to the snapshot installed and self-checked (the download record in weights/bert-base-uncased/.cache)
            revision="86b5e0934494bd15c9632b12f734a8a67f723594",
            dest="bert-base-uncased",
            files=("config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "model.safetensors"),
        ),
        Weight(
            key="sam-vit-b",
            sha256="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
            kind="url",
            source="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            dest="torch/hub/sam/sam_vit_b_01ec64.pth",
        ),
        Weight(
            key="deaot-r50",
            sha256="7e8a8d83310739bac02817f6bf48b6bbe2bbd7d5325722f1084088eb3aee1e06",
            kind="url",
            source="https://drive.usercontent.google.com/download?id=1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ&export=download&confirm=t",
            dest="torch/hub/aot/R50_DeAOTL_PRE_YTB_DAV.pth",
        ),
    ) + tuple(
        hf_file(
            repo, revision, filename,
            key=key,
            dest=dest,
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
