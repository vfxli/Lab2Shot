"""Video Depth Anything: temporally consistent depth for a whole shot.

Small and Metric-Small are Apache-2.0. Base / Large and Metric-Base /
Metric-Large are CC-BY-NC-4.0 (non-commercial, research use): downloaded too,
because the user is a researcher; the node marks them non-commercial.
"""

from __future__ import annotations


from lab2shot.sdk import COMMERCIAL, NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

VDA_URL = "https://github.com/DepthAnything/Video-Depth-Anything.git"
# 2025-10-07 (main): streaming mode shared with the metric models (#93) + point cloud fix.
VDA_COMMIT = "4f5ae23172ba60fd7bc11ef671cca678842c7072"

# model param -> (checkpoint file in weights/, Hugging Face repo, pinned revision, LFS sha256,
#                 model card licence); what each one is: extension.videodepthanything.weight.vda-<param>.note
CHECKPOINTS = {
    "small": (  # ViPE runs this model too and pins the same file in its own declaration
        "video_depth_anything_vits.pth",
        "depth-anything/Video-Depth-Anything-Small",
        "256875362cff76724b920335dfb4b29dd611f66e",
        "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609",
        "Apache-2.0",
    ),
    "metric_small": (
        "metric_video_depth_anything_vits.pth",
        "depth-anything/Metric-Video-Depth-Anything-Small",
        "273d090f2ce17df50c2872d82c8322c45da5b4dd",
        "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33",
        "Apache-2.0",
    ),
    # CC-BY-NC-4.0 (model cards: license: cc-by-nc-4.0): non-commercial.
    "base": (
        "video_depth_anything_vitb.pth",
        "depth-anything/Video-Depth-Anything-Base",
        "7231d0c6e260f54f103eba5005d01f6efa4f43db",
        "775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b",
        "CC-BY-NC-4.0",
    ),
    "metric_base": (
        "metric_video_depth_anything_vitb.pth",
        "depth-anything/Metric-Video-Depth-Anything-Base",
        "f6a245abad4b5a5b0d26722c8e1767ef310c547d",
        "f6f58576b9680a112f2428d4f39aff92656d3ae85745b0164675ace8d5b1fade",
        "CC-BY-NC-4.0",
    ),
    "large": (
        "video_depth_anything_vitl.pth",
        "depth-anything/Video-Depth-Anything-Large",
        "7aafbcb5c6af0bac741aad2b6471894fb4761afa",
        "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa",
        "CC-BY-NC-4.0",
    ),
    "metric_large": (
        "metric_video_depth_anything_vitl.pth",
        "depth-anything/Metric-Video-Depth-Anything-Large",
        "607fcdbd454b95c3bd39abbd3054142869a527d3",
        "24eba25342e7ee0f054be25da6a852ebd7cfa40cd6bf41a353ecb6abfe24e620",
        "CC-BY-NC-4.0",
    ),
}


# 「模型」选哪些是非商用（模型卡 CC-BY-NC-4.0 的那几个）：节点的 OptionTrait 从这里生成（licence_traits）
OPTION_LICENCES = {"model": {k: NONCOMMERCIAL for k, (*_, licence) in CHECKPOINTS.items() if "NC" in licence}}

class VideoDepthAnything(Extension):
    name = "videodepthanything"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Video Depth Anything"
    homepage = "https://github.com/DepthAnything/Video-Depth-Anything"
    source = GitSource(url=VDA_URL, commit=VDA_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/DepthAnything/Video-Depth-Anything/blob/main/LICENSE",
    )
    generative = False
    env = EnvSpec(
        python="3.11",
        # Upstream pins torch 2.1.1 + xformers 0.0.23 (no sm_89-era wheels needed): the
        # code runs unchanged on 2.9.0; attention goes through torch SDPA, no xformers.
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu128",
    )
    weights = tuple(
        hf_file(
            repo, revision, filename,
            key=f"vda-{key.replace('_', '-')}",
            dest=f"checkpoints/{filename}",
            sha256=sha256,
        )
        for key, (filename, repo, revision, sha256, _licence) in CHECKPOINTS.items()
    )


EXTENSION = VideoDepthAnything()
