"""VideoMaMa (KAIST / Korea University / Adobe Research, CVPR 2026): mask-guided
video matting with a video diffusion prior (Stable Video Diffusion, one step).
Turns a coarse per-frame mask into a soft alpha matte, several frames at a time."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file


VIDEOMAMA_URL = "https://github.com/cvlab-kaist/VideoMaMa.git"
VIDEOMAMA_COMMIT = "d5cce3e0ffe3b6429c147e658bb28bcfb576374c"  # 2026-04-01 (main)

# (key, Hugging Face repo, pinned revision, file in the repo, sha256 of the file)
# Only what inference uses: VideoMaMa's fine-tuned UNet and SVD-XT's temporal VAE.
# Not downloaded: SVD's own UNet and CLIP image encoder (upstream zeroes the CLIP
# embedding), and dino_projection_mlp.pth (training-time regulariser only).
VIDEOMAMA_REPO, VIDEOMAMA_REV = "SammyLim/VideoMaMa", "e289a7acc8403c4fbe4dea2a1de5a9749ebc9bf5"
SVD_REPO, SVD_REV = "stabilityai/stable-video-diffusion-img2vid-xt", "9e43909513c6714f1bc78bcb44d96e733cd242aa"
FILES = (
    ("videomama_unet", VIDEOMAMA_REPO, VIDEOMAMA_REV, "unet/diffusion_pytorch_model.safetensors",
     "f2442bf16ededad25c1c272ae7535b6411c43cee5c27b012bb6f7fda72d07b8c"),
    ("videomama_unet_config", VIDEOMAMA_REPO, VIDEOMAMA_REV, "unet/config.json",
     "d93f866daa31851058ca16a18e35b22dc9d3655d61e991b67d120ff333bf8176"),
    ("svd_vae", SVD_REPO, SVD_REV, "vae/diffusion_pytorch_model.fp16.safetensors",
     "af602cd0eb4ad6086ec94fbf1438dfb1be5ec9ac03fd0215640854e90d6463a3"),
    ("svd_vae_config", SVD_REPO, SVD_REV, "vae/config.json",
     "8f34272db69f7e2c615da6142ca3f9fdcd7b682bcfd903ceb15035fea79a8303"),
    # The Stability AI Community License must travel with the weights.
    ("svd_license", SVD_REPO, SVD_REV, "LICENSE.md",
     "d6f6b1a4dce5c852bd6d7d9482d002baf0ccdb71e662250b73be9eec8764ee8d"),
)


class VideoMaMa(Extension):
    name = "videomama"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "VideoMaMa"
    homepage = "https://github.com/cvlab-kaist/VideoMaMa"
    source = GitSource(url=VIDEOMAMA_URL, commit=VIDEOMAMA_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        url="https://github.com/cvlab-kaist/VideoMaMa/blob/main/License.md",
    )
    generative = False  # processes footage, not tagged 生成式扩散: only QwenImage (diffusers) carries it
    import_repo = ""
    env = EnvSpec(
        python="3.12",
        # Upstream used torch 2.4.0 (cu124) on Python 3.9; diffusers' SVD classes run
        # unchanged on 2.10.0 (cu128 includes sm_89). No compiled ops.
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
    )
    weights = tuple(
        hf_file(
            repo, rev, file,
            key=key,
            dest=f"{repo}/{file}",
            sha256=sha256,
        )
        for key, repo, rev, file, sha256 in FILES
    )


EXTENSION = VideoMaMa()
