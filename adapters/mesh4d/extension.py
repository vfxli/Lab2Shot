"""Mesh4D (VGG Oxford + Naver Labs Europe): one cut-out object, six frames at a
time, producing a deforming mesh with the same topology on every frame (a point cache).

Two nets, both from the Hunyuan3D-2.1 code base the authors built on:

1. the shape half (Tencent's Hunyuan3D-2.1 image-to-3D) turns one frame into a mesh;
2. Mesh4D's own deformation VAE + denoiser (a flow-matching DiT over 6 frames) move that one
   mesh's vertices to every frame of the window, conditioned on the six pictures.

The topology is therefore fixed once, which is what a DCC calls a point cache; for this reason
the output uses the existing 模型 type rather than a new one.

The checkout is never modified: it is read only, and the tree upstream's scripts compile in is built
from symlinks beside it (codebase.py, build_mesh4d.py).
"""

from __future__ import annotations

from lab2shot.sdk import RESEARCH, EnvSpec, Extension, GitSource, LicenseInfo, Weight

MESH4D_URL = "https://github.com/jzr99/Mesh4D"
MESH4D_COMMIT = "30269e31f193b13131cfdc3bf9ec8a3595c1ac4e"  # the only commit of the release

# The authors' two checkpoints (their README's two `gdown` lines). Both are full Lightning
# training checkpoints (the denoiser is 24 GB on disk), so the build step writes the weights
# out once as fp16 safetensors and the worker loads those instead (build_mesh4d.py).
_DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
CKPTS = {
    "deform_vae": (_DRIVE.format("1e9YGQAuFr5BDN2--srDEMnoULb4ZP5sx"), "ckpt/deform_vae.ckpt",
                   "d85c3107e2b022e3f2b433974ad59d2754549ca0bd9876e46a5d46d05c737a82",
                   "Mesh4D 形变 VAE，3.3 GB（作者 Google Drive，无许可证声明，按仅限研究对待）"),
    "denoiser": (_DRIVE.format("1jeNwiP9-B1uyKvk3_yBN7Y9-D7YQxBc5"), "ckpt/denoiser.ckpt",
                 "ba7a6842a4943467f1faccf86d7cfb11b08b35eace92075bd9ae617931ca448c",
                 "Mesh4D 形变去噪网络（MoE DiT），24 GB：作者放出的是带优化器状态的训练存档"
                 "（作者 Google Drive，无许可证声明，按仅限研究对待）"),
}

# Hunyuan3D-2.1: the shape half (dit) and the shape VAE the deformation net conditions on.
# Upstream's smart_load_model looks for them at $HY3DGEN_MODELS/<repo id>/<subfolder>.
HY3D21 = "tencent/Hunyuan3D-2.1"
HY3D21_REVISION = "0b94677654c57bb9a6b6845cd7b704ccf551d327"


class Mesh4D(Extension):
    name = "mesh4d"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Mesh4D"
    summary = "单目 4D 网格重建的前馈模型；给一段动态物体的单目视频，恢复它完整的三维形状和运动（表示成形变场）"
    homepage = "https://mesh-4d.github.io/"
    source = GitSource(url=MESH4D_URL, commit=MESH4D_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,
        name="仓库没有许可证 + Tencent Hunyuan 3D 2.1 社区许可（不含欧盟、英国、韩国）",
        url="https://github.com/jzr99/Mesh4D",
        summary=(
            "仅限研究，标签按宁严勿松取最严的一档，三条理由叠在一起："
            "① Mesh4D 仓库本身**没有任何许可证文件**，作者也没在别处声明授权条款——没有授权就等于"
            "没有给出使用许可，只能当作论文附带的研究代码；两个权重（形变 VAE、去噪网络）同样没有声明。"
            "② 它的推理代码整个是腾讯 Hunyuan3D-2.1（hy3dshape / hy3dpaint），受「Tencent Hunyuan 3D 2.1"
            "社区许可协议」约束：该协议明确**不适用于欧盟、英国和韩国**，并禁止在这些地区使用它的代码、"
            "权重和输出结果；生成网格用的 hunyuan3d-dit-v2-1 权重也是这一份许可。"
            "③ 环境里的 PyMeshLab 是 GPL-3.0。"
            "另：Hunyuan3D-2.1 的 NOTICE 里列出的第三方组件（Stable Diffusion 的 MIT + CreativeML Open "
            "RAIL++-M 等）各自的条款也要一并遵守。"
        ),
    )
    import_repo = None  # the worker imports from the composed tree (worker_env PYTHONPATH), not from repo/
    worker_modules = ("codebase.py",)
    # torch 2.5.1 is the version pinned upstream and has no sm_120 (RTX 5090) kernels; this build runs on Ada (4090)
    # only. A newer torch would require replacing the pytorch-lightning 1.9.5 and torch_cluster wheels too (untested).
    env_archs = ("sm_89",)
    disk_gb = 45.0  # environment ~12 GB + weights ~36 GB (denoiser 24 GB + deformation VAE 3.3 GB + Hunyuan3D 8 GB)
    env = EnvSpec(
        python="3.10",  # upstream: Python 3.10.18
        torch=("torch==2.5.1", "torchvision==0.20.1"),  # upstream: PyTorch 2.5.1+cu124
        torch_backend="cu124",
        # torch_cluster (missing from upstream requirements.txt, imported by attention_blocks_deform.py):
        # the PyG wheel prebuilt for this torch / CUDA pair, so no CUDA kernels are compiled at install time
        compiled=(
            "torch_cluster @ https://data.pyg.org/whl/torch-2.5.0%2Bcu124/"
            "torch_cluster-1.6.3%2Bpt25cu124-cp310-cp310-linux_x86_64.whl",
        ),
        compiled_cuda=False,  # prebuilt wheel: nvcc is not needed at install time
        build="build_mesh4d.py",
        build_files=("codebase.py",),
        pickled_checkpoints=True,  # the authors' two .ckpt files are pickled Lightning archives
        imports=("trimesh", "open3d", "timm", "torch_cluster", "pymeshlab"),
    )
    weights = tuple(
        Weight(key=key, kind="url", source=url, dest=dest, sha256=sha, note=note)
        for key, (url, dest, sha, note) in CKPTS.items()
    ) + (
        Weight(key="hunyuan3d-shape", kind="hf", source=HY3D21, revision=HY3D21_REVISION,
               dest=f"hy3dgen/{HY3D21}", files=("hunyuan3d-dit-v2-1/*", "hunyuan3d-vae-v2-1/*"),
               note="Hunyuan3D-2.1 生成网格的那一半（DiT 7.4 GB + 形状 VAE 0.7 GB），"
                    "腾讯 Hunyuan 3D 2.1 社区许可（不含欧盟、英国、韩国）",
               notice="Powered by Tencent Hunyuan"),
        Weight(key="dinov2-large", kind="hf", source="facebook/dinov2-large",
               revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
               dest="hf/hub/models--facebook--dinov2-large", files=("*.json", "*.safetensors"),
               note="DINOv2-Large 图像编码器（Meta，Apache-2.0），1.2 GB：形变网络看画面用的就是它"),
    )

    def worker_env(self) -> dict[str, str]:
        root, cache = self.paths.root, self.paths.root / "cache"
        code = root / "codebase"
        return {
            # upstream's two sys.path entries (`cd hy3dshape` + `sys.path.insert(0, '../hy3dpaint')`),
            # pointing into the composed tree, not the original checkout
            "PYTHONPATH": f"{code / 'hy3dshape'}:{code / 'hy3dpaint'}",
            "MESH4D_CODE_BASE": str(code),
            # root under which upstream smart_load_model looks for the Hunyuan3D weights
            "HY3DGEN_MODELS": str(root / "weights" / "hy3dgen"),
            "HF_HOME": str(root / "weights" / "hf"),  # DINOv2 is read from here, without network access
            # every file in the composed tree is a symlink, and Python would write __pycache__ next to the real file,
            # i.e. into the pinned checkout. The checkout must stay unmodified, including .pyc files
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
            "HOME": str(cache / "home"),
            "MPLBACKEND": "Agg",  # upstream imports pyplot; there is no display here
        }


EXTENSION = Mesh4D()
