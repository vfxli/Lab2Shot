"""The tree Mesh4D's scripts expect, composed out of symlinks next to the pinned checkout.

Upstream's setup builds three things **inside** its own checkout — the Cython modules of
`hy3dshape/im2mesh` (`setup_im2mesh.py build_ext --inplace`), the CUDA rasteriser of
`hy3dpaint/custom_rasterizer` (`pip install -e .`) and the C++ mesh inpainter of
`hy3dpaint/DifferentiableRenderer` (`compile_mesh_painter.sh`) — and downloads its weights
into `ckpt/` and `hy3dpaint/ckpt/`. Lab2Shot never writes into a pinned checkout, so the same
tree is built beside it: every entry is a symlink to the original, except the three folders a
compiler writes into, which are real copies, and the weight folders, which are real folders of
links into `weights/`.

    third_party/mesh4d/
        repo/                  the pinned checkout (never written to)
        weights/
            ckpt/              deform_vae.ckpt, denoiser.ckpt (the authors' Google Drive)
            hy3dgen/tencent/Hunyuan3D-2.1/   HY3DGEN_MODELS: the Hunyuan3D-2.1 shape weights
            realesrgan/        RealESRGAN_x4plus.pth (only the texture pass wants it; not downloaded)
            hf/                the Hugging Face cache (DINOv2-large, the image encoder)
        codebase/              built here
            ckpt/              -> weights/ckpt/*.ckpt
            configs DATA assets torchvision_fix.py  -> repo/...
            hy3dshape/         mirrored; im2mesh/ is a real copy (its .so files are built in place)
            hy3dpaint/         mirrored; custom_rasterizer/ and DifferentiableRenderer/ are real
                               copies (compiled in place), ckpt/ -> weights/realesrgan/...

Imported by `build_mesh4d.py` (which composes the tree and compiles in it) and by `worker.py`
(which reads the paths). The composing itself is the worker SDK's (lab2shot_worker.codetree):
Pixel3DMM builds its tree the same way.
"""

from __future__ import annotations

from pathlib import Path

from lab2shot_worker.codetree import copy_tree, link, mirror

# Folders a compiler writes into: copied, not linked (relative to repo/)
BUILT = ("hy3dshape/im2mesh", "hy3dpaint/custom_rasterizer", "hy3dpaint/DifferentiableRenderer")


class Layout:
    """Where everything is, given the extension's third_party/mesh4d folder."""

    def __init__(self, root: Path):
        # the real path: a git worktree links third_party/<name> to the main checkout's, and the
        # symlinks this builds have to keep working when that worktree is gone
        self.root = Path(root).resolve()
        self.repo = self.root / "repo"
        self.weights = self.root / "weights"
        self.code_base = self.root / "codebase"
        self.cache = self.root / "cache"

    @property
    def shape(self) -> Path:
        """Upstream runs everything from here (`cd hy3dshape; python infer.py`)."""
        return self.code_base / "hy3dshape"

    @property
    def paint(self) -> Path:
        """`sys.path.insert(0, '../hy3dpaint')`: the texture pass's own package root."""
        return self.code_base / "hy3dpaint"

    @property
    def hy3dgen_models(self) -> Path:
        """HY3DGEN_MODELS: upstream's `smart_load_model` looks for <this>/<repo id>/<subfolder>."""
        return self.weights / "hy3dgen"

    def sys_paths(self) -> list[str]:
        """What the worker puts on sys.path, in order (upstream's own two entries)."""
        return [str(self.shape), str(self.paint)]


def compose(root: Path) -> Layout:
    """Build (or rebuild) the tree. Weight symlinks may dangle until the installer's weights step:
    nothing reads them before a worker runs."""
    lay = Layout(root)

    # the two package roots, mirrored; the compiled folders (None: copy_tree below) and the
    # weight folders are real
    mirror(lay.repo / "hy3dshape", lay.shape, {"im2mesh": None})
    mirror(lay.repo / "hy3dpaint", lay.paint, {
        "custom_rasterizer": None,
        "DifferentiableRenderer": None,
        # upstream's `wget … -P hy3dpaint/ckpt`
        "ckpt/RealESRGAN_x4plus.pth": lay.weights / "realesrgan" / "RealESRGAN_x4plus.pth",
    })
    for name in ("configs", "DATA", "assets", "torchvision_fix.py", "requirements.txt", "README.md"):
        link(lay.code_base / name, lay.repo / name)
    # upstream's `mkdir ckpt; gdown … -O ./ckpt/`, read as ../ckpt/<name> from hy3dshape/
    for ckpt in ("deform_vae.ckpt", "denoiser.ckpt"):
        link(lay.code_base / "ckpt" / ckpt, lay.weights / "ckpt" / ckpt)

    for rel in BUILT:
        copy_tree(lay.repo / rel, lay.code_base / rel)
    return lay
