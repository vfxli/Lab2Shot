"""The code base upstream's `pixel3dmm.env_paths` expects, composed out of symlinks.

Upstream's install_preprocessing_pipeline.sh clones facer, MICA and PIPNet over SSH
*into* the pixel3dmm checkout, copies its replacement files over theirs, and downloads
weights into the checkout as well. Lab2Shot never writes into a pinned checkout, so the
same tree is built next to them instead: every entry is a symlink to the original, and
only the directories leading to a replaced or added file are real. Nothing is copied.

    third_party/pixel3dmm/
        repo/ facer/ MICA/ PIPNet/   the four pinned checkouts (never written to)
        weights/                     the downloads
        codebase/                    PIXEL3DMM_CODE_BASE, built here
            assets configs scripts   -> repo/...
            pretrained_weights/      -> weights/uv.ckpt, weights/normals.ckpt
            preproc/facer            facer with its two replaced files
            preproc/MICA             MICA with its two replaced files, its checkpoint and FLAME
            preproc/PIPNet           PIPNet with its snapshot and a real folder for the built NMS
            src/                     -> repo/src, with preprocessing/{MICA,PIPNet} added
        cache/home/                  the worker's HOME (insightface only looks in ~/.insightface)

Imported by build_p3dmm.py (which composes the tree) and by worker.py (which reads the
paths). The composing itself is the worker SDK's (lab2shot_worker.codetree): Mesh4D builds
its tree the same way.
"""

from __future__ import annotations

from pathlib import Path

from lab2shot_worker.codetree import link, mirror

# Upstream file -> the file of pixel3dmm's preprocessing/replacement_code/ that replaces it
# (install_preprocessing_pipeline.sh's `cp` lines). replacement_code/pipnet_demo.py is copied
# nowhere: pixel3dmm's own preprocessing/pipnet_utils.py is what run_cropping.py calls.
FACER_REPLACED = {"facer/face_parsing/farl.py": "farl.py", "facer/transform.py": "facer_transform.py"}
MICA_REPLACED = {"demo.py": "mica_demo.py", "micalib/models/mica.py": "mica.py"}

# PIPNet's snapshot, as pipnet_utils.demo_image builds the path: snapshots/<data>/<experiment>/epoch<n-1>.pth
PIPNET_SNAPSHOT = "snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth"
# MICA's configs/config.py; env_paths.FLAME_ASSETS is the same data/ folder
MICA_CHECKPOINT = "data/pretrained/mica.tar"
# the Cython NMS build_p3dmm.py compiles; naming a file inside it makes nms/ a real folder
PIPNET_NMS = "FaceBoxesV2/utils/nms/__init__.py"


class Layout:
    """Where everything is, given the extension's third_party/pixel3dmm folder."""

    def __init__(self, root: Path):
        # the real path: a git worktree links third_party/<name> to the main checkout's, and the symlinks this
        # builds have to keep working when that worktree is gone
        self.root = Path(root).resolve()
        self.repo = self.root / "repo"
        self.facer_repo = self.root / "facer"
        self.mica_repo = self.root / "MICA"
        self.pipnet_repo = self.root / "PIPNet"
        self.weights = self.root / "weights"
        self.code_base = self.root / "codebase"
        self.cache = self.root / "cache"

    @property
    def replacements(self) -> Path:
        return self.repo / "src" / "pixel3dmm" / "preprocessing" / "replacement_code"

    @property
    def facer(self) -> Path:
        """The composed facer checkout; `import facer` needs this folder on sys.path."""
        return self.code_base / "preproc" / "facer"

    @property
    def mica(self) -> Path:
        """The composed MICA; on sys.path too (its code imports `models`, `micalib`, `datasets`)."""
        return self.code_base / "preproc" / "MICA"

    @property
    def pipnet(self) -> Path:
        return self.code_base / "preproc" / "PIPNet"

    @property
    def faceboxes(self) -> Path:
        """PIPNet's detector; on sys.path too (`from detector import Detector`, `from utils.config import cfg`)."""
        return self.pipnet / "FaceBoxesV2"

    @property
    def flame_assets(self) -> Path:
        """env_paths.FLAME_ASSETS: FLAME2020/generic_model.pkl and FLAME2023/ are linked here at run time."""
        return self.mica / "data"

    @property
    def home(self) -> Path:
        return self.cache / "home"

    def sys_paths(self) -> list[str]:
        """What the worker puts on sys.path, in order."""
        return [str(self.code_base / "src"), str(self.facer), str(self.mica), str(self.faceboxes)]


def compose(root: Path) -> Layout:
    """Build (or rebuild) the code base. Weight symlinks may dangle until the installer's weights step:
    nothing reads them before a worker runs."""
    lay = Layout(root)
    rep = lay.replacements

    mirror(lay.facer_repo, lay.facer, {k: rep / v for k, v in FACER_REPLACED.items()})
    mirror(lay.mica_repo, lay.mica, {**{k: rep / v for k, v in MICA_REPLACED.items()},
                                     MICA_CHECKPOINT: lay.weights / "mica" / "mica.tar"})
    mirror(lay.pipnet_repo, lay.pipnet, {PIPNET_SNAPSHOT: lay.weights / "pipnet" / "epoch59.pth",
                                         PIPNET_NMS: lay.pipnet_repo / "FaceBoxesV2" / "utils" / "nms" / "__init__.py"})
    # the package tree: preprocessing/ must be a real folder, so MICA and PIPNet can be added inside it
    mirror(lay.repo / "src", lay.code_base / "src",
           {"pixel3dmm/preprocessing/MICA": lay.mica, "pixel3dmm/preprocessing/PIPNet": lay.pipnet})

    for name in ("assets", "configs", "scripts"):
        link(lay.code_base / name, lay.repo / name)
    for ckpt in ("uv.ckpt", "normals.ckpt"):
        link(lay.code_base / "pretrained_weights" / ckpt, lay.weights / ckpt)
    link(lay.home / ".insightface" / "models", lay.weights / "insightface" / "models")
    return lay
