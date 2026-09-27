"""One preprocessing step of Pixel3DMM, run in a process of its own.

    python steps.py <step> <json arguments>

Upstream runs its four preprocessing steps through `os.system` (scripts/run_preprocessing.py), one process each,
and that is not an accident: MICA, PIPNet and facer each carry a top-level `utils`, `configs`, `models` and
`datasets` package, and the three collide as soon as two of them are imported into one interpreter (MICA's
`from utils.masking import Masking` finds PIPNet's FaceBoxesV2/utils). So each step gets its own process here
too, with only the paths that step needs on sys.path — never the whole set.

The step's own prints go to the worker's log; what the node shows (stage, progress) the worker says around it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codebase import Layout  # noqa: E402

VIDEO = "shot"  # upstream keys every folder by a "video name"


def load_script(path: Path):
    """One of upstream's scripts/*.py as a module (they guard only tyro.cli behind __main__)."""
    spec = importlib.util.spec_from_file_location(f"p3dmm_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def on_path(*folders: Path) -> None:
    for folder in folders:
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))


def no_downloads() -> None:
    """The two things upstream would fetch at run time: MICA's unused landmark library (it imports
    face_alignment at module level and uses it only in its FAN branch; 1.3.3 also calls scipy.integrate.simps,
    gone since scipy 1.14) and the ViT backbone timm would pull from the Hub — Pixel3DMM's own checkpoints
    carry it (net.img_encoder.model.*)."""
    import timm

    sys.modules.setdefault("face_alignment", types.ModuleType("face_alignment"))
    create_model = timm.create_model
    timm.create_model = lambda *a, **kw: create_model(*a, **{**kw, "pretrained": False})


# ------------------------------------------------------------------ the steps


def crop(lay: Layout, args: dict) -> None:
    """PIPNet's FaceBoxes on every frame -> one square crop box for the whole shot, 512 x 512 crops and the
    98 WFLW landmarks. run_cropping.py puts FaceBoxesV2 on sys.path itself."""
    load_script(lay.code_base / "scripts" / "run_cropping.py").run(
        "experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py", args["rgb"], start_frame=-1,
        vertical_crop=False, static_crop=True, max_bbox=True, disable_cropping=False)


def mica(lay: Layout, args: dict) -> None:
    """The identity prior (300 FLAME shape coefficients) from about ten frames of the shot."""
    from argparse import Namespace

    on_path(lay.mica)
    from configs.config import get_cfg_defaults  # MICA's own

    cfg = get_cfg_defaults()
    cfg.model.testing = True
    data = Path(args["data"])
    load_script(lay.mica / "demo.py").main(
        cfg, Namespace(i=str(data / "cropped"), o=str(data / "mica"), a=str(data / "arcface"),
                       m=str(lay.mica / "data" / "pretrained" / "mica.tar")))


def segment(lay: Layout, args: dict) -> None:
    """facer / FaRL: the face parsing map the UV and normal losses are masked by."""
    on_path(lay.facer)
    load_script(lay.code_base / "scripts" / "run_facer_segmentation.py").main(VIDEO)


def priors(lay: Layout, args: dict) -> None:
    """The two ViT predictions: the normal map, then the canonical-face UV map."""
    from omegaconf import OmegaConf

    script = load_script(lay.code_base / "scripts" / "network_inference.py")
    base = OmegaConf.load(lay.code_base / "configs" / "base.yaml")
    script.main(OmegaConf.merge(base, {"model": {"prediction_type": args["kind"]}, "video_name": VIDEO}))


def track(lay: Layout, args: dict) -> None:
    """The two-stage fit. `focal_norm`: the node's focal length over the crop's width, upstream's own
    normalisation (focal_length = f / size, and f = focal_px * size / crop width); null lets it solve one.
    Writes the tracker's own folder name, so the worker knows where to read the result."""
    import torch
    from omegaconf import OmegaConf
    from pixel3dmm.tracking.tracker import Tracker

    focal_norm = args["focal_norm"]
    settings = dict(args["settings"])
    if focal_norm is not None:
        settings["lr_f"] = 0.0  # a known lens: keep it, do not optimise it
    cfg = OmegaConf.merge(OmegaConf.load(lay.code_base / "configs" / "tracking.yaml"),
                          {"video_name": VIDEO, "output_folder": args["out"], **settings})

    class LockedLens(Tracker):
        """The focal length the node was given, in place of the one upstream starts from and optimises."""

        def create_parameters(self, timestep, mica_shape):
            super().create_parameters(timestep, mica_shape)
            self.focal_length = torch.tensor([[focal_norm]], dtype=torch.float32, device=self.device)
            self.focal_length.requires_grad = True  # a leaf in the optimiser at lr 0, as the rest of its code expects

    tracker = (Tracker if focal_norm is None else LockedLens)(cfg)
    tracker.run()
    Path(args["wrote"]).write_text(json.dumps({"folder": tracker.actor_name, "size": int(tracker.config.size)}),
                                   encoding="utf-8")


STEPS = {"crop": crop, "mica": mica, "segment": segment, "priors": priors, "track": track}


def main() -> None:
    """Runs one step and records what it used of the GPU: the models live in these processes, so the worker's own
    torch would report almost nothing (it never loads one)."""
    step, args = sys.argv[1], json.loads(sys.argv[2])
    lay = Layout(Path(os.environ["PIXEL3DMM_EXT_ROOT"]))
    on_path(lay.code_base / "src")
    no_downloads()
    STEPS[step](lay, args)
    peaks = Path(os.environ["PIXEL3DMM_PEAKS"])
    import torch

    with peaks.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"step": step, "mb": round(torch.cuda.max_memory_allocated() / 2**20)}) + "\n")


if __name__ == "__main__":
    main()
