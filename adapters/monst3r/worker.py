"""MonST3R worker: dynamic-scene reconstruction (cameras + depth). Runs inside
third_party/monst3r/.venv with the pinned repo on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>        (job["node"] == "monst3r.reconstruct")

Frames (every `step`-th), in chunks of `max_frames` (the global alignment keeps every
frame pair of a chunk on the GPU, one chunk at a time) -> per chunk, what upstream demo.py
does: pairs of frames in a sliding window with strides ("swinstride") -> MonST3R point maps
of every pair -> optical flow of every pair (SEA-RAFT) -> moving-object masks (flow that the
camera motion cannot explain, refined through the chunk by SAM 2.1) -> global alignment
(point maps + flow on static pixels + camera smoothness) -> one camera per frame, one
focal length, a depth map per frame. Chunks are stitched by a similarity fitted on their
shared frames (lab2shot_worker.recon); later chunks keep the first chunk's focal.

Not upstream's window_wise continuation (demo.py --window_wise --prev_output_dir: a window's
first frames frozen at the previous window's results, the rest optimized against them),
even streamed one window at a time: measured (RTX 5090, 24 frames, max_frames 12,
focal given, ATE Sim3 vs ground truth) it drifts more than independent chunks + stitching,
C05 1.13 vs 0.80 cm, R04 0.60 vs 0.57 cm (a 2/3 overlap gave the same, 1.12 / 0.61 cm). The
frozen frames are the previous window's last ones, its least reliable cameras.

    raw/cameras.npz   frames [F], K [F,3,3] (pixels, input resolution), cam_to_world [F,4,4]
                      (OpenCV; world = first solved frame's camera; ARBITRARY scale)
    raw/frame_<n>.npz depth [H,W] (camera Z, same arbitrary unit), confidence [H,W]
                      (MonST3R conf >= 1), mask [H,W] (confident and static),
                      moving [H,W] (MonST3R's moving-object mask, bool)

Moving objects: MonST3R finds them itself; the optional "mask" input is added to its own
mask after the SAM 2.1 refinement (so it is kept out of the flow loss, the output mask and
the chunk stitching, without SAM 2.1 growing it). There is no "boxes" input: upstream
takes a mask image only; boxes become one through the explicit 「人物框转遮罩」 node.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lab2shot_worker import MemoryBound, fail, progress, read_frame, reason, recon, resident, say, serve
from lab2shot_worker.run import Run

# the cut3r adapter's module (MonST3R requires cut3r: its folder is on the worker's path)
import dust3r_input as di  # noqa: E402

MODEL_DIR = "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"
RAFT_FILE = "Tartan-C-T-TSKH-spring540x960-M/model.safetensors"
SAM2_FILE = "sam2.1-hiera-large/sam2.1_hiera_large.pt"
# Frames per chunk that keep the batched global alignment under ~18 GB of GPU memory at 288 x 512
# (measured: 40 frames -> 18.9 GB peak); scaled by pixel count for other input sizes.
FRAMES_AT_288X512 = 36
# the parameter memory grows with, reported when memory runs out: 每段最多帧数, stepping down 40, 32, 24, 16, 12 (40 measured 18.9 GB)
MAX_FRAMES = MemoryBound.parameter("max_frames", (40, 32, 24, 16, 12))
# Upstream demo settings, kept unchanged (not tuned per shot):
WINDOW = 5  # scene graph "swinstride-5": each frame paired with the next 5 frames at stride 2
TEMPORAL_SMOOTHING = 0.01  # weight of the camera smoothness term
FLOW_LOSS_WEIGHT = 0.01  # weight of the optical-flow term on static pixels
BATCH_SIZE = 16  # frame pairs per network batch
OVERLAP = 8  # frames two consecutive chunks share (to stitch them)
RESOLUTION = 512  # long side of the network input: the training size


def read_params(params: dict) -> dict:
    """The node's parameters (max_frames None: what fits in ~18 GB at this resolution, see default_max_frames;
    focal_px: a known lens in pixels, fixed during the solve), "auto" resolution filled in."""
    return {**params, "resolution": params["resolution"] or RESOLUTION}


def default_max_frames(geo: di.Geometry) -> int:
    return max(8, min(64, int(FRAMES_AT_288X512 * 288 * 512 / (geo.w * geo.h))))


def stub_evo() -> None:
    """dust3r/utils/misc.py imports dust3r/utils/vo_eval.py, which imports `evo` (trajectory
    benchmark tool, GPL-3.0) at module level; only the evaluation scripts use it. Not
    installed: empty stand-ins let the import succeed and fail loudly if anything calls them."""
    import types

    class Missing:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return Missing(f"{self.name}.{attr}")

        def __call__(self, *args, **kwargs):
            raise RuntimeError(f"{self.name}: evo (GPL-3.0) is not installed; only MonST3R's evaluation uses it")

    def module_getattr(name):
        def get(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return Missing(f"{name}.{attr}")
        return get

    for name in ("evo", "evo.main_ape", "evo.main_rpe", "evo.core", "evo.core.sync", "evo.core.metrics",
                 "evo.core.trajectory", "evo.tools", "evo.tools.file_interface", "evo.tools.plot"):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__getattr__ = module_getattr(name)
        sys.modules.setdefault(name, module)  # a resident process keeps the first ones


def import_upstream(repo: Path, weights: Path, loaded: dict):
    """The pinned code, with its hard-coded checkpoint paths pointed at weights/ (repo untouched).

    Upstream builds SEA-RAFT and SAM 2.1 inside every PointCloudOptimizer (optimizer.py:265 load_RAFT, :382
    build_sam2_video_predictor), so a shot in N chunks would load both N times. `loaded` (one dict per job) keeps the
    first ones for the job's later chunks: the same networks, eval mode, no state carried between calls (SAM 2.1's
    per-video state lives in init_state's return value)."""
    stub_evo()
    for folder in (repo, repo / "third_party" / "sam2"):  # sam2: upstream installs it with pip -e
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    from dust3r.cloud_opt import optimizer
    from sam2.build_sam import build_sam2_video_predictor
    from third_party import raft

    optimizer.sam2_checkpoint = str(weights / SAM2_FILE)

    def sam2_once(config, checkpoint, *args, **kwargs):
        key = ("sam2", config, checkpoint, str(kwargs.get("device")))
        if key not in loaded:
            loaded[key] = build_sam2_video_predictor(config, checkpoint, *args, **kwargs)
        return loaded[key]

    optimizer.build_sam2_video_predictor = sam2_once

    def load_sea_raft(*_args, **_kwargs):
        if "raft" not in loaded:
            loaded["raft"] = build_sea_raft()
        return loaded["raft"]

    def build_sea_raft():
        """third_party/raft.load_RAFT for the SEA-RAFT (RAFT2) checkpoint, from the Hugging Face
        safetensors; its ResNet backbone is not pre-initialized from torchvision (every weight
        comes from the checkpoint), so nothing is downloaded."""
        import argparse

        from extractor import ResNetFPN
        from safetensors.torch import load_model

        cfg = raft.json_to_args(str(repo / "third_party/RAFT/core/configs/congif_spring_M.json"))
        args = argparse.Namespace(**{**vars(cfg), "model": None, "device": "cuda"})
        orig_init = ResNetFPN.__init__

        def init(self, args, input_dim=3, output_dim=256, ratio=1.0, norm_layer=torch.nn.BatchNorm2d, init_weight=False):
            orig_init(self, args, input_dim, output_dim, ratio, norm_layer, False)

        ResNetFPN.__init__ = init
        try:
            net = raft.RAFT2(args)
        finally:
            ResNetFPN.__init__ = orig_init
        # The Hub file stores each shared tensor once (bn3 = downsample.1): load_model restores the aliases
        # (it only trips over the unused BatchNorm step counters of the aliases).
        missing, unexpected = load_model(net, str(weights / RAFT_FILE), strict=False)
        if missing or any(not k.endswith("num_batches_tracked") for k in unexpected):
            fail("E-MONST3R-RAFTWEIGHTS", missing=list(missing[:3]), unexpected=list(unexpected[:3]))
        return net.eval()

    optimizer.load_RAFT = load_sea_raft
    return optimizer


@resident
def load_model(model_dir: Path, device: torch.device):
    """As upstream demo.py with its .pth checkpoint (dust3r.model.load_model): plain patch embedding, portrait
    frames fed as they are. The Hugging Face config says ManyAR_PatchEmbed + landscape_only, which only accepts
    landscape tensors. (SEA-RAFT and SAM 2.1 are built by upstream's optimizer, once per job: import_upstream.)"""
    from dust3r.model import AsymmetricCroCo3DStereo

    return AsymmetricCroCo3DStereo.from_pretrained(str(model_dir), patch_embed_cls="PatchEmbedDust3R",
                                                   landscape_only=False).to(device).eval()


def make_views(images: list[np.ndarray], moving: np.ndarray) -> list[dict]:
    """Upstream dust3r.utils.image.load_images output, from frames already at the model size."""
    views = []
    for i, rgb in enumerate(images):
        img = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        views.append({
            "img": img.sub(0.5).div(0.5)[None],
            "true_shape": np.int32([rgb.shape[:2]]),
            "idx": i,
            "instance": str(i),
            "mask": ~(img.sum(0) <= 0.01)[None],
            "dynamic_mask": torch.from_numpy(moving[i])[None],
        })
    return views


def grown(masks: np.ndarray) -> np.ndarray:
    """upstream enlarge_seg_masks: grow the moving masks a little (3x3)."""
    return np.stack([cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for m in masks])


def run_chunk(model, optimizer, images: list[np.ndarray], moving_in: np.ndarray, p: dict, known_focal: float | None,
              device: torch.device) -> tuple[recon.Chunk, bool]:
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.cloud_opt.base_opt import global_alignment_iter
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference

    n = len(images)
    views = make_views(images, moving_in)
    graph = f"swinstride-{min(WINDOW, max(1, (n - 1) // 2))}-noncyclic"
    pairs = make_pairs(views, scene_graph=graph, prefilter=None, symmetrize=True)
    progress(0, 1, "pairs", count=len(pairs))
    output = inference(pairs, model, device, batch_size=BATCH_SIZE, verbose=False)

    # MonST3R's own moving-object mask (flow check, refined through the chunk by SAM 2.1), then the
    # job's masks added on top: after the refinement, so SAM 2.1 does not grow the (boxy) inputs.
    cls = optimizer.PointCloudOptimizer
    last_step = "refine_motion_mask_w_sam2" if p["sam2_refine"] else "get_motion_mask_from_pairs"
    orig_step = getattr(cls, last_step)

    def with_inputs(self, *args):
        orig_step(self, *args)
        # The mask upstream exports: `base_opt.py:349 save_dynamic_masks` exports the SAM 2.1 mask when available,
        # otherwise the flow mask. The 「运动物体遮罩」 output delivers this mask. The union below is used internally by
        # the upstream optimizer (`optimizer.py:434`) and only excludes pixels from the solve; exporting the union would
        # include spurious flow fragments (nearby ground or shrubs detected as moving).
        official = getattr(self, "sam2_dynamic_masks", None) or self.dynamic_masks
        self.official_masks = [m.to(self.device).detach().bool().clone() for m in official]
        self.dynamic_masks = [m.to(self.device).bool() | torch.from_numpy(moving_in[i]).to(self.device)
                              for i, m in enumerate(self.dynamic_masks)]

    setattr(cls, last_step, with_inputs)
    try:
        scene = global_aligner(
            output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False,
            shared_focal=True, temporal_smoothing_weight=TEMPORAL_SMOOTHING, translation_weight=1.0,
            flow_loss_weight=FLOW_LOSS_WEIGHT, flow_loss_start_epoch=0.1, flow_loss_thre=25,
            use_self_mask=True, sam2_mask_refine=p["sam2_refine"], motion_mask_thre=p["motion_threshold"],
            num_total_iter=p["niter"], empty_cache=n > 72, batchify=True,
        )
    finally:
        setattr(cls, last_step, orig_step)
    del output
    if known_focal is not None:
        scene.preset_focal([known_focal])  # shared focal: one parameter
        scene.im_focals.requires_grad_(False)
    # upstream compute_global_alignment(init="mst", niter, schedule="linear", lr=0.01), with progress
    scene.compute_global_alignment(init="mst", niter=0)
    params = [q for q in scene.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=0.01, betas=(0.9, 0.9))
    for it in range(p["niter"]):
        global_alignment_iter(scene, it, p["niter"], 0.01, 1e-3, opt, "linear")
        if (it + 1) % 25 == 0 or it + 1 == p["niter"]:
            progress(it + 1, p["niter"], "global_optimization")

    with torch.no_grad():
        c2w = scene.get_im_poses().detach().double().cpu().numpy()
        focal = scene.get_focals().detach().double().cpu().numpy().reshape(-1)
        pp = scene.get_principal_points().detach().double().cpu().numpy()
        depth = np.stack([d.detach().float().cpu().numpy() for d in scene.get_depthmaps()])
        scene.min_conf_thr = p["conf_threshold"]
        scene.thr_for_init_conf = True  # as upstream demo's get_3D_model_from_scene
        valid = np.stack([m.detach().cpu().numpy() for m in scene.get_masks()])
        conf = np.stack([c.detach().float().cpu().numpy() for c in scene.init_conf_maps])
        # the solve excludes the upstream internal union (flow ∪ SAM 2.1 ∪ wired masks); the output is the mask upstream exports
        solving = np.stack([m.detach().bool().cpu().numpy() for m in scene.dynamic_masks])
        official = np.stack([m.cpu().numpy() for m in scene.official_masks])
    # upstream enlarge_seg_masks: grow the moving masks a little (3x3)
    moving, official = grown(solving), grown(official)
    flow_dropped = bool(getattr(scene, "flow_loss_flag", False))
    del scene
    res = recon.Chunk(cam_to_world=c2w, K=di.model_K(focal, pp), depth=depth, confidence=conf, usable=valid & ~moving,
                      extra={"moving": official})
    return res, flow_dropped


def main(job_path: str) -> None:
    run = Run.start(job_path, "monst3r.reconstruct", "MonST3R")
    job = run.job
    all_frames = job.frames
    height, width = read_frame(all_frames[0][1]).shape[:2]
    p = read_params(job.params)
    weights = job.weights_dir
    run.weights(*(weights / rel for rel in (MODEL_DIR + "/model.safetensors", MODEL_DIR + "/config.json",
                                             RAFT_FILE, SAM2_FILE)))
    if not (job.repo_dir / "croco" / "models").is_dir():
        fail("E-MONST3R-CROCO")

    used = all_frames[::p["step"]]
    if len(used) < 3:
        fail("E-WORKER-TOOFEWFRAMES", least=3, have=len(used), why=reason("I-WORKER-WHYSTEP", step=p["step"]))
    frames = [f for f, _ in used]
    raw = job.raw_dir
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # as upstream demo.py

    run.stage("read_frames")
    geo = di.Geometry.for_size(width, height, p["resolution"])
    p["max_frames"] = p["max_frames"] or default_max_frames(geo)
    images = di.read_images(used, geo, read_frame, progress)
    moving_in = di.moving_at_model_size(job, frames, geo, width, height)

    loaded: dict = {}  # SEA-RAFT and SAM 2.1, built once for the job's chunks (import_upstream)

    def load():
        """The pinned code with its checkpoint paths, then the model (both under the loading stage's clock)."""
        return import_upstream(job.repo_dir, weights, loaded), load_model(weights / MODEL_DIR, device)

    # run.model caps the GPU first: an allocation beyond the free memory fails instead of spilling into RAM
    optimizer, model = run.model("load_model", load, stage_params={"model": "MonST3R"})

    def solve(max_frames: int):
        """The whole shot in chunks of at most `max_frames`, stitched; the chunks whose flow check was dropped."""
        overlap = min(OVERLAP, max_frames - 1)
        known_focal = geo.focal_to_model(p["focal_px"]) if p["focal_px"] else None
        chunks = recon.plan_chunks(len(frames), max_frames, overlap)
        stitch = recon.Stitcher(rotation="points")  # the chunk-end cameras are less reliable than the points
        flow_dropped = []
        for ci, (a, b) in enumerate(chunks):
            run.stage("reconstruct_segment", segment=ci + 1, segments=len(chunks)) if len(chunks) > 1 else run.stage("reconstruct")
            res, dropped = run_chunk(model, optimizer, images[a:b], moving_in[a:b], p, known_focal, device)
            if dropped:
                flow_dropped.append([frames[a], frames[b - 1]])
            if known_focal is None:
                known_focal = float(np.median(res.K[:, 0, 0]))  # one lens for the whole shot
            try:
                stitch.add(a, res).update(frames=[frames[a], frames[b - 1]])
            except ValueError:
                fail("E-MONST3R-STITCH")
            torch.cuda.empty_cache()
        return max_frames, chunks, stitch, flow_dropped

    t_inf = time.time()
    try:
        p["max_frames"], chunks, stitch, flow_dropped = run.fit(MAX_FRAMES, solve, min(p["max_frames"], len(frames)))
    finally:
        loaded.clear()  # a resident worker keeps the MonST3R model only, not the job's flow and SAM 2.1 networks
        torch.cuda.empty_cache()
    infer_seconds = time.time() - t_inf
    run.frame_seconds.extend([infer_seconds / len(frames)] * len(frames))  # the chunks are one pass over the shot: shared out per frame
    for f0, f1 in flow_dropped:
        say("W-MONST3R-FLOWDROPPED", first=f0, last=f1)

    run.stage("write_results")
    # the moving-object mask is written as its own array too (MonST3R estimates it; useful downstream)
    stitched = list(stitch.pop())
    summary = di.write_outputs(raw, frames, geo, stitched, np.array([f.K[0, 0] for f in stitched]), progress)
    for info in stitch.alignments[1:]:
        if info["residual_rms_relative"] > 0.05:
            say("W-MONST3R-SEAM", first=info["frames"][0], last=info["frames"][1], residual=float(info["residual_rms_relative"]))

    run.finish(
        frames,
        kind="reconstruction",
        extension="monst3r",
        model="MonST3R PO-TA-S-W ViT-L 512 DPT + SEA-RAFT spring-M + SAM 2.1 Hiera-L",
        files="cameras.npz: frames, K, cam_to_world, width, height; frame_<n>.npz: depth, confidence, mask, moving",
        convention=recon.CONVENTION,
        metric=False,
        units="arbitrary (MonST3R is scale-free; one unit is the same everywhere in the shot)",
        confidence="MonST3R per-frame confidence before alignment (>= 1, higher = surer)",
        mask="confidence > conf_threshold, inside the network's crop, minus moving objects",
        # the exported mask is the one upstream save_dynamic_masks exports (SAM 2.1 when available, otherwise flow).
        # The solve additionally uses the upstream internal union (plus wired masks), which is not exported
        moving_objects="SAM 2.1" if p["sam2_refine"] else "MonST3R flow check",
        solve_mask="MonST3R flow check" + (" + SAM 2.1" if p["sam2_refine"] else "")
        + (" + job mask" if "mask" in job.inputs else ""),
        params=p,
        focal_source="user" if p["focal_px"] else "monst3r (first chunk; later chunks keep it)",
        frames=frames,  # the whole list, not the standard [first, last]: the converter reads every frame's file by it
        width=width,
        height=height,
        geometry=geo.describe(),
        chunks=[[frames[a], frames[b - 1]] for a, b in chunks],
        chunk_alignment=stitch.alignments,
        **summary,
        inference_seconds=round(infer_seconds, 1),
    )


if __name__ == "__main__":
    serve(main)
