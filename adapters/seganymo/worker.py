"""SegAnyMo worker: masks of the objects that really move in a shot. Runs inside third_party/seganymo/.venv with the
pinned repo's core/, its SAM 2 copy (sam2/) and its PyTorch TAPIR (preproc/tapnet_torch) on PYTHONPATH; never
imports Lab2Shot core.

    python worker.py <job.json>

Upstream's three command-line steps (core/utils/run_inference.py --depths --tracks --dinos, --motion_seg_infer,
--sam2), run in one process on frames in memory instead of through folders of files. Upstream's models and functions
are used as they are; what the worker does itself is upstream's glue, query_step by query_step:

1. Frames at the processing size: the long side down to params.resolution (upstream's efficiency mode --e: 1000 px,
   cv2 INTER_AREA). Motion is analysed on up to params.analysis_frames of them, spread evenly over the shot (--e keeps 100
   frames of a video); the masks are then made for every frame.
2. Depth of the analysed frames: Depth Anything V2 Small through upstream's run_depth.get_depth_anything_disp
   (upstream uses Large, which is CC-BY-NC), stored as upstream does (uint16 over the frame's min..max) and read back
   through Pillow's I;16 -> L conversion, then normalised to 0..1 per frame, as upstream's inference.py reads it. The
   classifier hardly uses it (traj_oa_depth.gather_point reads the depth at trajectories already normalised to
   -1..1: points of the upper-left quarter all read the corner pixel, the others the wrong place): Video Depth
   Anything's or MoGe's depth instead gave the same masks on every DAVIS / FBMS / SegTrackV2 sequence tried, so the
   node takes no depth input.
3. DINOv2 ViT-B/14 key features (layer 11, stride 7) of every params.query_step-th analysed frame (the query frames):
   upstream's ViTExtractor, rounded to fp16 as upstream saves them.
4. BootsTAPIR tracks at 256 x 256 of a grid of about 9000 points on each query frame (upstream's run_tapir.py grid).
   Upstream tracks every grid point and inference.py then keeps a random 1 / (query frames) of each query frame's
   points and a random 5000 of those; the worker draws the same random selection first and tracks only the points
   that are kept (same distribution, a tenth of the tracking), with the video's feature grids computed once.
5. The motion classifier (moseg.pth, configs/example_train.yaml) on those tracks, depth and DINO features, and
   upstream's threshold ladder -> the dynamic tracks.
6. SAM 2 Hiera-L groups the dynamic tracks into objects (run_sam2.process_invisible_traj) and tracks each object
   through the whole shot from point prompts every 16 analysed frames (run_sam2.main's loop), forwards and
   backwards; objects overlapping on most frames are merged (run_sam2.analyze_frame_merges / merge_masks). SAM 2
   runs on every frame of the shot: prompts on analysed frames go to their place in the shot.

Output: raw/frame_<n>.npz labels uint8 [H,W] at the plate size (0 = static, k = moving object k; an object's edge
is its mask resized bilinearly and cut at 0.5), raw/objects.json [{"id", "frames"}] (frames: how many frames it is
in), raw/result.json.
"""

from __future__ import annotations

import json
import os
import resource
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lab2shot_worker import fail, local_hub, progress, quiet, reason, resident, save_npz, say, serve, set_seed
from lab2shot_worker.run import Run

TAPIR_SIZE = 256  # run_tapir.py: frames resized to 256 x 256
GRID_POINTS = 9000  # run_tapir.py: max_grid_points
TRACKS_KEPT = 5000  # inference.py: total_num
TAPIR_CHUNK = 128  # run_tapir.py: points per model call
DINO_LAYER, DINO_STRIDE = 11, 7  # dino_feat.py
SAM2_PROMPT_EVERY = 16  # run_sam2.py: q_ts = range(0, T, 16)
MERGE_IOU = 0.9  # run_sam2.py: analyze_frame_merges(..., iou_threshold=0.9)
SEED = 0  # upstream draws its point selection unseeded; the worker seeds it so a shot gives the same masks every time


# --------------------------------------------------------------------------- models


@resident
def load_depth_model(model_dir: Path):
    from transformers import pipeline

    return pipeline(task="depth-estimation", model=str(model_dir), device="cuda")


@resident
def load_dino(hub_dir: Path):
    from core.utils.dino_feat import ViTExtractor

    local_hub({"dinov2": hub_dir}, "SegAnyMo")
    with quiet():  # ViTExtractor prints the whole network
        return ViTExtractor(model_type="dinov2_vitb14", stride=DINO_STRIDE)


@resident
def load_tapir(checkpoint: Path):
    from tapnet_torch import tapir_model

    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to("cuda").eval()


@resident
def load_moseg(checkpoint: Path, config: Path):
    from core.network.traj_oa_depth import traj_oa_depth
    from core.utils.utils import load_config_file

    cfg = load_config_file(str(config))  # train_seq.setup_model for model_name traj_oa_depth
    model = traj_oa_depth(cfg.extra_info, cfg.oanet, cfg.pos_embed, cfg.dino, cfg.target_feature_dim, cfg.dino_later,
                          cfg.dino_woatt, cfg.time_att, cfg.tracks, cfg.depths)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
    return model.to("cuda").eval()


@resident
def load_sam2(checkpoint: Path):
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor("sam2_hiera_l.yaml", str(checkpoint))  # run_sam2.py


def moseg_config(repo: Path):
    from core.utils.utils import load_config_file

    return load_config_file(str(repo / "configs" / "example_train.yaml"))


def load(run: Run, what: str, loader, *args):
    """run.model for a worker of five models: Run.loading keeps the last load's time only, so the earlier loads' time
    is added back (load_seconds = all of them)."""
    before = run.load_seconds
    model = run.model(what, loader, *args)
    run.load_seconds += before
    return model


# --------------------------------------------------------------------------- frames


def processing_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """run_inference.resize_images: the long side down to resolution (int() of the scaled size), never up."""
    if max(width, height) <= resolution:
        return width, height
    scale = resolution / max(width, height)
    return int(width * scale), int(height * scale)


def analysed(count: int, most: int) -> list[int]:
    """Indices of the frames motion is analysed on: all, or `most` spread evenly from the first to the last."""
    if count <= most:
        return list(range(count))
    return sorted({int(round(i)) for i in np.linspace(0, count - 1, most)})


def depth_as_read(disp16: np.ndarray) -> torch.Tensor:
    """What inference.py reads: the uint16 PNG run_depth.py saves (imageio), opened with Pillow and converted to "L",
    normalised to 0..1."""
    import io

    import imageio.v2 as iio
    from PIL import Image

    png = io.BytesIO()
    iio.imwrite(png, disp16, format="png")
    png.seek(0)
    img = np.array(Image.open(png).convert("L"))
    rng = img.max() - img.min()
    return torch.from_numpy((img - img.min()) / rng if rng else np.zeros_like(img, np.float64))


# --------------------------------------------------------------------------- tracks


def grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """run_tapir.py's query grid: every g-th pixel, g so that about GRID_POINTS points cover the frame."""
    g = max(1, int(np.sqrt((h * w) / GRID_POINTS)))
    y, x = np.mgrid[0:h:g, 0:w:g]
    return y.ravel(), x.ravel()


def choose_points(n_grid: int, queries: list[int], rng: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    """inference.py's selection, drawn before tracking: a random int(n / Q) grid points per query frame, then a random
    TRACKS_KEPT of all of them. (which query frame each kept track starts on, its grid index), in the kept order."""
    per_query = [torch.randperm(n_grid, generator=rng)[: int(n_grid * (1 / len(queries)))].numpy() for _ in queries]
    owner = np.concatenate([np.full(len(p), q) for q, p in enumerate(per_query)])
    index = np.concatenate(per_query)
    keep = torch.randperm(len(index), generator=rng)[:TRACKS_KEPT].numpy()
    return owner[keep], index[keep]


def track_points(model, frames: np.ndarray, queries: list[int], owner: np.ndarray, index: np.ndarray,
                 grid_yx: tuple[np.ndarray, np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """BootsTAPIR as run_tapir.py runs it, for the kept points only: [N, T, 4] (x, y at the processing size, occlusion
    logit, expected-distance logit), in the kept order. The video's feature grids are computed once."""
    import mediapy as media
    from tapnet_torch import transforms

    w, h = size
    t_count = len(frames)
    video = torch.from_numpy(np.asarray(media.resize_video(frames, (TAPIR_SIZE, TAPIR_SIZE)))).to("cuda")
    video = (video.float() / 255 * 2 - 1)[None]  # run_tapir.preprocess_frames
    y, x = grid_yx
    y_r, x_r = y / (h - 1) * (TAPIR_SIZE - 1), x / (w - 1) * (TAPIR_SIZE - 1)
    out = np.zeros((len(index), t_count, 4), np.float32)
    with torch.inference_mode():
        feature_grids = model.get_feature_grids(video, False, None)
        rows = np.arange(len(index))
        chunks = [rows[i:i + TAPIR_CHUNK] for i in range(0, len(rows), TAPIR_CHUNK)]
        for c, chunk in enumerate(chunks):
            qt = np.array(queries)[owner[chunk]]
            pts = np.stack([qt, y_r[index[chunk]], x_r[index[chunk]]], -1).astype(np.float32)
            pts = torch.from_numpy(pts)[None].to("cuda")
            query_features = model.get_query_features(video, False, pts, feature_grids, None)
            traj = model.estimate_trajectories(video.shape[-3:-1], False, feature_grids, query_features, pts, 64)
            p = model.num_pips_iter
            tracks = torch.mean(torch.stack(traj["tracks"][p::p]), 0)[0].cpu().numpy()
            occ = torch.mean(torch.stack(traj["occlusion"][p::p]), 0)[0].cpu().numpy()
            dist = torch.mean(torch.stack(traj["expected_dist"][p::p]), 0)[0].cpu().numpy()
            tracks = transforms.convert_grid_coordinates(tracks, (TAPIR_SIZE - 1, TAPIR_SIZE - 1), (w - 1, h - 1))
            out[chunk] = np.concatenate([tracks, occ[..., None], dist[..., None]], -1)
            # run_tapir.py: on its own query frame a point sits exactly on its grid pixel
            out[chunk, qt, 0], out[chunk, qt, 1] = x[index[chunk]], y[index[chunk]]
            progress(c + 1, len(chunks), "跟踪点")
    del feature_grids, video
    return out


# --------------------------------------------------------------------------- motion


def dynamic_tracks(run: Run, frames: np.ndarray, depth: list[torch.Tensor], queries: list[int], dinos: dict[int, np.ndarray],
                   size: tuple[int, int], rng: torch.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """inference.py for BootsTAPIR tracks: [2, N, T] dynamic tracks, [N, T] visible, [N, T] confidence."""
    from core.dataset.data_utils import normalize_point_traj_torch
    from core.dataset.kubric import parse_tapir_track_info
    from core.utils.utils import get_feat

    w, h = size
    repo, weights = run.job.repo_dir, run.job.weights_dir
    cfg = moseg_config(repo)
    grid_yx = grid(h, w)
    owner, index = choose_points(len(grid_yx[0]), queries, rng)

    tapir = load(run, "BootsTAPIR", load_tapir, weights / "bootstapir" / "bootstapir_checkpoint_v2.pt")
    run.stage("跟踪点（BootsTAPIR）")
    tracks_2d = torch.from_numpy(track_points(tapir, frames, queries, owner, index, grid_yx, size))
    track_2d, occs, dists = tracks_2d[..., :2], tracks_2d[..., 2], tracks_2d[..., 3]
    visibles, _, confidences, visib_value, confi_value = parse_tapir_track_info(occs, dists, 0.5)

    model = load(run, "moseg", load_moseg, weights / "moseg" / "moseg.pth", repo / "configs" / "example_train.yaml")
    run.stage("判断哪些点在动")
    dino = None
    if cfg.dino:
        feats = []
        for q, qt in enumerate(queries):
            mine = owner == q
            dino_map = torch.from_numpy(dinos[qt]).unsqueeze(0).unsqueeze(0)
            factor = (dino_map[0].shape[1] / h, dino_map[0].shape[2] / w)
            feats.append((mine, get_feat(factor, track_2d[mine].permute(2, 0, 1)[..., qt:qt + 1], dino_map)))
        dino = torch.zeros((1, feats[0][1].shape[1], len(owner), 1), dtype=feats[0][1].dtype)
        for mine, f in feats:
            dino[:, :, torch.from_numpy(mine)] = f

    rows_all_false = torch.all(~visibles, dim=1)
    keep = ~rows_all_false
    track_2d, visibles, confidences = track_2d[keep], visibles[keep], confidences[keep]
    visib_value, confi_value = visib_value[keep], confi_value[keep]
    if dino is not None:
        dino = dino[:, :, keep, :]
    for t in torch.nonzero(torch.all(~visibles.permute(1, 0), dim=1)).flatten().tolist():
        visibles[torch.max(confidences[:, t], dim=0)[1], t] = True  # a frame with no visible point: its surest one

    track = track_2d.permute(2, 0, 1).unsqueeze(0)  # [1, 2, N, T]
    mask = (~visibles).unsqueeze(0).unsqueeze(0)
    depths = torch.stack(depth, dim=0).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
    batch = {"traj": normalize_point_traj_torch(track, [h, w]).float().cuda(), "mask": mask.float().cuda(),
             "depth": depths.float().cuda()}
    if cfg.extra_info:
        batch["visib_value"] = visib_value.unsqueeze(0).unsqueeze(0).float().cuda()
        batch["confi_value"] = confi_value.unsqueeze(0).unsqueeze(0).float().cuda()
    if dino is not None:
        batch["dino"] = dino.float().cuda()
    with torch.no_grad(), quiet():  # the model prints a line per fully hidden track
        pred = model(batch).detach().cpu()
    del batch

    # inference.py's ladder: the highest threshold at least 10 tracks pass (a fixed lower one takes in the road and
    # parked scooters on DAVIS scooter-black before it finds a second mover). Upstream then falls back to the 3 surest
    # tracks, as if something always moved (its benchmarks always have a moving object); here fewer than 10 tracks
    # over the lowest threshold means nothing moves: no objects.
    ladder = [0.99, 0.98, 0.97, 0.96, 0.95] if track.shape[-1] > 300 else [0.95, 0.93, 0.9, 0.85, 0.8, 0.75, 0.7]
    chosen = next((pred > t for t in ladder if (pred > t).sum() >= 10), torch.zeros_like(pred, dtype=torch.bool))
    d_mask = chosen.squeeze(0).squeeze(0)
    visible = visibles & (confidences > 0.9)
    return (track.squeeze(0)[:, d_mask, :].numpy(), visible[d_mask, :].numpy(), confidences[d_mask, :].numpy())


# --------------------------------------------------------------------------- objects (SAM 2)


class ShotFrames:
    """The SAM 2 predictor seen from the analysed frames: a prompt on analysed frame i goes to frame shot[i]."""

    def __init__(self, predictor, shot: list[int]):
        self.predictor, self.shot = predictor, shot

    def reset_state(self, state):
        return self.predictor.reset_state(state)

    def add_new_points_or_box(self, inference_state, frame_idx, **kwargs):
        return self.predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=self.shot[frame_idx], **kwargs)


def objects(run: Run, frame_dir: Path, shot: list[int], traj: np.ndarray, visible: np.ndarray, confi: np.ndarray,
            count: int) -> dict[int, dict[int, np.ndarray]]:
    """run_sam2.main: {shot frame index: {object id: mask [H, W] bool}} at the processing size, objects merged."""
    import run_sam2

    run_sam2.args = SimpleNamespace(vis=False)  # process_points_with_memory reads the script's global args
    predictor = load(run, "SAM 2", load_sam2, run.job.weights_dir / "sam2-hiera-large" / "sam2_hiera_large.pt")
    mapped = ShotFrames(predictor, shot)
    _, _, t_count = traj.shape
    max_iterations = min(max(len(range(0, t_count, 2 * 8)), 5), 10)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        with quiet():
            state = predictor.init_state(str(frame_dir), offload_video_to_cpu=True)
        memory = run_sam2.process_invisible_traj(traj, visible, confi, state, mapped, dilation_size=6,
                                                 max_iterations=max_iterations, timestep=t_count)
    segments: dict[int, dict[int, np.ndarray]] = {}
    for n, (obj, pkg) in enumerate(memory.items()):
        run.stage(f"跟踪运动物体 {n + 1}/{len(memory)}（SAM 2）")
        predictor.reset_state(state)
        prompts = list(range(0, t_count, SAM2_PROMPT_EVERY))
        if pkg["time"] in prompts:
            prompts.remove(pkg["time"])
        prompts.insert(0, pkg["time"])
        pts_trajs, vis_trajs = pkg["pts_trajs"], pkg["vis_trajs"]
        reverse = True
        for t in prompts:
            points = pts_trajs[:, :, t][vis_trajs[:, t]]
            if points.shape[0] == 0:
                continue
            if shot[t] == 0:
                reverse = False
            prompt = run_sam2.find_dense_pts(points)
            _, _, logits = mapped.add_new_points_or_box(inference_state=state, frame_idx=t, obj_id=obj, points=prompt,
                                                        labels=np.ones(len(prompt), np.int32))
            inside = run_sam2.find_pts_in_mask((logits[0] > 0.0).cpu().numpy(), points)
            if inside.sum() < points.shape[0] * 0.7:  # the densest point alone misses the object: centre + far points
                near, _, far, _, _ = run_sam2.find_centroid_and_nearest_farthest(points)
                prompt = np.concatenate((near, far), axis=0)
                mapped.add_new_points_or_box(inference_state=state, frame_idx=t, obj_id=obj, points=prompt,
                                             labels=np.ones(len(prompt), np.int32))
        directions = [False, True] if reverse else [False]
        done = 0
        for backwards in directions:
            for frame, ids, logits in predictor.propagate_in_video(state, reverse=backwards):
                per = segments.setdefault(frame, {})
                for i, oid in enumerate(ids):
                    m = (logits[i] > 0.0).cpu().numpy()
                    per[oid] = per[oid] | m if oid in per else m
                done += 1
                progress(done, count * len(directions), f"运动物体 {n + 1}")
    predictor.reset_state(state)
    del state
    if not segments:
        return {}
    segments = dict(sorted(segments.items()))
    return run_sam2.merge_masks(segments, run_sam2.analyze_frame_merges(segments, iou_threshold=MERGE_IOU))


# --------------------------------------------------------------------------- job


def main(job_path: str) -> None:
    import cv2

    run = Run.start(job_path, "seganymo.moving_objects", "SegAnyMo")
    job, p = run.job, run.params
    repo, weights = job.repo_dir, job.weights_dir
    dinov2 = Path(os.environ.get("LAB2SHOT_DINOV2_DIR", repo.parent / "dinov2"))
    run.weights(weights / "moseg" / "moseg.pth", weights / "sam2-hiera-large" / "sam2_hiera_large.pt",
                weights / "bootstapir" / "bootstapir_checkpoint_v2.pt",
                weights / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth", dinov2 / "hubconf.py")
    rng = set_seed(SEED)

    frames = job.frames
    count = len(frames)
    if count < 2:
        fail("E-WORKER-TOOFEWFRAMES", least=2, have=count, why=reason("I-SEGANYMO-WHYMOTION"))
    size = processing_size(job.width, job.height, p["resolution"])
    w, h = size
    shot = analysed(count, p["analysis_frames"])
    queries = list(range(0, len(shot), p["query_step"]))
    with tempfile.TemporaryDirectory(prefix="seganymo_", dir=job.dir) as tmp:
        frame_dir = Path(tmp) / "frames"
        frame_dir.mkdir()
        run.stage("读取画面")
        analysis = []
        want = set(shot)
        for i, (_, path) in enumerate(frames):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                fail("E-SEGANYMO-UNREADABLE", frame=frames[i][0])
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(frame_dir / f"{i:05d}.png"), img)
            if i in want:
                analysis.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            progress(i + 1, count, "读取画面")
        analysis = np.stack(analysis)

        from core.utils.run_depth import get_depth_anything_disp

        pipe = load(run, "Depth Anything V2 Small", load_depth_model, weights / "Depth-Anything-V2-Small-hf")
        run.stage("深度（Depth Anything V2 Small）")
        depth = []
        for n, i in enumerate(shot):
            depth.append(depth_as_read(get_depth_anything_disp(pipe, str(frame_dir / f"{i:05d}.png"), "uint16")))
            progress(n + 1, len(shot), "深度")
        del pipe

        from torchvision import transforms

        extractor = load(run, "DINOv2", load_dino, dinov2)
        run.stage("DINOv2 特征")
        shape = ((h + 13) // 14 * 14, (w + 13) // 14 * 14)  # dino_feat.py: a multiple of 14
        prep = transforms.Compose([transforms.ToTensor(), transforms.Resize(list(shape)),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        from PIL import Image

        dinos = {}
        for n, qt in enumerate(queries):
            batch = prep(Image.fromarray(analysis[qt])).unsqueeze(0).to("cuda")
            with torch.no_grad():
                desc = extractor.extract_descriptors(batch, [DINO_LAYER], "key", include_cls=False)
            desc = desc.reshape(desc.shape[0], extractor.num_patches[0], extractor.num_patches[1], -1).squeeze()
            dinos[qt] = desc.cpu().numpy().astype(np.float16)
            progress(n + 1, len(queries), "DINOv2 特征")
        torch.cuda.empty_cache()

        traj, visible, confi = dynamic_tracks(run, analysis, depth, queries, dinos, size, rng)
        del analysis, depth, dinos
        torch.cuda.empty_cache()
        found = objects(run, frame_dir, shot, traj, visible, confi, count) if traj.shape[1] else {}

    run.stage("写出遮罩")
    ids = sorted({o for per in found.values() for o in per})
    if not ids:
        say("N-SEGANYMO-EMPTY")
    number = {o: k + 1 for k, o in enumerate(ids)}  # merged ids -> 1..K
    seen = {k: 0 for k in number.values()}
    for i, (f, _) in enumerate(frames):
        labels = np.zeros((job.height, job.width), np.uint8)
        # run_sam2.put_per_obj_mask: lower ids win where objects overlap (written last)
        for o in sorted(found.get(i, {}), reverse=True):
            m = found[i][o].reshape(h, w).astype(np.float32)
            if not m.any():
                continue
            if (w, h) != (job.width, job.height):
                m = cv2.resize(m, (job.width, job.height), interpolation=cv2.INTER_LINEAR)
            labels[m > 0.5] = number[o]
            seen[number[o]] += 1
        save_npz(job.raw_dir / f"frame_{f}.npz", labels=labels)
        progress(i + 1, count, "写出遮罩")
    (job.raw_dir / "objects.json").write_text(json.dumps([{"id": k, "frames": n} for k, n in seen.items()]), encoding="utf-8")
    run.finish([f for f, _ in frames], objects=len(ids), dynamic_tracks=int(traj.shape[1]), analysed_frames=len(shot),
               processing_size=[w, h],
               peak_ram_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1))


if __name__ == "__main__":
    serve(main)
