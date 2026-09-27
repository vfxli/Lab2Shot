"""Robbyant LingBot-Map worker: streaming reconstruction. Runs inside
third_party/lingbotmap/.venv with the pinned repo on sys.path; never imports
Lab2Shot core.

    python worker.py <job.json>

node "lingbotmap.reconstruct". Frames (every `step`-th, plus always the last) are
read one at a time and streamed through the model: the first `anchor_frames`
together (they fix the coordinate system and the scale), then one frame per
forward pass against a KV cache (PyTorch SDPA) that holds the anchor frames, the last
`context_window` keyframes in full and 6 summary tokens per older keyframe.
Results leave in small pieces (lab2shot_worker.recon.Stitcher) and go straight
to disk: memory does not grow with the length of the shot.

Output: the reconstruction contract (lab2shot_worker.recon; node side
lab2shot/nodes/results.py reconstruction()):

    raw/cameras.npz   frames [F], K [F,3,3] (pixels at the input resolution, fx / fy from the
                      model's per-frame horizontal / vertical field of view, principal point at
                      the image centre), cam_to_world [F,4,4] (OpenCV, world = first frame's
                      camera, RELATIVE units), plus keyframe [F] bool (kept in the model's
                      memory) and segment [F] int (which stream segment)
    raw/frame_<n>.npz depth [H,W] (camera Z, relative units, raw prediction also outside the
                      mask), confidence [H,W] (>= 1, higher = better), mask [H,W] (confidence
                      >= conf_threshold, not a depth edge, not sky)

Scale: not metric. Training normalised every scene so that the anchor frames'
points lie on average 1 unit from the first camera: the first frames fix the unit.

Long shots (upstream's two inference modes):
  one stream   ("Direct Output", the most accurate) up to 3000 frames, upstream's stable
               range. Keyframe interval (auto): 1 up to 320 frames, else ceil(frames / 320),
               so the memory never holds more than ~320 keyframes (the training length).
  segments     ("VO mode") beyond that, or with max_frames set: the stream restarts every
               max_frames frames; consecutive segments share max(anchor_frames, 16 keyframes)
               frames and are joined by recon.Stitcher (cameras' mean relative rotation, scale
               and translation on the shared frames' points, cross-fade across them).

LingBot-Map has no input for masks: it sees the whole frame, moving objects
included, and nothing is taken out afterwards. To reconstruct only part of the
picture, black the rest out before the 「图像」 input (人物检测 → 人物框转遮罩 →
图像相乘), where it is visible on the node graph.
"""

from __future__ import annotations

import math
import os
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lab2shot_worker import fail, fit_size, progress, reason, recon, require_weights, resident, say, serve
from lab2shot_worker.frame_io import FrameReader, Writer
from lab2shot_worker.run import Run

NODE = "lingbotmap.reconstruct"
WEIGHTS = {  # params["model"] -> checkpoint in weights/
    "long": "lingbot-map-long.pt",  # upstream: better for long sequences and large scenes
    "balanced": "lingbot-map.pt",  # upstream: the paper / benchmark checkpoint
}
NATIVE_RESOLUTION = 518  # training: longest side 518 px
PATCH = 14
TRAIN_VIEWS = 320  # longest training sequence: more keyframes in memory degrade the poses
STREAM_MAX_FRAMES = 3000  # upstream: one stream is stable up to ~10x the training length
OVERLAP_KEYFRAMES = 16  # segments share this many keyframes (upstream README's windowed example)
PIECE = 32  # frames handed to the stitcher at a time
CHAIN = 2  # frames consecutive pieces of one segment share (identical: they only carry the transform on)
WRITERS = 3  # threads turning stitched frames into input-resolution npz files
SKYSEG_SIZE = 320
SKY_THRESHOLD = 0.1  # upstream: non-sky confidence above this is kept
# Focal: K has square pixels and the focal from the field of view along the frame's LONG side. The network predicts horizontal and
# vertical field of view separately; it was trained on landscape views (long side = width, 518 px), and on an
# upright portrait plate its horizontal estimate comes out far too wide (a 9:16 plate: fx 2100 px vs fy 3900 px, true
# 5480 px) while the long side stays the closer one. Turning portrait plates 90 degrees instead makes the camera path
# worse (13.6 % vs 3.3 % RMS against ViPE). Both raw estimates go to result.json (focal_px_model).

# Upstream's defaults, kept: the first frames the stream anchors its scale on (the node's smallest segment, 16
# frames, is twice this), and the camera head's refinement passes (its maximum).
ANCHOR_FRAMES = 8
CAMERA_ITERATIONS = 4


def read_params(params: dict) -> dict:
    """The node's parameters (max_frames None: one stream up to STREAM_MAX_FRAMES frames; keyframe_interval 0: auto,
    at most ~320 keyframes per segment), "auto" resolution filled in, and the fixed upstream settings."""
    return {**params, "resolution": params["resolution"] or NATIVE_RESOLUTION,
            "anchor_frames": ANCHOR_FRAMES, "camera_iterations": CAMERA_ITERATIONS}


# ---------------------------------------------------------------------- frames and sky


def to_model(rgb: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """uint8 [H,W,3] -> float [3,h,w] in 0..1 (the model normalises itself). Downscaling uses area
    filtering: no aliasing from 1920 px plates."""
    interp = cv2.INTER_AREA if size[0] < rgb.shape[1] else cv2.INTER_CUBIC
    return torch.from_numpy(cv2.resize(rgb, size, interpolation=interp)).permute(2, 0, 1).float().div_(255.0)


class SkySegmenter:
    """Upstream's sky filter (lingbot_map/vis/sky_segmentation.py; importing it pulls in the viser viewer):
    skyseg.onnx at 320x320, ImageNet-normalised, output min-max normalised per frame; non-sky confidence
    = 1 - map, kept above 0.1."""

    def __init__(self, onnx_path: Path):
        import onnxruntime

        require_weights("lingbotmap", onnx_path, what="天空分割模型")
        self.session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp = self.session.get_inputs()[0].name
        self.out = self.session.get_outputs()[0].name

    def sky(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 RGB [H,W,3] -> bool [H,W], True on sky."""
        x = cv2.resize(rgb, (SKYSEG_SIZE, SKYSEG_SIZE)).astype(np.float32) / 255.0
        x = ((x - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])).transpose(2, 0, 1)[None]
        r = np.asarray(self.session.run([self.out], {self.inp: x})).squeeze().astype(np.float32)
        r = (r - r.min()) / max(float(r.max() - r.min()), 1e-8)
        r = cv2.resize(r, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return (1.0 - r) <= SKY_THRESHOLD


# ---------------------------------------------------------------------- model and plan


@resident
def load_model(repo: Path, checkpoint: Path, context_window: int, anchor_frames: int, camera_iterations: int,
               max_frame_num: int, device: torch.device):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from lingbot_map.models.gct_stream import GCTStream

    with torch.device(device):  # built on the GPU: the checkpoint goes there straight from the mapped file
        model = GCTStream(
            img_size=NATIVE_RESOLUTION,
            patch_size=PATCH,
            enable_3d_rope=True,
            max_frame_num=max_frame_num,
            kv_cache_sliding_window=context_window,
            kv_cache_scale_frames=anchor_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=True,  # PyTorch attention with a contiguous KV cache (no FlashInfer, see requirements.txt)
            camera_num_iterations=camera_iterations,
        )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        fail("E-LINGBOTMAP-MISSINGKEYS", file=checkpoint.name, count=len(missing), keys=list(missing[:3]))
    # 上游的 GCTBase 有两个点图头（world_points / cam_points，gct_base.py:206-247），GCTStream 默认
    # 两个都不建（enable_point / enable_local_point = False，gct_stream.py:118-119）。放出来的两份权重
    # 里也没有这两个头：lingbot-map.pt 和 lingbot-map-long.pt 的键只有
    # aggregator (1211) / camera_head (69) / depth_head (62)，一个 point_head. 都没有。所以
    # LingBot-Map 交不出世界点图：官方没放这部分权重。
    unexpected = [k for k in unexpected if not k.startswith(("point_head.", "local_point_head."))]
    if unexpected:
        say("I-LINGBOTMAP-EXTRAKEYS", count=len(unexpected), keys=list(unexpected[:2]))
    del ckpt, state
    model = model.to(device).eval()  # a few buffers are created on the CPU whatever the default device
    # Upstream demo: the trunk in bf16 (the heads stay fp32 and run without autocast).
    model.aggregator = model.aggregator.to(dtype=torch.bfloat16)
    # 这里逐帧跑（上游的 `inference_streaming` 把全片 cat 成一个大字典，长镜头会撑爆内存），
    # 所以上游那个函数末尾的 `if self.pred_normalization: predictions = self._normalize_predictions(...)`
    # （`gct_stream.py:541-542`）这条路走不到。它默认 False、两边都没打开，等价；
    # 若上游在配置里打开而这里不跟，结果会静默跑偏，所以在这里守门
    if getattr(model, "pred_normalization", False):
        fail("E-LINGBOTMAP-NORMALIZATION")
    return model


def auto_keyframe_interval(frames: int) -> int:
    """Upstream: every frame up to 320, else ceil(frames / 320), so that at most ~320 keyframes are remembered."""
    return 1 if frames <= TRAIN_VIEWS else math.ceil(frames / TRAIN_VIEWS)


def current_rss_mb() -> int:
    with open("/proc/self/statm") as fh:
        return round(int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20)


# ---------------------------------------------------------------------- main


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "LingBot-Map")
    job = run.job
    p = read_params(job.params)

    if len(job.frames) < 2:
        fail("E-WORKER-TOOFEWFRAMES", least=2, have=len(job.frames), why=reason("I-WORKER-WHYCOMPARE"))
    selected = run.frames(step=p["step"])  # every step-th frame and always the last: nothing to extrapolate
    frame_numbers = selected.numbers
    n = len(selected)

    width, height = selected.width, selected.height
    mw, mh = size = fit_size(width, height, p["resolution"], PATCH)  # the whole frame: no crop, no padding
    if p["resolution"] != NATIVE_RESOLUTION:
        say("W-LINGBOTMAP-RESOLUTION", native=NATIVE_RESOLUTION, resolution=p["resolution"])

    anchors = min(p["anchor_frames"], n)
    length = min(p["max_frames"] or STREAM_MAX_FRAMES, n)
    kf = p["keyframe_interval"] or auto_keyframe_interval(length)
    overlap = min(max(anchors, OVERLAP_KEYFRAMES * kf), length // 2)
    segments = recon.plan_chunks(n, length, overlap)
    overlap = segments[0][1] - segments[1][0] if len(segments) > 1 else 0
    length = max(b - a for a, b in segments)
    memory_frames = anchors + math.ceil(max(length - anchors, 0) / kf)  # keyframes remembered per segment
    if memory_frames > TRAIN_VIEWS * 1.1:
        say("W-LINGBOTMAP-MEMORY", keyframes=memory_frames, trained=TRAIN_VIEWS)
    max_frame_num = max(1024, memory_frames + 16)  # length of the temporal 3D RoPE table (keyframes)

    checkpoint = job.weights_dir / WEIGHTS[p["model"]]
    run.weights(checkpoint)
    raw = job.raw_dir
    device = torch.device("cuda")
    sky = SkySegmenter(job.weights_dir / "skyseg.onnx") if p["mask_sky"] else None

    model = run.model("LingBot-Map 模型", load_model, job.repo_dir, checkpoint, p["context_window"], p["anchor_frames"],
                      p["camera_iterations"], max_frame_num, device)
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    stitch = recon.Stitcher()
    segment_info: list[dict] = []
    K_in = np.zeros((n, 3, 3))
    c2w = np.zeros((n, 4, 4))
    keyframe = np.zeros(n, dtype=bool)
    segment_of = np.zeros(n, dtype=np.int64)
    depth_median = [float("nan")] * n
    focal_model = np.zeros((n, 2))  # the model's own fx, fy (input pixels)
    valid_fraction = [float("nan")] * n  # share of each frame's pixels in the output mask
    memory_trace: list[dict] = []
    sx, sy = width / mw, height / mh
    reader = FrameReader(selected.paths, threads=2, ahead=1)
    writer = Writer(threads=WRITERS, max_pending=2 * WRITERS)  # back-pressure: a few frames in flight at most
    written = 0

    def write(fr: recon.Frame) -> None:
        """A stitched frame at the input resolution (runs in the writer threads)."""
        i = fr.index
        if fr.usable.any():
            depth_median[i] = float(np.median(fr.depth[fr.usable]))
        # excluded: sky at the full resolution (sharper than at the model's)
        d, c, m = recon.at_input_size(fr, width, height, fr.extra.get("excluded"))
        m &= (c >= p["conf_threshold"]) & (d > 0)
        valid_fraction[i] = float(m.mean())
        recon.save_frame(raw, frame_numbers[i], d, c, m)

    def flush(frames: list[recon.Frame]) -> None:
        nonlocal written
        for fr in frames:
            K_in[fr.index] = fr.K
            K_in[fr.index, 0, :] *= sx
            K_in[fr.index, 1, :] *= sy
            c2w[fr.index] = fr.cam_to_world
            writer.submit(write, fr)
            written += 1
            if written % 5 == 0 or written == n:
                progress(written, n, "重建")
            if written % 50 == 0 or written == n:
                memory_trace.append({"frames": written, "rss_mb": current_rss_mb(),
                                     "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 2**20)})

    def predict(pred: dict, j: int, i: int, rgb: np.ndarray, is_kf: bool, si: int) -> dict:
        """One frame's prediction, segment-local, at the model resolution: camera, depth, confidence, usable."""
        ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"][:, j:j + 1].float(), (mh, mw))
        w2c = np.eye(4)
        w2c[:3, :4] = ext[0, 0].double().cpu().numpy()  # pose_encoding_to_extri_intri gives world-to-camera (OpenCV), as VGGT's
        cam = np.linalg.inv(w2c)  # camera-to-world, what cam_to_world below and every consumer expects (adapters/vggt/worker.py does the same)
        k = intr[0, 0].double().cpu().numpy()
        focal_model[i] = k[0, 0] * sx, k[1, 1] * sy  # input pixels
        f = focal_model[i, 0] if mw >= mh else focal_model[i, 1]  # see "Focal" above PARAMS
        depth = np.nan_to_num(pred["depth"][0, j, ..., 0].float().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        conf = pred["depth_conf"][0, j].float().cpu().numpy()
        K = np.array([[f / sx, 0, mw / 2], [0, f / sy, mh / 2], [0, 0, 1.0]])  # model pixels; square at the input
        usable = (conf >= p["conf_threshold"]) & (depth > 0) & ~recon.depth_edges(depth)
        excluded = sky.sky(rgb) if sky is not None else None
        if excluded is not None:  # never align on sky
            usable &= ~(cv2.resize(excluded.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0)
        keyframe[i], segment_of[i] = is_kf, si
        return {"i": i, "cam": cam, "K": K, "depth": depth, "conf": conf,
                "usable": usable, "excluded": excluded}

    def hand_over(piece: list[dict], si: int, until: int | None) -> None:
        """Frames of one segment -> the stitcher; frames no later piece shares -> disk."""
        chunk = recon.Chunk(
            cam_to_world=np.stack([f["cam"] for f in piece]), K=np.stack([f["K"] for f in piece]),
            depth=np.stack([f["depth"] for f in piece]), confidence=np.stack([f["conf"] for f in piece]),
            usable=np.stack([f["usable"] for f in piece]))
        if sky is not None:
            chunk.extra["excluded"] = np.stack([f["excluded"] for f in piece])
        try:
            info = stitch.add(piece[0]["i"], chunk)
        except ValueError:
            fail("E-LINGBOTMAP-NOOVERLAP", segment=si + 1)
        if "scale" in info and piece[0]["i"] in segment_starts:  # a new segment joined onto the previous one
            info["frames"] = [frame_numbers[piece[0]["i"]], frame_numbers[piece[-1]["i"]]]
            segment_info.append(info)
        flush(list(stitch.pop(until)))

    segment_starts = {a for a, _ in segments[1:]}
    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    def stream(si: int, s0: int, s1: int):
        """One segment through the model: each frame's prediction, in order."""
        model.clean_kv_cache()
        wa = min(anchors, s1 - s0)
        # Anchor frames together (full attention among themselves): coordinate system and scale.
        rgbs = reader.take(range(s0, s0 + wa))
        reader.request(range(s0 + wa, min(s0 + wa + 1, s1)))
        t = run.frame_started()
        batch = torch.stack([to_model(r, size) for r in rgbs])[None].to(device)
        with autocast:
            pred = model.forward(batch, num_frame_for_scale=wa, num_frame_per_block=wa, causal_inference=True)
        items = [predict(pred, j, s0 + j, rgbs[j], True, si) for j in range(wa)]
        del pred, batch, rgbs
        run.frame_seconds.extend([(time.time() - t) / wa] * wa)  # the anchors go through together: shared out per frame
        yield from items
        # Then one frame per forward pass; non-keyframes look at the memory but are not added to it.
        for i in range(s0 + wa, s1):
            rgb = reader.get(i, range(i + 1, s1))
            is_kf = kf <= 1 or (i - s0 - wa) % kf == 0
            t = run.frame_started()
            image = to_model(rgb, size)[None, None].to(device, non_blocking=True)
            if not is_kf:
                model._set_skip_append(True)
            with autocast:
                pred = model.forward(image, num_frame_for_scale=wa, num_frame_per_block=1, causal_inference=True)
            if not is_kf:
                model._set_skip_append(False)
            item = predict(pred, 0, i, rgb, is_kf, si)
            del pred
            run.frame_done(t)
            yield item

    run.stage(f"流式重建（{n} 帧，关键帧间隔 {kf}）" if len(segments) == 1 else
              f"分段重建（{n} 帧，{len(segments)} 段，关键帧间隔 {kf}）")
    t_run = time.time()
    with torch.no_grad():
        for si, (s0, s1) in enumerate(segments):
            if len(segments) > 1:
                run.stage(f"第 {si + 1}/{len(segments)} 段（第 {frame_numbers[s0]}–{frame_numbers[s1 - 1]} 帧）")
            # The stitcher gets the segment in pieces of ~PIECE frames; consecutive pieces share CHAIN
            # frames, and a joined segment's first piece holds every frame it shares with the previous one.
            first_piece_end = (segments[si - 1][1] if si else s0) + PIECE
            next_start = segments[si + 1][0] if si + 1 < len(segments) else None
            piece: list[dict] = []
            for item in stream(si, s0, s1):
                piece.append(item)
                i = item["i"]
                if i + 1 < s1 and i + 1 >= max(piece[0]["i"] + PIECE, first_piece_end):
                    until = i + 1 - CHAIN
                    hand_over(piece, si, until if next_start is None else min(until, next_start))
                    piece = piece[-CHAIN:]
            # the rest; frames the next segment shares stay in the stitcher
            hand_over(piece, si, next_start)
    model.clean_kv_cache()  # the model stays loaded for the next job: without this shot's memory
    torch.cuda.synchronize()
    run_seconds = time.time() - t_run
    writer.close()
    reader.close()

    recon.save_cameras(raw, frame_numbers, K_in, c2w, width, height, keyframe=keyframe, segment=segment_of)
    numbers = recon.summary(c2w, K_in, width, depth_median)
    fx_model, fy_model = np.median(focal_model, axis=0)
    if abs(fx_model / fy_model - 1) > 0.1:
        say("W-LINGBOTMAP-FOCALXY", fx=float(fx_model), fy=float(fy_model), used=float(fx_model if width >= height else fy_model))
    run.finish(
        frame_numbers,
        kind="reconstruction",
        cameras="cameras.npz",
        files="cameras.npz: frames, K, cam_to_world, width, height, keyframe, segment; frame_<n>.npz: depth, confidence, mask",
        model="LingBot-Map",
        weights=p["model"],
        checkpoint=WEIGHTS[p["model"]],
        weights_license="Apache-2.0",
        convention=recon.CONVENTION,
        intrinsics_convention="K in pixels at the input resolution; pixel (i, j) covers [j, j+1] x [i, i+1]; "
                              "principal point fixed at the image centre (W/2, H/2); fx, fy from the model's "
                              "per-frame horizontal / vertical field of view",
        depth="camera Z at the input resolution in the same relative units as cam_to_world; raw prediction also "
              "outside the mask",
        metric=False,
        units="relative, not metres: the model normalises every shot so that the anchor frames' points lie on "
              "average ~1 unit from the first camera; consistent within the shot",
        confidence="LingBot-Map depth confidence, >= 1, unitless, higher = more reliable",
        mask=f"True = reliable geometry: confidence >= {p['conf_threshold']}, not a depth edge (3x3 depth "
             f"ratio > {1 + recon.EDGE_RTOL}), not sky (mask_sky); the model sees the whole frame, moving objects "
             "included, and nothing is taken out afterwards",
        points="not predicted: the released checkpoints carry no point head (aggregator / camera_head / "
               "depth_head only), so LingBot-Map gives depth and cameras, no 3D point map",
        width=width,
        height=height,
        fps=job.fps,
        step=p["step"],
        frames=frame_numbers,  # the whole list, not the standard [first, last]: the converter reads every frame's file by it
        model_input={"width": mw, "height": mh, "resize": "whole frame, no crop / padding"},
        segment_frames=length,
        segments=len(segments),
        segment_overlap=overlap,
        segment_alignment=segment_info,
        alignment="segments share `segment_overlap` frames, joined by lab2shot_worker.recon.Stitcher (rotation = "
                  "mean relative camera orientation of the shared frames, scale + translation on their points, "
                  "cross-fade across them); inside a segment the stream itself is one world",
        keyframe_interval=kf,
        keyframes=int(keyframe.sum()),
        anchor_frames=anchors,
        context_window=p["context_window"],
        camera_iterations=p["camera_iterations"],
        attention="PyTorch SDPA",
        focal_px_model={"fx_median": float(fx_model), "fy_median": float(fy_model),
                        "note": "the network's own horizontal / vertical focal; K uses the long side's (square pixels)"},
        mask_sky=p["mask_sky"],
        conf_threshold=p["conf_threshold"],
        valid_fraction_median=float(np.nanmedian(valid_fraction)),
        **numbers,
        run_seconds=round(run_seconds, 1),
        frames_per_second=round(n / max(run_seconds, 1e-6), 2),
        seconds_per_frame_median=round(float(np.median(run.frame_seconds)), 4),
        # the peak includes the checkpoint pages read (memory-mapped) while loading
        cpu_peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        memory_trace=memory_trace,
    )


if __name__ == "__main__":
    serve(main)
