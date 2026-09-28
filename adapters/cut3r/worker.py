"""CUT3R worker: dynamic-scene reconstruction (cameras + depth). Runs inside
third_party/cut3r/.venv with the pinned repo's src/ on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>        (job["node"] == "cut3r.reconstruct")

Frames (every `step`-th) -> CUT3R online inference (the recurrent state reads the frames
one by one, updated by CUT3R's rule or TTT3R's; each chunk of `max_frames` starts from a fresh state) -> per frame: camera
pose from the pose head, depth + confidence from the self-view point map, focal from
that point map (upstream's Weiszfeld fit) -> chunks stitched by a similarity fitted on
their shared frames (lab2shot_worker.recon) -> the raw `reconstruction` contract:

    raw/cameras.npz   frames [F], K [F,3,3] (pixels, input resolution), cam_to_world [F,4,4]
                      (OpenCV; world = first solved frame's camera; metres, CUT3R's metric scale)
    raw/frame_<n>.npz depth [H,W] (camera Z, metres), confidence [H,W] (CUT3R conf >= 1),
                      mask [H,W] (confident), points [H,W,3] (CUT3R's own world point map
                      pts3d_in_other_view, put back into that frame's camera)

Moving objects: CUT3R has no mask input (it was trained on dynamic scenes and sees the
whole frame), and nothing is taken out of the result afterwards. To reconstruct only part
of the picture, black the rest out before the 「RGB」 input (「ViTDet 人物框」 → 「人物框转遮罩」 →
「图像合成」 set to 留下), where it is visible on the node graph.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import MemoryBound, fail, progress, read_frame, reason, recon, resident, say, serve
from lab2shot_worker.run import Run

# this adapter's own module (Extension.worker_modules)
import dust3r_input as di  # noqa: E402

CHECKPOINT = "cut3r_512_dpt_4_64.pth"
# CUT3R keeps O(1) memory per frame (one frame encoded at a time), so the chunk length is about accuracy, not memory:
# the 512 model was trained on 4-64 views, its state forgets / drifts on much longer runs. TTT3R's state update holds
# on far longer runs, but not on any: on a 792-frame take a single pass loses its orientation, 200-frame chunks work.
MAX_FRAMES = {"cut3r": 64, "ttt3r": 200}
# what the memory grows with, when it runs out: 每段最多帧数 (the node offers 32..792; the steps start at the longest
# measured default, 200)
MAX_FRAMES_STEPS = (200, 128, 64, 32, 16, 8)
RESOLUTION = 512  # long side of the network input: the training size


def read_params(params: dict) -> dict:
    """The node's parameters, its "auto" (None) values filled in."""
    p = dict(params)
    p["max_frames"] = p["max_frames"] or MAX_FRAMES[p["update"]]
    p["resolution"] = p["resolution"] or RESOLUTION
    if p["overlap"] >= p["max_frames"]:
        fail("E-CUT3R-OVERLAP", overlap=p["overlap"], max_frames=p["max_frames"])
    return p


def patch_rope() -> str:
    """CUT3R gives its pose token the RoPE position -1. The compiled cuRoPE kernel (which
    upstream asks you to build) computes cos/sin(position x frequency) directly, so that
    works; the pure-PyTorch fallback looks positions up in a table and fails on -1 (CUDA
    device-side assert). Without the kernel, the fallback is replaced by the kernel's
    formula in PyTorch: identical results for every position, negative ones included."""
    from models import pos_embed  # croco's, on sys.path via dust3r.utils.path_to_croco

    if pos_embed.RoPE2D.__name__ == "cuRoPE2D":
        return "cuRoPE (compiled)"

    def forward(self, tokens, positions):
        # tokens: B x heads x N x D; positions: B x N x 2 (y, x)
        d = tokens.size(3) // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, d, 2, device=tokens.device, dtype=torch.float32) / d))
        out = []
        for part, pos in zip(tokens.chunk(2, dim=-1), (positions[:, :, 0], positions[:, :, 1])):
            f = pos.float()[..., None] * inv_freq
            f = torch.cat((f, f), dim=-1)[:, None]
            out.append(part * f.cos().to(part.dtype) + self.rotate_half(part) * f.sin().to(part.dtype))
        return torch.cat(out, dim=-1)

    pos_embed.RoPE2D.forward = forward
    return "PyTorch (cuRoPE formula)"


@resident
def load_model(repo: Path, checkpoint: Path, device: torch.device):
    """Upstream dust3r.model.load_model, with torch.load(weights_only=False): the checkpoint
    (the authors' file, sha256-checked at install) pickles its training args, which torch>=2.6
    refuses by default."""
    sys.path.insert(0, str(repo / "src"))  # src/dust3r and src/croco, as upstream's add_path_to_dust3r
    import dust3r.model as upstream

    rope = patch_rope()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ckpt["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace("landscape_only=True", "landscape_only=False")
    model = eval(args, vars(upstream))  # e.g. "ARCroco3DStereo(ARCroco3DStereoConfig(...))"
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:  # upstream only prints these
        print(f"checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys: {missing[:5]} {unexpected[:5]}", flush=True)
    del ckpt
    return model.to(device).eval(), rope


def make_views(images: list[np.ndarray]) -> list[dict]:
    """Upstream demo.prepare_input for images only (no ray maps)."""
    views = []
    for i, rgb in enumerate(images):
        img = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).sub(0.5).div(0.5)[None]
        h, w = rgb.shape[:2]
        views.append({
            "img": img,
            "ray_map": torch.full((1, 6, h, w), torch.nan),
            "true_shape": torch.tensor([[h, w]], dtype=torch.int32),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.eye(4, dtype=torch.float32)[None],
            "img_mask": torch.tensor([True]),
            "ray_mask": torch.tensor([False]),
            "update": torch.tensor([True]),
            "reset": torch.tensor([False]),
        })
    return views


class StateLearningRate:
    """TTT3R's state update (Chen et al., "TTT3R: 3D Reconstruction as Test-Time Training", ICLR 2026; MIT,
    github.com/Inception3D/TTT3R): instead of replacing the whole state with the decoder's new one on every frame,
    each state token moves by a learning rate, sigmoid of the mean pre-softmax cross-attention logit from that
    token (query) to the frame's tokens (keys), over every key, head and layer of the state decoder.

    TTT3R computes those logits by editing CUT3R's attention to return them; here pre-hooks on the pinned CUT3R
    state decoder's cross-attention compute the same numbers (the mean over keys of q.k is q.(mean of k), with the
    same projections and RoPE), leaving upstream's attention untouched. Installed per job, removed afterwards."""

    def __init__(self, model) -> None:
        self.logits: list[torch.Tensor] = []  # per layer [B, heads, state tokens]
        self.handles = [blk.cross_attn.register_forward_pre_hook(self._record) for blk in model.dec_blocks_state]

    def _record(self, attn, args) -> None:
        query, key, _value, qpos, kpos = args
        b, nq, c = query.shape
        heads = attn.num_heads
        q = attn.projq(query).reshape(b, nq, heads, c // heads).permute(0, 2, 1, 3).float()
        k = attn.projk(key).reshape(b, key.shape[1], heads, c // heads).permute(0, 2, 1, 3).float()
        if attn.rope is not None:  # as upstream's CrossAttention.forward
            with torch.autocast(device_type="cuda", enabled=False):
                q = attn.rope(q, qpos) if qpos is not None else q
                k = attn.rope(k, kpos) if kpos is not None else k
        self.logits.append((q * k.mean(dim=2, keepdim=True)).sum(-1) * attn.scale)

    def take(self) -> torch.Tensor:
        """The rate [B, state tokens, 1] of the frame just decoded; clears the record for the next one."""
        rate = torch.sigmoid(torch.stack(self.logits).mean(dim=(0, 2)))[..., None]
        self.logits.clear()
        return rate

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_chunk(model, images: list[np.ndarray], conf_threshold: float, device: torch.device,
              update: str, report) -> recon.Chunk:
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.utils.camera import pose_encoding_to_camera

    views = make_views(images)
    for v in views:
        for k in ("img", "ray_map", "true_shape", "camera_pose", "img_mask", "ray_mask", "update", "reset"):
            v[k] = v[k].to(device, non_blocking=True)
    n = len(images)
    h, w = images[0].shape[:2]
    c2w = np.zeros((n, 4, 4))
    focal = np.zeros(n)
    depth = np.zeros((n, h, w), np.float32)
    conf = np.zeros((n, h, w), np.float32)
    points = np.zeros((n, h, w, 3), np.float32)
    pp = torch.tensor([[w // 2, h // 2]], dtype=torch.float32, device=device)

    def keep(i: int, res: dict) -> None:
        pts = res["pts3d_in_self_view"].float()
        focal[i] = float(estimate_focal_knowing_depth(pts, pp, focal_mode="weiszfeld").item())
        depth[i] = pts[0, ..., 2].cpu().numpy()
        conf[i] = res["conf_self"][0].float().cpu().numpy()
        c2w[i] = pose_encoding_to_camera(res["camera_pose"].float()).cpu().numpy()[0]
        # 上游的第二张点图：世界坐标系里的 pts3d_in_other_view（这一段自己的世界）。用这一帧上游自己的
        # 位姿放回相机空间（无损，只是换坐标系），由 recon.Stitcher 跟深度一起摆进整片的世界
        # （Chunk.scaled），节点那边再按相机放回世界
        world = res["pts3d_in_other_view"][0].float().cpu().numpy()
        w2c = np.linalg.inv(c2w[i])
        points[i] = (world @ w2c[:3, :3].T + w2c[:3, 3]).astype(np.float32)
        report(i + 1, len(views))

    rate = StateLearningRate(model) if update == "ttt3r" else None
    try:
        _forward_recurrent(model, views, keep, rate)
    finally:
        if rate is not None:
            rate.remove()
    return recon.Chunk(cam_to_world=c2w, K=di.model_K(focal, np.tile([w // 2, h // 2], (n, 1)).astype(np.float64)),
                       depth=depth, confidence=conf, usable=conf > conf_threshold,
                       extra={"points": points}, scaled=("points",))


def _forward_recurrent(model, views, emit, rate: StateLearningRate | None) -> None:
    """model.forward_recurrent (pinned upstream code), but each view's result goes to
    `emit(index, result)` instead of a list on the GPU. Only the image path (no ray maps). `rate`: TTT3R's state
    update (each state token moves by its learning rate after the first frame), else CUT3R's (replaced whole)."""
    state_feat = state_pos = init_state_feat = mem = init_mem = None
    for i, view in enumerate(views):
        img_out, img_pos, _ = model._encode_image(view["img"], view["true_shape"])
        feat_i, pos_i = img_out[-1], img_pos
        if i == 0:
            state_feat, state_pos = model._init_state(feat_i, pos_i)
            mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            init_state_feat, init_mem = state_feat.clone(), mem.clone()
        global_img_feat_i = model._get_img_level_feat(feat_i)
        if i == 0:
            pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1)
        else:
            pose_feat_i = model.pose_retriever.inquire(global_img_feat_i, mem)
        pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        new_state_feat, dec = model._recurrent_rollout(
            state_feat, state_pos, feat_i, pos_i, pose_feat_i, pose_pos_i, init_state_feat,
            img_mask=view["img_mask"], reset_mask=view["reset"], update=view.get("update"),
        )
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = model.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
        head_input = [
            dec[0].float(),
            dec[model.dec_depth * 2 // 4][:, 1:].float(),
            dec[model.dec_depth * 3 // 4][:, 1:].float(),
            dec[model.dec_depth].float(),
        ]
        res = model._downstream_head(head_input, view["true_shape"], pos=pos_i)
        emit(i, res)
        update_mask = (view["img_mask"] & view["update"])[:, None, None].float()
        state_mask = update_mask
        if rate is not None:
            learning = rate.take()
            if i > 0:
                state_mask = update_mask * learning
        state_feat = new_state_feat * state_mask + state_feat * (1 - state_mask)
        mem = new_mem * update_mask + mem * (1 - update_mask)
        reset_mask = view["reset"][:, None, None].float()
        state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
        mem = init_mem * reset_mask + mem * (1 - reset_mask)


def main(job_path: str) -> None:
    run = Run.start(job_path, "cut3r.reconstruct", "CUT3R")
    job = run.job
    p = read_params(job.params)
    checkpoint = job.weights_dir / CHECKPOINT
    run.weights(checkpoint)

    all_frames = job.frames
    used = all_frames[::p["step"]]
    if len(used) < 2:
        fail("E-WORKER-TOOFEWFRAMES", least=2, have=len(used), why=reason("I-WORKER-WHYSTEP", step=p["step"]))
    frames = [f for f, _ in used]
    raw = job.raw_dir
    device = torch.device("cuda")

    run.stage("读取画面")
    height, width = read_frame(used[0][1]).shape[:2]
    geo = di.Geometry.for_size(width, height, p["resolution"])
    images = di.read_images(used, geo, read_frame, progress)

    # run.model caps the GPU first: an allocation beyond the free memory fails instead of spilling into RAM (WSL/Windows)
    model, rope = run.model("CUT3R 模型", load_model, job.repo_dir, checkpoint, device)

    def solve(max_frames: int):
        """The whole shot in chunks of at most `max_frames`, stitched."""
        chunks = recon.plan_chunks(len(frames), max_frames, p["overlap"])
        stitch = recon.Stitcher(rotation="points")  # the chunk-end cameras are less reliable than the points
        for ci, (a, b) in enumerate(chunks):
            label = f"重建（第 {ci + 1}/{len(chunks)} 段）" if len(chunks) > 1 else "重建"
            run.stage(label)
            res = run_chunk(model, images[a:b], p["conf_threshold"], device, p["update"],
                            lambda d, t, _l=label: progress(d, t, _l) if d % 4 == 0 or d == t else None)
            try:
                stitch.add(a, res).update(frames=[frames[a], frames[b - 1]])
            except ValueError:
                fail("E-CUT3R-STITCH")
            torch.cuda.empty_cache()
        return max_frames, chunks, stitch

    t_inf = time.time()
    bound = MemoryBound.parameter("max_frames", [s for s in MAX_FRAMES_STEPS if s > p["overlap"]])
    p["max_frames"], chunks, stitch = run.fit(bound, solve, min(p["max_frames"], len(frames)))
    infer_seconds = time.time() - t_inf
    run.frame_seconds.extend([infer_seconds / len(frames)] * len(frames))  # the chunks are one pass over the shot: shared out per frame

    stitched = list(stitch.pop())
    per_frame_focal = np.array([f.K[0, 0] for f in stitched])
    focal = np.full_like(per_frame_focal, np.median(per_frame_focal)) if p["shared_focal"] else per_frame_focal
    run.stage("写出结果")
    summary = di.write_outputs(raw, frames, geo, stitched, focal, progress)
    spread = float(per_frame_focal.std() / np.median(per_frame_focal))
    if spread > 0.1:
        say("W-CUT3R-FOCALJUMP", low=int(round(per_frame_focal.min() / geo.sx)),
            high=int(round(per_frame_focal.max() / geo.sx)))
    for info in stitch.alignments[1:]:
        if info["residual_rms_relative"] > 0.05:
            say("W-CUT3R-STITCHERROR", first=int(info["frames"][0]), last=int(info["frames"][1]),
                error=float(info["residual_rms_relative"]))

    run.finish(
        frames,
        kind="reconstruction",
        extension="cut3r",
        model="CUT3R cut3r_512_dpt_4_64",
        state_update="TTT3R (sigmoid of the state-to-frame cross-attention, per state token)" if p["update"] == "ttt3r"
                     else "CUT3R (the state replaced every frame)",
        rope=rope,
        files="cameras.npz: frames, K, cam_to_world, width, height; frame_<n>.npz: depth, confidence, mask, points",
        convention=recon.CONVENTION,
        metric=True,
        units="metres (CUT3R predicts metric-scale point maps; approximate)",
        scale_cm=100.0,
        confidence="CUT3R self-view confidence (>= 1, higher = surer)",
        mask="confidence > conf_threshold, inside the network's crop",
        points="CUT3R's own world point map pts3d_in_other_view, stored per frame in that frame's camera, metres",
        moving_objects="not estimated by CUT3R and no mask input: the model sees the whole frame",
        params=p,
        frames=frames,  # the whole list, not the standard [first, last]: the converter reads every frame's file by it
        width=width,
        height=height,
        geometry=geo.describe(),
        chunks=[[frames[a], frames[b - 1]] for a, b in chunks],
        chunk_alignment=stitch.alignments,
        focal_px_per_frame=(per_frame_focal / (0.5 * (geo.sx + geo.sy))).round(2).tolist(),
        **summary,
        inference_seconds=round(infer_seconds, 1),
    )


if __name__ == "__main__":
    serve(main)
