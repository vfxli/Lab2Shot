"""VidEoMT worker: video panoptic segmentation (VIPSeg, 124 classes) with ids that stay the same through the shot. Runs
inside third_party/videomt/.venv; never imports Lab2Shot core.

    python worker.py <job.json>

The network is upstream's (videomt/modeling/backbone: VidEoMT_CLASS, MIT), loaded by path: nothing else of the repo
runs. Its training framework (detectron2 registries, which the network file names at import) and the DINOv3 loader
(transformers) are not used, so they are stand-ins; the DINOv2 ViT is built without timm's pretrained download, since
the VidEoMT checkpoint holds every weight.

How the video is segmented is upstream's video panoptic inference (videomt_online with window size 1, as evaluated),
done in passes so a whole shot fits in memory at the plate's size:

1. Every frame at upstream's test size (short side params.resolution, long side at most 1333 / 720 of it; PIL bilinear),
   normalised and padded to a multiple of 32, goes through the network one frame at a time; the queries carry on from
   frame to frame (resume). A query is one segment through the whole shot: that is what keeps ids stable.
2. Each query's class is its class logits averaged over every frame of the shot (upstream's rule), then softmax;
   queries that are an object (best class not "nothing") on at least MIN_FRAMES frames, and whose best class scores
   over params.threshold, are kept. Because the average runs over every frame, an object present on only part of the
   shot scores lower than one present throughout.
3. The network runs again (it is deterministic) and each frame's kept masks are resized to the plate (bilinear from
   the padded input size, cropped, sigmoid, bilinear to the plate); every pixel goes to the kept query with the
   highest score x mask, if that query's own mask is at least 0.5 there.
4. Over the whole shot, a query whose pixels are under params.min_coverage of its own mask area is dropped (it lost most of
   its mask to others); stuff classes (sky, road, ...) are one segment per class; things (people, cars, ...) keep a
   segment per query.

The network's classes are VIPSeg's things in id order, then its stuff in id order (upstream's dataset mapper), not
VIPSeg's ids: params.things (VIPSeg's thing ids) gives the order, and each segment is reported by its VIPSeg id.

Output: raw/frame_<n>.npz segments uint16 [H,W] at the plate size (0 = none), raw/segments.json
[{"id", "category", "isthing", "query", "pixels"}] in id order, raw/result.json.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lab2shot_worker import fail, nothing, progress, resident, save_npz, serve, stub_module
from lab2shot_worker.run import Run

CHECKPOINT = "VidEoMT/vipseg_vit_large_55.2.pth"
# configs/VIPSeg/videomt/dinov2/vit-large/videomt_Online_ViTL.yaml and its bases
IMG_SIZE, NUM_CLASSES, NUM_QUERIES = 1280, 124, 200
BACKBONE, SEGMENTER_BLOCKS, NUM_FRAMES = "vit_large_patch14_reg4_dinov2", (20, 21, 22, 23), 5
PIXEL_MEAN, PIXEL_STD = (123.675, 116.280, 103.530), (58.395, 57.120, 57.375)  # RGB, 0..255
SIZE_DIVISIBILITY = 32
MAX_SIZE_RATIO = 1333 / 720  # MAX_SIZE_TEST (detectron2's default) over MIN_SIZE_TEST
# An additional filter not present upstream: objects that appear on only two or three frames are not treated as
# objects in the shot. It does not affect the scores, only whether a query is kept; scoring follows upstream (see the
# averaging below). The filter is recorded in the result description, visible to the user.
MIN_FRAMES = 3
UPSAMPLE_CHUNK = 16  # kept queries resized to the plate at a time (plate pixels x 4 bytes each)


# --------------------------------------------------------------------------- network


def _backbone_module(repo: Path):
    """videomt/modeling/backbone/videomt.py as a package of its own (its relative imports: vit, scale_block), without
    importing the videomt package (whose __init__ loads the training framework)."""
    from torch import nn

    class Backbone(nn.Module):  # detectron2.modeling.Backbone: only a base class here
        pass

    class Registry:  # detectron2's BACKBONE_REGISTRY: registering is all the file does with it
        def register(self):
            return lambda cls: cls

    stub_module("detectron2")
    stub_module("detectron2.modeling", Backbone=Backbone, BACKBONE_REGISTRY=Registry())
    stub_module("transformers", AutoModel=None)  # the DINOv3 path only
    folder = repo / "videomt" / "modeling" / "backbone"
    name = "videomt_backbone"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, folder / "__init__.py", submodule_search_locations=[str(folder)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    return importlib.import_module(f"{name}.videomt")


@resident
def load_model(checkpoint: Path, repo: Path):
    import timm

    module = _backbone_module(repo)
    create = timm.create_model

    def no_download(*args, **kwargs):  # every weight comes from the VidEoMT checkpoint
        return create(*args, **{**kwargs, "pretrained": False})

    timm.create_model = no_download
    try:
        model = module.VidEoMT_CLASS(img_size=IMG_SIZE, num_classes=NUM_CLASSES, name=BACKBONE, num_frames=NUM_FRAMES,
                                     num_q=NUM_QUERIES, segmenter_blocks=list(SEGMENTER_BLOCKS))
    finally:
        timm.create_model = create
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    own = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(own, strict=False)
    missing = [k for k in missing if k != "attn_mask_probs"]
    if missing or unexpected:
        fail("E-WORKER-WEIGHTSMISMATCH", project="VidEoMT", extension="videomt", model=checkpoint.name,
             missing=len(missing), unexpected=len(unexpected), examples=[*missing[:3], *unexpected[:3]])
    return model.to("cuda").eval()


# --------------------------------------------------------------------------- frames


def test_size(width: int, height: int, short: int) -> tuple[int, int]:
    """detectron2 ResizeShortestEdge.get_output_shape: (w, h) with the short side `short`, the long side at most
    short x 1333 / 720."""
    longest = short * MAX_SIZE_RATIO
    scale = short / min(height, width)
    newh, neww = (short, scale * width) if height < width else (scale * height, short)
    if max(newh, neww) > longest:
        s = longest / max(newh, neww)
        newh, neww = newh * s, neww * s
    return int(neww + 0.5), int(newh + 0.5)


class Frames:
    """The plate's frames as the network's input: resized (PIL bilinear, as detectron2's ResizeTransform does for 8-bit
    images), normalised, padded to a multiple of 32 (ImageList.from_tensors pads the normalised image with 0)."""

    def __init__(self, paths: list[Path], size: tuple[int, int]):
        self.paths, self.size = paths, size
        w, h = size
        self.padded = (-(-h // SIZE_DIVISIBILITY) * SIZE_DIVISIBILITY, -(-w // SIZE_DIVISIBILITY) * SIZE_DIVISIBILITY)
        self.mean = torch.tensor(PIXEL_MEAN, device="cuda").view(3, 1, 1)
        self.std = torch.tensor(PIXEL_STD, device="cuda").view(3, 1, 1)

    def __getitem__(self, i: int) -> torch.Tensor:
        from PIL import Image

        with Image.open(self.paths[i]) as im:
            img = np.asarray(im.convert("RGB").resize(self.size, Image.BILINEAR), np.float32)
        x = (torch.from_numpy(img).to("cuda").permute(2, 0, 1) - self.mean) / self.std
        w, h = self.size
        return F.pad(x, (0, self.padded[1] - w, 0, self.padded[0] - h))[None]


def sweep(model, frames: Frames, count: int, what: str):
    """The network over the shot, one frame at a time, queries carried from frame to frame: yields (index, class
    logits [Q, C+1], mask logits [Q, h, w])."""
    with torch.no_grad():
        for i in range(count):
            out = model(frames[i], resume=i > 0)
            progress(i + 1, count, what)
            yield i, out["pred_logits"][0, 0], out["pred_masks"][0, :, 0]


# --------------------------------------------------------------------------- job


def main(job_path: str) -> None:
    run = Run.start(job_path, "videomt.panoptic", "VidEoMT")
    job, p = run.job, run.params
    checkpoint = job.weights_dir / CHECKPOINT
    run.weights(checkpoint)
    paths = [path for _, path in job.frames]
    count = len(paths)
    size = test_size(job.width, job.height, p["resolution"])
    frames = Frames(paths, size)
    things = sorted(p["things"])
    # network class -> VIPSeg id: the things in id order, then the stuff
    vipseg = things + sorted(set(range(NUM_CLASSES)) - set(things))
    model = run.model("VidEoMT", load_model, checkpoint, job.repo_dir)

    run.stage("分割：每个物体整段的类别")
    # As upstream (`videomt/videomt.py:307 post_processing`: `out_logits = sum(out_logits)/len(out_logits)`):
    # each query's class score is the plain average of its logits over all frames, divided by the frame count.
    # Averaging only over frames where the query is an object would raise briefly visible objects to the same score
    # as objects present throughout.
    logits_sum = torch.zeros((NUM_QUERIES, NUM_CLASSES + 1), dtype=torch.float64, device="cuda")
    present = torch.zeros(NUM_QUERIES, dtype=torch.float64, device="cuda")  # number of frames on which each query is an object (used only for the flicker filter)
    seen = 0
    for _, cls_logits, _ in sweep(model, frames, count, "分类"):
        logits_sum += cls_logits.double()
        present += (cls_logits.argmax(-1) != NUM_CLASSES).double()
        seen += 1
    scores, labels = F.softmax((logits_sum / max(seen, 1)).float(), dim=-1).max(-1)
    kept = torch.nonzero((present >= min(MIN_FRAMES, count)) & (labels != NUM_CLASSES) & (scores > p["threshold"])).flatten()
    k_count = len(kept)
    if k_count == 0:  # nothing kept: an empty segmentation, not an error
        nothing("N-VIDEOMT-EMPTY")
    if k_count > 254:
        fail("E-VIDEOMT-TOOMANY", count=k_count)
    k_scores = scores[kept].view(-1, 1, 1)
    height, width = job.height, job.width

    area = torch.zeros(k_count, dtype=torch.float64, device="cuda")  # pixels each query wins
    own_area = torch.zeros_like(area)  # pixels of its own mask >= 0.5
    inside = torch.zeros_like(area)  # pixels it wins where its own mask >= 0.5
    with tempfile.TemporaryDirectory(prefix="videomt_", dir=job.dir) as tmp:
        tmp = Path(tmp)
        run.stage("分割：逐帧的像素")
        w, h = size
        for i, _, masks in sweep(model, frames, count, "逐帧分割"):
            masks = masks[kept][None]
            best = torch.full((height, width), -1.0, device="cuda")
            winner = torch.zeros((height, width), dtype=torch.int64, device="cuda")
            winner_mask = torch.zeros((height, width), device="cuda")
            for c in range(0, k_count, UPSAMPLE_CHUNK):
                m = F.interpolate(masks[:, c:c + UPSAMPLE_CHUNK], size=frames.padded, mode="bilinear", align_corners=False)
                m = m[:, :, :h, :w].sigmoid()
                m = F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)[0]
                own_area[c:c + len(m)] += (m >= 0.5).flatten(1).sum(1).double()
                prob = k_scores[c:c + len(m)] * m
                top, arg = prob.max(0)
                better = top > best  # strictly: on a tie the earlier query wins, as argmax over all of them
                best = torch.where(better, top, best)
                winner = torch.where(better, arg + c, winner)
                winner_mask = torch.where(better, m.gather(0, arg[None])[0], winner_mask)
            area += torch.bincount(winner.flatten(), minlength=k_count).double()
            own = winner_mask >= 0.5
            inside += torch.bincount(winner[own], minlength=k_count).double()
            save_npz(tmp / f"{i}.npz", compression=1, win=torch.where(own, winner, 255).to(torch.uint8).cpu().numpy())

        run.stage("写出分割")
        segment_of = np.zeros(256, np.uint16)  # kept query -> segment id (0: dropped); 255 = no one
        segments, stuff = [], {}
        area_, own_, inside_ = area.cpu().numpy(), own_area.cpu().numpy(), inside.cpu().numpy()
        for k in range(k_count):
            label = int(labels[kept[k]])
            category, thing = vipseg[label], label < len(things)
            if not (area_[k] > 0 and own_[k] > 0 and inside_[k] > 0) or area_[k] / own_[k] < p["min_coverage"]:
                continue
            if not thing and category in stuff:
                segment_of[k] = stuff[category]
                continue
            sid = len(segments) + 1
            segment_of[k] = sid
            if not thing:
                stuff[category] = sid
            segments.append({"id": sid, "category": category, "isthing": thing, "query": int(kept[k]), "pixels": 0})
        if not segments:  # every kept object dropped by 完整度门槛: same as none kept
            nothing("N-VIDEOMT-EMPTY")
        for i, (f, _) in enumerate(job.frames):
            seg = segment_of[np.load(tmp / f"{i}.npz")["win"]]
            for sid, n in zip(*np.unique(seg[seg > 0], return_counts=True)):
                segments[sid - 1]["pixels"] += int(n)
            save_npz(job.raw_dir / f"frame_{f}.npz", segments=seg)
            progress(i + 1, count, "写出分割")
    (job.raw_dir / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    run.finish([f for f, _ in job.frames], segments=len(segments), kept_queries=k_count, processing_size=list(size),
               class_score_rule="每个物体的类别分数 = 它在所有帧上的 logits 简单平均再 softmax（官方 "
                                "videomt.py post_processing 的口径）",
               flicker_filter=f"只在少于 {MIN_FRAMES} 帧里出现的物体不留（这一条是 Lab2Shot 加的，官方没有；"
                              f"它不碰分数，只决定留不留）",
               peak_ram_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1))


if __name__ == "__main__":
    serve(main)
