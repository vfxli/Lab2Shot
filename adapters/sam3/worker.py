"""SAM 3 worker: video segmentation + tracking. Runs inside third_party/sam3/.venv
with the original repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>

Two prompt types, both run SAM 3's video models so objects are tracked through
the shot (memory of past frames), not segmented frame by frame:

* text concept (params.prompt, e.g. "person"): SAM 3's detector + tracker video
  model finds every instance on every frame, starts a masklet for new ones and
  tracks them. Object ids are assigned in order of appearance.
* per-person boxes (inputs.boxes, the "人物框" JSON): SAM 3's tracker alone,
  each person's box on the first frame they appear is a box prompt for that
  person's id (the text prompt is ignored).

Long shots are cut into overlapping chunks, one inference session each, so GPU
memory is bounded by the chunk length, not the shot length (SAM 3 keeps every
tracked frame's memory features, ~1 MB per object per frame, plus the masks it
has output). Frames stay on the CPU (offload_video_to_cpu). Across chunks:
text mode re-detects and links ids by mask overlap on OVERLAP_TEXT shared
frames; box mode hands each person's last mask to the next chunk as a mask
prompt (a person lost at a chunk border is re-acquired from their next box).

Output: raw/frame_<n>.npz (masks uint8 [K,H,W] 0/255, ids int32 [K], same order
on every frame) and raw/objects.json.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lab2shot_worker import fail, nothing, progress, require_weights, resident, save_npz, say, serve
from lab2shot_worker.frame_io import FrameReader
from lab2shot_worker.run import Run

IMAGE_SIZE = 1008  # SAM 3's input resolution (frames are squashed to 1008x1008)
# Chunk length: the tracker keeps ~0.8 MB per object per frame on the GPU (memory
# features 64x72x72 + low-res mask logits), on top of ~4.7 GB for the model and
# its activations; text mode also holds the chunk's frames in CPU RAM (6 MB each).
# 8000 MB of state: 8 objects -> 800 frames, ~11 GB peak on the GPU.
STATE_BUDGET_MB = 8000
MIN_CHUNK, MAX_CHUNK = 60, 800
OVERLAP_TEXT = 8  # frames shared by consecutive text-mode chunks (id linking)
LINK_IOU = 0.3  # min mask IoU on the shared frames to keep an id across chunks


def log(message: str) -> None:
    print(message, flush=True)


def chunk_length(max_objects: int) -> int:
    override = int(os.environ.get("SAM3_CHUNK_FRAMES", "0") or 0)
    if override > 0:
        return max(override, OVERLAP_TEXT + 2)
    return int(min(MAX_CHUNK, max(MIN_CHUNK, STATE_BUDGET_MB // max(1, max_objects))))


def chunks(n: int, length: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) index ranges covering 0..n, consecutive ranges share `overlap` frames."""
    out, start = [], 0
    while True:
        end = min(n, start + length)
        out.append((start, end))
        if end >= n:
            return out
        start = end - overlap


# --------------------------------------------------------------------------- model


@resident
def build_model(checkpoint: Path, repo: Path):
    """SAM 3 detector + tracker video model (sam3.pt), everything local."""
    from sam3.model_builder import build_sam3_video_model

    bpe = repo / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_video_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        bpe_path=str(bpe),
        device="cuda",
    )
    return model.eval()


# --------------------------------------------------------------------------- frames


def load_frame_tensor(path: Path) -> torch.Tensor:
    """Like SAM 3's tracker loader (sam2_utils): squash to 1008x1008, normalize to [-1, 1]."""
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (t - 0.5) / 0.5


def frame_seq(paths: list[Path]) -> FrameReader:
    """A chunk's frames for the tracker (it indexes them itself), decoded on demand with a small read-ahead; the
    frame before the current one stays (the tracker steps back one)."""
    return FrameReader(paths, load_frame_tensor, threads=2, ahead=4, keep=2)


def link_frames(paths: list[Path], folder: Path, ext: str) -> Path:
    """SAM 3's video loaders read a folder of <index>.<ext> images: symlink the chunk's frames."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for i, path in enumerate(paths):
        (folder / f"{i:05d}{ext}").symlink_to(path.resolve())
    return folder


def free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- mask store


class MaskStore:
    """Per-frame masks keyed by object id, spilled to disk as packed bits, so the
    final npz files can be written once the number of objects K is known."""

    def __init__(self, folder: Path, height: int, width: int):
        self.folder = folder
        self.h, self.w = height, width
        folder.mkdir(parents=True, exist_ok=True)
        # id -> [first index, last index, pixel count, frames present]
        self.stats: dict[int, list[int]] = {}

    def put(self, index: int, masks: dict[int, np.ndarray], merge: bool = False) -> None:
        """Store a frame's masks; merge=True adds objects to what the frame already has."""
        new = {i: m for i, m in masks.items() if m.any()}
        masks = {**self.get(index), **new} if merge else new
        ids = sorted(masks)
        for oid in new:
            s = self.stats.setdefault(oid, [index, index, 0, 0])
            s[0], s[1] = min(s[0], index), max(s[1], index)
            s[2] += int(masks[oid].sum())
            s[3] += 1
        bits = np.stack([np.packbits(masks[i], axis=None) for i in ids]) if ids else np.zeros((0, 0), np.uint8)
        save_npz(self.folder / f"{index:06d}.npz", ids=np.asarray(ids, np.int64), bits=bits)

    def get(self, index: int) -> dict[int, np.ndarray]:
        path = self.folder / f"{index:06d}.npz"
        if not path.exists():
            return {}
        data = np.load(path)
        n = self.h * self.w
        return {
            int(oid): np.unpackbits(bits, count=n).reshape(self.h, self.w).astype(bool)
            for oid, bits in zip(data["ids"], data["bits"])
        }


# --------------------------------------------------------------------------- text mode


def mask_iou_sum(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    inter = sum(int(np.logical_and(x, y).sum()) for x, y in zip(a, b))
    union = sum(int(np.logical_or(x, y).sum()) for x, y in zip(a, b))
    return inter / union if union else 0.0


def run_text(model, job, prompt: str, max_objects: int, threshold: float, store: MaskStore, tmp: Path):
    """Detector + tracker over the shot, chunk by chunk. Returns {id: best detection score}."""
    frames = job.frames
    n = len(frames)
    model.max_num_objects = max_objects
    model.score_threshold_detection = threshold
    # New masklets need a more confident detection than matching does (upstream: 0.5 / 0.7).
    model.new_det_thresh = min(0.95, threshold + 0.2)

    spans = chunks(n, chunk_length(max_objects), OVERLAP_TEXT)
    log(f"[sam3] text prompt {prompt!r}: {n} frames in {len(spans)} chunk(s) {spans}")
    scores: dict[int, float] = {}
    next_id = 1
    prev_end = 0
    done = 0
    for c, (start, end) in enumerate(spans):
        t_chunk = time.time()
        overlap = max(0, prev_end - start)  # indices [start, start+overlap) were output by the previous chunk
        folder = link_frames([p for _, p in frames[start:end]], tmp / "chunk", ".png")
        state = model.init_state(resource_path=str(folder), offload_video_to_cpu=True, async_loading_frames=True)
        model.add_prompt(state, frame_idx=0, text_str=prompt)

        mapping: dict[int, int] = {}  # local id -> global id
        stash: list[tuple[int, dict[int, np.ndarray], dict[int, float]]] = []

        def emit(index: int, local: dict[int, np.ndarray], local_scores: dict[int, float]) -> None:
            nonlocal next_id
            shared = index < start + overlap  # the previous chunk already wrote this frame
            have = store.get(index) if shared else {}
            out = {}
            for lid, mask in local.items():
                if lid not in mapping:
                    mapping[lid] = next_id
                    next_id += 1
                gid = mapping[lid]
                if gid in have:  # keep the previous chunk's mask on shared frames
                    continue
                out[gid] = mask
                scores[gid] = max(scores.get(gid, 0.0), local_scores.get(lid, 0.0))
            store.put(index, out, merge=shared)

        def link() -> None:
            """Match this chunk's objects to the previous chunk's on the shared frames."""
            prev = [store.get(i) for i, _, _ in stash]
            local_ids = sorted({lid for _, m, _ in stash for lid in m})
            global_ids = sorted({gid for p in prev for gid in p})
            pairs = []
            for lid in local_ids:
                for gid in global_ids:
                    a = [m.get(lid, np.zeros((store.h, store.w), bool))[::4, ::4] for _, m, _ in stash]
                    b = [p.get(gid, np.zeros((store.h, store.w), bool))[::4, ::4] for p in prev]
                    iou = mask_iou_sum(a, b)
                    if iou >= LINK_IOU:
                        pairs.append((iou, lid, gid))
            used_l, used_g = set(), set()
            for iou, lid, gid in sorted(pairs, reverse=True):
                if lid not in used_l and gid not in used_g:
                    mapping[lid] = gid
                    used_l.add(lid)
                    used_g.add(gid)
            log(f"[sam3] chunk {c}: linked {len(used_l)}/{len(local_ids)} object(s) to the previous chunk")

        for local_idx, out in model.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=None, reverse=False):
            # Only needed for later interactive refinement, which never happens here: keep GPU memory flat.
            state["cached_frame_outputs"].pop(local_idx, None)
            index = start + local_idx
            ids = [int(i) for i in out["out_obj_ids"]]
            local = {i: np.asarray(m, bool) for i, m in zip(ids, out["out_binary_masks"])}
            local_scores = {i: float(p) for i, p in zip(ids, out["out_probs"])}
            if local_idx < overlap:
                stash.append((index, local, local_scores))
                if local_idx == overlap - 1:
                    link()
                    for item in stash:
                        emit(*item)
                    stash.clear()
                continue
            emit(index, local, local_scores)
            done += 1
            progress(done, n, "分割跟踪")
        for item in stash:  # chunk shorter than the overlap (cannot happen with MIN_CHUNK, kept for safety)
            emit(*item)
        prev_end = end
        state.clear()
        del state
        log(f"[sam3] chunk {c}: {end - start} frames in {time.time() - t_chunk:.1f}s, "
            f"GPU peak so far {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
        free_cuda()
    return scores


# --------------------------------------------------------------------------- box mode


def run_boxes(model, job, people: dict[int, dict[int, list[float]]], store: MaskStore) -> dict[int, float]:
    """SAM 3 tracker with one box prompt per person. Returns {id: mean presence score}.

    SAM 3's tracker (like SAM 2) treats every prompted frame as a conditioning frame
    for *all* objects of a session: a person without a prompt there would be
    remembered as "absent" on that frame. So people prompted on different frames get
    separate sessions, run in lockstep over one shared image-feature cache (the image
    encoder still runs once per frame, like SAM 3's own detector + tracker model).
    """
    tracker = model.tracker
    own = tracker.backbone
    tracker.backbone = model.detector.backbone  # the tracker has no image encoder of its own
    try:
        return _run_boxes(tracker, job, people, store)
    finally:
        tracker.backbone = own  # the model stays loaded: text mode gets it as built


def _run_boxes(tracker, job, people: dict[int, dict[int, list[float]]], store: MaskStore) -> dict[int, float]:
    frames = job.frames
    n = len(frames)
    h, w = store.h, store.w
    spans = chunks(n, chunk_length(len(people)), 1)
    log(f"[sam3] box prompts for {len(people)} person(s): {n} frames in {len(spans)} chunk(s) {spans}")
    score_sum: dict[int, float] = {}
    score_n: dict[int, int] = {}
    carry: dict[int, np.ndarray] = {}  # person -> mask on the previous chunk's last frame
    started: set[int] = set()
    done = 0
    for c, (start, end) in enumerate(spans):
        t_chunk = time.time()
        n_local = end - start
        seq = frame_seq([p for _, p in frames[start:end]])
        shared: dict = {}  # frame -> (image, image features), the current frame only

        def features(local_idx: int) -> None:
            if local_idx not in shared:
                shared.clear()
                image = seq[local_idx].cuda().float().unsqueeze(0)
                shared[local_idx] = (image, tracker.forward_image(image))

        # person -> (local frame, "mask" | "box", prompt)
        prompts: dict[int, tuple[int, str, object]] = {}
        for pid, boxes in people.items():
            if pid in carry:
                prompts[pid] = (0, "mask", torch.from_numpy(carry[pid]))
                continue
            # First box of the shot, or (after the person was lost) their first box in this chunk.
            later = sorted(i for i in boxes if start <= i < end and (c == 0 or i > start or pid not in started))
            if later:
                x1, y1, x2, y2 = boxes[later[0]]
                box = torch.tensor([[x1 / w, y1 / h], [x2 / w, y2 / h]], dtype=torch.float32).clamp(0, 1)
                prompts[pid] = (later[0] - start, "box", box)
                started.add(pid)
        carry = {}
        groups: dict[int, list[int]] = {}
        for pid, (f, _, _) in prompts.items():
            groups.setdefault(f, []).append(pid)
        states = {}
        for f, pids in sorted(groups.items()):
            state = tracker.init_state(video_height=h, video_width=w, num_frames=n_local, cached_features=shared)
            state["images"] = seq
            features(f)
            for pid in pids:
                _, kind, value = prompts[pid]
                if kind == "mask":
                    tracker.add_new_mask(state, frame_idx=f, obj_id=pid, mask=value)
                else:
                    tracker.add_new_points_or_box(state, frame_idx=f, obj_id=pid, box=value, rel_coordinates=True)
            state["cached_features"] = shared  # a cache miss would have swapped in a private one
            states[f] = state
        log(f"[sam3] chunk {c}: prompts on local frames {sorted(groups)} for {sorted(prompts)}")

        gens = {}
        first_out = 0 if c == 0 else 1  # a chunk's first frame is the previous chunk's last
        for local_idx in range(n_local):
            local: dict[int, np.ndarray] = {}
            if local_idx in states:
                gens[local_idx] = tracker.propagate_in_video(
                    states[local_idx], start_frame_idx=local_idx, max_frame_num_to_track=n_local,
                    reverse=False, tqdm_disable=True, propagate_preflight=True,
                )
            if gens:
                features(local_idx)
            for gen in gens.values():
                out_idx, obj_ids, _, video_res_masks, obj_scores = next(gen)
                assert out_idx == local_idx, (out_idx, local_idx)
                masks = (video_res_masks[:, 0] > 0.0).cpu().numpy()
                presence = torch.sigmoid(obj_scores.float().reshape(-1)).cpu().numpy()
                for k, pid in enumerate(obj_ids):
                    local[pid] = masks[k]
                    if masks[k].any():
                        score_sum[pid] = score_sum.get(pid, 0.0) + float(presence[k])
                        score_n[pid] = score_n.get(pid, 0) + 1
            if local_idx == n_local - 1 and end < n:
                carry = {pid: m for pid, m in local.items() if m.any()}
            if local_idx >= first_out:
                store.put(start + local_idx, local)
                done += 1
                progress(done, n, "分割跟踪")
        for gen in gens.values():
            gen.close()
        seq.close()
        for state in states.values():
            state.clear()
        del states, gens
        shared.clear()
        log(f"[sam3] chunk {c}: {n_local} frames in {time.time() - t_chunk:.1f}s, "
            f"GPU peak so far {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
        free_cuda()
    return {pid: score_sum[pid] / score_n[pid] for pid in score_sum}


# --------------------------------------------------------------------------- main


def main(job_path: str) -> None:
    run = Run.start(job_path, "sam3.segment", "SAM 3")
    job, params = run.job, run.params
    prompt, max_objects, threshold = params["prompt"].strip(), params["max_objects"], params["threshold"]
    frames = run.frames()
    frame_numbers, width, height = frames.numbers, frames.width, frames.height

    people = None
    if "boxes" in job.inputs:  # person id -> {frame index in this job: [x1, y1, x2, y2] in pixels}
        people = job.people_by_index()
        if not people:
            nothing("N-SAM3-NOBOXES", first=frame_numbers[0], last=frame_numbers[-1])
        if len(people) > max_objects:
            keep = sorted(people, key=lambda p: -len(people[p]))[:max_objects]
            say("W-SAM3-TOOMANYPEOPLE", count=len(people), max_objects=max_objects, people=sorted(keep))
            people = {p: people[p] for p in sorted(keep)}
        if prompt:
            log(f"[sam3] boxes given: ignoring the text prompt {prompt!r}")
    elif not prompt:
        fail("E-SAM3-NOPROMPT")

    checkpoint = job.weights_dir / "sam3" / "sam3.pt"
    # not run.weights: the extension is "sam3", the project "SAM 3"
    require_weights("sam3", checkpoint, page="https://huggingface.co/facebook/sam3")

    raw = job.raw_dir
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = run.model("SAM 3", build_model, checkpoint, job.repo_dir)

    tmp = Path(tempfile.mkdtemp(prefix=".sam3_", dir=raw))
    try:
        store = MaskStore(tmp / "masks", height, width)
        run.stage("分割跟踪")
        t1 = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if people is not None:
                scores = run_boxes(model, job, people, store)
                label = prompt or "person"
            else:
                scores = run_text(model, job, prompt, max_objects, threshold, store, tmp)
                label = prompt
        t_track = time.time() - t1
        run.frame_seconds.extend([t_track / len(frame_numbers)] * len(frame_numbers))  # tracking is one pass over the shot: shared out per frame

        # K objects: those that were ever visible, at most max_objects (longest-lived first), in id order.
        present = {oid: s for oid, s in store.stats.items() if s[3] > 0}
        if people is not None:
            for pid in people:
                if pid not in present:
                    say("N-SAM3-PERSONEMPTY", person=pid)
        chosen = sorted(present, key=lambda o: (-present[o][3], present[o][0]))[:max_objects]
        if len(present) > len(chosen):
            say("W-SAM3-TOOMANYOBJECTS", count=len(present), max_objects=max_objects)
        ids = sorted(chosen) if people is not None else sorted(chosen, key=lambda o: (present[o][0], o))
        # Text mode: renumber 1..K in order of appearance.
        out_ids = list(ids) if people is not None else list(range(1, len(ids) + 1))

        run.stage("写出遮罩")
        coverage = {oid: 0.0 for oid in out_ids}
        for i, frame in enumerate(frame_numbers):
            masks = store.get(i)
            stack = np.zeros((len(ids), height, width), np.uint8)
            for k, oid in enumerate(ids):
                m = masks.get(oid)
                if m is not None:
                    stack[k][m] = 255
                    coverage[out_ids[k]] += float(m.mean())
            save_npz(raw / f"frame_{frame}.npz", 1, masks=stack, ids=np.asarray(out_ids, np.int32))  # fast zlib: masks shrink ~20x
            if (i + 1) % 10 == 0 or i + 1 == len(frame_numbers):
                progress(i + 1, len(frame_numbers), "写出遮罩")
        objects = [
            {
                "id": int(out_ids[k]),
                "label": label,
                "score": round(float(scores.get(oid, 0.0)), 4),
                "frames": [frame_numbers[present[oid][0]], frame_numbers[present[oid][1]]],
            }
            for k, oid in enumerate(ids)
        ]
        (raw / "objects.json").write_text(json.dumps(objects, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    run.finish(
        frame_numbers,
        kind="masks",
        objects=len(ids),
        width=width,
        height=height,
        prompt_type="boxes" if people is not None else "text",
        prompt=label,
        max_objects=max_objects,
        threshold=threshold,
        mask_format="raw/frame_<n>.npz: masks uint8 [K,H,W] (0/255), ids int32 [K]; raw/objects.json",
        # mean fraction of the frame covered by each object, over the whole shot
        coverage={str(k): round(v / len(frame_numbers), 4) for k, v in coverage.items()},
        model="SAM 3 (facebook/sam3, sam3.pt)",
        chunk_frames=chunk_length(len(people) if people is not None else max_objects),
    )


if __name__ == "__main__":
    serve(main)
