"""Mesh4D worker: a cut-out object -> a mesh that deforms (one topology, points per frame).

Upstream's own path, `hy3dshape/infer.py` on an "in-the-wild" sequence, driven from here instead
of from its hard-coded evaluation loop. Nothing of the method is reimplemented: its dataset class
prepares the pictures, its shape pipeline generates the mesh, its deformation pipeline moves it.

  1. stage: every frame is written as the RGBA PNG upstream reads (plate + the 「前景遮罩」 in
     alpha; its loader composites that over white itself);
  2. shape: Hunyuan3D-2.1 turns frame 1 into one mesh (optionally simplified first, to the
     node's 「面数」, with upstream's own simplifier);
  3. deform: its flow-matching denoiser + deformation VAE move that mesh's vertices to each of
     the 6 frames of a window, conditioned on the six pictures. A longer shot is windows of 6,
     the same mesh in every one; the last window is the shot's last 6 frames, so every frame is
     solved inside a real 6-frame window (the node warns about the seams).

Raw contract (what the node reads back):

    raw/mesh.npz
        faces      int32  [T, 3]      triangles, the same for every frame
        vertices   float32[F, V, 3]   the point cache, in upstream's deformation space:
                                      Z up, bounding box min at the origin, longest side 0.9
        rest       float32[V, 3]      the generated mesh itself, same topology, same space
        frames     int32  [F]         the original frame numbers, in order
    raw/result.json
        windows      how many 6-frame windows it took
        seams        the frame numbers a window starts at, the first one left out
        target_faces the 「面数」 the mesh was simplified to before deforming (0: not simplified)
        verts faces  of the delivered mesh
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lab2shot_worker import WEIGHTS_ENV, fail, progress, save_npz, serve, set_seed, stub_module
from lab2shot_worker.files import read_frame, read_mask
from lab2shot_worker.run import Run
from lab2shot_worker.serving import resident

NODE = "mesh4d.solve"

WINDOW = 6  # the model's window (configs num_frames / length_sequence): not a setting

# The dinov2-large weight's dest in extension.py (under weights/): the installer downloads it there as plain files
DINOV2 = "hf/hub/models--facebook--dinov2-large"


def _code_base() -> Path:
    """The composed tree (extension.py worker_env MESH4D_CODE_BASE); upstream runs from its
    hy3dshape/ folder and reads ./configs/infer.yaml and ../ckpt/* relative to it."""
    value = os.environ.get("MESH4D_CODE_BASE", "")
    if not value or not Path(value, "hy3dshape", "configs", "infer.yaml").is_file():
        fail("E-MESH4D-NOCODEBASE", name="MESH4D_CODE_BASE")
    return Path(value)


@resident
def _shapegen():
    """Hunyuan3D-2.1's image-to-3D pipeline (infer.py's `pipeline_shapegen`)."""
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")


def _dinov2_from_weights(params: dict) -> None:
    """Point the image encoder at the DINOv2 files the installer wrote, when they are there.

    upstream's infer.yaml names the encoder by repository (`version: 'facebook/dinov2-large'`), which
    transformers resolves in the Hugging Face cache layout (refs/, snapshots/) under HF_HOME. The
    installer downloads the weight as plain files into its dest (snapshot_download(local_dir=...)), so on a
    fresh install that name resolves to nothing and loading fails offline. The version handed to upstream's
    own ImageEncoder (AutoModel.from_pretrained(version)) becomes that folder instead; only the config this
    worker read is changed, never upstream's files. Without the files (an environment that holds only the
    cache layout) the name is left as it is."""
    folder = Path(os.environ.get(WEIGHTS_ENV, "")) / DINOV2
    if (folder / "config.json").is_file() and (folder / "model.safetensors").is_file():
        params["cond_stage_config"]["params"]["main_image_encoder"]["kwargs"]["version"] = str(folder)


@resident
def _deform(weights: str):
    """Mesh4D's deformation pipeline, built exactly as infer.py builds it, except that the
    denoiser's weights come from the fp16 safetensors the install step wrote out of the authors'
    24 GB training archive (build_mesh4d.py; the same tensors, the same key rewrite)."""
    import torch
    import yaml
    from safetensors.torch import load_file

    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    from hy3dshape.utils.misc import instantiate_from_config, instantiate_non_trainable_model

    params = yaml.safe_load(open("./configs/infer.yaml", encoding="utf-8"))["model"]["params"]
    _dinov2_from_weights(params)
    model = instantiate_from_config(params["denoiser_cfg"])
    missing, _ = model.load_state_dict(load_file(weights), strict=False)
    if missing:
        fail("E-MESH4D-STEP", step="装载形变网络", detail=f"缺 {len(missing)} 个权重，第一个是 {missing[0]}")
    model = model.cuda().half()

    conditioner = instantiate_from_config(params["cond_stage_config"])
    conditioner.disable_drop = True
    pipeline = instantiate_from_config(
        params["pipeline_cfg"],
        vae=instantiate_non_trainable_model(params["first_stage_config"]),
        model=model,
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        conditioner=conditioner,
        image_processor=instantiate_from_config(params["image_processor_cfg"]),
        cond_stage_model_2=instantiate_non_trainable_model(params["cond_stage_config_2"]),
        z_scale_factor=params["z_scale_factor"],
    )
    pipeline.device, pipeline.dtype = torch.device("cuda"), torch.float16
    return pipeline


def _stage_frames(job, folder: Path) -> list[int]:
    """The plate and its matte -> the RGBA PNGs upstream's loader reads, named 0.png, 1.png, …
    in frame order (it sorts by that number). Returns the frame numbers, in the same order."""
    from PIL import Image

    masks = job.listing("mask")
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (frame, path) in enumerate(job.frames):
        rgb = read_frame(path)  # uint8 RGB, the plate as the models see it
        alpha = masks.get(frame)
        if alpha is None:
            fail("E-MESH4D-EMPTYMASK", frame=frame)
        matte = read_mask(alpha)  # float 0..1, the plate's size
        if float(matte.max()) <= 0.003:  # nothing to cut out: upstream would raise "input image is empty"
            fail("E-MESH4D-EMPTYMASK", frame=frame)
        rgba = np.concatenate([rgb, (np.clip(matte, 0, 1) * 255).astype(np.uint8)[..., None]], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(folder / f"{i}.png")
        frames.append(frame)
    return frames


def _windows(count: int) -> list[int]:
    """Where each 6-frame window starts: 0, 6, 12 … and, when the shot is not a multiple of 6,
    one more at the end so its last frames are solved inside a real window too (it overlaps the
    one before; only its own tail is kept)."""
    starts = list(range(0, max(count - WINDOW, 0) + 1, WINDOW))
    if starts[-1] + WINDOW < count:
        starts.append(count - WINDOW)
    return starts


def _dataset(code: Path, data_root: Path, name: str, starts: list[int], logs: Path):
    """Upstream's own dataset, told which windows to make: its constructor enumerates windows by
    a fixed stride (interval_between_sequence, 40 in its config), so the list is replaced with
    the windows computed here. Everything it then does to the pictures is its own code."""
    import yaml

    # its dataloader's first line is a stray `from tkinter import S` (S is never used); there is no
    # Tk in this environment and none is wanted, so the name is stood in for instead of installing one
    stub_module("tkinter", S="s")

    from hy3dshape.utils.misc import instantiate_from_config

    spec = yaml.safe_load(open("./configs/infer.yaml", encoding="utf-8"))["dataset"]
    spec["params"] = {**spec["params"], "dataset_path": str(data_root), "num_workers": 0, "val_num_workers": 0,
                      "log_dir": str(logs),  # nothing of this path writes into the checkout
                      "cfg": str(code / "configs" / "OBJVERSE" / "train" / "infer.yaml")}
    module = instantiate_from_config(spec)
    module.val_dataset.models = [{"model": name, "start_idx": s, "stride": 1} for s in starts]
    return module


def _simplify(mesh, target: int, work: Path):
    """Upstream's own simplifier (hy3dshape.pipelines_…_infer.mesh_simplify_trimesh), at the face
    count the node's 「面数」 asks for. Once, before any deformation: the whole shot then shares
    this topology."""
    from hy3dshape.pipelines_video_newvae_all_nonalign_infer import mesh_simplify_trimesh

    import trimesh

    raw, small = work / "generated.obj", work / "simplified.obj"
    mesh.export(raw)
    mesh_simplify_trimesh(str(raw), str(small), target_count=target)
    return trimesh.load(small, process=False)


def main(job_path: str) -> None:
    import torch
    import trimesh
    from PIL import Image

    run = Run.start(job_path, NODE, "Mesh4D")
    job, params = run.job, run.params
    code = _code_base()
    os.chdir(code / "hy3dshape")  # upstream reads ./configs and ../ckpt from here

    run.stage("整理画面")
    work = job.scratch("mesh4d")
    name = "shot/seq"
    data_root = work / "DATA"
    frames = _stage_frames(job, data_root / name)
    if len(frames) < WINDOW:  # the node refuses this before cooking (min_frames); a direct call could still get here
        fail("E-MESH4D-STEP", step="整理画面", detail=f"Mesh4D 一次要 {WINDOW} 帧，这一段只有 {len(frames)} 帧")
    starts = _windows(len(frames))

    shapegen = run.model("Hunyuan3D-2.1 形状模型", _shapegen)
    run.stage("生成网格")
    generator = set_seed(int(params["seed"]))
    mesh, _ = shapegen(image=Image.open(data_root / name / "0.png").convert("RGBA"),
                       preset_latent=None, return_init_latent=True, generator=generator)
    mesh = mesh[0]
    target = int(params["target_faces"])
    if target:
        mesh = _simplify(mesh, target, work)
    torch.cuda.empty_cache()

    pipeline = run.model("Mesh4D 形变模型", _deform, str(job.weights_dir / "denoiser.fp16.safetensors"))
    run.stage("解形变")
    module = _dataset(code, data_root, name, starts, work / "logs")
    out = work / "out"
    done: dict[int, np.ndarray] = {}
    rest: np.ndarray | None = None
    faces: np.ndarray | None = None
    for window, batch in enumerate(module.val_dataloader()):
        model_name = batch["model_name"][0]
        batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.amp.autocast(device_type="cuda"):
            pipeline(batch=batch, output_path=str(out), generator=generator, gen_mesh=mesh,
                     gen_mesh_list=None, not_simplify=True, combine_glb=False,
                     num_inference_steps=int(params["steps"]))
        folder = out / model_name
        for j in range(WINDOW):
            solved = trimesh.load(folder / "gen" / f"gen_{j:02d}.obj", process=False)
            if faces is None:
                faces = np.asarray(solved.faces, np.int32)
            # overlapping frames (only the last window can have them) keep the result computed first:
            # every frame comes from a genuine 6-frame window
            done.setdefault(starts[window] + j, np.asarray(solved.vertices, np.float32))
        if rest is None:
            rest = np.asarray(trimesh.load(folder / "gen_mesh" / "registered_gen_mesh.obj",
                                           process=False).vertices, np.float32)
        torch.cuda.empty_cache()
        progress(window + 1, len(starts), "解形变")

    vertices = np.stack([done[i] for i in range(len(frames))])
    save_npz(job.raw_dir / "mesh.npz", faces=faces, vertices=vertices, rest=rest,
             frames=np.asarray(frames, np.int32))
    run.finish(frames, windows=len(starts), seams=[frames[s] for s in starts[1:]],
               target_faces=target, verts=int(vertices.shape[1]), faces=int(faces.shape[0]))


if __name__ == "__main__":
    serve(main)
