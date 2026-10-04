"""The LTX-2.5 runtime shared by the LTX feature workers (AlphaGen, Clean Plate, ...). It knows nothing of any one
feature: a feature is its LoRA plus a fixed `Recipe`, written in its worker as one `Feature`, never chosen by the artist.
A feature's worker is that constant and what it writes per frame; the whole job is run_feature here:

    MATTE = Feature(node="alphagen.matte", title="LTX-2.5 Alpha Gen", lora=..., recipe=Recipe(...), stage="matte",
                    shape_error="E-ALPHAGEN-SHAPE")
    def main(job_path): run_feature(job_path, MATTE, RESOLUTION, open_output)

run_feature: the base where the ltx extension installed it (LtxBase.installed) with the feature's LoRA fused in
(@resident), the fixed prompt encoded once and cached, the transformer built, the shot run window by window (run_shot)
under the one out-of-memory policy (Run.fit on the processing size), every output frame scaled back to the input size
and checked (one per input frame, in order, of its size), handed to the feature's output, and the job finished with the
facts every feature reports (processing size, window, seed).

One IC-LoRA pass, as the LoRA model cards recommend (ltx_pipelines.ic_lora with --skip-stage-2): the distilled
transformer with the LoRAs fused in, the input clip VAE-encoded as the reference latent
(VideoConditionByReferenceLatent, strength 1), the distilled 8-step Euler schedule at the native resolution, no
guidance (one transformer call per step), video only (no audio latent: as ltx_pipelines.alpha_gen), then the video
VAE decoder. Built from upstream's own blocks (ltx_core / ltx_pipelines at the pinned commit); the only differences
are that the models stay loaded between segments and jobs (upstream builds and frees them on every call), and that
the reference frames are passed as tensors instead of being read from a video file (no lossy re-encode).

The output follows the input strictly: every input frame gets exactly one output frame of the same size, and no other
frame is made. What the model needs is added and removed here:
* size: the frame is scaled down (never up) to fit recipe.max_side × recipe.max_short, then padded (reflected, never
  stretched) to multiples of 32; the padding is cropped off the result;
* length: a shot is cut into windows of at most recipe.max_frames frames (8n+1), overlapping by recipe.overlap and
  cross-faded linearly; a window shorter than 8n+1 is padded by repeating its last frame, and those frames are dropped;
* every window's output is checked against its input (frames, height, width); a mismatch stops the job.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lab2shot_worker import resident

MULTIPLE = 32  # the transformer's spatial patch on the latent grid: frame sides are multiples of 32
TIME_STRIDE = 8  # the VAE's temporal compression: frame counts are 8n+1
# The reference clip is VAE-encoded in one pass up to this many pixels × frames (measured on a 32 GB card with the
# transformer resident: 1920×1088×25 fits). A larger window is encoded in tiles, with upstream's own conv-VAE tile
# layout (ltx_pipelines.utils.helpers: long side 768 / overlap 64, 80 frames / overlap 24), as its tiled IC-LoRA
# stages do (ic_lora._reference_encode_tiling).
ENCODE_UNTILED_VOLUME = 1920 * 1088 * 25
# The most latent tokens one window may have (the reference adds as many again): measured on an RTX 5090 (32 GB) with
# the fp8 transformer resident, 1920×1088×57 frames (16 320 tokens) peaked at 31.6 GB of the card's 32.6; this keeps a
# margin (1920×1088 -> 49 frames, 1280×736 -> 113, 960×544 -> 145).
MAX_WINDOW_TOKENS = 14_400
DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)  # ltx_pipelines constants

BASE_REPO_DIR = "LTX-2.5"  # where the installer puts Lightricks/LTX-2.5's files (lab2shot_worker.hf_dest)
# the ltx extension's weights folder, set in every feature's worker environment (adapters/ltx/extension.py worker_env)
BASE_ENV = "LAB2SHOT_LTX_WEIGHTS"
TRANSFORMER = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = "vae/ltx-2.5-video-vae-bf16.safetensors"

log = logging.getLogger("ltx_runtime")


class ShapeMismatch(Exception):
    """The model gave a different number of frames or a different frame size than it was given."""

    def __init__(self, expected: tuple[int, int, int], got: tuple[int, ...]):
        super().__init__(f"expected {expected}, got {got}")
        self.expected, self.got = expected, got


@dataclass(frozen=True)
class LtxBase:
    """The base model files (adapters/ltx/extension.py BASE_FILES)."""

    transformer: Path
    text_encoder: Path
    video_vae: Path

    @classmethod
    def under(cls, weights: Path) -> LtxBase:
        root = weights / BASE_REPO_DIR
        return cls(root / TRANSFORMER, root / TEXT_ENCODER, root / VIDEO_VAE)

    @classmethod
    def installed(cls) -> LtxBase:
        """The base where the ltx extension installed it (BASE_ENV), shared by every feature."""
        import os

        return cls.under(Path(os.environ[BASE_ENV]))

    @property
    def files(self) -> tuple[Path, ...]:
        return (self.transformer, self.text_encoder, self.video_vae)


@dataclass(frozen=True)
class Lora:
    path: Path
    strength: float = 1.0


@dataclass(frozen=True)
class Recipe:
    """One feature's fixed way of running the model. Written as a constant in the feature's worker; no part of it is
    a node parameter."""

    prompt: str = ""
    sigmas: tuple[float, ...] = DISTILLED_SIGMAS
    seed: int = 1234
    fps: float = 24.0  # the transformer's time base for the latent positions
    max_frames: int = 145  # 8n+1: frames per window
    overlap: int = 17  # frames shared by consecutive windows (cross-faded)
    max_side: int = 1920  # the processing size is at most max_side × max_short (either orientation)
    max_short: int = 1088

    def __post_init__(self) -> None:
        if (self.max_frames - 1) % TIME_STRIDE or self.max_frames < TIME_STRIDE + 1:
            raise ValueError(f"max_frames must be 8n+1, got {self.max_frames}")
        if not 0 <= self.overlap < self.max_frames // 2:
            raise ValueError(f"overlap must be below half a window, got {self.overlap}")


# --------------------------------------------------------------------------- geometry


def processing_size(width: int, height: int, recipe: Recipe, side: int | None = None) -> tuple[int, int]:
    """(w, h) the frames are computed at before padding: scaled down (never up) so the long side is at most
    `side` (default recipe.max_side) and the short side at most recipe.max_short, aspect kept."""
    long_cap = min(side or recipe.max_side, recipe.max_side)
    short_cap = min(recipe.max_short, long_cap)
    long_, short = max(width, height), min(width, height)
    scale = min(1.0, long_cap / long_, short_cap / short)
    return max(1, round(width * scale)), max(1, round(height * scale))


def padded(n: int) -> int:
    return -(-n // MULTIPLE) * MULTIPLE


def frames_8n1(n: int) -> int:
    """The smallest 8k+1 that holds n frames."""
    return -(-(n - 1) // TIME_STRIDE) * TIME_STRIDE + 1


def window_frames(height: int, width: int, recipe: Recipe) -> int:
    """Frames per window (8n+1) at a padded size: recipe.max_frames, or fewer so that the latent tokens stay within
    MAX_WINDOW_TOKENS."""
    per_latent_frame = (height // MULTIPLE) * (width // MULTIPLE)
    latent_frames = max(1, MAX_WINDOW_TOKENS // per_latent_frame)
    return min(recipe.max_frames, (latent_frames - 1) * TIME_STRIDE + 1)


def windows(n: int, recipe: Recipe, size: int | None = None) -> list[tuple[int, int]]:
    """[start, end) frame ranges of at most `size` (default recipe.max_frames) frames, 8n+1 each where the shot is
    long enough; consecutive ones share at least recipe.overlap frames (at most half a window). The fewest windows
    that cover the shot, each as short as that allows (less work in the overlaps)."""
    import math

    size = size or recipe.max_frames
    overlap = min(recipe.overlap, (size - 1) // 2)
    if n <= size:
        return [(0, n)]
    count = math.ceil((n - overlap) / (size - overlap))
    size = min(size, frames_8n1(math.ceil((n + (count - 1) * overlap) / count)))
    step = (n - size) / (count - 1)
    return [(round(k * step), round(k * step) + size) for k in range(count)]


# --------------------------------------------------------------------------- the model


def _resident_stage_class():
    """Upstream's DiffusionStage builds the transformer on every call and frees it afterwards; this one builds it on
    the first call and keeps it."""
    from ltx_pipelines.utils.blocks import DiffusionStage

    class ResidentStage(DiffusionStage):
        """Also parks the transformer in RAM while the VAE decoder runs, when the decoder would otherwise tile the
        video differently for lack of memory (park / unpark): the weights never change, so parking only points every
        tensor at its RAM copy (made once) and unparking uploads it again (~20 GB, a few seconds). When the card holds
        both (a small window, a large card) nothing moves. The RAM copy is ordinary (pageable) memory: under WSL2, pinned
        memory uploads only a little faster but takes seconds per GB to allocate (minutes for the transformer) and
        locks that RAM, so it is not pinned."""

        _held = None
        _ram: dict | None = None
        _parked = False

        def _transformer_ctx(self, **kwargs):
            if self._held is None:
                self._held = self._build_transformer(**kwargs)
            self.unpark()
            return nullcontext(self._held)

        def _tensors(self):
            return [*self._held.parameters(), *self._held.buffers()]

        def held_bytes(self) -> int:
            """What parking would free on the card (0 when nothing is on it)."""
            if self._held is None or self._parked:
                return 0
            return sum(t.numel() * t.element_size() for t in self._tensors() if t.is_cuda)

        def park(self) -> None:
            if self._held is None or self._parked:
                return
            if self._ram is None:
                self._ram = {id(t): t.data.to("cpu") for t in self._tensors()}
            for t in self._tensors():
                t.data = self._ram[id(t)]
            self._parked = True
            torch.cuda.empty_cache()

        def unpark(self) -> None:
            if self._held is None or not self._parked:
                return
            for t in self._tensors():
                t.data = self._ram[id(t)].to(self._device, non_blocking=True)
            torch.cuda.synchronize()
            self._parked = False

    return ResidentStage


@dataclass
class LtxModel:
    base: LtxBase
    loras: tuple[Lora, ...]
    device: torch.device
    dtype: torch.dtype = torch.bfloat16
    quantization: str = "fp8-cast"  # bf16 weights downcast to fp8 at load (upstream --quantization fp8-cast)
    _contexts: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_pipelines.iclora_utils import (read_lora_reference_downscale_factor,
                                                read_lora_reference_temporal_scale_factor)
        from ltx_pipelines.utils.blocks import ImageConditioner, VideoDecoder

        loras = tuple(LoraPathStrengthAndSDOps(str(l.path), l.strength, LTXV_LORA_COMFY_RENAMING_MAP) for l in self.loras)
        for lora in loras:  # every LoRA used here conditions on a full-size, full-rate reference
            if read_lora_reference_downscale_factor(lora.path) != 1 or read_lora_reference_temporal_scale_factor(lora.path) != 1:
                raise ValueError(f"{lora.path}: a scaled reference (downscale / temporal factor) is not supported")
        policy = None
        if self.quantization == "fp8-cast":
            from ltx_core.quantization.fp8_cast import build_policy

            policy = build_policy(str(self.base.transformer))
        self.stage = _resident_stage_class().from_checkpoint(
            str(self.base.transformer), self.dtype, self.device, loras=loras, quantization=policy)
        self.encoder = ImageConditioner(str(self.base.video_vae), self.dtype, self.device)._build_encoder()
        self._decoder_block = VideoDecoder(str(self.base.video_vae), self.dtype, self.device)
        self.decoder = self._decoder_block._prepared_builder().build(device=self.device, dtype=self.dtype).eval()

    def warm(self) -> None:
        """Build the transformer now (the first window would otherwise pay for it)."""
        self.stage._transformer_ctx()

    # ------------------------------------------------------------------ the fixed prompt

    def context(self, recipe: Recipe, cache_dir: Path) -> torch.Tensor:
        """The video text context of recipe.prompt. Encoded once with the Gemma text encoder (streamed through the
        GPU a block at a time, so it never needs its 26 GB at once) and kept on disk: later runs never load Gemma."""
        if recipe.prompt in self._contexts:
            return self._contexts[recipe.prompt]
        key = hashlib.sha256("\0".join([recipe.prompt, *(f"{p.name}:{p.stat().st_size}" for p in
                                                           (self.base.transformer, self.base.text_encoder))])
                             .encode()).hexdigest()[:24]
        path = cache_dir / f"prompt-{key}.pt"
        if path.is_file():
            ctx = torch.load(path, map_location="cpu", weights_only=True)
        else:
            if self.stage._held is not None:  # Gemma and the resident transformer would not share the card
                raise RuntimeError("a new prompt must be encoded before the transformer is built (context before warm)")
            from ltx_pipelines.utils.blocks import PromptEncoder
            from ltx_pipelines.utils.model_paths import ModelPaths
            from ltx_pipelines.utils.types import OffloadMode

            paths = ModelPaths.from_split(transformer_path=str(self.base.transformer),
                                          text_encoder_path=str(self.base.text_encoder))
            encoder = PromptEncoder(paths, self.dtype, self.device, offload_mode=OffloadMode.CPU)
            (out,) = encoder([recipe.prompt])
            ctx = out.video_encoding.detach().to("cpu")
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            torch.save(ctx, tmp)
            tmp.replace(path)
        self._contexts[recipe.prompt] = ctx
        return ctx

    # ------------------------------------------------------------------ one window

    @torch.inference_mode()
    def window(self, frames: torch.Tensor, context: torch.Tensor, recipe: Recipe) -> torch.Tensor:
        """frames uint8 [T, H, W, 3] on the GPU, T = 8n+1, H and W multiples of 32 -> float32 [T, H, W, 3] 0..1."""
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import AUTO_TILING
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.iclora_utils import append_ic_lora_reference_video_conditionings
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_core.devices import activation_budget_bytes
        from ltx_pipelines.utils.helpers import (create_initial_video_latent, ensure_tiling_config,
                                                 tiling_scale_factors_for_vae)
        from ltx_pipelines.utils.types import ModalitySpec, VideoAudio

        t, h, w = frames.shape[:3]
        if (t - 1) % TIME_STRIDE or h % MULTIPLE or w % MULTIPLE:
            raise ValueError(f"window {t}x{h}x{w}: frames must be 8n+1 and sides multiples of {MULTIPLE}")
        generator = torch.Generator(device=self.device).manual_seed(recipe.seed)
        v_context = context.to(self.device, self.dtype)

        # the VAE encode runs beside the resident transformer (ENCODE_UNTILED_VOLUME was measured that way; larger
        # windows are encoded in tiles)
        conditionings: list = []
        append_ic_lora_reference_video_conditionings(
            conditionings, [("<frames>", 1.0)], height=h, width=w, num_frames=t,
            video_encoder=self.encoder, dtype=self.dtype, device=self.device,
            reference_downscale_factor=1, reference_temporal_scale_factor=1,
            conditioning_attention_strength=1.0, conditioning_attention_mask=None,
            tiling_config=encode_tiling(t, h, w),
            frames=(frames[i:i + 1] for i in range(t)),
        )
        torch.cuda.empty_cache()
        latent = create_initial_video_latent(width=w, height=h, frames=t, fps=recipe.fps, device=self.device,
                                             dtype=self.dtype, scale_factors=self.stage.video_scale_factors)
        sigmas = torch.tensor(recipe.sigmas, dtype=torch.float32, device=self.device)
        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(v_context, None),
            sigmas=sigmas,
            noiser=GaussianNoiser(generator=generator),
            modalities=VideoAudio(video=ModalitySpec(latent=latent, conditioning_fps=recipe.fps, context=v_context,
                                                     conditionings=conditionings, noise_scale=float(sigmas[0]))),
        )
        del conditionings, latent
        # The decoder sizes its tiles by the memory free: it gets the tiling it would get with the whole card (as
        # upstream, which frees the transformer before decoding). The transformer is parked only when the memory
        # free beside it would give a different tiling.
        vae = str(self.base.video_vae)

        def tiling_for(free_bytes=None):
            return ensure_tiling_config(
                AUTO_TILING, scale_factors=tiling_scale_factors_for_vae(vae), vae_checkpoint_path=vae,
                video_shape=VideoPixelShape(batch=1, frames=t, height=h, width=w, fps=recipe.fps),
                diffvae_optimization=self._decoder_block.diffvae_optimization, device=self.device,
                free_bytes=free_bytes)

        torch.cuda.empty_cache()
        beside = tiling_for()
        if beside != tiling_for(activation_budget_bytes(self.device) + self.stage.held_bytes()):
            self.stage.park()
            beside = tiling_for()
        tiling = beside
        out = torch.cat([c.float() for c in self.decoder.decode_video(
            video_state.latent.to(self.dtype), tiling, generator)], dim=0)
        if tuple(out.shape) != (t, h, w, 3):
            raise ShapeMismatch((t, h, w), tuple(out.shape))
        return out.clamp_(0, 1)


def encode_tiling(t: int, h: int, w: int):
    """None (one pass) for a small window, else upstream's conv-VAE tile layout (ENCODE_UNTILED_VOLUME)."""
    if t * h * w <= ENCODE_UNTILED_VOLUME:
        return None
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig
    from ltx_core.types import VIDEO_SCALE_FACTORS

    return TileSizeConfig.from_long_side(long_side=DimensionSizeConfig(tile_size=768, overlap=64), height=h, width=w,
                                         scale_factors=VIDEO_SCALE_FACTORS,
                                         frames=DimensionSizeConfig(tile_size=80, overlap=24))


def load(base: LtxBase, loras: tuple[Lora, ...], device: torch.device, quantization: str = "fp8-cast") -> LtxModel:
    """The base with `loras` fused in. Wrap it in the worker SDK's @resident. The transformer is built on first use:
    call context() before it, so that encoding a prompt the first time (Gemma, streamed through the GPU) does not have
    to share the card with the resident transformer."""
    return LtxModel(base, tuple(loras), device, quantization=quantization)


# --------------------------------------------------------------------------- a whole shot


def run_shot(model: LtxModel, recipe: Recipe, context: torch.Tensor, read: Callable[[int], np.ndarray], n: int,
             size: tuple[int, int], *, side: int | None = None,
             progress: Callable[[int, int], None] | None = None) -> Iterator[tuple[int, torch.Tensor]]:
    """Every frame of a shot of `n` frames, once each and in order: (index, float32 [H, W, 3] 0..1 on the GPU at the
    processing size, padding cropped). `read(i)`: input frame i, uint8 [H, W, 3] of `size` (h, w). The caller
    scales the result back to the input size (processing_size tells whether it was scaled)."""
    height, width = size
    pw, ph = processing_size(width, height, recipe, side)
    H, W = padded(ph), padded(pw)
    spans = windows(n, recipe, window_frames(H, W, recipe))
    tail: dict[int, torch.Tensor] = {}
    for k, (start, end) in enumerate(spans):
        clip = []
        for i in range(start, end):
            x = torch.from_numpy(np.ascontiguousarray(read(i))).to(model.device).permute(2, 0, 1)[None].float()
            if (ph, pw) != (height, width):
                x = F.interpolate(x, size=(ph, pw), mode="bilinear", align_corners=False, antialias=True)
            if (H, W) != (ph, pw):
                x = F.pad(x, (0, W - pw, 0, H - ph), mode="reflect" if H - ph < ph and W - pw < pw else "replicate")
            clip.append(x.round_().clamp_(0, 255).to(torch.uint8)[0].permute(1, 2, 0))
        count = len(clip)
        clip += clip[-1:] * (frames_8n1(count) - count)  # 8n+1: repeat the last frame; those outputs are dropped
        # clone: a plain tensor (window() runs in inference mode), so the cross-fade and the caller may change it
        out = model.window(torch.stack(clip), context, recipe)[:count, :ph, :pw].clone()
        if tuple(out.shape[:3]) != (count, ph, pw):
            raise ShapeMismatch((count, ph, pw), tuple(out.shape))
        next_start = spans[k + 1][0] if k + 1 < len(spans) else end
        shared = [i for i in range(start, end) if i in tail]
        for j, i in enumerate(shared):  # linear cross-fade from the previous window into this one
            a = (j + 1) / (len(shared) + 1)
            out[i - start] = (1 - a) * tail.pop(i) + a * out[i - start]
        for i in range(start, min(end, next_start)):
            yield i, out[i - start]
        tail = {i: out[i - start].clone() for i in range(next_start, end)}
        if progress:
            progress(min(next_start, n), n)


# --------------------------------------------------------------------------- a feature's whole job


@dataclass(frozen=True)
class Feature:
    """One IC-LoRA feature, a constant in its worker: which node it runs for, the model's name in its progress words,
    its LoRA (relative to its own weights folder), its fixed recipe, the stage id of the pass over the shot, and the
    failure code when a frame count or size comes out other than it went in."""

    node: str
    title: str
    lora: str
    recipe: Recipe
    stage: str
    shape_error: str
    strength: float = 1.0


class FrameOutput:
    """What a feature writes for one attempt at the shot (run_feature opens one per attempt: Run.fit may run the shot
    again at a smaller processing size). put(index, rgb): input frame `index`'s result, float32 [H, W, 3] 0..1 on the
    GPU at the input size, clamped; close(): the attempt ended (failed or not; idempotent); finish(run, numbers,
    **facts): result.json, with the facts run_feature gives and the feature's own."""

    def put(self, index: int, rgb: torch.Tensor) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def finish(self, run, numbers: list[int], **facts) -> None:
        raise NotImplementedError


@resident
def load_feature(base_weights: Path, lora: Path, strength: float, device: torch.device) -> LtxModel:
    """The base with one feature's LoRA fused in, kept loaded between jobs."""
    return load(LtxBase.under(base_weights), (Lora(lora, strength),), device)


def run_feature(job_path: str, feature: Feature, bound, open_output: Callable[[object, list[int]], FrameOutput]) -> None:
    """A feature's whole job (module doc). `bound`: the processing size parameter as a lab2shot_worker MemoryBound
    (the node's choices, largest first); `open_output(job, numbers)`: a new FrameOutput for one attempt."""
    import os

    from lab2shot_worker import fail, progress
    from lab2shot_worker.matte import frame_reader
    from lab2shot_worker.run import Run

    run = Run.start(job_path, feature.node, feature.title)
    job, recipe = run.job, feature.recipe
    base_weights = Path(os.environ[BASE_ENV])
    base, lora = LtxBase.under(base_weights), job.weights_dir / feature.lora
    run.weights(*base.files, lora)

    frames = run.frames()
    numbers, height, width = frames.numbers, frames.height, frames.width
    n = len(frames)

    device = torch.device("cuda")
    model = run.model("load_model", load_feature, base_weights, lora, feature.strength, device,
                      stage_params={"model": feature.title})
    run.stage("encode_prompt")
    context = model.context(recipe, job.weights_dir.parent / "cache" / "ltx")
    with run.loading("load_model", model="LTX-2.5 22B"):
        model.warm()  # the transformer, after the prompt (they would not share the card)

    run.stage(feature.stage)
    reader = frame_reader(frames.paths, (height, width))

    def wrong_shape(frames_: int, h: int, w: int, got: tuple) -> None:
        fail(feature.shape_error, frames=frames_, width=w, height=h, got_frames=int(got[0]),
             got_width=int(got[2]) if len(got) > 2 else 0, got_height=int(got[1]) if len(got) > 1 else 0)

    def one_shot(side: int):
        out = open_output(job, numbers)
        written = 0
        try:
            order = list(range(n))
            for i, rgb in run_shot(model, recipe, context, lambda i: reader.get(i, order[i:]), n, (height, width),
                                   side=side, progress=lambda done, total: progress(done, total, feature.stage)):
                if tuple(rgb.shape[:2]) != (height, width):
                    rgb = F.interpolate(rgb.permute(2, 0, 1)[None], size=(height, width), mode="bilinear",
                                        align_corners=False)[0].permute(1, 2, 0)
                if i != written or tuple(rgb.shape) != (height, width, 3):
                    wrong_shape(n, height, width, (written, *rgb.shape))
                out.put(i, rgb.clamp_(0, 1))
                written += 1
        except ShapeMismatch as e:
            out.close()
            wrong_shape(e.expected[0], e.expected[1], e.expected[2], e.got)
        except BaseException:
            out.close()
            raise
        if written != n:
            out.close()
            wrong_shape(n, height, width, (written, height, width))
        return side, out

    try:
        resolution, out = run.fit(bound, one_shot, run.params[bound.param])
    finally:
        reader.close()

    pw, ph = processing_size(width, height, recipe, resolution)
    out.finish(run, numbers, width=width, height=height, processing_size={"width": pw, "height": ph},
               window={"frames": window_frames(padded(ph), padded(pw), recipe), "overlap": recipe.overlap,
                       "blend": "linear cross-fade"},
               params={bound.param: resolution}, seed=recipe.seed)
