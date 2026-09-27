"""StableMotion worker: the rig-and-model contract (lab2shot_worker.rig_motion) with StableMotion-BrokenAMASS. Runs
inside third_party/stablemotion/.venv with the pinned repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>        (node "stablemotion.cleanup")

The rig comes in as joint-to-world transforms on its own frames (cm, Y up, the shot's frame rate). This worker:

  1. retargets it onto SMPL's 24-joint skeleton (the contract's Retarget, rest joints from upstream's own
     `smpl_neutral_nobetas_24J.npz`: the model knows only the neutral body, so a shaped one would be wrong anyway)
     and resamples to the model's 20 fps;
  2. turns that into SMPL parameters (lab2shot_shared.smpl to_params / from_params, the project's single conversion)
     and into AMASS's Z-up world, then into upstream's own 232-feature representation
     (`smpldata_to_alignglobsmplrifkefeats`: the features are lossless, max round-trip error 0.5 µm);
  3. detects over the whole clip in one pass: the model's 233rd channel is a per-frame "this frame is broken"
     label, and detection holds up at any length (a clean 2400-frame clip flags 8 frames);
  4. fixes in 100-frame windows, the length it was trained on: repair quality falls off beyond it (a 94 cm root
     teleport comes back to 1.5 cm at 100–200 frames, 6.6 cm at 400, 33 cm at 800). Only windows that hold a flagged
     frame are run at all, and each is canonicalised and un-canonicalised on its own (the representation puts every
     window's first frame at the origin facing +X, so a window stitched back without undoing that teleports the
     character).

`labels` go back through the contract beside the motion: the node shows them as 「问题帧」 and, by default, keeps the
frames the model called good exactly as the artist brought them in.

Parameters: quality ("basic" / "best": upstream's plain path, or its ensemble selection), seed.
"""

from __future__ import annotations

import os
import sys
import types
import warnings
from contextlib import contextmanager
from pathlib import Path

from functools import partial

import numpy as np
import torch

from lab2shot_worker import fail, progress, resident, say, serve, set_seed
from lab2shot_worker import rig_motion as rm
from lab2shot_worker.run import Run
from lab2shot_shared import motion as mo
from lab2shot_shared import smpl as S

NODE = "stablemotion.cleanup"
EXT = "stablemotion"
BODY = "smpl"
MODEL_FPS = 20.0  # BrokenAMASS is resampled to 20 fps; the model knows no other rate
WINDOW = 100  # frames it was trained on
STRIDE = 50
DETECT_CHUNK = 1000  # frames the detection pass looks at in one go: it holds up at any length (a clean 2400-frame
# clip flags 8 frames), but attention costs the square of it, so a ten-minute take is read in pieces
LEAST_FRAMES = 8  # below this a window is mostly padding and the label means nothing
FEATURES = 232  # the motion features; the 233rd channel is the quality label
REST_JOINTS = "data_loaders/amasstools/smpl_neutral_nobetas_24J.npz"

# AMASS puts gravity on Z; Lab2Shot (and SMPL's own rest pose) put it on Y.
Y_TO_Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
Z_TO_Y = Y_TO_Z.T


@contextmanager
def in_repo(repo: Path):
    """Upstream reads its own npz files by relative path."""
    old = os.getcwd()
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(old)


def _prepare(repo: Path) -> None:
    """The pinned repo on sys.path, and the one import it makes that inference never uses stubbed out: `sample.utils` pulls
    in `eval.eval_motion` for the paper's metrics, which pulls in TMR (a text-to-motion retrieval model with its own
    weights). Only the evaluation needs it; inference never calls a line of it."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    for name in ("tmr_utils", "tmr_utils.guofeats", "tmr_utils.model", "tmr_utils.model.tmr",
                 "tmr_utils.load_tmr_model"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["tmr_utils.guofeats"].joints_to_guofeats = lambda *a, **k: None
    sys.modules["tmr_utils.model.tmr"].get_score_matrix = lambda *a, **k: None
    sys.modules["tmr_utils.load_tmr_model"].load_tmr_model_easy = lambda *a, **k: None
    warnings.filterwarnings("ignore", category=FutureWarning)  # diffusion/nn.py's torch.cuda.amp calls


class Args:
    """What upstream's model builder and samplers read; the values are the released checkpoint's own args.json."""

    layers = 8
    heads = 8
    zero_init = False
    diffusion_steps = 50
    noise_schedule = "cosine"
    sigma_small = True
    predict_xstart = 1
    ts_respace = ""  # ancestral sampling, upstream's default
    skip_timesteps = 0
    ProbDetTh = 0.5
    ProbDetNum = 0
    enable_sits = False
    ensemble = False
    classifier_scale = 0.0
    seed = 10


# per-channel training mean and standard deviation, included in the upstream repository (checked into git)
STATS_DIR = "dataset/meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano"


class Normalizer:
    """upstream's utils.normalizer.Normalizer, reading `mean.pt` / `std.pt` (232 numbers each) from the upstream
    repository and appending 0.5 / 0.5 for the label channel, the values upstream's `add_label_channel` writes."""

    def __init__(self, repo: Path, eps: float = 1e-12) -> None:
        stats = Path(repo) / STATS_DIR

        def read(name: str) -> torch.Tensor:
            path = stats / name
            if not path.is_file():
                fail("E-STABLEMOTION-NOSTATS", path=str(path))
            values = torch.load(path, map_location="cpu", weights_only=True).flatten().to(torch.float32)
            if len(values) != FEATURES:
                raise ValueError(f"{name}: {len(values)} numbers, expected {FEATURES}")
            # label channel (the 233rd): upstream add_label_channel writes 0.5 / 0.5
            return torch.cat([values, torch.tensor([0.5], dtype=torch.float32)])

        self.mean, self.std, self.eps = read("mean.pt"), read("std.pt"), eps

    def to(self, device):
        self.mean, self.std = self.mean.to(device), self.std.to(device)
        return self

    def __call__(self, x):
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

    def inverse(self, x):
        return x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)


@resident
def load_model(repo: str, weights: str, device: torch.device):
    """The DiT and its diffusion, with the released EMA weights.

    Upstream loads them through `ema_pytorch.EMA(model, include_online_model=False)`, which only puts the weights
    under an `ema_model.` prefix; stripping it loads all 88 tensors into the plain model with nothing missing, so
    that dependency is not installed. The five unexpected keys are dead weights of an older revision of the
    architecture, which upstream's own `strict=False` also ignores."""
    from utils.model_util import create_model_and_diffusion

    with in_repo(Path(repo)):
        model, diffusion = create_model_and_diffusion(Args())
    state = torch.load(Path(weights) / "ema001000000.pt", map_location="cpu")
    online = {k[len("ema_model."):]: v for k, v in state.items() if k.startswith("ema_model.")}
    info = model.load_state_dict(online, strict=False)
    if info.missing_keys:
        raise RuntimeError(f"StableMotion checkpoint is missing {len(info.missing_keys)} tensors: {info.missing_keys[:4]}")
    return model.to(device).eval(), diffusion


# --------------------------------------------------------------------------- SMPL <-> AMASS's world


def to_amass(pose: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SMPL parameters in Lab2Shot's Y-up world -> AMASS's Z-up one. Only the root's rotation and the translation
    know about the world: every other joint's pose is relative to its parent."""
    out = np.asarray(pose, np.float64).copy()
    out[:, 0] = mo.matrix_to_rotvec(Y_TO_Z @ mo.rotvec_to_matrix(out[:, 0]))
    return out, np.asarray(transl, np.float64) @ Y_TO_Z.T


def from_amass(pose: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(pose, np.float64).copy()
    out[:, 0] = mo.matrix_to_rotvec(Z_TO_Y @ mo.rotvec_to_matrix(out[:, 0]))
    return out, np.asarray(transl, np.float64) @ Z_TO_Y.T


class Features:
    """Upstream's 232-feature representation, and the rigid transform it takes out.

    `smpldata_to_alignglobsmplrifkefeats` canonicalises its input (the ground to z = 0, the first frame's XY to the
    origin, the first frame's heading to +X), so a decoded window comes back in that frame and must be put back
    (`unalign`), otherwise the character teleports. It also returns `trans` as the pelvis position, while SMPL
    puts the pelvis at rest_joints[0] + transl; the two differ by 22.5 cm on the neutral body, without any error if
    the offset is missed."""

    def __init__(self, rest: np.ndarray) -> None:
        from data_loaders.amasstools import globsmplrifke_feats as feats
        from data_loaders.amasstools import geometry as geo

        self.feats, self.geo, self.rest = feats, geo, np.asarray(rest, np.float64)

    def joints(self, pose: np.ndarray, transl: np.ndarray) -> np.ndarray:
        """The 24 joint positions the encoder checks its own inputs against: forward kinematics on the neutral body,
        which is what upstream's data loader does when a motion carries no joints of its own."""
        return S.to_motion(BODY, pose, transl, self.rest)[..., :3, 3]

    def smpldata(self, pose: np.ndarray, transl: np.ndarray) -> dict:
        return {"poses": torch.as_tensor(pose.reshape(len(pose), -1)[:, :66], dtype=torch.float32),
                "trans": torch.as_tensor(transl, dtype=torch.float32),
                "joints": torch.as_tensor(self.joints(pose, transl), dtype=torch.float32)}

    def encode(self, pose: np.ndarray, transl: np.ndarray) -> torch.Tensor:
        return self.feats.smpldata_to_alignglobsmplrifkefeats(self.smpldata(pose, transl)).to(torch.float32)

    def place(self, pose: np.ndarray, transl: np.ndarray) -> tuple[float, np.ndarray, float]:
        """What `encode` takes out of this motion: (ground height, first frame's XY, first frame's heading)."""
        joints = self.joints(pose, transl)
        heading = self.geo.matrix_to_euler_angles(
            torch.as_tensor(mo.rotvec_to_matrix(pose[:1, 0]), dtype=torch.float32), "ZYX")[0, 0].item()
        return float(joints[:, :, 2].min()), joints[0, 0, :2].copy(), heading

    def decode(self, values: torch.Tensor, place: tuple[float, np.ndarray, float]) -> tuple[np.ndarray, np.ndarray]:
        """Features -> (pose [F,24,3], transl [F,3]) back where `place` says this window came from."""
        out = self.feats.globsmplrifkefeats_to_smpldata(torch.as_tensor(values, dtype=torch.float32))
        pose = np.asarray(out["poses"], np.float64).reshape(-1, 22, 3)
        pose = np.concatenate([pose, np.zeros((len(pose), S.body(BODY).joints - 22, 3))], axis=1)
        ground, xy, heading = place
        turn = mo.rotvec_to_matrix(np.array([0.0, 0.0, heading]))
        pelvis = np.asarray(out["trans"], np.float64) @ turn.T + np.array([xy[0], xy[1], ground])
        pose[:, 0] = mo.matrix_to_rotvec(turn @ mo.rotvec_to_matrix(pose[:, 0]))
        return pose, pelvis - self.rest[0]  # the pelvis position is not SMPL's transl


# --------------------------------------------------------------------------- the model


def normalized(feats: torch.Tensor, norm: Normalizer, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[F,232] -> what the model reads: ([1,233,N] on the device, the frames that are real [1,N], the length [1]).

    A clip shorter than the trained window is padded to it with zeros and masked off, which is exactly how
    upstream's own data loader hands the model a short item (data_loaders/smpl_collate.py: zeros in normalized space
    up to the window, length_to_mask beside it). Without the padding a 42-frame take comes back with every
    frame called broken; with it, the same take is called clean."""
    with_label = torch.cat([feats, torch.zeros(len(feats), 1)], dim=-1)
    x = norm(with_label).T[None].contiguous()
    frames = x.shape[2]
    if frames < WINDOW:
        x = torch.cat([x, torch.zeros(1, x.shape[1], WINDOW - frames)], dim=2)
    attention = torch.zeros(1, x.shape[2], dtype=torch.bool)
    attention[:, :frames] = True
    return x.to(device), attention.to(device), torch.tensor([frames], device=device)


def detect_all(model, diffusion, norm: Normalizer, features, whole: torch.Tensor, device) -> np.ndarray:
    """The whole take's per-frame score (0..1, the model's label channel), in pieces of at most DETECT_CHUNK frames that
    overlap by a window (each frame is judged by the piece whose middle it is nearest, so no frame is judged by a
    piece's unreliable last frames). The cut (the node's 「检测阈值」) is made by the caller, and the score itself goes
    back as the 「问题帧」 curve, so the artist's threshold on the node and the worker's agree by construction."""
    total = len(whole)
    if total <= DETECT_CHUNK:
        return detect(model, diffusion, norm, normalized(whole, norm, device))
    out = np.zeros(total, np.float32)
    step = DETECT_CHUNK - WINDOW
    starts = list(range(0, total - WINDOW, step))
    for n, start in enumerate(starts):
        end = min(start + DETECT_CHUNK, total)
        got = detect(model, diffusion, norm, normalized(whole[start:end], norm, device))
        first = start if n == 0 else start + WINDOW // 2
        last = end if end == total else end - WINDOW // 2
        out[first:last] = got[first - start:last - start]
        progress(n + 1, len(starts), "找问题帧")
    return out


@torch.no_grad()
def detect(model, diffusion, norm: Normalizer, sent) -> np.ndarray:
    """Upstream's detection pass: everything but the label channel is kept, the label channel is predicted; its value
    per frame (upstream cuts it at ProbDetTh; here the caller does)."""
    x, attention, length = sent
    keep = torch.ones_like(x).bool()
    keep[:, -1] = False
    painted = x.clone()
    painted[:, -1] = 1.0
    kwargs = {"y": {"inpainting_mask": keep, "inpainted_motion": painted},
              "inpaint_cond": (~keep) & attention.unsqueeze(-2),
              "length": length, "attention_mask": attention}
    sample = diffusion.p_sample_loop(model, tuple(x.shape), clip_denoised=False, model_kwargs=kwargs,
                                     skip_timesteps=0, init_image=None, progress=False, dump_steps=None,
                                     noise=None, const_noise=False)
    got = norm.inverse(sample.transpose(1, 2).cpu())[0, :, -1].numpy().astype(np.float32)
    return got[:int(length[0])]


@torch.no_grad()
def repaint(model, diffusion, norm: Normalizer, args: Args, sent, bad: np.ndarray) -> torch.Tensor:
    """Upstream's fix pass: the flagged frames (widened by one on each side, the last frame always kept) are
    repainted, everything else is held to what came in."""
    x, attention, length = sent
    frames = x.shape[2]
    label = torch.zeros(1, frames, dtype=torch.bool, device=x.device)
    label[0, :len(bad)] = torch.as_tensor(bad, dtype=torch.bool, device=x.device)
    grown = label.clone()
    grown[:, 1:] |= label[:, :-1]
    grown[:, :-1] |= label[:, 1:]
    grown[0, int(length[0]) - 1] = False  # the last real frame is always kept, as upstream does
    keep = torch.zeros_like(x).bool()
    keep[..., (~grown[0]).nonzero().flatten()] = True
    keep[:, -1] = True  # the label channel is never repainted
    painted = x.clone()
    painted[:, -1] = -1.0  # ... it is pinned to "this frame is fine"
    kwargs = {"y": {"inpainting_mask": keep.clone(), "inpainted_motion": painted.clone()},
              "inpaint_cond": ((~keep) & attention.unsqueeze(-2)).clone(),
              "length": length, "attention_mask": attention}
    if args.ensemble:
        from sample.utils import prepare_cond_fn, run_cleanup_selection

        # The README's "Enhanced inference" sets `--ensemble --enable_sits --classifier_scale 100` together
        # (`third_party/stablemotion/repo/README.md:150-158`); all three are set here to match upstream behaviour:
        #  · `--enable_sits` has no effect in that command: `fix_globsmpl.py:181` requires
        #    `args.enable_sits and args.ProbDetNum`, and ProbDetNum defaults to 0 and is not given in the README,
        #    so soft_inpaint_ts is always None;
        #  · `--classifier_scale 100` is the effective one: it adds foot-locking classifier guidance to sampling
        #    (`sample/utils.py prepare_cond_fn` -> `footlocking_fn`), an upstream function called directly.
        cond_fn = prepare_cond_fn(args, norm, x.device)
        if cond_fn is not None:
            cond_fn = partial(cond_fn, model=model)  # upstream binds the model only once it is ready (see the prepare_cond_fn comment)
        det_keep = torch.ones_like(x).bool()
        det_keep[:, -1] = False
        det_in = x.clone()
        det_in[:, -1] = 1.0
        sample = run_cleanup_selection(
            model=model,
            model_kwargs_detmode={"y": {"inpainting_mask": det_keep, "inpainted_motion": det_in},
                                  "inpaint_cond": (~det_keep) & attention.unsqueeze(-2),
                                  "length": length, "attention_mask": attention},
            model_kwargs=kwargs, motion_normalizer=norm, args=args, bs=1, nfeats=x.shape[1], nframes=frames,
            sample_fn=diffusion.p_sample_loop, cond_fn=cond_fn)
    else:
        sample = diffusion.p_sample_loop(model, tuple(x.shape), clip_denoised=False, model_kwargs=kwargs,
                                         skip_timesteps=0, init_image=kwargs["y"]["inpainted_motion"], progress=False,
                                         dump_steps=None, noise=None, const_noise=False)
    return norm.inverse(sample.transpose(1, 2).cpu())[0, :int(length[0]), :-1]


def windows(bad: np.ndarray, total: int) -> list[tuple[int, int]]:
    """The windows the fix pass runs: one per stretch of flagged frames, WINDOW frames long, the stretch a quarter
    of the way in (the model's answer is least reliable in a window's last frames). A take with three bad frames in
    one place costs one window, not the whole shot."""
    out: list[tuple[int, int]] = []
    todo = [int(f) for f in np.flatnonzero(bad)]
    while todo:
        first = todo[0]
        start = min(max(first - WINDOW // 4, 0), max(total - WINDOW, 0))
        end = min(start + WINDOW, total)
        out.append((start, end))
        rest = [f for f in todo if f >= end - 2]  # the tail frames go to the next window
        if rest and rest[0] <= first:  # nothing left to move forward to (the clip's own end)
            break
        todo = rest
    return out


def cleanup(run: Run) -> None:
    job = run.job
    repo, weights = job.repo_dir, job.weights_dir
    _prepare(repo)
    run.weights(weights / "ema001000000.pt",  # the normaliser is the upstream repo's own mean.pt / std.pt (Normalizer)
                what="StableMotion 权重", page="https://github.com/Murrol/StableMotion")
    p = job.params
    device = torch.device("cuda")
    motion = rm.read_job(job.inputs["motion"])

    with in_repo(repo):
        rest = np.load(REST_JOINTS)["J"].astype(np.float64)  # the neutral, no-betas body, metres, Y up
    run.stage("对齐骨骼")
    try:
        retarget = motion.retarget(S.skeleton(BODY, rest))
    except ValueError as exc:
        fail("E-STABLEMOTION-SKELETON", reason=str(exc))

    at = motion.model_times(MODEL_FPS)  # the shot's frames on the model's 20 fps timeline
    total = int(np.floor(at[-1])) + 1
    if total < LEAST_FRAMES:
        fail("E-STABLEMOTION-SHORT", frames=total, least=LEAST_FRAMES)
    if total < WINDOW:  # a 2-second take gets 70–100 % of its frames called broken, a 5-second one 2–3 %
        say("W-STABLEMOTION-SHORTTAKE", seconds=total / MODEL_FPS, frames=total, window=WINDOW, port="character")
    rotations, root = retarget.to_model(motion.poses)
    t = np.arange(total, dtype=np.float64)
    world = mo.resample_rotations(at, rotations, t)
    root_t = mo.resample_values(at, root, t)
    pose, transl = to_amass(*S.to_params(BODY, world, root_t, rest))
    features = Features(rest)

    model, diffusion = run.model("StableMotion 模型", load_model, str(repo), str(weights), device)
    norm = Normalizer(repo)  # normalisation statistics from the upstream repository's mean.pt / std.pt
    args = Args()
    args.seed, args.ensemble = int(p["seed"]), p["quality"] == "best"
    args.ProbDetTh = float(p["threshold"])  # the node's 「检测阈值」 (rig_motion.py cleanup_job hands it over): what gets repainted
    if args.ensemble:  # the 「增强」 setting = the three flags of upstream's Enhanced inference command
        args.enable_sits = True
        args.classifier_scale = 100.0
    set_seed(args.seed)

    run.stage("找出有问题的帧")
    with in_repo(repo):
        whole = features.encode(pose, transl)
        score = detect_all(model, diffusion, norm, features, whole, device)  # per frame, 0..1
    bad = score > args.ProbDetTh
    say("I-STABLEMOTION-FOUND", bad=int(bad.sum()), total=total)
    if p.get("repaint_all"):  # 「只改问题帧」 off: the whole clip is repainted by the model, including frames judged good
        bad = np.ones_like(bad)

    parts = windows(bad, total)
    if parts:
        run.stage("重画有问题的帧")
    if len(parts) > 1:
        say("N-STABLEMOTION-PARTS", seconds=total / MODEL_FPS, parts=len(parts), overlap=WINDOW - STRIDE)
    fixed_pose, fixed_transl = pose.copy(), transl.copy()
    for n, (start, end) in enumerate(parts):
        with in_repo(repo):
            place = features.place(pose[start:end], transl[start:end])
            sent = normalized(features.encode(pose[start:end], transl[start:end]), norm, device)
            out = repaint(model, diffusion, norm, args, sent, bad[start:end])
            got_pose, got_transl = features.decode(out, place)
        take = np.flatnonzero(bad[start:end])  # only the frames this pass was asked to repaint
        fixed_pose[start + take] = got_pose[take]
        fixed_transl[start + take] = got_transl[take]
        progress(n + 1, len(parts), "重画")

    run.stage("换回人物骨骼")
    rot, pos = S.from_params(BODY, *from_amass(fixed_pose, fixed_transl), rest)
    rm.write_result(run, retarget, MODEL_FPS, at, rot, pos, np.zeros((total, 4), np.float32),
                    labels=score.astype(np.float32), label_names=("问题帧",), method="StableMotion",
                    quality=p["quality"], seed=args.seed, bad_frames=int(bad.sum()), model_frames=total,
                    windows=len(parts))


def main(job_path: str) -> None:
    cleanup(Run.start(job_path, NODE, "StableMotion"))


if __name__ == "__main__":
    serve(main)
