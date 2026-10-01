"""Kimodo worker: the in-betweening contract (lab2shot_worker.rig_motion) with Kimodo's seven released weights on
three skeletons: SOMA 30 joints (rp / seed / rp_v1 / seed_v1), SMPL-X 22 joints (smplx), Unitree G1 34 joints
(g1 / g1_seed). Runs inside third_party/kimodo/.venv with the pinned repo on PYTHONPATH; never imports Lab2Shot core.

No skeleton is hard-coded: joint names, parents and the neutral pose are read from the skeleton of each upstream
weight (`third_party/kimodo/repo/kimodo/skeleton/base.py:72-101`), and upstream also decides which weights return
their result on a different skeleton (see `unpack`; only SOMA does).

    python worker.py <job.json>        (node "kimodo.motion")

Two tasks (job["params"]["task"]):

text      the prompt's embedding with Kimodo's own text encoder (LLM2VecEncoder, the preset load_model builds: LLM2Vec on
          Meta Llama 3 8B Instruct, bfloat16) on the CPU -> raw/text.npy [1,4096]. Kimodo's _generate asks its text
          encoder for exactly this per prompt (one pooled vector); the node keys this job by the prompt alone, so an
          embedding is computed once and reused, like the embedding cache of Kimodo's own demo. The 16 GB model is
          freed as soon as it is used.
generate  with an animation wired: the rig's keys turned onto the model's own skeleton (30 / 22 / 34 joints by weight; its zero pose is a standing T-pose facing +Z), placed
          on Kimodo's 30 fps timeline, and generated as upstream's CLI does (Kimodo.__call__ with post-processing):
          every key a full-body keyframe constraint (joint positions) plus end-effector constraints for the hands and
          feet (their positions and rotations), as Kimodo's own multi-prompt transitions constrain frames. A part
          spans at most 10 s and fewer than 20 constrained frames; a longer shot is generated in parts that share
          their boundary key, each later part also constrained on the previous part's last 5 frames and cross-faded
          over them (Kimodo's num_transition_frames). Each part is canonicalised as Kimodo's CLI starts a motion: the
          first constrained frame at the origin, heading 0 (first_heading_angle 0), and turned back afterwards. With
          no prompt the text features are zero, which is what Kimodo itself does with an empty prompt (and the
          unconditional branch of its guidance): the text encoder is not needed at all.

          Without an animation (no "motion" in the job): an empty constraint list (upstream `kimodo_model.py`
          documents `constraint_lst` as "Pass an empty list for unconstrained generation"), generated from the
          text and the frame count only. Motions longer than 10 s are generated in parts, each later part
          constrained on the previous part's last 5 frames (the same TRANSITION as above). The result is on the
          model's own skeleton (rig_motion.write_model_result), resampled to the node's frame rate and count.

Parameters: task; text: prompt; generate: model (one of seven: "rp" / "seed" / "rp_v1" / "seed_v1" / "smplx" /
"g1" / "g1_seed"), checkpoint (that weight's folder under weights/, from the node: extension.py MODELS is the one
table), prompt ("": none), steps, guidance, seed, post_process.
"""

from __future__ import annotations

import gc
import os

import numpy as np
import torch

from lab2shot_worker import Failure, check_node, fail, load_job, progress, reason, resident, say, serve, set_seed
from lab2shot_worker import rig_motion as ib
from lab2shot_worker.run import Run
from lab2shot_shared import motion as mo
from lab2shot_shared.units import M_TO_CM

NODE = "kimodo.motion"
EXT = "kimodo"
MODEL_FPS = 30  # the model's own frame rate (its motion is at 30 fps)
MAX_FRAMES = 300  # 10 s: the longest motion Kimodo generates at once
TRANSITION = 5  # frames a part takes over from the part before it (Kimodo's num_transition_frames)
MAX_KEYS = 20 - TRANSITION - 1  # fewer than 20 constrained frames per constraint type ("best practices")
# Upstream expects these abstract names, not SOMA joint names: `expand_joint_names` takes the set
# `["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"]` and each skeleton maps it to real joints through
# `left_foot_joint_names` and the like (`third_party/kimodo/repo/kimodo/skeleton/base.py:135-173`).
# This line therefore applies to all three skeletons (upstream maps SMPL-X left_wrist and G1 left_hand_roll_skel
# itself) and must not be changed to concrete joint names.
END_EFFECTORS = ["LeftHand", "RightHand", "LeftFoot", "RightFoot"]
TEXT_GUIDANCE = 2.0  # Kimodo's default text weight of its separated guidance
TEXT_DIM = 4096


class KnownText:
    """Kimodo's text encoder slot: the prompt's embedding the text task computed, or zeros without a prompt (Kimodo
    zeroes an empty prompt's features and masks them out). Called as LLM2VecEncoder is: texts -> ([B,1,4096], [1]*B)."""

    def __init__(self, embedding: np.ndarray | None) -> None:
        self.embedding = None if embedding is None else np.asarray(embedding, np.float32).reshape(1, TEXT_DIM)

    def __call__(self, texts):
        n = 1 if isinstance(texts, str) else len(texts)
        feat = np.zeros((n, 1, TEXT_DIM), np.float32) if self.embedding is None else np.repeat(self.embedding[None], n, 0)
        return torch.from_numpy(feat), [1] * n

    def to(self, *args, **kwargs):
        return self


@resident
def load_kimodo(name: str, device: torch.device):
    """A Kimodo model (config.yaml + model.safetensors below CHECKPOINT_DIR) without its text encoder."""
    from kimodo import load_model

    return load_model(name, device=str(device), text_encoder=KnownText(None))


def encode_text(run: Run) -> None:
    """The text task: Kimodo's LLM2Vec encoder on the CPU, loaded for this prompt and freed again."""
    from kimodo.model.llm2vec import LLM2VecEncoder
    from kimodo.model.load_model import TEXT_ENCODER_PRESETS
    from kimodo.sanitize import sanitize_texts

    job = run.job
    preset = TEXT_ENCODER_PRESETS["llm2vec"]["kwargs"]
    weights = job.weights_dir
    run.weights(*(weights / preset[k] for k in ("base_model_name_or_path", "peft_model_name_or_path")),
                weights / "meta-llama" / "Meta-Llama-3-8B-Instruct", what="文字编码模型",
                page="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
    cwd = os.getcwd()
    os.chdir(weights)  # the adapters name their base model "meta-llama/Meta-Llama-3-8B-Instruct": the folder here
    try:
        encoder = run.model("文字编码模型（Llama 3 8B，CPU）", LLM2VecEncoder, **{**preset, "device": "cpu"})
        run.stage("编码文字描述")
        text = sanitize_texts([job.params["prompt"]])
        feat, _ = encoder(text)
    finally:
        os.chdir(cwd)
    embedding = feat.float().cpu().numpy().reshape(1, TEXT_DIM)
    del encoder, feat
    gc.collect()
    np.save(job.raw_dir / "text.npy", embedding)
    run.finish([], prompt=text[0], encoder="LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised (bfloat16, CPU)")


def model_skeleton(model) -> mo.Skeleton:
    """The weight's own skeleton in its neutral pose (joint-to-world [J,4,4], metres): SOMA 30 joints, SMPL-X 22
    joints or G1 robot 34 joints, all built from the same upstream declaration with no names hard-coded here
    (`bone_order_names` / `joint_parents` / `neutral_joints`,
    `third_party/kimodo/repo/kimodo/skeleton/base.py:72-101`)."""
    skel = model.skeleton
    rest = np.repeat(np.eye(4)[None], skel.nbjoints, 0)
    rest[:, :3, 3] = skel.neutral_joints.detach().cpu().numpy()
    return mo.Skeleton(list(skel.bone_order_names), skel.joint_parents.cpu().numpy(), rest)


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def generate_part(model, rotations: np.ndarray, root: np.ndarray, frames: np.ndarray, length: int, p: dict,
                  seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Kimodo generation of `length` frames constrained at `frames` (their world rotations [n,J,3,3] and root
    positions [n,3]). Returns world rotations [length,J,3,3], root [length,3], contacts [length,4], in the frame the
    constraints came in. An empty `frames` means unconstrained generation (upstream `constraint_lst=[]`)."""
    from kimodo.constraints import EndEffectorConstraintSet, FullBodyConstraintSet
    from kimodo.motion_rep.feature_utils import compute_heading_angle

    skel, device = model.skeleton, model.device
    parents = skel.joint_parents.cpu().numpy()

    def fk(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
        local = torch.as_tensor(mo.local_from_world(rot, parents), dtype=torch.float32, device=device)
        return skel.fk(local, torch.as_tensor(pos, dtype=torch.float32, device=device))[1].cpu().numpy()

    if not len(frames):  # unconstrained: no first constrained frame to canonicalise, and no constraint_lst is passed
        set_seed(seed)
        with torch.inference_mode():
            out = model(p["prompt"].strip(), length, num_denoising_steps=p["steps"], constraint_lst=[],
                        cfg_weight=[TEXT_GUIDANCE, p["guidance"]], cfg_type="separated", post_processing=p["post_process"],
                        num_samples=1, return_numpy=True, first_heading_angle=0.0, progress_bar=lambda x: x)
        return unpack(skel, out, np.eye(3), np.zeros(3))

    # canonical: the first constrained frame at the origin facing heading 0, as a motion starts in Kimodo's CLI
    joints = fk(rotations[:1], root[:1])
    heading = float(compute_heading_angle(torch.as_tensor(joints[None], device=device), skel)[0, 0])
    turn = rot_y(-heading)
    shift = np.array([root[0, 0], 0.0, root[0, 2]])
    rot_c = turn @ rotations
    root_c = (root - shift) @ turn.T
    joints_c = torch.as_tensor(fk(rot_c, root_c), dtype=torch.float32, device=device)
    rots_c = torch.as_tensor(rot_c, dtype=torch.float32, device=device)
    index = torch.as_tensor(frames, dtype=torch.long, device=device)
    constraints = [c.to(device=device) for c in (  # as load_constraints_lst moves them (joint index tensors included)
        FullBodyConstraintSet(skel, index, joints_c, rots_c),
        EndEffectorConstraintSet(skel, index, joints_c, rots_c, None, joint_names=END_EFFECTORS))]
    set_seed(seed)
    with torch.inference_mode():
        out = model(p["prompt"].strip(), length, num_denoising_steps=p["steps"], constraint_lst=constraints,
                    cfg_weight=[TEXT_GUIDANCE, p["guidance"]], cfg_type="separated", post_processing=p["post_process"],
                    num_samples=1, return_numpy=True, first_heading_angle=0.0, progress_bar=lambda x: x)  # a batch of one, as the CLI
    return unpack(skel, out, turn, shift)


def unpack(skel, out, turn: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kimodo's result in the frame the constraints came in: world rotations [T,J,3,3], root [T,3], contacts [T,4].

    Upstream decides which skeleton the result is on; only SOMA's differs
    (`output_skeleton` in `third_party/kimodo/repo/kimodo/model/kimodo_model.py:61-65`:
    "somaskel77 for SOMA, else unchanged"; both conversion sites,
    `third_party/kimodo/repo/kimodo/model/kimodo_model.py:371-373` and `:553-555`, check
    `if isinstance(self.skeleton, SOMASkeleton30)`):
      · the four SOMA weights: the result is on 77 joints and is cut back to the model's own 30;
      · the three SMPL-X (22 joints) and G1 robot (34 joints) weights: unchanged.
    Contacts likewise: the SOMA conversion adds the toe ends, turning 4 channels into 6
    (`third_party/kimodo/repo/kimodo/skeleton/definitions.py:277-281`); upstream reduces them back to 4 by channel
    count (`third_party/kimodo/repo/kimodo/metrics/foot_skate.py:108-111`), and so does this function.
    The check must be on the skeleton type: SMPLXSkeleton22 / G1Skeleton34 have no `somaskel77` attribute."""
    from kimodo.skeleton import SOMASkeleton30

    world = np.asarray(out["global_rot_mats"][0], np.float64)
    contacts = np.asarray(out["foot_contacts"][0], np.float32)
    if isinstance(skel, SOMASkeleton30):
        world = world[:, skel.get_skel_slice(skel.somaskel77)]
        contacts = contacts[:, [0, 1, 3, 4]]  # heels and toes (77: toe ends added)
    return turn.T @ world, np.asarray(out["root_positions"][0], np.float64) @ turn + shift, contacts


def loaded(run: Run) -> tuple[object, str]:
    """Load the selected weight onto the GPU and attach this job's text embedding (shared by both paths)."""
    job = run.job
    p = job.params
    name = p["checkpoint"]  # the selected weight's folder, from the node (the one table is extension.py MODELS)
    run.weights(job.weights_dir / name)
    embedding = np.load(job.inputs["text"]) if p["prompt"].strip() else None
    model = run.model(name, load_kimodo, name, torch.device("cuda"))
    model.text_encoder = KnownText(embedding)  # this job's prompt
    return model, name


def world_joints(model, rotations: np.ndarray, root: np.ndarray) -> np.ndarray:
    """World position of every joint [T,J,3] (metres): world rotations and root through the model's own FK."""
    skel, device = model.skeleton, model.device
    parents = skel.joint_parents.cpu().numpy()
    local = torch.as_tensor(mo.local_from_world(rotations, parents), dtype=torch.float32, device=device)
    return skel.fk(local, torch.as_tensor(root, dtype=torch.float32, device=device))[1].cpu().numpy()


def generate_free(run: Run) -> None:
    """Without an animation: an empty constraint list, generated from the text and frame count only (upstream
    `constraint_lst=[]`). Motions longer than 10 s are generated in parts, each later part constrained on the
    previous part's last 5 frames. Only the inputs are arranged this way; the model itself is unchanged."""
    p = run.params
    model, name = loaded(run)
    skeleton = model_skeleton(model)
    want, fps = int(p["length"]), float(p["fps"])
    total = max(int(round(want / fps * MODEL_FPS)), 1)  # frame count at the model's own 30 fps
    joints = len(skeleton.names)
    rotations = np.zeros((total, joints, 3, 3))
    root = np.zeros((total, 3))
    contacts = np.zeros((total, 4), np.float32)
    step = MAX_FRAMES - TRANSITION  # frames each part after the first adds (it re-generates TRANSITION frames of the one before)
    parts = 1 + -(-max(total - MAX_FRAMES, 0) // step)  # ceil: the last part may be short, never left ungenerated
    run.stage("Kimodo 生成")
    done = 0
    for n in range(parts):
        lead = TRANSITION if done else 0  # the previous part's last frames constrain this part
        first = done - lead
        length = min(MAX_FRAMES, total - first)
        rot, pos, contact = generate_part(model, rotations[first:done], root[first:done], np.arange(lead),
                                          length, p, p["seed"] + n)
        end = first + length
        rotations[done:end], root[done:end], contacts[done:end] = rot[lead:], pos[lead:], contact[lead:]
        done = end
        progress(n + 1, parts, "生成")
    if parts > 1:
        say("N-KIMODO-PARTS", seconds=total / MODEL_FPS, parts=parts)
    run.stage("重采样到镜头帧率")
    times, at = np.arange(total) / MODEL_FPS, np.arange(want) / fps
    rot_out = mo.resample_rotations(times, rotations, at)
    pos_out = world_joints(model, rot_out, mo.resample_values(times, root, at)) * M_TO_CM
    skeleton = mo.Skeleton(skeleton.names, skeleton.parents, _rest_cm(skeleton.rest))  # output is always in centimetres
    ib.write_model_result(run, skeleton, want, rot_out, pos_out, mo.resample_values(times, contacts, at),
                          method="Kimodo", model=name, text=p["prompt"].strip(), steps=p["steps"],
                          model_fps=MODEL_FPS, fps=fps, parts=parts, seed=p["seed"],
                          post_process=p["post_process"], unconstrained=True)


def _rest_cm(rest: np.ndarray) -> np.ndarray:
    """Rest pose from metres to centimetres (translation column only; rotations unchanged)."""
    out = np.asarray(rest, np.float64).copy()
    out[:, :3, 3] *= M_TO_CM
    return out


def _why(exc: Exception):
    """The {reason} of this node's own message: a Failure's message passed on as a message (its words stay in the
    catalogue, lab2shot_worker.reason), anything else as its text."""
    return reason(exc.code, **exc.params) if isinstance(exc, Failure) else str(exc)


def generate(run: Run) -> None:
    p = run.params
    motion = ib.read_job(run.job.inputs["motion"])
    model, name = loaded(run)
    run.stage("对齐骨骼")
    try:
        retarget = motion.retarget(model_skeleton(model))
    except (Failure, ValueError) as exc:  # MotionJob.retarget / Retarget.align raise Failure (E-MOTION-NOJOINTS / NOROOT / NOLEG)
        fail("E-KIMODO-SKELETON", reason=_why(exc))
    key_rot, key_root = retarget.to_model(motion.poses)
    keys = motion.model_frames(MODEL_FPS)
    if (np.diff(keys) < 1).any():
        k = int(np.argmax(np.diff(keys) < 1))
        fail("E-KIMODO-KEYSTOOCLOSE", first=int(motion.keys[k]), second=int(motion.keys[k + 1]))
    try:
        parts = mo.windows([int(k) for k in keys], MAX_FRAMES - TRANSITION, MAX_KEYS)
    except (Failure, ValueError) as exc:  # motion.windows raises Failure (E-MOTION-KEYGAP)
        fail("E-KIMODO-KEYGAP", reason=_why(exc))

    total = int(keys[-1]) + 1
    rotations = np.zeros((total, len(retarget.model.names), 3, 3))
    root = np.zeros((total, 3))
    contacts = np.zeros((total, 4), np.float32)
    run.stage("Kimodo 生成")
    for n, (a, b) in enumerate(parts):
        start, end = int(keys[a]), int(keys[b])
        lead = min(TRANSITION, start) if n else 0  # the previous part's last frames, constrained and cross-faded
        first = start - lead
        frames = np.concatenate([np.arange(lead), keys[a:b + 1] - first])
        rot_in = np.concatenate([rotations[first:start], key_rot[a:b + 1]])
        root_in = np.concatenate([root[first:start], key_root[a:b + 1]])
        rot, pos, contact = generate_part(model, rot_in, root_in, frames, end - first + 1, p, p["seed"] + n)
        blend = (np.arange(1, lead + 1) / (lead + 1))[:, None] if lead else np.zeros((0, 1))
        if lead:  # previous part -> this one over the shared frames
            q_old, q_new = mo.matrix_to_quat(rotations[first:start]), mo.matrix_to_quat(rot[:lead])
            rotations[first:start] = mo.quat_to_matrix(mo.slerp(q_old, q_new, blend))
            root[first:start] = root[first:start] * (1 - blend) + pos[:lead] * blend
        rotations[start:end + 1], root[start:end + 1], contacts[start:end + 1] = rot[lead:], pos[lead:], contact[lead:]
        progress(n + 1, len(parts), "生成")
    if len(parts) > 1:
        say("N-KIMODO-PARTS", seconds=total / MODEL_FPS, parts=len(parts))
    ib.write_result(run, retarget, MODEL_FPS, keys, rotations, root, contacts, method="Kimodo", model=name,
                    text=p["prompt"].strip(), steps=p["steps"], guidance=[TEXT_GUIDANCE, p["guidance"]],
                    seed=p["seed"], post_process=p["post_process"], parts=len(parts))


def main(job_path: str) -> None:
    job = load_job(job_path)
    check_node(job, NODE)
    # the job decides whether this is a GPU run (Run.start would decide before the job is read): the text task is CPU
    if job.params["task"] == "text":
        encode_text(Run(job, "Kimodo", gpu=False))
    elif "motion" in job.inputs:  # animation wired: the keys are constraints (in-betweening)
        generate(Run(job, "Kimodo"))
    else:  # no animation: empty constraints, generated from text and frame count
        generate_free(Run(job, "Kimodo"))


if __name__ == "__main__":
    serve(main)
