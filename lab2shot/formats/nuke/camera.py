"""A camera as Nuke's Camera3 node (.nk text).

Lab2Shot and Nuke describe a camera the same way: Y up, looking down -Z, right-handed, film back and focal length in
millimetres. Lab2Shot positions are in centimetres and Nuke has no unit of its own, so the values are written
unchanged (the Nuke comp's world is then in centimetres, consistent with every camera Lab2Shot writes from the same
solve).

Rotation: Nuke's `rot_order` knob names the order the rotations are applied in, first to last, so the default ZXY
applies Z first: R = Ry·Rx·Rz with column vectors. The angles written here are decomposed in exactly that order
(euler_zxy) and `rot_order` is written explicitly, so the result does not depend on the default.

Every knob that changes over the shot is written as an animation curve with the plate's own frame numbers; one that
does not is written as a plain number (a locked-off camera then has no keys).
"""

from __future__ import annotations

import numpy as np

from . import script

CAMERA_NODE = "Camera3"  # Nuke 13 and later (Camera2 before it)
ROT_ORDER = "ZXY"  # what the angles below are decomposed in, written out on the node


def euler_zxy(rotations: np.ndarray) -> np.ndarray:
    """Rotation matrices [F,3,3] -> Nuke's rotate knob [F,3] in degrees (rx, ry, rz) for rot_order ZXY. Nuke builds
    R = Ry·Rx·Rz from them: ZXY names the order in
    which the rotations are applied, Z first, so Rz is rightmost. At gimbal lock (cos rx = 0) Z and Y rotate about the
    same axis; rz is then set to 0 and ry carries the whole rotation, which yields the same matrix."""
    m = np.asarray(rotations, np.float64).reshape(-1, 3, 3)
    rx = np.arcsin(np.clip(-m[:, 1, 2], -1.0, 1.0))
    locked = np.abs(np.cos(rx)) < 1e-9
    rz = np.where(locked, 0.0, np.arctan2(m[:, 1, 0], m[:, 1, 1]))
    ry = np.where(locked, np.arctan2(-m[:, 2, 0], m[:, 0, 0]), np.arctan2(m[:, 0, 2], m[:, 2, 2]))
    return np.degrees(np.stack([rx, ry, rz], axis=-1))


def write_camera(samples, name: str = "camera", note: str = "") -> str:
    """A camera as the text of a Nuke Camera3 node, ready to paste (used for the .nk in a data packet and by
    「复制到 Nuke」).

    `samples`: data/camera.py CameraSamples (centimetres, Y up, looking down -Z). `note`: text appended to the node
    label after the Lab2Shot mark (the solving method, so the camera is not mistaken for a hand-tracked one).
    """
    from lab2shot_shared.motion import orthonormal

    frames = [int(f) for f in samples.frames] or [1]
    placed = samples.at(frames) if samples.frames else samples  # cam_to_world as one [4,4] per frame
    poses = np.asarray(np.eye(4)[None] if placed.cam_to_world is None else placed.cam_to_world, np.float64).reshape(-1, 4, 4)
    poses = np.repeat(poses, len(frames), 0) if len(poses) == 1 else poses
    translate = poses[:, :3, 3]
    rotate = euler_zxy(orthonormal(poses[:, :3, :3]))
    focal = np.asarray(samples.focal_mm, np.float64).reshape(-1)
    h_ap = np.asarray(samples.h_aperture_mm, np.float64).reshape(-1)
    v_ap = np.asarray(samples.v_aperture_mm, np.float64).reshape(-1)
    centre = np.asarray(samples.center_mm, np.float64).reshape(-1, 2)

    def over_frames(values: np.ndarray) -> np.ndarray:
        """One value per frame; a single value is repeated over all frames."""
        return np.repeat(values[:1], len(frames), 0) if len(values) == 1 else values

    focal, h_ap, v_ap, centre = over_frames(focal), over_frames(h_ap), over_frames(v_ap), over_frames(centre)
    knobs = [
        ("inputs", "0"),
        # Both orders are written explicitly rather than left to Nuke's defaults; the angles are decomposed in these.
        ("xform_order", "SRT"),
        ("rot_order", ROT_ORDER),
        ("translate", script.channels(frames, translate.T)),
        ("rotate", script.channels(frames, rotate.T)),
        ("focal", script.channel(frames, focal)),
        ("haperture", script.channel(frames, h_ap)),
        ("vaperture", script.channel(frames, v_ap)),
    ]
    if np.abs(centre).max(initial=0.0) > 1e-9:
        # Principal point off the image centre: Nuke's window translate is in half-apertures, +x right, +y up.
        shift = np.stack([centre[:, 0] / (h_ap / 2.0), centre[:, 1] / (v_ap / 2.0)], axis=-1)
        knobs.append(("win_translate", script.channels(frames, shift.T)))
    said = note or "camera"
    if samples.width and samples.height:
        said = f"{said}, {samples.width}x{samples.height}"
    knobs += [("name", script.node_name(name, "Lab2Shot_camera")), ("label", script.label(f"{said}, cm, frames {frames[0]}-{frames[-1]}"))]
    return script.script([script.block(CAMERA_NODE, knobs)])


