"""The Nuke module's node: 「Nuke 相机输出设置」 (interface: nodes/output.py) writes a camera as a Camera3 in a .nk
script (camera.py), and declares 「复制到 Nuke」 (OutputSettings.clipboard) so that the page can copy it from the result."""

from __future__ import annotations

from ...messages import Msg
from ...nodes.base import NodeParams, Port
from ...nodes.output import Format, OutputSettings, Writes, learned_projects, name_param


class NukeCameraOutput(OutputSettings):
    id = "nuke.output"
    format = Format("nuke_camera")
    category = "out_scene"
    inputs = (Port("camera", "scene.camera"),)
    on_node = ("name",)
    clipboard = "nuke"
    writes = {
        "model": Writes.no(Msg("I-NUKECAM-NOMODEL")),
        "camera": Writes.full(),
        "points": Writes.no(Msg("I-NUKECAM-NOPOINTS")),
        "curves": Writes.no(Msg("I-NUKECAM-NOCURVES")),
        "skeleton": Writes.no(Msg("I-NUKECAM-NOSKELETON")),
        "character": Writes.no(Msg("I-NUKECAM-NOCHARACTER")),
        "gaussian": Writes.no(Msg("I-NUKECAM-NOGAUSSIAN")),
        "light": Writes.no(Msg("I-NUKECAM-NOLIGHT")),
    }

    class Params(NodeParams):
        name: str = name_param("camera")

    @classmethod
    def write(cls, ctx) -> str:
        from ...data.camera import CameraSamples
        from .camera import write_camera

        src = ctx.input("camera")
        samples = CameraSamples.from_packet(src)
        projects = learned_projects(ctx.provenance)
        if (samples.lens or {}).get("distortion"):
            ctx.say("N-NUKE-CAMERADISTORTION", model=(samples.lens["distortion"] or {}).get("model", ""), port="camera")
        out = cls.out_file(ctx, ".nk")
        out.write_text(write_camera(samples, ctx.params["name"], ", ".join(projects)), encoding="utf-8")
        return out.name


NODES = (NukeCameraOutput,)
