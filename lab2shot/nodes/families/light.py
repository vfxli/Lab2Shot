"""Environment light probe: an HDRI (a lat-long EXR, which is what the upstream methods output) estimated from one plate frame."""

from __future__ import annotations

from typing import Literal

from lab2shot_worker.light_probe import ENVMAP, PREVIEW

from ...data.contracts import NEW_PICTURE

from ...data.packet import Packet
from ...data.payloads import display_rgb, image_packet, still_packet
from ...errors import Invalid
from ...messages import Msg
from ..base import NodeParams, P, Port
from ..lens import NO_LENS, takes_lens
from .base import Job, RawOutput, WorkerNode
from ..kit.cameras import plate_lens
from ..applies import Cost


def probe_plate(ctx, image: Packet, frame: int) -> Packet:
    """Only the probe frame goes to a lighting worker: one sRGB PNG (the working space clamped), not the whole shot."""
    from ...io import images
    from ...io.color import working_space

    probe_dir = ctx.work / "probe"
    probe_dir.mkdir(exist_ok=True)
    png = probe_dir / f"frame.{frame}.png"
    images.write_png(png, images.as_uint8(display_rgb(image, frame)))
    return image_packet(probe_dir, {frame: png}, image.meta["width"], image.meta["height"], working_space())


class LightProbeParams(NodeParams):
    """Parameters every light-probe node shares; each adds its model's own."""

    seed: int = P(0, label="随机种子", ge=0, group="环境光")
    envmap_width: Literal[512, 1024, 2048] = P(
        1024, label="环境图宽度", group="环境光",
        option_labels={"512": "512", "1024": "1024", "2048": "2048"},
    )


class LightProbe(WorkerNode):
    """An HDRI (lat-long EXR) from one plate frame: exactly what the upstream methods output, nothing else.
    A method that unfolds with the lens has the lens parameters (LensParams): its Focal Length, else the camera's, at that frame
    (Job.lens; the method adds what it sends of it in prepare, DiffusionLight its fov_deg).

    There is no "light" output and no light data type: a USD DomeLight wrapping the HDRI would carry no new
    information and no renderer-neutral guarantee. The HDRI image itself is what every renderer accepts.

    Raw contract (lab2shot_worker.light_probe): raw/envmap.exr, a lat-long HDR in the camera's frame, its centre column
    where the camera looks, top = up; raw/preview.png. Only the probe frame is sent (Job.send); the outputs span the
    plate's frames. Job.notes: "frame", the frame sampled (the plate's one frame)."""

    on_node = ("envmap_width",)
    # No camera input: the worker never reads one. The HDRI is in the plate camera's frame (see the raw contract).
    inputs = (Port("image", "image.3", "RGB"),)
    # the HDRI and its display-referred PNG are pictures of their own, not the plate's geometry
    outputs = (Port("hdri", "image.3", "HDRI", shape=NEW_PICTURE),
               Port("preview", "image.3", "显示图", shape=NEW_PICTURE,
                    help="HDRI 转成屏幕上看的样子（PNG，显示参考），检查环境对不对；渲染用「HDRI」那根线"))
    cost = Cost(gpu=True)
    most_frames = 1  # one frame in, one HDRI out: a sequence is refused before submission, with a hint to insert 「FrameHold」
    # An HDRI holds radiance, not a picture: the packet is tagged linear Rec.709 (io/color.LINEAR_REC709) and is not
    # converted to the working space. Relighting workers read its file as linear; the viewer converts it to the working
    # space for display by this tag (the exception noted in io/color.py).
    colorspace = "lin_rec709_scene"

    @classmethod
    def prepare(cls, ctx) -> Job:
        # A pure solver: one frame in, one HDRI out, with no frame selection here. To take one frame of a
        # sequence, connect 「FrameHold」 upstream.
        image = ctx.input("image")
        frames = image.meta["frames"]
        # Second check for most_frames: submission already refuses by it (nodes/expects.py FrameCount(most=));
        # this ensures the first frame is never used silently at run time.
        if cls.most_frames and len(frames) > cls.most_frames:
            raise Invalid(Msg("E-LIGHT-ONEFRAME", frames=len(frames)))
        frame = frames[0]
        lens = plate_lens(ctx, image, [frame]) if takes_lens(cls) else NO_LENS
        return Job(image, send=probe_plate(ctx, image, frame), extra={"frame": frame}, lens=lens,
                   notes={"frame": frame})

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        """The HDRI and its display-referred preview: the two files upstream writes."""
        import shutil

        from ...io import images
        from ...io.color import working_space, space

        image = job.plate
        frames = image.meta["frames"]

        ctx.stage("写出 HDRI")
        hdri = ctx.outputs["hdri"] / ENVMAP
        shutil.copyfile(raw.file(ENVMAP), hdri)
        out = {
            "hdri": still_packet(ctx.outputs["hdri"], hdri, frames, *images.image_size(hdri), colorspace=space(cls.colorspace)),
        }
        preview = ctx.outputs["preview"] / PREVIEW
        shutil.copyfile(raw.file(PREVIEW), preview)
        out["preview"] = still_packet(ctx.outputs["preview"], preview, frames, *images.image_size(preview), colorspace=working_space())
        return out
