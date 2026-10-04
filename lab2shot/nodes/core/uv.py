"""规范坐标转 UV：位置图 → UV 坐标图。

A position map in a canonical space (FaceAnything's 规范坐标, or any other body or object model) records which point
of the model each pixel shows. Projected consistently, that position serves as a texture coordinate: the same point of
the model receives the same UV in every frame and every shot, so it can carry a texture. The projection maths lives in
nodes/kit/projection.py; this node only reads the input, writes the output and reports what it did. It is not tied to any
project: any node that produces a position map can feed it.
"""

from __future__ import annotations

from typing import Literal

from ..kit.ports import values_port
from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port
from ..expects import SameShot
from ...data.contracts import meant

CANONICAL = "canonical"  # canonical space (the `space` of map.position in data/contracts.py; shared by face, body and object models)


class PositionToUv(NodeDef):
    id = "uv_from_position"
    on_node = ("way",)
    category = "geometry_tools"
    inputs = (
        values_port("position", "image.3"),
        Port("mask", "image.1", optional=True, expects=(SameShot("position"),)),
    )
    outputs = (Port("uv", "image.2", means=("projection",)),)

    class Params(NodeParams):
        way: Literal["cylinder", "sphere", "plane"] = P(
            "cylinder", group="projection",
        )

    @classmethod
    def cook(cls, ctx):
        import numpy as np

        from ...data.maps import map_at, same_size
        from ...data.payloads import UNIT, ExrWriter, window_of
        from ..kit.projection import grown, project
        from ...data.summary import SPACE_SAID

        src, mask = ctx.input("position"), ctx.input("mask")
        same_size({ctx.node_type.port_label("position"): src, ctx.node_type.port_label("mask"): mask})  # 2D inputs used together must come from the same plate
        space = meant(ctx, src, "space", ctx.node_type.port_label("position"))
        if not space.startswith(CANONICAL):
            raise Invalid(Msg("E-UV-SPACE", space=Msg(SPACE_SAID[space]).text))
        way, frames = ctx.params["way"], src.meta["frames"]
        window = window_of(src)

        def valid_at(frame, alpha):
            """Return the weight of pixels this node uses: the map's own validity, multiplied by the mask when one is wired."""
            if mask is None:
                return alpha
            got = map_at(mask, frame, window.data)
            return alpha if got is None else alpha * got[0][..., 0]

        ctx.stage("measure_canonical_range")

        def bounds(f):  # 一帧的范围：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            values, alpha = map_at(src, f)
            return grown(None, values, valid_at(f, alpha))

        box = None
        for one in ctx.each_done(frames, bounds):  # 按帧序并入：一帧的范围即其两个角上的位置，取最小与最大值不受先后影响
            if one is not None:
                box = grown(box, np.array([one.low, one.high], np.float64))
        if box is None:
            ctx.say("N-UV-NOPOSITIONS", port="position")

        ctx.stage("project_uv")
        out = ExrWriter(ctx.outputs["uv"], 2, validity=True, value_range=UNIT, window=window, projection=way,
                        **({"box": box.json()} if box is not None else {}))
        def uv(f):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            values, alpha = map_at(src, f)
            keep = valid_at(f, alpha)
            out.add(f, project(values, way, box) if box is not None else values[..., :2] * 0.0, keep)

        list(ctx.each_done(frames, uv))
        packet = out.packet()
        if src.meta.get("still"):
            packet.meta["still"] = True
        return {"uv": packet}


NODES = (PositionToUv,)
