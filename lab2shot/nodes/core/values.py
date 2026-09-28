"""数值：五种基本值各一个常量节点（浮点、整数、布尔、向量、文字），和从相机、曲线里取出数值的节点。

A value is fed to any parameter of its type through a wire (via the input created by 提升到节点, NodeDef.param_port).
These nodes are the general-purpose ones; a project node outputs its own values directly (for example AnyCalib's
Focal Length and Filmback) rather than through a dedicated node.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port
from ...data.values import BOOL, FLOAT, INT, TEXT, VECTOR, value_meta, value_packet

Unit = Literal["", "mm", "cm", "m", "px", "°", "帧", "秒", "EV"]
UNIT_LABELS = {"": "无", "mm": "mm", "cm": "cm", "m": "m", "px": "px", "°": "°", "帧": "帧", "秒": "秒", "EV": "EV"}


def unit_param(default: str = "") -> str:
    return P(default, label="单位", group="数值", option_labels=UNIT_LABELS)


class Constant(NodeDef):
    """A single value entered on the node, fed to any parameter of its type or to a node input that accepts it."""

    category = "value"
    on_node = ("value",)
    value_type = FLOAT

    @classmethod
    def known_outputs(cls, params):
        return {"value": value_meta(cls.value_type, params["value"], unit=params.get("unit", ""))}

    @classmethod
    def cook(cls, ctx):
        return {"value": value_packet(ctx.outputs["value"], cls.value_type, ctx.params["value"], unit=ctx.params.get("unit", ""))}


class FloatValue(Constant):
    id = "core.value_float"
    on_node = ("value", "unit")
    outputs = (Port("value", FLOAT, "值", unit="param:unit"),)
    value_type = FLOAT

    class Params(NodeParams):
        value: float = P(0.0, label="值", group="数值")
        unit: Unit = unit_param()


class IntValue(Constant):
    id = "core.value_int"
    on_node = ("value", "unit")
    outputs = (Port("value", INT, "值", unit="param:unit"),)
    value_type = INT

    class Params(NodeParams):
        value: int = P(0, label="值", group="数值")
        unit: Unit = unit_param()


class BoolValue(Constant):
    id = "core.value_bool"
    outputs = (Port("value", BOOL, "值"),)
    value_type = BOOL

    class Params(NodeParams):
        value: bool = P(False, label="值", group="数值")


class VectorValue(Constant):
    id = "core.value_vector"
    on_node = ("value", "unit")
    outputs = (Port("value", VECTOR, "值", unit="param:unit"),)
    value_type = VECTOR

    class Params(NodeParams):
        value: tuple[float, float, float] = P((0.0, 0.0, 0.0), label="值", widget="vec3", group="数值")
        unit: Unit = unit_param()


class TextValue(Constant):
    id = "core.value_text"
    outputs = (Port("value", TEXT, "值"),)
    value_type = TEXT

    class Params(NodeParams):
        value: str = P("", label="值", group="数值")


# ------------------------------------------------------------------ values taken out of data


def _per_frame_param():
    return P(True, label="逐帧", group="数值")


class SplitCamera(NodeDef):
    id = "core.split_camera"
    on_node = ("per_frame",)
    category = "camera_tools"
    inputs = (Port("camera", "scene.camera", "相机"),)
    # 四个输出对应 DCC（Houdini、Nuke）中相机的四项属性。主点、画面宽高、镜头模型、畸变系数不单独输出：
    # 三维相机不包含这些信息，从相机拆出的结果始终为空。
    outputs = (
        Port("focal", FLOAT, "Focal Length", unit="mm"),
        Port("filmback", FLOAT, "Filmback", unit="mm"),
        Port("translate", VECTOR, "位置", unit="cm"),
        Port("rotate", VECTOR, "旋转", unit="°"),
    )

    class Params(NodeParams):
        per_frame: bool = _per_frame_param()

    @classmethod
    def cook(cls, ctx):
        from ...data.camera import CameraSamples
        from ...data.scene import xyz_euler_deg

        camera = ctx.input("camera")
        frames = camera.meta["frames"]
        s = CameraSamples.from_packet(camera, frames or [0])
        m = s.cam_to_world
        # 画面宽高不作为输出，但 Focal Length 和 Filmback 的数值须附带该信息：
        # 毫米换算为 Focal Length（px）需要画面宽度（`nodes/lens.py units.focal_px`），接收方从值的元数据中读取。
        w, h = int(camera.meta.get("width") or 0), int(camera.meta.get("height") or 0)
        picture = {"width": w, "height": h} if w and h else {}
        series = {
            "focal": (FLOAT, s.focal_mm, "mm", picture),
            "filmback": (FLOAT, s.h_aperture_mm, "mm", picture),
            "translate": (VECTOR, m[:, :3, 3], "cm", {}),
            "rotate": (VECTOR, xyz_euler_deg(s.rotations()), "°", {}),
        }
        out = {}
        for port, (kind, values, unit, more) in series.items():
            values = np.asarray(values, np.float64)
            same = np.allclose(values, values[:1], rtol=1e-9, atol=1e-9)
            if frames and ctx.params["per_frame"] and not same:
                out[port] = value_packet(ctx.outputs[port], kind, frames=frames, values=list(values), unit=unit,
                                         **more)
            else:
                one = np.median(values, axis=0)
                out[port] = value_packet(ctx.outputs[port], kind, one.tolist() if kind == VECTOR else float(one), unit=unit, **more)
        return out


class CurveChannel(NodeDef):
    id = "core.curve_channel"
    category = "value"
    inputs = (Port("curves", "curves", "曲线"),)
    outputs = (Port("value", FLOAT, "值", unit="param:unit"),)
    on_node = ("curve", "unit")

    class Params(NodeParams):
        curve: str = P("", label="曲线", group="数值", placeholder="第一条")
        unit: Unit = unit_param()

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import read_curves

        curves = ctx.input("curves")
        names, want = curves.meta["names"], ctx.params["curve"].strip() or "1"
        k = int(want) - 1 if want.isdigit() else (names.index(want) if want in names else -1)
        if not 0 <= k < len(names):
            raise Invalid(Msg("E-CURVES-NONAME", name=want, count=len(names), names=list(names[:3])))
        values = read_curves(curves)[:, k]
        return {"value": value_packet(ctx.outputs["value"], FLOAT, frames=curves.meta["frames"], values=[float(v) for v in values],
                                      unit=ctx.params["unit"], curve=names[k])}


NODES = (FloatValue, IntValue, BoolValue, VectorValue, TextValue, CurveChannel, SplitCamera)
