"""好几个家族、好几个项目共用的口和参数：画面里的人和 mask、去掉光照的「基础色」，以及不止一个家族的 worker 都有的那些参数（处理尺寸、帧数上限、置信度门槛……）。

这里的口不一定属于某个家族：`basecolor_port()` 的两个用户（OpenDelight、DiffusionRenderer）
各自是独立节点，没有共同的家族，共用的只有这一个口的名字和标签，所以它落在零件里，不是家族里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...data.units import M_TO_CM
from ..base import P, Port
from ..expects import EachPerson, SameShot
from ...availability import Not
from ..applies import Cond, Param, fact


def people_port(optional: bool = False, each: bool = True) -> Port:
    """The people of the plate: each: the node works once per person (a character, a pair of hands, a matte object),
    so boxes nobody chose that hold several people are warned about, with 「选人」 to insert; else it takes them all as
    one thing (areas kept out of a reconstruction, crops to segment). A required one has nothing to do with boxes that
    hold nobody (takes_empty: the node gives nothing and says so); an optional one takes them as they are."""
    return Port("boxes", "boxes", optional=optional, expects=(EachPerson(), SameShot()) if each else (SameShot(),),
                takes_empty=optional,
                # 从这个口往外拉一根线时，节点菜单第一个提「选人」（Port.recommend → describe()["inserts"]；
                # EachPerson.fix 也是「选人」）。
                recommend="select_people" if each else "")


def basecolor_port() -> Port:
    """去掉光照之后的颜色的输出口。名字和标签只在这里写一次：CG 流程里 basecolor 和 albedo 是同一样东西，
    用两个名字用户在节点图上会看见同一样东西两个叫法。中文有统一译法（Houdini / Substance 的中文界面都叫
    「基础色」），所以不留英文原名。写出这张图走 kit/maps.py 的 basecolor_map()。"""
    return Port("basecolor", "image.3")


def plate_mask_port(optional: bool = True, every_frame: bool = True, words: str = "") -> Port:
    """A mask of the plate (moving things kept out, where to track, a rough matte). every_frame False: some frames are
    enough (a matte guided from its first mask, a tracker's grid placed on one frame), so only its size is checked.
    Its name is node.<type>.port.mask.label."""
    return Port("mask", "image.1", optional=optional, expects=(SameShot(frames=every_frame),), words=words)


@dataclass(frozen=True)
class Measured:
    """What one setting of a heavy parameter measured (the settings follow the cards that can take the job, placed by
    the scheduler). Exactly one of: `gb`, its peak VRAM measured on the node's card
    (Cost.measured_on); `below`, the setting whose measurement bounds it (fewer frames, a smaller size: never more
    memory); `flat`, the setting changes the time only (the node's own measured cost stands)."""

    gb: float | None = None
    below: Any = None
    flat: bool = False

    def __post_init__(self) -> None:
        if (self.gb is not None) + (self.below is not None) + bool(self.flat) != 1:
            raise TypeError(f"a measured setting says exactly one of gb, below, flat: {self!r}")


def measured_bounds(measured: dict[Any, Measured]) -> dict[str, float | None]:
    """Each setting (as the page keys it) -> the most VRAM it can take (GB): its own measurement, the one it is below
    (followed along), None for a setting that changes the time only."""
    from ..applies import option_key

    def bound(key, seen=()):
        m = measured[key]
        if m.below is None:
            return m.gb
        if m.below in seen or m.below not in measured:
            raise TypeError(f"setting {key!r} is below {m.below!r}, which is not a measured setting of the same parameter")
        return bound(m.below, (*seen, key))

    return {option_key(k): bound(k) for k in measured}


def measured_param(measured: dict[Any, Measured], *, default: Any = None, group: str = "", **kw):
    """A parameter that can blow the GPU's or the machine's memory, or make a cook take many times longer (frames per
    segment, processing size, point and object counts, iterations): only settings with a measurement
    (Measured) are offered, and the node's annotation is the same Literal, so the server refuses any other number
    exactly as the page does (E-GRAPH-PARAMS). Every declared setting is offered; its measured VRAM
    goes into the node's resolved cost, by which the scheduler places the job on a card that holds it (nodes/applies.py
    setting_vram). `measured`: the settings. Its words are the parameter's (node.<type>.param.<p>.label; .placeholder:
    what the empty value does, when it may be left empty, default None)."""
    if not measured or not all(isinstance(m, Measured) for m in measured.values()):
        raise TypeError("every setting of a measured parameter is a Measured (what it measured, or which measurement bounds it)")
    return P(default, group=group, measured=measured_bounds(measured), **kw)


def resolution_param(measured: dict[int, Measured], default: int):
    """The processing size (long side, pixels) a model runs at, as measured settings (measured_param).

    `default` 是必填的，没有「自动」这一档（同 max_frames_param）。每个节点填模型自己的那个尺寸
    （训练尺寸 / 官方默认），界面上就是那个数字，不留空。"""
    return measured_param(measured, default=default, group="solve", words="kit.resolution")


def max_frames_param(measured: dict[int, Measured], default: int):
    """每段最多帧数 of a reconstruction, as measured settings (measured_param): a one-shot (non-streaming) model's memory
    grows with the whole segment at once, so each setting is one measured on this model.

    `default` 是必填的，没有「自动」这一档：留空显示「自动」的话用户在界面上看不到任何一个数，
    不知道自己实际在用多少帧。每个节点把测过的那一档直接填成默认值，界面上就是那个数字。"""
    return measured_param(measured, default=default, group="solve", words="kit.max_frames")


def conf_threshold_param(default: float, hi: float = 50.0):
    return P(default, ge=1.0, le=hi, group="solve")


def loops_param():
    """Loop closure of a chunked reconstruction (lab2shot_worker.recon.Stitcher, keep=...): for the workers that do it."""
    return P(False, group="solve", applies=fact("segments").gt(2), words="kit.loops")


def precision_level_param():
    """0–9 internal resolution level (MoGe, UniDepth, UniK3D): the output is always the plate's size."""
    return P(9, ge=0, le=9, group="geometry")


def unit_cm_param(applies: Cond | None = None) -> float:
    """「尺度」：相对尺度的方法解出来的 1 个单位是多少厘米（单位换算，不是尺度对齐）。`applies`：什么时候它才起作用；
    输出本身是真实距离的方法不该被它缩放（整段家族的 RELATIVE_ONLY 让它变灰并写原因，位置不跳）。"""
    return P(M_TO_CM, unit="cm", gt=0, group="scene",
             worker=False, applies=applies)


def point_size_param(default: float = 1.0):
    return P(default, unit="cm", gt=0, le=100, group="tracking", worker=False)


def flow_resolution_param(measured: dict[int, Measured], default: int | None = None):
    """The long side an optical-flow model runs at (never above the plate's), as measured settings; None: the plate's
    own size, capped at the measured 1920 (flow.py clamped_flow_side)."""
    return measured_param(measured, default=default, group="optical_flow", words="kit.flow_resolution")


def static_camera_param(rotation_wire: bool = True):
    """Whether the camera is locked off; the 「相机旋转」 wire says it itself (WorldHumans hands the worker that rotation:
    families/humans.py prepare, kit/cameras.py rotation_is_still).

    上游只吃旋转、不吃位移，线接在参数「相机旋转」上（不是一个输入口），所以变灰的条件是 `Param(...).wired()`，
    不是 `Wired(...)`。

    `rotation_wire=False`：这个节点没有「相机旋转」可接（上游自己解整台相机，`camera_to_worker` 不是 "rotation"，
    families/humans.py 把那个参数从它身上去掉了）。提示里不能提那条线，也不能挂「接了就变灰」的条件：
    条件指着一个不存在的参数，nodes/applies.py check_declarations 当场拒绝。"""
    if not rotation_wire:
        return P(False, group="camera")
    return P(False, group="camera",
             applies=Not(Param("camera_rotate").wired()))


def follow_camera_param():
    """For methods that solve bodies in their own world and then align it to the camera (GVHMR, WHAM)."""
    return P(False, group="camera")


# ------------------------------------------------------------------ 读取节点的「帧率」口

def fps_port(may_be_empty: bool = False) -> Port:
    """读取节点的「帧率」输出口：文件自己记的帧率，接到要帧率的参数上（输出设置的「帧率」、动作模型的「帧率」，
    data/units.py DEFAULT_FPS 的说明）。may_be_empty：格式可以不记帧率（视频、USD），没记时口给空包，
    接着的参数照样可填、用填的；总记着的格式（BVH 的 Frame Time）不是。值由 fps_meta / fps_packet 给出。"""
    from ...data.values import FLOAT

    return Port("fps", FLOAT, unit="fps", may_be_empty=may_be_empty)


def fps_meta(fps: float | None) -> dict:
    """「帧率」口的描述（known_outputs 用：计算前就能在接着的参数上显示）；None：文件没记，空包。不是有限正数的帧率
    是读取器的错（它该拒收那个文件），报 ValueError，不当成「没记」，也不把 NaN 交给下游。"""
    import math

    from ...data.values import FLOAT, value_meta

    if fps is None:
        return {"empty": True}
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError(f"a frame rate is a positive number, not {fps!r}")
    return value_meta(FLOAT, float(fps), unit="fps")


def fps_packet(ctx, fps: float | None):
    """「帧率」口的包（cook 用），与 fps_meta 同一个描述；没记帧率时是 empty_packet（空包只从那一处造）。"""
    from ...data.packet import Packet
    from ...data.values import FLOAT
    from ..base import empty_packet

    if fps is None:
        return empty_packet(ctx, "fps")
    return Packet(ctx.outputs["fps"], FLOAT, fps_meta(fps))


def normal_port() -> Port:
    """A normal map output: three channels of values (Port.data), not a picture, and which space they are in said on
    the packet (means space: 相机 / 世界). One declaration for every node that gives normals."""
    return Port("normal", "image.3", means=("space",), data=True)


def rgb_port(name: str = "image", **kw) -> Port:
    """A picture input (a plate a model looks at): only a picture goes in (Port.data False), a normal map or motion
    vectors are refused at the wire rather than read as a photograph. Its name: node.<type>.port.<name>.label, else
    port.<name>.label."""
    return Port(name, "image.3", data=False, **kw)


def values_port(name: str, type_: str, **kw) -> Port:
    """An input of values (normals, positions, motion vectors: Port.data True): a picture is refused at the wire rather
    than read as numbers (a photograph's colours taken for motion vectors)."""
    return Port(name, type_, data=True, **kw)
