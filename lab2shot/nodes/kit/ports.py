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
    return Port("boxes", "boxes", "人物框", optional=optional, expects=(EachPerson(), SameShot()) if each else (SameShot(),),
                takes_empty=optional,
                # 从这个口往外拉一根线时，节点菜单第一个提「选人」（Port.recommend → describe()["inserts"]；
                # EachPerson.fix 也是「选人」）。
                recommend="core.select_people" if each else "")


def basecolor_port() -> Port:
    """去掉光照之后的颜色的输出口。名字和标签只在这里写一次：CG 流程里 basecolor 和 albedo 是同一样东西，
    用两个名字用户在节点图上会看见同一样东西两个叫法。中文有统一译法（Houdini / Substance 的中文界面都叫
    「基础色」），所以不留英文原名。写出这张图走 kit/maps.py 的 basecolor_map()。"""
    return Port("basecolor", "image.3", "基础色")


def plate_mask_port(label: str, optional: bool = True, every_frame: bool = True) -> Port:
    """A mask of the plate (moving things kept out, where to track, a rough matte). every_frame False: some frames are
    enough (a matte guided from its first mask, a tracker's grid placed on one frame), so only its size is checked."""
    return Port("mask", "image.1", label, optional=optional, expects=(SameShot(frames=every_frame),))


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


def measured_param(label: str, measured: dict[Any, Measured], *, default: Any = None, auto: str = "",
                   group: str = "", **kw):
    """A parameter that can blow the GPU's or the machine's memory, or make a cook take many times longer (frames per
    segment, processing size, point and object counts, iterations): only settings with a measurement
    (Measured) are offered, and the node's annotation is the same Literal, so the server refuses any other number
    exactly as the page does (E-GRAPH-PARAMS). Every declared setting is offered; its measured VRAM
    goes into the node's resolved cost, by which the scheduler places the job on a card that holds it (nodes/applies.py
    setting_vram). `auto`: what the empty value does, when it may be left empty (default None)."""
    if not measured or not all(isinstance(m, Measured) for m in measured.values()):
        raise TypeError(f"{label}: every setting is a Measured (what it measured, or which measurement bounds it)")
    labels = kw.pop("option_labels", None) or {str(k): str(k) for k in measured}
    return P(default, label=label, group=group, option_labels=labels, measured=measured_bounds(measured),
             **({"placeholder": auto} if auto else {}), **kw)


def resolution_param(measured: dict[int, Measured], default: int):
    """The processing size (long side, pixels) a model runs at, as measured settings (measured_param).

    `default` 是必填的，没有「自动」这一档（同 max_frames_param）。每个节点填模型自己的那个尺寸
    （训练尺寸 / 官方默认），界面上就是那个数字，不留空。"""
    return measured_param("处理分辨率", measured, default=default, group="解算")


def max_frames_param(measured: dict[int, Measured], default: int):
    """每段最多帧数 of a reconstruction, as measured settings (measured_param): a one-shot (non-streaming) model's memory
    grows with the whole segment at once, so each setting is one measured on this model.

    `default` 是必填的，没有「自动」这一档：留空显示「自动」的话用户在界面上看不到任何一个数，
    不知道自己实际在用多少帧。每个节点把测过的那一档直接填成默认值，界面上就是那个数字。"""
    return measured_param("每段最多帧数", measured, default=default, group="解算")


def conf_threshold_param(default: float, hi: float = 50.0):
    return P(default, label="置信度门槛", ge=1.0, le=hi, group="解算")


def loops_param():
    """Loop closure of a chunked reconstruction (lab2shot_worker.recon.Stitcher, keep=...): for the workers that do it."""
    return P(False, label="回环闭合", group="解算", applies=fact("segments").gt(2))


def precision_level_param():
    """0–9 internal resolution level (MoGe, UniDepth, UniK3D): the output is always the plate's size."""
    return P(9, label="精度等级", ge=0, le=9, group="几何")


def unit_cm_param(applies: Cond | None = None) -> float:
    """「尺度」：这个方法解出来的 1 个单位是多少厘米。`applies`：什么时候它才起作用。节点能从别处
    算出尺度时（接了一台相机）这个参数不起作用，变灰并写原因，位置不跳。"""
    return P(M_TO_CM, label="尺度", unit="cm", gt=0, group="场景",
             worker=False, applies=applies)


def point_size_param(default: float = 1.0):
    return P(default, label="点的大小", unit="cm", gt=0, le=100, group="跟踪", worker=False)


def flow_resolution_param(measured: dict[int, Measured], default: int | None = None):
    """The long side an optical-flow model runs at (never above the plate's), as measured settings; None: the plate's
    own size, capped at the measured 1920 (flow.py clamped_flow_side)."""
    return measured_param("处理分辨率", measured, default=default, auto="原尺寸", group="光流")


def static_camera_param(rotation_wire: bool = True):
    """Whether the camera is locked off; the 「相机旋转」 wire says it itself (WorldHumans hands the worker that rotation:
    families/humans.py prepare, kit/cameras.py rotation_is_still).

    上游只吃旋转、不吃位移，线接在参数「相机旋转」上（不是一个输入口），所以变灰的条件是 `Param(...).wired()`，
    不是 `Wired(...)`。

    `rotation_wire=False`：这个节点没有「相机旋转」可接（上游自己解整台相机，`camera_to_worker` 不是 "rotation"，
    families/humans.py 把那个参数从它身上去掉了）。提示里不能提那条线，也不能挂「接了就变灰」的条件：
    条件指着一个不存在的参数，nodes/applies.py check_declarations 当场拒绝。"""
    if not rotation_wire:
        return P(False, label="固定机位", group="相机")
    return P(False, label="固定机位", group="相机",
             applies=Not(Param("camera_rotate").wired()))


def follow_camera_param():
    """For methods that solve bodies in their own world and then align it to the camera (GVHMR, WHAM)."""
    return P(False, label="逐帧贴合画面", group="相机")
