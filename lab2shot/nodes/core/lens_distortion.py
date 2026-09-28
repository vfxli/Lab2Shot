"""LensDistortion：把一个镜头烘焙为两张 ST-map（去畸变、加畸变）。

本节点不修改像素。它只计算镜头（模型、系数、内参、扩边 → 两张 ST-map），图像变形一律在「STMap」中进行，
因此重采样次数在节点图上可见：

    读取序列 ─┬─→ STMap（源）──→ 结果（已去畸变的画面）
              └─→ LensDistortion ──去畸变 ST-map──→ STMap（ST-map）

本节点不输出「结果」图：同一操作若有两条路径，使用者容易把结果再次送入 STMap 造成二次变形。

镜头参数在节点上填写，或由连线驱动（Focal Length、Filmback 两个参数端口，其余打包在「镜头内参」输入端口中）。
「图像」端口可选：接入时从画面获取分辨率和像素比，未接入时手动填写「画面宽度 / 画面高度」
（仅有 3DE 镜头数据、尚未导入素材时也可烘焙）。

数值计算（模型、画布、ST-map、反函数）见 lab2shot/data/lens_models.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...errors import Invalid
from ...messages import Msg
from ..applies import LENS_ANY
from ..base import Info, NodeDef, P, Port, empty_packet
from ..clipboard import Pasteable
from ..expects import Checked, Expect, Seen
from ...data.lens_models import MODELS
from ..lens import DISTORTED, LensParamEntry, LensSheet, LensSheetParams, lens_identity, model_label, sheet_field, table_of, unpacked_lens
from ...availability import All, Not
from ...data.lens_models import distorts
from ...data.values import LENS, read as read_value
from ..applies import Wired

from ...data.contracts import Shape
from ...data.windows import Window

# 画面的镜头状态：原始拍摄、已去畸变、未声明
RAW, UNDISTORTED, UNKNOWN = "raw", "undistorted", "unknown"
OVERSCAN = {"auto": "auto", "none": "none", "5": 0.05, "10": 0.10, "20": 0.20}
LEVEL_SAID = {"measured": "I-LENS-MEASURED", "solved": "I-LENS-SOLVED", "estimated": "P-LENS-ESTIMATED"}
STMAP_PORTS = (("undistort_stmap", "undistort"), ("distort_stmap", "distort"))


@dataclass(frozen=True)
class Maps:
    """某一帧的两张 ST-map（lens_models.encode 的条目，无值处为 INVALID）。"""

    undistort: np.ndarray | None
    distort: np.ndarray | None  # None：本次没有需要该方向的下游


@dataclass(frozen=True)
class Plan:
    """节点重采样所用的数据：按帧的映射图（None 表示所有帧共用一张）、画布，以及移除（或加回）的镜头描述。"""

    maps: dict
    canvas: object
    removed: dict


@dataclass(frozen=True)
class HasSource(Expect):
    """该镜头对接入的画面是否成立（提交之前即在图上提示，使用与计算时相同的 B- 编号）：
    在其他画面比例上测得的镜头不适用于此画面（B-LENS-RASTER）。

    「镜头模型」未填写不属于接线问题：节点仍可连接，计算时输出恒等 ST-map 并在节点上提示
    （W-LENS-NOLENS），因为使用者通常先连接节点，再填写参数或接线。"""

    def check(self, got: Seen, node: Checked) -> Msg | None:
        typed = distorts(table_of(str(node.params.get("lens_group") or "colmap"), str(node.params.get("lens_model") or "")))
        # 宽和高都填写时才构成画面尺寸（只填一个表示尚未填完，不构成约束）
        w, h = node.params.get("width"), node.params.get("height")
        raster = [w, h] if typed and w and h else None
        if raster and got.size and raster[0] * got.size[1] != raster[1] * got.size[0]:
            return Msg("B-LENS-RASTER", width=got.size[0], height=got.size[1],
                       lens_width=int(raster[0]), lens_height=int(raster[1]))
        return None


@dataclass(frozen=True)
class LensModelMatches(Expect):
    """接入的「镜头内参」的模型必须与本节点所选的「镜头模型」一致：
    不一致时在提交前拦下（B- 码），而不是在计算中途才发现。上游尚未计算（模型未知）时不提示。"""

    def check(self, got: Seen, node: Checked) -> Msg | None:
        value = (got.meta or {}).get("value") if got.meta else None
        if not isinstance(value, dict) or not value.get("model"):
            return None
        upstream = lens_identity(value)
        here = (str(node.params.get("lens_group") or "colmap"), str(node.params.get("lens_model") or ""))
        if not here[1] or upstream == here:
            return None
        return Msg("B-LENS-MODELMISMATCH", node=node.label, here=model_label(*here), upstream=model_label(*upstream), source=got.source)


class LensDistortion(Pasteable, LensSheet, NodeDef):
    id = "core.lens_distortion"
    category = "camera_tools"
    lens = LENS_ANY  # 本节点负责去除（或加回）畸变，从不假定针孔模型
    on_node = ("lens_model",)  # 节点上只显示最需要看到的一项：所用的镜头模型
    # 只实现「粘贴」方向（不向数据包写入 Nuke 文本）：从 3DE 导出的 .nk（或从 Nuke 复制的 LD_3DE4 节点）中
    # 读出整个镜头，一次填满此表。生产中镜头由 3DE 反求得出，手工抄写十几个数值既慢又容易出错。
    paste = "nuke"
    # 求解镜头的节点（COLMAP、AnyCalib、GeoCalib）输出三项：Focal Length、Filmback 各一根连线接到这两个常驻参数端口，
    # 其余（模型、畸变系数、主点、像素比）打包为「镜头内参」接到下方的 `lens` 输入端口（nodes/lens.py packed_lens）。
    wired_ports = ("focal_mm", "filmback_mm")
    # 「图像」可选：接入时从画面获取分辨率（及像素比），未接入时手动填写「画面宽度 / 画面高度」。
    # 输入端口的可用性取决于「镜头模型」：选择两种无畸变模型时「镜头内参」端口置灰且不可连接（DISTORTED，与 COLMAP
    # 端「镜头内参」输出端口置灰的条件相同）；接入后，表中的系数、主点、像素比置灰（见下方 Params 中的 sheet_field 各项）。
    # 接入的模型与此处所选不一致时在提交前拦下（LensModelMatches → B-LENS-MODELMISMATCH）。
    inputs = (Port("image", "image", "图像", alpha=True, optional=True, expects=(HasSource(),)),
              Port("lens", LENS, "镜头内参", optional=True, applies=DISTORTED, expects=(LensModelMatches(),),
                   help="上游解出来的镜头内参（COLMAP、AnyCalib、GeoCalib 的「镜头内参」口）：模型 + 畸变系数 + 主点 + 像素比一份。"
                        "接了它，表上的畸变参数、主点、像素比就从它来（变灰）；它的模型要和这里选的「镜头模型」一样，不一样提交前拦下"))
    # 视图下方的数值控件：显示所用镜头的四项参数（Focal Length、Filmback、镜头内参组、镜头模型）当前的值（填写的或由连线提供的）。
    # 这些值不作为透传输出端口：输出等于输入的端口无法表明是否经过隐式处理，需要这些值时从上游获取。
    strip = {"focal_mm": "Focal Length", "filmback_mm": "Filmback", "lens_group": "镜头内参组", "lens_model": "镜头模型"}
    # 两张 ST-map 是本节点唯一的输出。ST-map 本身不是该镜头的画面（没有镜头状态），但它携带烘焙所用的镜头：
    # 「STMap」据此变形得到的画面因此同样携带该畸变，在该画面上解出的相机交付时可以带回畸变，无需使用者重新填写。
    # 画布由节点自行决定（window="node"），不附加「尺寸随镜头扩边」的常驻提示。
    outputs = (
        Port("undistort_stmap", "image.2", "去畸变 ST-map", shape=Shape(window="node", lens="node")),
        Port("distort_stmap", "image.2", "加畸变 ST-map", shape=Shape(window="node", lens="node")),
    )

    class Params(LensSheetParams):
        # 默认使用有畸变的模型：这样新建节点时系数和内参不会全部置灰（置灰参数的端口无法操作，「镜头模型」也无法接入），
        # 且 AnyCalib 默认的「径向 k1」映射到核心表正是 SIMPLE_RADIAL，连接后两端一致。
        lens_model: str = P(
            "SIMPLE_RADIAL", label="镜头模型", group="镜头", widget="choice", choices_from=("lens_group",), derived_from=("lens_group",))
        # 系数行随默认模型一同给出：新建节点的面板中即显示 k，无需等待编辑器向服务器查询
        distortion: list[LensParamEntry] = P(
            [{"name": spec.name, "value": float(spec.default)} for spec in MODELS["SIMPLE_RADIAL"].params],
            label="畸变参数", widget="table", group="镜头", derived_from=("lens_group", "lens_model"), validate_default=True,
            applies=All(DISTORTED, Not(Wired("lens"))))
        # 主点和像素比：接入「镜头内参」时同样取自该输入（同一份声明，增加一条置灰条件；nodes/lens.py sheet_field）
        center_x_mm: float = sheet_field("center_x_mm", "lens")
        center_y_mm: float = sheet_field("center_y_mm", "lens")
        pixel_aspect: float = sheet_field("pixel_aspect", "lens")
        overscan: Literal["auto", "none", "5", "10", "20"] = P(
            "auto", label="扩边", group="畸变",
            option_labels={"auto": "自动", "none": "无", "5": "5%", "10": "10%", "20": "20%"})

    @classmethod
    def read_pasted(cls, text: str) -> dict:
        """一段 Nuke 文本 → 本节点的一组参数值。

        生产中镜头由 3DE 反求得出，交付形式为 Nuke 脚本或一段 LD_3DE4 节点文本；手工抄写
        十几个数值既慢又容易出错，因此整个镜头一次填满，一步即可撤销。

        本函数不解析 Nuke 语法：解析在 `formats/nuke/parse.py` 中完成，
        此处只负责将解析结果映射到本表的参数。文本中有几个镜头即为几个，
        多于一个时不做猜测（E-NUKE-PASTEMANY）：一次粘贴一个镜头由使用者明确，猜错则会产生无提示的错误数值。

        画面宽高不从文本中推测：LD_3DE4 的参数中不含宽高，3DE 导出的脚本中即使有 Reformat 也可能是代理
        尺寸。留空时，接入「图像」则跟随画面，未接入则按镜头表手动填写（由 N-NUKE-PASTEDRASTER 提示）。
        """
        from ...formats.nuke import parse

        found = parse.lens_nodes(text)
        if not found:
            raise Invalid(Msg("E-NUKE-PASTENOLENS"))
        if len(found) > 1:
            raise Invalid(Msg("E-NUKE-PASTEMANY", count=len(found), names=[f["name"] for f in found]))
        # 宽高仅用于让解析器把系数置于同一坐标系；此处不采用其读回的 raster（见文档字符串）
        got = parse.lens(text, found[0]["name"], 1, 1)
        out: dict = {"lens_group": "3de4", "lens_model": got["model"],
                     "distortion": [{"name": n, "value": float(v)} for n, v in got["params"].items()
                                    if not isinstance(v, dict)]}
        if got.get("focal_mm") is not None and not isinstance(got["focal_mm"], dict):
            out["focal_mm"] = float(got["focal_mm"])
        if got.get("filmback_mm"):
            out["filmback_mm"] = float(got["filmback_mm"][0])  # 该表只有水平宽度：高度由画面比例决定
        if got.get("center_mm"):
            out["center_x_mm"], out["center_y_mm"] = float(got["center_mm"][0]), float(got["center_mm"][1])
        if got.get("pixel_aspect") is not None:
            out["pixel_aspect"] = float(got["pixel_aspect"])
        return out

    @classmethod
    def _wired_lens(cls, ctx) -> tuple[dict, str] | None:
        """接入的「镜头内参」（值及提供它的节点名称），未接入或接入为空时为 None。"""
        pk = ctx.input("lens")
        if pk is None or pk.meta.get("empty"):
            return None
        value = read_value(pk).value
        if not isinstance(value, dict):
            return None
        from ..registry import node_types

        by = node_types().get(str(getattr(pk, "node", "") or ""))
        return value, (by.label if by is not None else "接进来的镜头内参")

    @classmethod
    def _level(cls, ctx) -> dict:
        """镜头的来源：节点上填写的是他人测量的镜头表；由连线（「镜头内参」输入）提供的是该节点的求解结果，
        属于估计值，数据包会注明这一点。"""
        wired = cls._wired_lens(ctx)
        return {"level": "estimated", "by": wired[1]} if wired else {"level": "measured", "by": "节点上填的镜头表"}

    @classmethod
    def _sheet(cls, ctx) -> dict:
        """本次使用的镜头表：取节点上填写的值；接入「镜头内参」时系数、主点、像素比改用其值（模型不一致已在提交前拦下，
        若在此处仍出现不一致，说明绕过了页面，同样拒绝，B-LENS-MODELMISMATCH）。"""
        params = dict(ctx.params)
        wired = cls._wired_lens(ctx)
        if wired is None:
            return params
        model, coeffs, center, aspect = unpacked_lens(wired[0])
        here = (str(params.get("lens_group") or "colmap"), str(params.get("lens_model") or ""))
        if lens_identity(wired[0]) != here:
            raise Invalid(Msg("B-LENS-MODELMISMATCH", node=cls.label, here=model_label(*here),
                              upstream=model_label(*lens_identity(wired[0])), source=wired[1]))
        params["distortion"] = [{"name": n, "value": float(v)} for n, v in coeffs.items()]
        params["center_x_mm"], params["center_y_mm"], params["pixel_aspect"] = center[0], center[1], aspect
        return params

    @classmethod
    def said_shot(cls, params):
        """两张 ST-map 携带烘焙所用的镜头：它完全由节点参数确定，因此在计算之前即可声明（契约据此核对，
        引擎也据此填写）。「STMap」据此变形得到的画面同样携带该畸变，在其上解出的相机交付时可以带回畸变，
        无需使用者重新填写。"""
        lens = cls.lens_of(params) or {}
        if lens and "raster" not in lens and params.get("width") and params.get("height"):
            lens["raster"] = [int(params["width"]), int(params["height"])]
        return {"lens": lens}

    @classmethod
    def info(cls, params, inputs):
        """映射图覆盖的帧：接入画面时为画面的帧，否则整个镜头共用一张。"""
        return inputs["image"][0] if inputs.get("image") else Info()

    @classmethod
    def cook(cls, ctx):
        """镜头 → 两张 ST-map。不读取任何像素：画面仅用于获取分辨率（未接入画面时使用「画面宽度 / 画面高度」）。"""
        from ...data import lens_models as L

        src = ctx.input("image")
        size = cls._raster(ctx, src)
        if size is None:  # 无法确定该镜头适用的画面尺寸：输出空结果，并在节点上说明下一步（空结果不属于错误）
            ctx.say("N-LENS-NOPLATE", node=cls.label)
            return {port: empty_packet(ctx, port) for port, _ in STMAP_PORTS if port in ctx.wanted}
        sheet = cls._sheet(ctx)  # 节点上填写的值；接入「镜头内参」时系数、主点、像素比取自该输入
        lens_typed = cls.lens_of(sheet) or None
        if lens_typed is not None:
            lens_typed.setdefault("raster", list(size))  # 未填写「画面宽高」表示该畸变即在此画面上测得
            lens_typed["source"] = cls._level(ctx)
        # 未填写任何系数（新建节点即为此状态：已选模型，尚无数值）：输出恒等 ST-map，并在节点上提示从何处获取这些数值。
        # 这不属于错误（空结果不属于错误），因为使用者通常先连接节点，再接线或填写数值。
        # （仅对有系数的模型如此判断：SIMPLE_FISHEYE / FISHEYE 没有系数，其等距投影本身即为畸变。）
        if lens_typed is not None and MODELS[cls.table_model(sheet)].params and not any(float(row["value"]) for row in sheet["distortion"] or []):
            lens_typed = None
        if lens_typed is None:
            ctx.say("W-LENS-NOLENS", node=cls.label)
            identity = L.identity_stmap(*size)
            plan = Plan({None: Maps(identity, identity)}, Window(*size), {})
        else:
            plan = cls._from_lens(ctx, lens_typed, size)
        return cls._stmaps(ctx, src, plan, size)

    @classmethod
    def _raster(cls, ctx, src) -> tuple[int, int] | None:
        """该镜头适用的画面尺寸：接入画面时为画面尺寸，未接入（或接入为空）时为填写的
        「画面宽度 / 画面高度」；两者都没有时为 None。"""
        if src is not None and src.meta.get("width"):
            return int(src.meta["width"]), int(src.meta["height"])
        w, h = ctx.params.get("width"), ctx.params.get("height")
        return (int(w), int(h)) if w and h else None

    # -- 由镜头生成的映射图

    @classmethod
    def _from_lens(cls, ctx, typed: dict, size: tuple[int, int]) -> Plan:
        """节点上填写的镜头在该尺寸画面上的映射：去畸变图覆盖其「扩边」所需的画布，加畸变图即原画面框本身。"""
        from ...data import lens_models as L

        lens = L.Lens.from_meta(typed)
        for said in lens.range_notes():
            ctx.say(said.code, **said.params)
        ctx.say(LEVEL_SAID[lens.source.get("level", "measured")], by=lens.source.get("by") or lens.label)
        lens = lens.on_plate(*size)
        canvas, capped = L.fit_canvas(lens, [None], OVERSCAN[ctx.params["overscan"]])
        pair = L.lens_stmaps(lens, canvas, None)
        if pair.folded:
            ctx.say("W-LENS-FOLD", count=pair.folded)
        if capped:
            ctx.say("W-LENS-OVERSCANCAP", cap=L.OVERSCAN_CAP, outside=pair.outside)
        return Plan({None: Maps(pair.undistort, pair.distort)}, canvas, lens.meta())

    @classmethod
    def _stmaps(cls, ctx, src, plan: Plan, size: tuple[int, int]) -> dict:
        """两张 ST-map，32 位浮点 EXR：整段共用一张（镜头为一组固定值；变焦镜头使用「STMap」接逐帧序列）。"""
        from lab2shot_worker.files import write_exr

        from ...data.payloads import UNIT, channel_names, still_packet

        frames = src.meta["frames"] if src is not None else (1001,)
        maps = plan.maps[None]
        out = {}
        for port, direction in STMAP_PORTS:
            if port not in ctx.wanted:
                continue
            # 去畸变图覆盖画布（原画面框为其显示窗口）；加畸变图即原画面尺寸
            window = plan.canvas if direction == "undistort" else Window(*plan.canvas.plate)
            values = maps.undistort if direction == "undistort" else maps.distort
            path = ctx.outputs[port] / f"frame.{frames[0]}.exr"
            write_exr(path, values.astype(np.float32), channel_names(2), half=False,
                      windows=window.exr_windows)
            packet = still_packet(ctx.outputs[port], path, frames, *window.plate, channels=2, value_range=UNIT,
                                  window=window)
            packet.meta["direction"] = direction  # 标明方向：合成端据此判断该图用于去畸变还是加畸变
            # 烘焙所用的镜头由 said_shot 声明、由引擎填写（契约：节点自行写入的内容必须先声明）
            out[port] = packet
        return out


NODES = (LensDistortion,)
