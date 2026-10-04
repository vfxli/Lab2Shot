"""镜头参数：所有接收镜头的节点遵循同一规则。先验是值，可手填或接线；显式优先于隐式；节点标明每个值的来源。

节点使用的 Focal Length 按以下顺序确定：
  1. 节点上已设置的 Focal Length，手填或接自任何输出浮点的节点（AnyCalib、「拆分相机」、「浮点」）；
  2. 否则取所接相机的值；
  3. 否则由方法自行估计（或采用方法假定的镜头，如 HaMeR、SMIRK）。
Filmback 同理：节点上的 Filmback，否则取相机的值，否则按 36 mm 全画幅。以毫米表示的 Focal Length 必须配合 Filmback
才有意义，换算为像素为 Focal Length (mm) / Filmback (mm) × 画面宽度。

同时给定相机与 Focal Length 时，相机提供位置和朝向，Focal Length 提供镜头，节点会标明这一点：参数通过
P(overrides=("camera",)) 声明，引擎为节点、参数面板、worker 任务和结果的来源记录生成相应说明（Engine.sources）。

「已知 Focal Length」与「Filmback」在每个具有这两个参数的节点上都有常驻输入口（`P(wired=True)`，见 nodes/base.py
`wired_ports`）。没有相机输入口的解算器（SAM 3D Body、Fast SAM 3D Body、HaMeR、SMIRK、TRAM、Pixel3DMM 及若干深度 /
重建方法）按以下优先级获取 Focal Length：① 节点上的「已知 Focal Length」（手填或接线）；② 方法的默认镜头
（`default_focal_mm`）；③ 方法自行估计（即界面上的「自动」）。若留空而采用第 ③ 种，人物按方法估计的 Focal Length
解算，而相机由其他节点解算，两者数值不一致会导致人物与画面无法对齐。常驻输入口用于接入已解算相机的 Focal Length：
「ViPE 相机解算」的「相机」→「拆分相机」的「Focal Length」→ 解算器的「已知 Focal Length」口。接线后节点底行标明
来源（「已知 Focal Length 35 mm · 来自 拆分相机」，engine/evaluation.py sources）。
此处增加的是参数输入口而非相机输入口：解算器的相机输入口仅提供给上游确实接收相机的方法；只使用内参的方法，内参即为两个普通参数。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .. import i18n

import numpy as np

from ..data import lens_models
from ..data.lens_models import MODELS, STMAP_MODEL, distortion, distorts
from ..data.units import FILMBACK_MM
from ..availability import All, Because, Cond, Not
from ..messages import Msg
from .applies import Param, Wired
from .base import NodeParams, P

# ------------------------------------------------------------------ 以下两个参数不设置 `assumed`
#
# 设置 `assumed=` 后，参数留空时 `engine/evaluation.py sources()` 会生成「没给，按……算」的说明，显示在节点下方的
# 提示区和参数面板该行下方。此处不需要该说明：「已知 Focal Length」的名称已表明「已知则填写，否则由方法估计」，
# 「Filmback」输入框中的占位值 36（`placeholder="36"`）已表明留空时按 36 计算。留空时的计算方式不受影响：`lens()`
# 仍按 `FILMBACK_MM` 计算，解算出的相机仍带有实际的 Focal Length 与 Filmback。


def focal_param(*, overrides: tuple[str, ...] = (), per_frame: bool = False,
                applies: Cond | None = None) -> Any:
    """接收镜头的节点的「已知 Focal Length」参数，留空表示自动。

    参数命名为「已知 Focal Length」，输出口命名为「Focal Length」：输出口传递的是 Focal Length 数据本身，使用最简洁的
    名称；参数是调用方提供给节点的先验值，需加以限定。COLMAP / GeoCalib / AnyCalib 同时具有该参数和解算出的 Focal Length
    输出口，使用同一名称将无法区分。所有使用 focal_param 的节点，该参数均命名为「已知 Focal Length」。

    节点自行换算该值（worker=False），并将得出的镜头（Lens）发送给 worker。`applies`：参数变灰的条件（有相机输入口的
    节点在相机接入后将其变灰，因为相机自带镜头，见 CameraLensParams）。
    `wired=True`：具有该参数的每个节点上都常驻其输入口，这是已解算相机的 Focal Length 进入无相机输入口的解算器的唯一
    显式途径（「拆分相机」的「Focal Length」→ 该输入口）。在此处而非各节点中声明的原因见模块文档与 NodeDef.wired_ports。

    名称在 param.focal_mm.label（各节点自己的在 node.<类型>.param.focal_mm.label）：对于自身也输出「Focal Length」的节点（标定节点：AnyCalib、GeoCalib、COLMAP），该参数表示调用方已知的
    Focal Length（上游支持将已知内参作为先验传入），输出口表示估计结果，二者含义不同，不使用同一名称。"""
    return P(None, unit="mm", gt=0, group="lens", worker=False, words="family.lens.focal_mm",
             overrides=overrides, per_frame=per_frame, applies=applies, wired=True)


def filmback_param(*, overrides: tuple[str, ...] = (), applies: Cond | None = None) -> Any:
    return P(None, unit="mm", gt=0, group="lens", worker=False, words="family.lens.filmback_mm",
             overrides=overrides, applies=applies, wired=True)


# ------------------------------------------------------------------ 手填的镜头表
# 对应 3DE 或标定程序提供的镜头表：模型及其系数、测量畸变时的画面尺寸、Focal Length、Filmback、主点。
# 「LensDistortion」的参数即这张表（LensSheetParams），镜头表只定义这一份。

# 可填写的模型：核心模型表中所有具有公式的模型。「ST-map（查表）」不在其中：测得的映射是一张图像，由「STMap」处理。
LENS_MODELS = tuple(m for m in MODELS if m != STMAP_MODEL)


# ------------------------------------------------------------------ 镜头内参组
#
# 一个组代表一种命名来源：COLMAP 使用其自身的模型名（表中的 id 即 COLMAP 官方名），AnyCalib 使用其 cam_id
# （simple_kb:4），3DE 使用镜头表上的三个模型名。组内每个模型名指向核心公式表（data/lens_models.py MODELS）中的
# 一个模型，并声明其系数按何种顺序对应该模型的参数。公式只有一份，名称随来源而定。
# 核心自带 3DE4 组；COLMAP 组及每个求解镜头的扩展各自声明组（Extension.lens_groups，如 AnyCalib、GeoCalib）。
# 「LensDistortion」按组列出模型，「镜头内参」值携带 (组, 模型名)，两端一致方可计算，不一致时在提交前拦截。
@dataclass(frozen=True)
class GroupModel:
    """组内的一个模型：名称、界面标签、对应的公式表模型，以及其系数依次对应的表模型参数。"""

    name: str
    table: str
    names: tuple[str, ...]  # 表模型的参数名，按本组系数的顺序排列；未列出的表参数取默认值（0）

    @property
    def label(self) -> str:
        """界面标签（当前语言）：lens.model.<name>，否则公式表模型的标签（核心组：名称即表 id），否则名称。"""
        from .. import i18n

        return i18n.Both.of(lambda: i18n.lookup(f"lens.model.{self.name}")
                            or (MODELS[self.table].label if self.table in MODELS else self.name))  # every language (messages)


@dataclass(frozen=True)
class LensGroup:
    id: str
    label: str
    models: Mapping[str, GroupModel]
    # the group 「LensDistortion」 starts on (listed first): its extension says so; the core names no extension
    default: bool = False

    def model(self, name: str) -> GroupModel | None:
        return self.models.get(str(name or ""))

    def model_label(self, name: str) -> str:
        """一个模型的界面标签（当前语言）：lens.model.<名称>；否则组所属扩展里列出这些模型的节点参数的选项名
        （node.<类型>.param.fit_model.option.<名称>，与标定节点的「镜头模型」同一份词）；否则 GroupModel 的（公式表模型的标签）。"""
        from .registry import node_types

        gm = self.models.get(str(name or ""))
        said = i18n.lookup(f"lens.model.{name}")
        if said is not None or gm is None:
            return said or str(name)
        for t in node_types().values():
            if t.runtime != self.id:
                continue
            spec = next((p for p in t.param_specs() if (p["options"] or ()) and name in p["options"]
                         and (p.get("option_labels") or {}).get(name)), None)
            if spec is not None:
                return spec["option_labels"][name]
        return gm.label


def _core_group(gid: str, label: str, ids: tuple[str, ...], default: bool = False) -> LensGroup:
    return LensGroup(gid, label, MappingProxyType({m: GroupModel(m, m, MODELS[m].names) for m in ids if m in MODELS}),
                     default)


# 核心仅自带 3DE4 组（手填或从 Nuke 粘贴）。COLMAP 组由 COLMAP 扩展声明（adapters/colmap/lens.py），其列表即
# 「COLMAP 相机解算」节点提供的选项；「LensDistortion」中选择 colmap 组后列出的模型与该节点的数量、名称完全一致，
# 二者来源相同，因此不会出现差异。
THREE_DE_GROUP = _core_group("3de4", "3DE4", tuple(m for m in LENS_MODELS if MODELS[m].family == "3de4"))
CORE_GROUPS = (THREE_DE_GROUP,)


def core_group(gid: str, label: str, ids: tuple[str, ...], default: bool = False) -> LensGroup:
    """以核心公式表 id 命名模型的组（供 COLMAP 扩展使用，其模型名即表 id）。"""
    return _core_group(gid, label, ids, default)


def lens_groups() -> Mapping[str, LensGroup]:
    """全部镜头内参组：核心自带的组，加上各扩展声明的组（Extension.lens_groups）。未安装或加载失败的扩展不提供组。"""
    found: dict[str, LensGroup] = {}
    try:
        from ..extensions import extensions

        for ext in extensions().values():
            for g in getattr(ext, "lens_groups", ()) or ():
                found.setdefault(g.id, g)
    except Exception:  # 扩展尚未加载（导入期间或命令行工具中）：仅返回核心的组
        pass
    # 解算器的组在前（声明 default 的居首，为 LensDistortion 的默认值：COLMAP），手填的 3DE4 在最后
    ordered = sorted(found.values(), key=lambda g: (not g.default, g.label))
    return {g.id: g for g in (*ordered, *CORE_GROUPS)}


def group_of(params: Mapping) -> LensGroup:
    """The lens group a node's 「镜头内参组」 names; left empty (or one not installed): the group that declares itself
    the default (LensGroup.default, listed first by lens_groups), the core naming none."""
    groups = lens_groups()
    return groups.get(str(params.get("lens_group") or "")) or next(iter(groups.values()))


def group_label(gid: str) -> str:
    """A group's name as shown (「COLMAP」): its own declaration's label; one not installed shows its id."""
    g = lens_groups().get(str(gid or ""))
    return g.label if g is not None else str(gid)


lens_models.name_groups(group_label)  # a lens value is said with its group's name (data/values.py, data/lens_models.py)


def group_model(group: str, name: str) -> GroupModel | None:
    """The model `name` of lens group `group` (empty: the default group, group_of); None when the group has no such."""
    if not group:
        return group_of({}).model(name)
    g = lens_groups().get(str(group))
    return g.model(name) if g is not None else None


def table_of(group: str, name: str) -> str:
    """(组, 模型名) → 公式表的模型 id。无法识别时按无畸变（SIMPLE_PINHOLE）处理，由校验步骤报告。"""
    gm = group_model(group, name)
    return gm.table if gm is not None else "SIMPLE_PINHOLE"


def model_label(group: str, name: str) -> str:
    gm = group_model(group, name)
    g = lens_groups().get(str(group or ""))
    return f"{g.label if g else group} · {g.model_label(name) if g and gm else name}"


@dataclass(frozen=True)
class LensDistorts(Cond):
    """条件：「镜头模型」（按「镜头内参组」解释）带有畸变。系数表、主点、「镜头内参」输入口仅在此时生效。"""

    def holds(self, f) -> bool | None:
        from .applies import WIRED

        group, name = f.params.get("lens_group"), f.params.get("lens_model")
        if group is WIRED or name is WIRED:
            return None
        return distorts(table_of(str(group or ""), str(name or "")))

    def why(self, f, t) -> Msg:
        return Msg("I-APPLIES-LENSDISTORTS")

    def names(self):
        return frozenset({"lens_group", "lens_model"}), frozenset(), frozenset()


DISTORTED = LensDistorts()  # 所选模型带有畸变


# ------------------------------------------------------------------ 「镜头内参」：镜头除 Focal Length、Filmback 之外的全部内参，合为一份
# Focal Length 与 Filmback 因使用频繁而单独提供；其余内参合为一份，接收端核对模型，不一致时报错。
# 数据类型为 value.lens（data/types.py）：{"group": 镜头内参组, "model": 组内的模型名, "table": 公式表的模型 id,
# "params": {系数名: 数值, 以及主点和像素比}}（packed_lens）。
# 系数使用模型自身的名称（COLMAP 的 k、k1、p1 等），主点和像素比使用镜头表上的三个固定键。接收端（「LensDistortion」）
# 接入后其「镜头内参组」「镜头模型」、系数、主点、像素比都取这份值（nodes/core/lens_distortion.py params_from_input）。
SHEET_KEYS = lens_models.SHEET_KEYS  # 镜头表上随系数一并传递的三项（值的定义在数据层：data/lens_models.py）


def packed_lens(dist: dict, center_mm=(0.0, 0.0), pixel_aspect: float = 1.0, *, group: str, name: str | None = None) -> dict:
    """由 {"model", "params"}（data/lens_models.py distortion() 的返回值，使用公式表的模型）、主点、像素比以及组和模型名
    构造一份「镜头内参」的值：{"group", "model"（组内名称）, "table"（公式表 id）, "params"（表参数名及主点、像素比）}。
    未提供 `name` 时使用表 id 本身（COLMAP 组的模型名即表 id）。"""
    return {"group": str(group), "model": str(name if name is not None else dist["model"]), "table": str(dist["model"]),
            "params": {**{str(k): float(v) for k, v in (dist.get("params") or {}).items()},
                       "center_x_mm": float(center_mm[0]), "center_y_mm": float(center_mm[1]), "pixel_aspect": float(pixel_aspect)}}


def lens_identity(value: dict) -> tuple[str, str]:
    """「镜头内参」值所属的 (组, 模型名)。"""
    return str(value["group"]), str(value["model"])


def unpacked_lens(value: dict) -> tuple[str, dict, tuple[float, float], float]:
    """「镜头内参」值 → (公式表的模型 id, 该模型的系数, 主点 (x, y)（毫米）, 像素比)。缺失的键分别取 0 / 1。"""
    model = str(value["table"])
    got = dict(value.get("params") or {})
    names = MODELS[model].names if model in MODELS else tuple(k for k in got if k not in SHEET_KEYS)
    coeffs = {n: float(got.get(n, 0.0)) for n in names}
    return (model, coeffs, (float(got.get("center_x_mm", 0.0)), float(got.get("center_y_mm", 0.0))),
            float(got.get("pixel_aspect", 1.0) or 1.0))


def sheet_field(name: str, wired: str):
    """镜头表的某一项（沿用 LensSheetParams 中的声明），并附加条件「接入 `wired` 输入口时变灰」：
    接入「镜头内参」时，系数、主点、像素比均取自该值，表中对应各项不再使用（变灰并显示原因，位置不变）。
    「LensDistortion」用它覆盖自身的对应项；LensSheetParams 中的原声明保持不变。"""
    import copy

    field = copy.deepcopy(LensSheetParams.model_fields[name])
    extra = field.json_schema_extra
    assert extra is not None and hasattr(extra, "applies")
    base = extra.applies
    extra.applies = All(base, Not(Wired(wired))) if base is not None else Not(Wired(wired))
    return field


class LensParamEntry(NodeParams):
    """畸变参数表的一行：参数名（沿用测量软件的命名，如 3DE；行随「镜头模型」变化）及镜头表上的数值。"""

    name: str = P(..., widget="fixed")
    value: float = P(0.0)


class LensSheetParams(NodeParams):
    """镜头表的参数，按面板顺序排列。画面宽度 / 高度留空表示跟随输入画面（lens_of 此时不含「raster」）。"""

    # 默认「无畸变」（SIMPLE_PINHOLE）；「LensDistortion」将其覆盖为 SIMPLE_RADIAL（见其 Params）。
    #
    # 命名为 `lens_model` 而非 `distortion_model`：它是组内的模型名，与「镜头内参」值里的 `model`
    # （packed_lens；「AnyCalib 镜头标定」「GeoCalib 镜头标定」「COLMAP 相机解算」的「镜头内参」输出口交出的）
    # 是同一个名字，界面标签同为「镜头模型」。
    # 它与 AnyCalib / GeoCalib / COLMAP 上的 `fit_model` 含义不同，因此分别命名：`fit_model`「拟合模型」表示
    # 要求按何种模型拟合（请求），本参数表示模型本身（结果，用于去畸变 / 加畸变）。
    # 两级下拉（见上文「镜头内参组」）：先选择组（COLMAP / AnyCalib / GeoCalib / 3DE4），
    # 再选择该组的模型名。二者均为 choice 控件，选项由 LensSheet.choices 按已安装的扩展提供，不在代码中写死。
    lens_group: str = P(
        "", group="lens", widget="choice")  # empty: group_of's default
    # derived_from=("lens_group",)：切换组时重新计算该值（不在新组中则改为该组的第一个模型，见 LensSheet.derive），
    # 否则旧值会以「· 找不到了」显示在列表首行。
    lens_model: str = P(
        "SIMPLE_PINHOLE", group="lens", widget="choice", choices_from=("lens_group",), derived_from=("lens_group",))
    distortion: list[LensParamEntry] = P(
        [], widget="table", group="lens", derived_from=("lens_group", "lens_model"), validate_default=True,
        applies=DISTORTED)
    # 接入「图像」后以下两个参数变灰，使用框架提供的参数与输入互斥机制（Param.applies + Wired）。
    # 未接入图像时可手填，使节点可独立使用：从 3DE 抄录一组镜头参数，无需素材即可烘焙 ST-map。
    # 画面宽高即烘焙出的 ST-map 尺寸（ST-map 的一个像素对应画面的一个像素），因此不另设「输出分辨率」参数。
    # 命名为 `width` / `height`，与项目中其他节点的「画面宽度 / 画面高度」参数一致。
    width: int | None = P(None, unit="px", gt=0, group="lens",
                          applies=All(DISTORTED, Not(Wired("image"))))
    height: int | None = P(None, unit="px", gt=0, group="lens",
                           applies=All(DISTORTED, Not(Wired("image"))))
    pixel_aspect: float = P(1.0, gt=0, group="lens", applies=DISTORTED)
    # 参数统一命名为「已知 Focal Length」（见 focal_param：输出口名为「Focal Length」，参数为调用方提供的先验值）
    focal_mm: float | None = P(None, unit="mm", gt=0, group="lens")
    filmback_mm: float = P(FILMBACK_MM, unit="mm", gt=0, group="lens")
    # 使用「主点」一词：与标定程序的输出以及 AnyCalib、COLMAP 输出口的命名一致，不使用「镜头中心」。
    center_x_mm: float = P(0.0, unit="mm", group="lens", applies=DISTORTED)
    center_y_mm: float = P(0.0, unit="mm", group="lens", applies=DISTORTED)


class LensSheet:
    """用于参数为镜头表的节点（「LensDistortion」）的 mixin：提供所选模型的参数行，并将镜头表读回为
    镜头描述。"""

    @classmethod
    def table_model(cls, params: dict) -> str:
        """(镜头内参组, 镜头模型) → 公式表的模型 id。"""
        return table_of(group_of(params).id, str(params.get("lens_model") or "SIMPLE_PINHOLE"))

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        """「镜头内参组」的选项（取决于已安装的扩展）以及当前组内「镜头模型」的选项。"""
        groups, g = lens_groups(), group_of(params)
        return {"lens_group": {"options": list(groups), "labels": {k: v.label for k, v in groups.items()}, "auto": g.id},
                "lens_model": {"options": list(g.models), "labels": {k: g.model_label(k) for k in g.models}}}

    @classmethod
    def derive(cls, params: dict) -> dict:
        """切换组时，若「镜头模型」不在该组中则改为该组的第一个模型；随后按所选模型重建畸变参数行，
        同名参数保留已填写的值。"""
        g = group_of(params)
        name = str(params.get("lens_model") or "")
        if name not in g.models:
            name = next(iter(g.models))
        params = {**params, "lens_model": name}
        typed = {row["name"]: row["value"] for row in params.get("distortion") or [] if isinstance(row, dict)}
        model = MODELS.get(cls.table_model(params))
        return {"lens_model": name,
                "distortion": [{"name": spec.name, "value": float(typed.get(spec.name, spec.default))}
                               for spec in (model.params if model else ())]}

    @classmethod
    def lens_of(cls, params) -> dict:
        """将镜头表转换为镜头描述；未填写畸变时返回 {}。画面宽度 / 高度留空时不含「raster」，由使用方按
        实际画面尺寸补充。"""
        model = cls.table_model(params)
        if not distorts(model):
            return {}
        # 仅取该模型自身的参数，缺失的按 0：切换模型后表格中可能仍保留上一个模型的行（编辑器需请求服务器才会重算
        # derive()，命令行和节点图文件则按原样提供）。derive() 采用相同规则。
        typed = {row["name"] if isinstance(row, dict) else row.name: row["value"] if isinstance(row, dict) else row.value
                 for row in params["distortion"] or []}
        rows = {spec.name: float(typed.get(spec.name, spec.default)) for spec in MODELS[model].params}
        raster = [params["width"], params["height"]] if params.get("width") and params.get("height") else None
        focal = params.get("focal_mm")
        return {"distortion": distortion(model, rows), **({"raster": raster} if raster else {}),
                **({"focal_mm": float(focal)} if focal else {}), "filmback_mm": float(params["filmback_mm"]),
                "pixel_aspect": float(params["pixel_aspect"]),
                "center_mm": [float(params["center_x_mm"]), float(params["center_y_mm"])],
                "source": {"level": "measured", "by": "sheet"}}  # by: lens.by.<by> when it has words (a name like COLMAP as it is)


def takes_lens(node_type) -> bool:
    """Whether a node takes a lens: its parameters have the Focal Length (LensParams, CameraLensParams)."""
    return "focal_mm" in node_type.Params.model_fields


class LensParams(NodeParams):
    """由参数提供镜头的解算器（无相机输入口）：Focal Length 与 Filmback。"""

    focal_mm: float | None = focal_param()
    filmback_mm: float | None = filmback_param()


class SolvedLensParams(LensParams):
    """自身也输出「Focal Length」的节点（标定与解算：AnyCalib、GeoCalib、COLMAP），参数命名为「已知 Focal Length」。

    参数表示调用方已知的 Focal Length（填写后固定使用，不再估计；上游均支持将已知内参作为先验传入），
    输出口表示计算结果，二者含义不同，不使用同一名称。"""

    focal_mm: float | None = focal_param()


# 接入相机后使用相机自带的镜头，节点上这两个参数变灰：接入的相机已经过跟踪部门质检，节点不得在使用者不知情的情况下
# 修改其 Focal Length 或重建后输出（这两个参数不声明 overrides=("camera",)，kit/cameras.py pass_camera
# 也不按参数重建相机）。如需修改已解算相机的 Focal Length，应使用「锁定 Focal Length」（lock_focal），
# 它会同时重算相机位置，以保证与画面对齐。
CAMERA_HAS_LENS = Because(Not(Wired("camera")), "N-LENS-CAMERAHASIT")


class CameraLensParams(NodeParams):
    """有相机输入口的节点：相机自带镜头，因此相机接入时 Focal Length 与 Filmback 变灰。"""

    focal_mm: float | None = focal_param(applies=CAMERA_HAS_LENS)
    filmback_mm: float | None = filmback_param(applies=CAMERA_HAS_LENS)


def without_camera_conditions(params: type[NodeParams]) -> type[NodeParams]:
    """去除所有「接了相机就变灰」条件后的同一组参数，用于没有相机输入口的节点（families/humans.py 仅在方法确实
    接收相机时保留该输入口），此时不存在可据以变灰的输入。Focal Length / Filmback / 固定机位均按此处理：凡条件中
    引用「camera」输入口的参数均清除条件，不逐个列出名称（逐个列出易遗漏；若某参数的条件仍引用已删除的相机输入口，
    节点将无法加载）。

    返回新的模型类，而不是就地清除字段的 `applies`：所有继承 CameraLensParams 的节点共享同一个 FieldInfo 对象，
    就地修改会使其他节点上的参数也不再变灰。规则是：有相机输入口的节点一律使用相机的镜头并将参数变灰。
    各节点自身的 per_frame 等属性均保留：复制字段，仅去除 `applies`。"""
    import copy

    from pydantic import create_model

    fields: dict[str, Any] = {}
    for name, field in params.model_fields.items():
        cond = getattr(field.json_schema_extra, "applies", None)
        if cond is None or "camera" not in cond.names()[1]:
            continue
        field = copy.deepcopy(field)
        field.json_schema_extra.applies = None
        fields[name] = (field.annotation, field)
    if not fields:
        return params
    return create_model(f"{params.__name__}WithoutCamera", __base__=params, **fields)


@dataclass(frozen=True)
class Lens:
    """节点使用的镜头及其来源。"""

    focal_px: float | None  # 按画面宽度计（随帧变化时取中位数）；None 表示由方法估计
    filmback_mm: float
    width: int  # focal_px 对应的画面宽度
    source: str  # "param"（手填或接线）、"camera"（所接相机）、"default"（方法假定值）、"estimate"
    said: str  # 节点显示的说明，如 "Focal Length 38.6 mm · 来自 AnyCalib 镜头标定（覆盖相机的 Focal Length）"
    frames: tuple[int, ...] = ()  # 画面的帧号，与 per_frame 对应
    per_frame: tuple[float, ...] = ()  # Focal Length 随帧变化（变焦）时各帧的值（px）

    @property
    def focal_mm(self) -> float | None:
        from ..data import units

        return None if self.focal_px is None else float(units.focal_mm(self.focal_px, self.filmback_mm, self.width))

    def focal_at(self, frames) -> np.ndarray:
        """给定帧上的 Focal Length（px）：变焦时插值，超出范围时保持端点值。"""
        if self.per_frame:
            return np.interp(np.asarray(list(frames), np.float64), self.frames, self.per_frame)
        return np.full(len(list(frames)), float(self.focal_px))

    @property
    def fov_x_deg(self) -> float | None:
        """水平视场角（度）。"""
        from ..data import units

        return None if self.focal_px is None else units.fov_x_deg(self.focal_px, self.width)

    @property
    def given(self) -> bool:
        """Focal Length 是否已知（来自节点、相机或方法默认值），而非交由方法估计。"""
        return self.focal_px is not None

    def record(self) -> dict | None:
        """写入 worker 任务和结果来源记录的内容；不接收镜头的节点（NO_LENS）返回 None。"""
        if self.source == "none":
            return None
        return {"focal_px": self.focal_px, "focal_mm": self.focal_mm, "filmback_mm": self.filmback_mm, "source": self.source,
                "said": self.said}

    def info(self) -> dict:
        """结果信息中的镜头说明 {"lens": said}；不接收镜头的节点（NO_LENS）返回空字典。"""
        return {} if self.source == "none" else {"lens": self.said}


NO_LENS = Lens(None, FILMBACK_MM, 0, "none", "")  # 不接收镜头的节点（如 FaceAnything）


def lens(ctx, width: int, frames: list[int], *, default_mm: float | None = None, port: str = "camera") -> Lens:
    """按模块文档所述规则确定镜头，基于画面宽度 `width` 与帧 `frames`：优先取节点的 Focal Length / Filmback（ctx.params，
    接线值已由引擎填入；参数支持逐帧时，逐帧值在 ctx.values 中），否则取 `port` 上所接的相机（按帧采样），
    否则按 36 mm Filmback 使用 `default_mm`，否则 Focal Length 为 None。"""

    from ..data import units
    from ..data.camera import CameraSamples

    given = ctx.input(port)
    have: dict[str, np.ndarray] = {}
    if given is not None and given.type.startswith("scene"):
        s = CameraSamples.from_packet(given, list(frames) or [0])
        have = {"focal_mm": s.focal_mm, "back": s.h_aperture_mm}
    params = ctx.params
    back = params.get("filmback_mm") or (float(np.median(have["back"])) if have else FILMBACK_MM)
    frames = tuple(int(f) for f in frames)
    said = ctx.sources.get("focal_mm", "")
    if params.get("focal_mm") is not None:
        wired = ctx.values.get("focal_mm")
        per = tuple(units.focal_px(wired.at(frames), back, width)) if wired is not None and not wired.constant() and frames else ()
        return Lens(float(units.focal_px(params["focal_mm"], back, width)), back, width, "param",
                    said or i18n.Word("lens.said.typed", focal=f"{params['focal_mm']:g}"), frames, per)
    if have:
        focal = units.focal_px(have["focal_mm"], have["back"], width)
        per = tuple(float(f) for f in focal) if len(focal) > 1 and not np.allclose(focal, focal[0], rtol=1e-6) else ()
        return Lens(float(np.median(focal)), back, width, "camera", said or i18n.Word("lens.said.camera"), frames if per else (), per)
    if default_mm:
        return Lens(float(units.focal_px(default_mm, FILMBACK_MM, width)), back, width, "default", i18n.Word("lens.said.default", focal=f"{default_mm:g}"))
    return Lens(None, back, width, "estimate", i18n.Word("lens.said.estimate"))
