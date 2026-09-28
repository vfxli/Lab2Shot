"""节点之间传递的数据类型。

类型 id 以点分层：接受 "scene" 的输入口也接受 "scene.camera" 和 "scene.character"（相机是三维数据的一种），
反之不成立。颜色属于契约的一部分：使用者靠颜色识别连线，发布后不再改变。各表只读，扩展不能为其他扩展修改类型。

任一类型的列表写作该类型加 "[]"：「图像序列[]」表示多个图像序列，各有自己的名称。表中没有列表的条目
（列表是同一种数据的多份），`accepts` 只在需要列表的位置接受列表（`is_list`、`element_of`、`list_of`）：
列表与单值不能直接连接，编辑器会在中间提供「合成列表」「取一条」或「逐项开始」（engine/graph.py B-WIRE-LIST）。
列表只有一层：`X[][]` 会被拒绝（B-LIST-NESTED），「逐项结束」会展平内层块收集的结果。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType


# 类型在视图各舞台中的角色。视图仅依据这些角色组织节点的显示，节点不编写视图代码。
#   2D："picture" 显示的画面（每次一张），"overlay" 叠加在画面上，"strip" 显示在两个舞台下方的曲线条中。
#       三维数据在 2D 中没有角色（None）：只输出三维结果的节点切换到 2D 时原样播放上游画面；
#       「透过某台相机看」属于 3D 舞台的视角选择，不是一种 2D 角色
#   3D："element" 场景中的对象，"backplate" 透过相机看场景时位于其后，"points" 以点预览显示
#       （数据位于相机空间时需配合相机），"strip"
#   两者："inputs" 显示为其来源（输出设置节点的文件：其写出内容即节点的输入）；
#         "value" 基本数值：单个值直接显示（节点上也显示），逐帧值在曲线条中显示为曲线
ROLES_2D = ("picture", "overlay", "strip", "inputs", "value")
ROLES_3D = ("element", "backplate", "points", "strip", "inputs", "value")


@dataclass(frozen=True)
class DataType:
    id: str
    label: str
    color: str
    description: str
    in_2d: str | None = None  # ROLES_2D 之一
    in_3d: str | None = None  # ROLES_3D 之一
    # 有意设计的终端类型：说明为何只有输出设置节点接受它（没有节点以它为计算输入）。不做校验，
    # 该声明仅作为类型自身的说明
    end: str = ""
    # 包含多项内容的数据（人物框含多个人，场景含多个组）中单项的中文名称。命名、拆分与合并方式统一登记于
    # data/items.py；此处未声明的类型不能逐项处理（「逐项开始」会拒绝并说明原因）。不含任何项（没有人物、
    # 没有跟踪点）的数据，对不接受空数据的输入（Port.takes_empty）而言无可计算（items.holds_nothing）
    items: str = ""
    # 摘要包含的行（data/summary.py ITEMS 的 id），按显示顺序排列：输入口提示、中键信息面板和「取信息」均读取此项。
    # 类型也会显示其父类型的摘要行
    summary: tuple[str, ...] = ()
    # 多层 EXR 中该类型的默认图层名（「多层 EXR 输出设置」的图层表在新增行时建议此名：webui/src/graph/edit.ts，
    # 已被占用时追加编号）；"" 表示该类型不会成为 EXR 图层
    layer_default: str = ""


# ------------------------------------------------------------------ 类型的颜色
#
# 规则：
#   1. 暖色 = 三维数据，冷色 = 二维数据，灰 = Mask，白 = 文件，紫红色系 = 数值；
#   2. 同组成员使用明显不同的色相或明度，不使用色相的微小偏移（相近色相难以区分）：
#      相机黄、角色正红、骨架浅粉、模型赭石、点云橙、三维曲线酒红；Mask 灰、UV 青、RGB 蓝、RGBA 海军蓝、任意通道浅蓝；
#      人物框品红、2D 跟踪点绿、视频灰青；
#   3. 列表与单值同色：列表已有方形端口和双线区分，不再以饱和度区分；
#   4. 导入时校验：任意两个类型的色差（CIELAB ΔE76）≥ 18，任一类型与画布底色的色差 ≥ 35；新增类型或修改颜色时若颜色过近，
#      导入即报错。
# 色值仅在此表中定义；网页从目录读取，不另存副本。
COLOURS: Mapping[str, str] = MappingProxyType({
    # 三维：暖色
    "scene": "#D9A066", "scene.camera": "#FFD60A", "scene.character": "#FF3B30", "scene.skeleton": "#FFB4A8",
    "scene.model": "#8C4A1E", "scene.points": "#FF8C1A", "scene.curves": "#8B1A4A",
    "curves": "#B8E356",  # 动画曲线：黄绿色，介于冷暖之间，因其属于动画数据，兼具两类特征
    # 二维像素：冷色
    "image": "#9FD8FF", "image.1": "#B0B0B6", "image.2": "#2FD5C8", "image.3": "#3E8EF7", "image.4": "#1F3F99",
    "video": "#2B6F7A",
    # 二维叠加物
    "boxes": "#FF2D95", "tracks2d": "#34C759",
    # 数值：紫红色系，以明度区分
    "value": "#DEC0FF", "value.float": "#9D7CFF", "value.int": "#5B3FD6", "value.bool": "#F0A6FF",
    "value.vector": "#C43BC9", "value.text": "#E6E6B8", "value.lens": "#7A1F7A",
    "files": "#FFFFFF",
})
UNKNOWN_COLOUR = "#6B6B70"  # 目录中不存在的类型：比 Mask 的灰色暗一档（ΔE 27），不分配给任何类型
CANVAS_COLOUR = "#1C1C1E"  # 节点图画布的底色（webui 的 --window），用于颜色校验
MIN_DELTA_E, MIN_DELTA_E_CANVAS = 18.0, 35.0


def _lab(hexv: str) -> tuple[float, float, float]:
    """sRGB 十六进制 → CIELAB（D65）。仅用于下方的校验，不追求色彩科学上的精确。"""
    r, g, b = (int(hexv[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    r, g, b = lin(r), lin(g), lin(b)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116  # noqa: E731
    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def _delta_e(a: str, b: str) -> float:
    la, lb = _lab(a), _lab(b)
    return sum((p - q) ** 2 for p, q in zip(la, lb)) ** 0.5


def colour_of(type_id: str) -> str:
    """类型的颜色（COLOURS）；列表（`[]`）使用其元素的颜色。"""
    return COLOURS[type_id.removesuffix("[]")]


def _check_colours() -> None:
    """校验任意两种颜色足够不同、所有颜色与画布足够不同：新增或修改的颜色若与其他颜色过近，在导入时即报错，
    而不是在截图中才被发现。"""
    names = list(COLOURS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (d := _delta_e(COLOURS[a], COLOURS[b])) < MIN_DELTA_E:
                raise ValueError(f"data/types.py COLOURS: {a} and {b} read alike (ΔE {d:.1f} < {MIN_DELTA_E})")
        if (d := _delta_e(COLOURS[a], CANVAS_COLOUR)) < MIN_DELTA_E_CANVAS:
            raise ValueError(f"data/types.py COLOURS: {a} vanishes on the canvas (ΔE {d:.1f} < {MIN_DELTA_E_CANVAS})")
    if _delta_e(UNKNOWN_COLOUR, COLOURS["image.1"]) < MIN_DELTA_E:
        raise ValueError("data/types.py UNKNOWN_COLOUR reads like Mask")


_check_colours()

# 二维像素数据的四个成员：id 即通道数，名称沿用 CG 中的叫法（仅在此处定义，其他位置不得重复此表）
PIXELS: Mapping[int, tuple[str, str]] = MappingProxyType({
    1: ("Mask", "单通道：深度、视差、遮罩、置信度、粗糙度、金属度、分割编号……数值不是颜色，不做色彩转换"),
    2: ("UV", "双通道：ST-map（去畸变、加畸变，Nuke 通用的坐标图）、模型的 UV 坐标"),
    3: ("RGB", "三通道：照片、渲染、法线图、位置图——带色彩空间的按图片显示（走 OCIO），不带的按数值显示"),
    4: ("RGBA", "四通道：带 alpha 的画面（和 Nuke 一样预乘）、运动矢量（前后各 u v）"),
})
# 像素图「数据信息」面板显示的内容：尺寸、帧数等始终显示，其后各项仅在数据自身声明时显示
# （即契约 MEANING 中的含义：尺度、坐标系、置信度来源、UV 投影方式、类别表、去畸变或加畸变）
PIXEL_SUMMARY = ('size', 'frames', 'channels', 'colorspace', 'values', 'range', 'scale', 'space',
                 'model', 'projection', 'classes', 'direction', 'pixel_aspect', 'data_window')


DATA_TYPES: Mapping[str, DataType] = MappingProxyType({
    t.id: t
    for t in (
        DataType("video", "视频", colour_of("video"), "原始视频文件，只能接视频转序列", "picture", None, summary=('size', 'frames')),
        # 家族根类型：摘要在此定义，四个成员经 lineage 继承（items_of），不重复定义
        DataType("image", "图像", colour_of("image"), "二维像素数据的总称，几条通道都行（Mask、UV、RGB、RGBA）：只有不挑通道数的口用它，包上永远是具体的那一种", "picture", None,
                 summary=PIXEL_SUMMARY),
        # 视图中的角色同样只取决于通道数：任意通道数的数据均作为画面显示（单通道即灰度图，视图中可再选择通道、
        # 调整黑白点、着色、叠加第二通道，构成一条预览链，见 webui/src/model/view2d.ts）；单通道（每像素一个距离，
        # 配合相机反投影）和三通道（每像素一个坐标）均可作为点云查看，是否需要相机由数据包声明的 space 决定
        # （server/view_data.py points_view）
        *(DataType(f"image.{n}", label, colour_of(f"image.{n}"), why, "picture",
                   "points" if n in (1, 3) else None,
                   # 单通道中存放类别编号时（自带类别表）即为分割图，每一项是一个物体
                   # （data/items.py 也仅为 image.1 登记了拆分与合并方式）；其他 Mask 无法拆分为项，
                   # 「逐项开始」会以不含任何项为由拒绝并说明
                   items="物体" if n == 1 else "",
                   layer_default={1: "float", 2: "uv", 3: "rgb", 4: "rgba"}[n])
          for n, (label, why) in PIXELS.items()),
        # 二维像素数据本质上只有一种，区别仅在于通道数（参考 DCC 的设计，底层不为每种分别实现）。
        # 因此只有一个家族 image，四个成员按通道数生成（PIXELS，见上方循环），读写、视图、连线均按通道数处理。
        # 通道的含义由使用方式决定：接到「粗糙度」口即为粗糙度，接到「高光」口即为高光；含义体现在输入口名称、
        # 数据自带的信息（色彩空间、分割编号表）以及使用它的节点参数上
        DataType("boxes", "人物框", colour_of("boxes"), "每个人的编号，以及他在每一帧画面里的框", "overlay", None, items="人物", summary=('people', 'chosen', 'frames', 'size')),
        DataType("scene", "场景", colour_of("scene"), "统称：几种三维数据合在一起的 USD 场景（「合成场景」的结果），里面可以有模型、相机、点云、骨架动画、蒙皮角色和灯光", None, "element", items="组", summary=('contents', 'top', 'subsets', 'attributes', 'frames')),
        DataType("scene.camera", "相机", colour_of("scene.camera"), "USD 相机：Focal Length、Filmback、分辨率、每帧位置（不动的相机也有它的位置）", None, "element", summary=('focal', 'filmback', 'size', 'distortion')),
        DataType("scene.character", "蒙皮角色", colour_of("scene.character"), "USD 蒙皮角色：网格蒙皮在骨骼上，骨骼逐帧的动画，网格的 blend shape 和逐帧权重（Maya 的 skinCluster + blendShape），解算人体、手、脸的节点给的就是它", None, "element"),
        DataType("scene.skeleton", "骨架动画", colour_of("scene.skeleton"), "USD 骨架动画：只有关节层级和逐帧的关节变换，没有网格（动捕、重定向、补帧）", None, "element"),
        DataType("scene.model", "模型", colour_of("scene.model"),
                 "USD 模型：网格（UV、法线），静止的、跟着变换动的，或每帧变形的（点缓存）；"
                 "网格可以带分区（GeomSubset，网格的一部分，如头皮、脸、脖子），用「按分区取出」取其中一块",
                 None, "element"),
        DataType("scene.points", "点云", colour_of("scene.points"), "USD 点云：每帧的点、颜色和附加属性", None, "element",
                 summary=('scale', 'points_from', 'attributes')),
        # 发丝、毛发导向线、相机路径、运动轨迹在 DCC 中都是场景中的对象（USD BasisCurves、Houdini 的 curves、
        # Maya 的 nurbsCurve），因此作为场景的一个种类，而非顶层类型。它与顶层的「动画曲线」curves 不同：
        # 后者是逐帧的数值曲线（如 52 条表情权重），前者是三维空间中的曲线
        DataType("scene.curves", "三维曲线", colour_of("scene.curves"), "USD 三维曲线（BasisCurves）：每条曲线的点，以及逐点的宽度、颜色、朝向；发丝、毛发导向线、运动轨迹都是它",
                 None, "element", summary=('strands', 'attributes')),
        DataType("tracks2d", "2D 跟踪点", colour_of("tracks2d"), "画面上的一组点，每个点在每一帧的位置、那一帧是否被挡住，给分的跟踪器还带它对每个点每帧的置信度（点跟踪、面部关键点）", "overlay", None,
                 end="交给 3DEqualizer、Nuke 的 2D 跟踪点，由「2D 跟踪点输出设置」写出", items="组", summary=('points', 'frames', 'size')),
        DataType("curves", "动画曲线", colour_of("curves"), "逐帧的数值曲线，每条有名字，如 52 条表情曲线", "strip", "strip",
                 end="交给 DCC 的动画数据：表情权重写成 CSV、.chan 或 USD，相机对比的误差画成曲线", summary=('curves', 'frames')),
        DataType("files", "要输出的文件", colour_of("files"), "输出设置节点按它的名字和格式写好的文件，接到「输出」交给你：每个输出设置一个子文件夹", "inputs", "inputs",
                 end="写好的文件，只交给「输出」", summary=('name', 'main', 'files', 'commercial', 'learned')),
        # 基本数值（data/values.py）：单个值或逐帧值；单位（mm、px、°）属于数据的一部分
        DataType("value", "数值", colour_of("value"), "一个值，或者每帧一个值（带帧号）；单位写在数据里", "value", "value", summary=('value', 'frames')),
        DataType("value.float", "浮点", colour_of("value.float"), "一个小数，或者每帧一个（一条曲线），可以带单位（mm、px、°……）：Focal Length、Filmback、强度", "value", "value"),
        DataType("value.int", "整数", colour_of("value.int"), "一个整数，或者每帧一个，可以带单位：分辨率、帧号、数量", "value", "value"),
        DataType("value.bool", "布尔", colour_of("value.bool"), "开或关，或者每帧一个", "value", "value"),
        DataType("value.vector", "向量", colour_of("value.vector"), "三个小数 X Y Z，或者每帧一组，可以带单位：位置（cm）、朝向（°）", "value", "value"),
        DataType("value.text", "文字", colour_of("value.text"), "一段文字，或者每帧一段：名字、编号列表", "value", "value"),
        # 一颗镜头的内参合并为一份：Focal Length 和 Filmback 因常用而各有输入口；其余（畸变系数、主点）与模型名
        # 一起经同一条连线传递，两侧模型一致时才可计算。
        # 接收端（「LensDistortion」）将 model 与自身的镜头模型比对，不一致即拒绝，比使用多个浮点输入口更可靠
        DataType("value.lens", "镜头内参", colour_of("value.lens"), "一颗镜头的内参打成一份：镜头模型的名字 + 这个模型的系数（畸变系数、主点），"
                 "Focal Length 和 Filmback 不在里面（它们各有自己的口）。接「LensDistortion」的「镜头内参」，模型对得上才算", "value", "value"),
    )
})

# 抽象父类型：没有数据直接属于它们，只属于其子类型（输入口仍可声明为 "map"）
ABSTRACT_TYPES = frozenset({"value", "image"})

# ------------------------------------------------------------------ lists

LIST = "[]"
# 所有根类型，按通用输入口的列出顺序排列：「逐项开始」「切换」「命名」及 nodes/core/flow.py 的其他节点声明此项，
# 因为它们适用于任何数据（ANY_LIST 为对应的列表形式）。在这些根下新增的类型自动适用，无需修改这些节点的代码。
ANY_TYPES = ("value", "image", "video", "boxes", "tracks2d", "scene", "curves", "files")
ANY = "|".join(ANY_TYPES)
ANY_LIST = "|".join(t + LIST for t in ANY_TYPES)


def is_list(type_id: str) -> bool:
    return type_id.endswith(LIST)


def element_of(type_id: str) -> str:
    """列表中单项的类型（非列表类型返回自身）。"""
    return type_id.removesuffix(LIST)


def list_of(type_id: str) -> str:
    """该类型的列表类型；列表的列表即列表本身（「逐项结束」会展平）。"""
    return type_id if is_list(type_id) else type_id + LIST


def type_label(type_id: str) -> str:
    """类型的显示名称，包括候选类型和列表：「图像序列」「图像序列列表」「深度图或遮罩」。"""
    said = []
    for t in type_id.split("|"):
        kind = DATA_TYPES.get(element_of(t))
        said.append((kind.label if kind else element_of(t)) + ("列表" if is_list(t) else ""))
    return "或".join(said)


@dataclass(frozen=True)
class SceneKind:
    """三维数据的种类，按 DCC 的区分方式划分（模型、相机、点云、骨架动画、蒙皮角色、灯光）：包括其单独传递时的
    类型及其内容。场景只是多个种类合并后的统称。三维输入口传递的种类在计算前即可由节点图得知
    （Graph.scene_kinds）；三维输出设置节点按种类声明写出方式（OutputSettings.writes）。"""

    id: str
    label: str
    type: str  # 单独传递时的数据类型：决定其颜色及可承载的连线


# 按编辑器的列出顺序排列（支持的数据）。新增三维数据种类时在此添加一行，每个三维输出设置节点随后须声明其写出方式。
SCENE_KINDS: Mapping[str, SceneKind] = MappingProxyType({
    k.id: k
    for k in (
        SceneKind("model", "模型", "scene.model"),
        SceneKind("camera", "相机", "scene.camera"),
        SceneKind("points", "点云", "scene.points"),
        SceneKind("curves", "三维曲线", "scene.curves"),
        SceneKind("skeleton", "骨架动画", "scene.skeleton"),
        SceneKind("character", "蒙皮角色", "scene.character"),
    )
})
# 逐帧变化的模型（点缓存）：与 "model" 一同传递；写出节点可以只接受静止模型
DEFORMING = "model.deforming"
KIND_ORDER = (*SCENE_KINDS, DEFORMING)  # 三维连线可承载的内容，按编辑器的显示顺序


# 所有节点上输入口和输出口的排列顺序：输出口与其所连接的输入口位于同一高度，连线不交叉（如 ViPE 的深度、相机
# 连到 TAPIP3D 的图像、深度、相机）。类型归入最长匹配的条目（map.normal 归入 map）。只读；未列出的类型在声明
# 输入口时被拒绝（port_rank）。
# 三维数据（scene.*）排在一起，与 DCC 中的对象对应；参数类数据（SMPL 人体、动画曲线、数值）排在其后，
# 因此解算人体的节点上依次为「相机 / 人物 / 网格 / SMPL 人体」，三维交付物在前，算法参数在后
PORT_ORDER = ("video", "image", "boxes", "tracks2d", "scene.camera",
              "scene.character", "scene.skeleton", "scene.model", "scene.points", "scene.curves", "scene",
              "curves",
              "value", "files")


def port_rank(port_type: str) -> int:
    """该类型的输入口在 PORT_ORDER 中的位置：按第一个候选类型（"image|map" 按 image）的最长匹配条目；
    列表与其元素位置相同（「图像序列[]」与图像序列相同）。"""
    t = element_of(port_type.split("|")[0])
    entries = [e for e in PORT_ORDER if t == e or t.startswith(e + ".")]
    if not entries:
        raise ValueError(f"type {t!r} has no place in PORT_ORDER (data/types.py)")
    return PORT_ORDER.index(max(entries, key=len))


def in_port_order(ports: tuple) -> tuple:
    """排列输入口：必需的在前、可选的在后，各自再按 PORT_ORDER 排序，同档内保持声明顺序。

    这样排列是出于使用需要（两个节点接入同一批数据时，输入口应对齐、连线不交叉）：二维数据同属一个家族，
    仅按 PORT_ORDER 排列时，可选的「遮罩」会排到「深度图」与「相机」之间，破坏「ViPE 的深度、相机 → TAPIP3D
    的深度、相机」这类同高不交叉的连接。可选输入口本就可接可不接，排在必需输入口之后也更符合 DCC 的习惯。"""
    return tuple(sorted(ports, key=lambda p: (bool(p.optional), port_rank(p.type))))


def kind_of(data_type: str) -> str:
    """单一种类的类型所对应的种类（"" 表示统称的场景，或不是三维数据）。"""
    return next((k.id for k in SCENE_KINDS.values() if k.type == data_type), "")


def kind_label(kind: str) -> str:
    """编辑器中种类（或变形模型）的显示名称。"""
    return "变形的模型" if kind == DEFORMING else SCENE_KINDS[kind].label


def accepts(port_type: str, data_type: str) -> bool:
    """判断 `data_type` 的数据能否连接到声明为 `port_type` 的输入口。

    输入口可用 "|" 列出候选类型（如 "image|map"）。类型尚未确定的输出（跟随尚未连接的输入，Port.type_from）
    同样可以：只要其任一候选类型或其子类型可接入即可连接；其输入连接后，节点图会以实际类型重新检查该连线。

    类型尚未确定的输出携带家族根类型（「STMap」的「源」未连接时为 `image`）：可连接到该家族的任何输入口，
    输入连接后输出具有确定的通道数，节点图会重新检查该连线。

    列表只能连接到需要列表的位置，单值只能连接到需要单值的位置：「图像序列[]」可连接到声明为「图像序列[]」
    的输入口，不能连接到声明为「图像序列」的输入口；元素须经由编辑器在连线上提供的「逐项开始」或「取一条」
    传递（B-WIRE-LIST）。
    """
    if "|" in data_type:
        return any(accepts(port_type, d)
                   or any(is_list(t) == is_list(d) and element_of(t).startswith(element_of(d) + ".")
                          for t in port_type.split("|"))
                   for d in data_type.split("|"))
    listed = is_list(data_type)
    carried = element_of(data_type)
    return any(carried == t or carried.startswith(t + ".") or t.startswith(carried + ".") and carried in ABSTRACT_TYPES
               or _enough_channels(t, carried)
               for t in (element_of(p) for p in port_type.split("|") if is_list(p) == listed))


def channels_of(type_id: str) -> int:
    """二维像素数据的通道数（0 表示不是二维像素数据）。id 即通道数（data/types.py PIXELS）。"""
    root, _, n = element_of(type_id).partition(".")
    return int(n) if root == "image" and n.isdigit() else 0


def _enough_channels(port_type: str, data_type: str) -> bool:
    """二维像素数据的唯一连线规则：通道数相同即可连接。
    仅有一个例外，且不属于隐式转换：RGBA 可接入接受 RGB 的输入口，第四通道 alpha 随之传递（与 Nuke 一致，
    带 alpha 的画面接入只使用 RGB 的节点时 alpha 仍保留）。

    反向（通道数更少）以及「三通道接入单通道输入口」均不可连接：后者丢弃哪两个通道应由连线者决定，
    须接入「取一条通道」；连线上会显示「这个口要 1 条通道，接进来的有 3 条」。"""
    want, have = channels_of(port_type), channels_of(data_type)
    return bool(want and have) and (have == want or (want >= 3 and have > want))


def describe_types() -> list[dict]:
    """编辑器读取的类型表，附带该通道数的结果写入多层 EXR 时使用的通道名
    （data/layers.py row_channels，即「多层 EXR 输出设置」图层表新增行时填入的值）。"""
    from .layers import row_channels  # layers 模块读取本表，因此在此处而非模块级导入

    return [{**asdict(t), "layer_channels": row_channels(t.layer_default, channels_of(t.id)) if channels_of(t.id) else []}
            for t in DATA_TYPES.values()]


def describe_layer_ports() -> dict[str, str]:
    """输出口名称 → 交付时对应的 EXR 图层（data/layers.py LAYER_FOR_PORT）：网页向「多层 EXR 输出设置」
    连线时据此填入图层名，规则与服务器写出时使用的相同。"""
    from .layers import LAYER_FOR_PORT

    return dict(LAYER_FOR_PORT)


def describe_scene_kinds() -> list[dict]:
    return [asdict(k) for k in SCENE_KINDS.values()]
