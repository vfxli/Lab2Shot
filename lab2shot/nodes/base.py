"""Node definitions: the contract between engine, command line and web UI.

A node type declares its ports, its parameters (a pydantic model), where it
runs and what it needs. Front ends are generated from `describe()`; nodes
never ship UI code.

This module is the facade for node declarations: `Port` lives in nodes/port.py, the parameter machinery (`NodeParams`,
`P`, parameter tables) in nodes/params.py and handle data parsing in nodes/handles.py. All are re-exported here so
that node authors import from this module only. The module itself defines `Info`, `NodeDef` (its class body is
divided into catalogue, ports and usage sections), `ReadsFile` and `empty_packet`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ValidationError

from ..errors import Invalid, NotFound
from ..messages import Msg
from ..availability import Cond
from .applies import Cost, Fact, Licence, OptionTrait, standing_marks  # noqa: F401 (re-exported)
from .handles import (HANDLE_KINDS, Handle, parse_corners, parse_figures, parse_picks, parse_shapes,  # noqa: F401 (re-exported)
                      say_bad_entries)
from .port import EITHER, PARAM, Port  # noqa: F401 (re-exported)
from .params import (  # noqa: F401 (re-exported)
    COOK_BUTTON, PICKED_IN_VIEW, Button, ON_NODE_MAX, pick_button, NodeParams, P, param_defaults, _entry_model, _param_list, always_wired, colorspace_param,
    fp16_param, person_ids, refusal, simple_kind, typed_list, without_params,
)
from .services import CORE_PROJECT, ProjectFacts, services, unloaded_project
from ..data.types import DATA_TYPES, KIND_ORDER, SCENE_KINDS, channels_of, element_of, in_port_order, type_label

if TYPE_CHECKING:
    from .expects import Expect
    from .handles import Places


@dataclass(frozen=True)
class Info:
    """What a node's result covers, known before it is cooked (like Nuke's info): its frames and picture size. The
    engine derives it through the graph to pick the frames a cook's frame range selects, to tell the frame range the
    inputs cover, and to scale time estimates. A still (one picture: an HDRI, a photo) has no frame range: it is
    neither cut by nor part of a cook's range."""

    frames: tuple[int, ...] = ()
    width: int = 0
    height: int = 0
    still: bool = False
    # Not the frame rate of the shot: frame rate is not data in Lab2Shot (see the DEFAULT_FPS comment in data/units.py).
    # Only 「读取视频」 sets this field (the frame rate recorded in the video file) and only 「视频转序列」 reads it (to
    # convert its 「区间（秒）」 into frame numbers of this file). Every other node leaves it at 0 and `merge` does not
    # propagate it.
    video_fps: float = 0.0

    @staticmethod
    def merge(infos: list[Info]) -> Info:
        """What a node covers by default: every frame its inputs have (stills only when there is nothing else) and the
        picture size of the first input that has one."""
        shots = [i for i in infos if not i.still] or infos
        sized = next((i for i in infos if i.width), Info())
        return Info(tuple(sorted({f for i in shots for f in i.frames})), sized.width, sized.height,
                    bool(shots) and all(i.still for i in shots))


class NodeDef:
    """Subclass per node type. Ids are '<extension>.<name>' (the extension's name, or core), English, stable."""

    # ==================================================================== 1. Catalogue
    # How the node is identified in the catalogue and on template cards: name, description, project, cost, licence,
    # handles. Nothing in this section affects how the node computes: changing it invalidates no cache and changes no
    # port.

    id: ClassVar[str]

    version: ClassVar[int] = 1  # bump when the node computes differently: results cached before are recomputed

    label: ClassVar[str] = ""  # the words come from the folder's nodes.json (nodes/text.py apply); none: the type id shows
    # the tool subcategory the node's author suggests (a word only: where a node sits in the node menu is the
    # administrator's file menu/nodes.json, lab2shot/categories.py; a new node is 未分类 until placed there)
    category: ClassVar[str] = ""

    description: ClassVar[str] = ""

    runtime: ClassVar[str] = "core"  # "core" or the extension name

    # its project (nodes/services.py): the core's, or stamped by the extension loader (lab2shot/adapters.py) on each node
    # class of an extension that loaded; a class of an extension never loaded keeps unloaded_project (never usable)
    project: ClassVar[ProjectFacts] = CORE_PROJECT

    handles: ClassVar[tuple[Handle, ...]] = ()  # how the node is worked on in the viewer (nodes/handles.py)

    # where it places what it gives, when that is a transform of its parameters: its transform handle (nodes/handles.py
    # Places, declared among `handles`), never declared separately; the viewer previews it while dragging, the cook
    # applies the same matrix
    places: ClassVar["Places | None"] = None

    # its key parameters, shown on the node's body (at most ON_NODE_MAX, each one a node can show: `simple`): the ones
    # changed most, editable there and always in the panel too.
    #
    # This is only the factory default, not the display criterion (it works exactly like parameter defaults in
    # `Params`): what takes effect is the node's own `ui.on_node` in the graph JSON. It is copied from here when the
    # node is created (webui/src/graph/edit.ts addNode); the 「在节点上显示」 marker in the parameter panel then edits
    # that JSON entry, and template JSON writes it directly (a template card is a node graph). This value is used only
    # when the JSON has no such entry.
    on_node: ClassVar[tuple[str, ...]] = ()
    # Button parameters of its own (nodes/params.py Button: a panel row that runs an action, holds no value), after its
    # parameters and after 「计算」 (COOK_BUTTON, which every node has): in the order they are used — cook first, then
    # what uses its result (「输出」's 「下载」) (interface_specs).
    buttons: ClassVar[tuple[Button, ...]] = ()
    # Parameters whose values are shown in the value strip below the viewer (Focal Length / Filmback / 镜头模型).
    # The strip normally shows numeric outputs only; a node that uses a lens without outputting one
    # (「LensDistortion」) declares the parameters here so the lens in use is visible. The value shown is the
    # parameter's current value, entered or supplied by a wire (engine/evaluation.py strip_values; same wording as the
    # panel's 「Focal Length 18.03 mm · 来自 COLMAP」). These are not made output ports: a pass-through port cannot show
    # whether the value was modified, and the values are available upstream, so they are displayed only.
    # Format: {parameter name: card label}; the labels match COLMAP's three cards (「Focal Length」, not the parameter
    # label 「已知 Focal Length」).
    strip: ClassVar[dict[str, str]] = {}

    # what running it costs (nodes/applies.py Cost): whether it always runs on a GPU, the VRAM and seconds per frame
    # measured on an RTX 4090 at its default parameters (every node that can run on a GPU gives them, taken from the
    # measurements in its adapter's docs.md, never guessed), the system memory its
    # worker takes at its peak when that is a lot. Where each of its instances runs (a card, or a CPU slot) follows
    # from the resolved cost alone (engine/cook.py, farm/scheduler).
    cost: ClassVar[Cost] = Cost()

    # its licence class when it differs from its extension's (nodes/tags.py), and why it is what it is
    licence: ClassVar[Licence] = Licence()

    # what a parameter value changes about it (applies.OptionTrait): a GPU, non-commercial parts, measured VRAM ...
    traits: ClassVar[tuple[OptionTrait, ...]] = ()

    # ==================================================================== 2. Ports
    # What the node takes and gives and how parameters become ports: declared ports, always-on parameter ports,
    # alternative wirings, ports made from parameters, the main result, parameter tables and parameter ports, refusals
    # and the default preview. Everything visible on the graph is determined here and must not be derived elsewhere.

    inputs: ClassVar[tuple[Port, ...]] = ()

    outputs: ClassVar[tuple[Port, ...]] = ()

    Params: ClassVar[type[BaseModel]] = NodeParams

    # its main result: the output a node shows by default and whose stage the viewer opens on ("" the first output).
    # Ports are listed by the type order (data/types.py PORT_ORDER), so the main result need not come first
    main: ClassVar[str] = ""

    # the input whose picture its 2D outputs follow: their size, their window and what was photographed come from it
    # (data/contracts.py settle). "" a node that makes pictures of its own (a source, a render)
    picture: ClassVar[str] = "image"

    # Parameters whose input port is always on the node, without the artist first enabling it in the panel
    # (「LensDistortion」's Focal Length, Filmback, 镜头模型, distortion parameters and principal point). When an upstream
    # node computes exactly these values (AnyCalib, COLMAP), the ability to wire them must not be hidden behind a pin
    # in the parameter panel: a port not visible on the graph is effectively absent. Other parameters still have to be
    # promoted by the user (「提升到节点」).
    #
    # Two declarations, resolved in one place (__init_subclass__ merges them into this attribute, the only one read):
    #   1. the list written by the node class itself (this ClassVar), for parameters specific to this node
    #      (「LensDistortion」's 镜头模型, distortion parameters, principal point);
    #   2. `P(wired=True)` declared by the parameter itself, for parameters that need a port on every node that has
    #      them, without each node having to list them (nodes/lens.py focal_param / filmback_param: 「已知 Focal
    #      Length」 and 「Filmback」 appear on many nodes, and listing them per node is error-prone).
    # The criterion is therefore whether the node has the parameter, not a hand-maintained list.
    wired_ports: ClassVar[tuple[str, ...]] = ()
    # Value inputs that set several of the node's parameters at once (「LensDistortion」's 「镜头内参」: the lens group,
    # model, coefficients, principal point and pixel aspect of the lens a calibration node solved). Once the wired
    # value is known (the node feeding it cooked, or gives it from its parameters: known_outputs), the parameters it
    # sets are what the node is cooked, fingerprinted and described with (engine/evaluation.py params ->
    # params_from_input): the node follows its source, rather than refusing a model typed differently on it.
    # The parameters it sets should be greyed while it is wired (applies Not(Wired(port))), so the panel says so.
    params_inputs: ClassVar[tuple[str, ...]] = ()

    # Alternative wirings of the node's inputs: exactly one must be used and at least one is required
    # (「序列图输出设置」: a whole 「图像」, or R G B A wired separately). From this one declaration the framework
    # derives three behaviours in one place, so nodes do not implement them individually:
    #   1. once one alternative is used, the ports of the others are inactive: greyed with the reason, kept in place,
    #      and not wirable (port_applies produces Port.applies);
    #   2. with no alternative wired, submission is blocked (Graph.check_inputs, B-GRAPH-NOWIREANY);
    #   3. when every port of the used alternative is empty (upstream failed or produced nothing), the node is marked
    #      「已跳过」 without reporting again (Evaluation.outcome, Engine._run_node; the error is reported only once, on
    #      the node that failed).
    # Each group has at least one name, groups share no name, and every name is an optional input declared by the node
    # (check_declarations).
    input_choice: ClassVar[tuple[tuple[str, ...], ...]] = ()

    @classmethod
    def choice_inputs(cls) -> tuple[str, ...]:
        """All ports of the alternative wirings, flattened (the engine and catalogue use it to check "at least one")."""
        return tuple(name for group in cls.input_choice for name in group)

    @classmethod
    def port_applies(cls, port: Port) -> Cond | None:
        """When this input applies: as the port declares (Port.applies), or as derived from the alternative wirings,
        in which case an alternative applies only while no port of any other alternative is wired. Wiring, greying and
        submission checks all read this."""
        if port.applies is not None:
            return port.applies
        mine = next((g for g in cls.input_choice if port.name in g), None)
        if mine is None:
            return None
        from ..availability import All, Not
        from .applies import Wired

        others = [n for g in cls.input_choice if g is not mine for n in g]
        return All(*(Not(Wired(n)) for n in others)) if others else None

    # a list parameter whose entries are more ports, one each, in row order: on the outputs (读取序列: a port per layer
    # of its file, typed as the user takes it, after the declared `outputs`) or on the inputs (「多层 EXR 输出设置」:
    # a port per row of its 图层 table, and no other input, since the node declares none). One declaration,
    # `ports_from_side` saying which; `output_ports`/`input_ports` are the only place either is resolved.
    ports_from: ClassVar[str] = ""

    ports_from_side: ClassVar[Literal["outputs", "inputs"]] = "outputs"

    # the port type every row gets when ports_from_side is "inputs" (a row's own entry has no `type` of its own then:
    # what it carries is whatever is wired into it, not chosen, unlike an output entry's `type`)
    ports_from_type: ClassVar[str] = ""

    # the rows of an input-making table when the node names them itself: the port names its rows may have, a new row
    # taking the first one not yet used, and the label the n-th row gets by default (「切换」: a…j, 第一路…第十路, so a
    # graph's wires to a / b / c keep their meaning however many ways it has). As many rows as names, no more: the page
    # offers no 「＋」 once they are all used, and the node's own table type refuses more (its row's `name` a Literal
    # of these, the table's max_length theirs). Empty: rows as many as wanted, named by the page (row1, row2…) and
    # labelled after what is wired in (「多层 EXR 输出设置」's 图层 depth, mask…).
    ports_from_names: ClassVar[tuple[str, ...]] = ()
    ports_from_labels: ClassVar[tuple[str, ...]] = ()
    # what one row is called where the page offers another (「加一层」, 「加一路」)
    ports_from_word: ClassVar[str] = "行"

    # (from type, to type): it turns data of the one into the other, a generic conversion (「置信度转遮罩」, 「烘焙成模型」).
    # A wire whose type an input does not take, when such a node takes it and gives what the input takes, is kept as a
    # wiring problem with this node as its one click in between (Graph.fix_for; store.ts converterFor): never converted
    # silently
    converts: ClassVar[tuple[str, str]] = ()

    @classmethod
    def main_output(cls) -> str:
        """The name of its main result: `main`, else its first output ("" a node without outputs)."""
        from .applies import all_outputs

        outs = all_outputs(cls)
        return cls.main or (outs[0].name if outs else "")

    @classmethod
    def made_ports(cls, params: dict) -> tuple[Port, ...]:
        """The ports the node has beyond the declared ones, computed from its parameters, on the `ports_from_side`
        side. By default this is the `ports_from` table, one port per row: outputs are typed by `row_type`, inputs are
        always the node's declared `ports_from_type` with alpha (「多层 EXR 输出设置」's 「图层」 table: a picture wired
        into a row carries its alpha like any other image input). Input rows are optional: a row added on the page
        (its 「＋」, the table's 「添加」) and not wired yet does not stop the cook; the node skips it and says so once
        (ExrOutput.write: N-LAYERS-UNWIRED), so every node that makes inputs from a table handles `ctx.input(row)` being None.

        Nodes may override it: 「读取序列」 uses no parameter table, and its outputs follow the layers present in the
        file. This method must never raise: it is called for every wire check and every status computation
        (nodes/applies.py output_ports, input_ports)."""
        entries = params.get(cls.ports_from) or () if cls.ports_from else ()
        if cls.ports_from_side == "inputs":
            return tuple(Port(e["name"], cls.ports_from_type, e["label"], optional=True, alpha=True, data=EITHER) for e in entries)
        return tuple(Port(e["name"], cls.row_type(e), e["label"]) for e in entries)

    @classmethod
    def row_type(cls, entry: dict) -> str:
        """The type of a port made from one ports_from row. By default it follows the number of channels the row picks
        (读取序列's layer table: a port picking n channels has n channels); a row with an explicit type uses that."""
        if "type" in entry:
            return entry["type"]
        picked = [c for c in str(entry.get("channels") or "").replace(",", " ").split() if c]
        return f"image.{len(picked)}" if 1 <= len(picked) <= 4 else "image"

    @classmethod
    def input_ports(cls, params: dict) -> tuple[Port, ...]:
        """The node's inputs with these parameters: the declared ones, then one per entry of the ports_from parameter
        when it makes inputs instead of outputs (「多层 EXR 输出设置」: a port per row of its 图层 table, in row order,
        which sets the EXR's layer order; 「切换」: one per way), then the ones its `wired_ports` parameters always
        carry (「切换」's 「走哪一路」 under its ways, as the page draws them: webui graph/rules.ts inputsOf).
        Graph.input_ports adds the parameters promoted on this instance on top; the web page reads the resolved ports
        from the status reply, and a row added since that reply from its document (webui graph/rules.ts inputsOf)."""
        return (cls.inputs + (cls.made_ports(params) if cls.ports_from_side == "inputs" else ())
                + tuple(cls.param_port(n) for n in cls.wired_ports))

    @classmethod
    def param_specs(cls) -> list[dict]:
        """Every parameter as front ends and the engine read it (label, type, options, rules): built once per type."""
        if cls not in _SPECS:
            specs = _param_list(cls.Params, cls.Params.model_json_schema())
            for spec in specs:
                # File parameters carry one more field: whether the node can answer its questions when the server has
                # only the file's header (`ReadsFile.head_is_enough`, which explains why the default is no). The web
                # page uses it to decide whether a selected file is declared first or uploaded at once
                # (webui/src/transfer/declare.ts). It is derived from the node's own declaration rather than written
                # per parameter or guessed from the file suffix: 「读取多条序列」 accepts images but needs the whole
                # folder on disk.
                if spec["widget"] in ("file", "sequence"):
                    spec["head_enough"] = bool(getattr(cls, "head_is_enough", False))
            _SPECS[cls] = specs
        return _SPECS[cls]

    @classmethod
    def interface_specs(cls) -> list[dict]:
        """What the parameter panel lists and a template's parameter interface may expose: every parameter
        (param_specs), each picks / canvas parameter followed by its own 「在视图里点选」 (pick_button), then the
        node's buttons in the order they are used (「计算」, then its own, e.g. 「输出」's 「下载」: nodes/params.py Button). Buttons are not parameters: nothing that reads
        values (loading, the fingerprint, the worker) sees them."""
        out: list[dict] = []
        for spec in cls.param_specs():
            out.append(spec)
            if spec["widget"] in PICKED_IN_VIEW:
                out.append(pick_button(spec).spec())
        return [*out, *(b.spec() for b in (COOK_BUTTON, *cls.buttons))]

    @classmethod
    def param_port(cls, name: str) -> Port:
        """The input a parameter gets when it is driven by a wire (promoted, like a Houdini channel reference or
        ComfyUI's widget turned into an input): "param:<name>", typed by the parameter's kind (a number takes a 浮点
        or 整数), in its unit, optional; a value of the shot it belongs to (SameShot) when the node takes a plate.
        ValueError for a parameter that cannot be driven by one (a file, a picker). A choice takes 文字 and a table of
        one number column a 浮点列表 (AnyCalib's 「镜头模型」 and 「畸变系数」 into LensDistortion)."""
        from .expects import SameShot
        from .values import CONSTANT_NODES

        spec = next((p for p in cls.param_specs() if p["name"] == name), None)
        if spec is None:
            raise Invalid(Msg("E-NODE-NOPARAM", node=cls.label, names=name))
        if not spec["wire"]:
            raise Invalid(Msg("E-NODE-NOTWIREABLE", node=cls.label, label=spec["label"]))
        from ..data.values import FLOAT, INT

        plate = any(p.name == "image" for p in cls.inputs)
        # a number with no unit of its own (a count, an index, a factor) takes a plain number only: a value with a
        # unit wired in would have it dropped on the quiet (2 mm as the second item), so the wire is refused (Port.plain).
        # Not on a node whose output carries a unit it is given (the constants: unit="param:unit"), which holds any
        carries = any(o.unit.startswith(PARAM) for o in cls.outputs)
        plain = not carries and not spec["unit"] and any(t in (INT, FLOAT) for t in spec["wire"].split("|"))
        return Port(PARAM + name, spec["wire"], spec["label"], optional=True, unit=spec["unit"], plain=plain,
                    expects=(SameShot(),) if plate else (),
                    recommend=CONSTANT_NODES.get(spec["wire"].split("|")[0], "core.make_list"))

    @classmethod
    def affecting_params(cls) -> set[str]:
        """Parameters that change the result (the others, e.g. an output folder, stay out of the fingerprint)."""
        return {p["name"] for p in cls.param_specs() if p["affects_result"]}

    @classmethod
    def fingerprint_params(cls, params: dict) -> dict:
        """What the fingerprint sees of the parameters: their effective values. A parameter the node fills in
        itself when it is left empty (a reader's 「色彩空间」: by the file's format) is the same result filled or not,
        so the fingerprint must not change with it; otherwise a graph cooked once with it empty and once with it
        filled (the page fills it only when the node's panel is open) cooks everything twice."""
        return params

    @classmethod
    def worker_params(cls, params: dict) -> dict:
        """What the worker gets: every parameter except those declared P(worker=False), which the node applies
        itself (units, thresholds, point density, a lens it converts ...). Nodes add values they compute."""
        names = {p["name"] for p in cls.param_specs() if p["worker"]}
        return {k: v for k, v in params.items() if k in names}

    @classmethod
    def load_params(cls, params: dict) -> dict:
        """Saved params -> every parameter (defaults for the ones not given). An unknown one is an error."""
        unknown = sorted(set(params) - set(cls.Params.model_fields))
        if unknown:
            raise Invalid(Msg("E-NODE-NOPARAM", node=cls.label, names=unknown))
        from .applies import output_ports

        try:
            loaded = cls.Params(**params).model_dump()
        except ValidationError as exc:
            raise Invalid(refusal(cls.param_specs(), exc)) from None
        names = [p.name for p in output_ports(cls, loaded)]
        if len(set(names)) != len(names):
            raise Invalid(Msg("E-NODE-DUPOUTPUT", node=cls.label, names=sorted({n for n in names if names.count(n) > 1})))
        in_names = [p.name for p in cls.input_ports(loaded)]
        if len(set(in_names)) != len(in_names):
            raise Invalid(Msg("E-NODE-DUPINPUT", node=cls.label, names=sorted({n for n in in_names if in_names.count(n) > 1})))
        return loaded

    @classmethod
    def expectations(cls, port: Port) -> tuple[Expect, ...]:
        """What the node expects of its input `port` beyond its type (checked by engine/lint.py): what the port
        declares, and, on 图像, what its frame-count declarations imply (min_frames / most_frames: FrameCount)."""
        # There are no lens-state checks (raw / undistorted / unknown, pixel aspect, undistortion): CG artists know
        # whether their input was undistorted. The NodeDef.lens declaration is kept as data (sent in the catalogue)
        # but produces no messages.
        implied = ()
        if port.name == "image" and (cls.min_frames or cls.most_frames):
            from .expects import FrameCount

            implied = (FrameCount(least=cls.min_frames, step=cls.min_frames_step),) if cls.min_frames else ()
            if cls.most_frames:
                implied = (*implied, FrameCount(most=cls.most_frames))
        return (*port.expects, *implied)

    @classmethod
    def refuses(cls, data_type: str, kinds: frozenset[str] | None = None) -> str:
        """Why the node cannot take a wire of `data_type` its input types would take ("": it can), as the words after
        its name: a wiring problem like a wrong type. `kinds`: the 3D data the wire carries (Graph.scene_kinds; None:
        its type's own kind). A 3D output-settings node refuses a kind its format cannot hold (OutputSettings)."""
        return ""

    @classmethod
    def param_refuses(cls, data_type: str, params: dict) -> Msg | None:
        """Why this node's current parameters cannot take a wire of `data_type`, beyond what its ports allow (None:
        they can): a choice that would really corrupt the data if the wire were cooked as it is (「多层 EXR 输出设置」
        set to a compression that touches every channel, i.e. DWAA, DWAB, PXR24, with a data layer wired in: unlike a
        picture, that layer must stay bit-exact). Only for a choice the engine cannot silently work around (compare
        ExrOutput's own 位深, always forced 32-bit for a data layer regardless of what this says): use it where the
        wrong choice would truly damage the result (「序列图输出设置」's format is not one: it always copies a data map's
        file untouched, and its 格式 says it does nothing then, greyed). Checked alongside the
        type refusal (Graph.wire_problem), the same way: the wire is kept a wiring problem until it is fixed."""
        return None

    @classmethod
    def wiring_notes(cls, params: dict, wires: dict[str, int]) -> list[tuple[Msg, str]]:
        """What the node says about how many wires it has (`wires`: input -> how many), beside its parameters: a wire
        or a setting that does nothing with them, said rather than passed over (「与或非」 set to 非 reads one wire,
        「合成列表」 more names than items). (message, input) pairs, notices only (engine/lint.py lists them)."""
        return []

    @classmethod
    def several_refused(cls, port: str, types: tuple[str, ...]) -> Msg | None:
        """Why the several wires into multi input `port`, carrying `types` (one per wire, as the graph has them), can't
        go in together (None: they can): a rule on the wires together, not on any one of them, checked on the graph
        before a cook (Graph._check_wires) and by the node's own cook on what really came (「逐项结束」: several
        results of one item pack into a scene only when all are 3D data)."""
        return None

    @classmethod
    def refusal_fix(cls, data_type: str, kinds: frozenset[str] | None = None) -> str:
        """The node type that turns what the node refuses into what it takes ("" none): offered as one click on the
        refused wire (Graph.fix_for)."""
        return ""

    @classmethod
    def fix_kind(cls, data_type: str, kinds: frozenset[str] | None = None) -> str:
        """The kind of 3D data refusal_fix lets through, named on its button ("" the fix names none: OutputSettings)."""
        return ""

    @staticmethod
    def _pixel_channels(port_type: str) -> int:
        """How many channels of pixel data the port gives (0: not pixel data). The `image` family (channel count follows
        the input) counts as 3, since it serves as a picture: what is viewed is its transformed result."""
        one = port_type.split("|")[0].replace("[]", "")
        return 3 if one == "image" else channels_of(one)

    @classmethod
    def default_preview(cls) -> str:
        """The node's own preview tag: when the node is displayed by double-click, the viewer switches to the preview
        mode that suits it best. The node carries the tag; the viewer switches 2D and 3D by it uniformly.

        One tag covers two stages, with four values:

        - ``scene``   the 3D view;
        - ``plate``   2D, plate only: nodes that produce no picture (pass-through, parameters, output settings) show
                      the plate (source nodes that make their own picture use 「仅结果」 instead, as explained below;
                      otherwise the file's layers would be locked in the slot that 「仅原图」 greys out);
        - ``result``  2D, result only: the result is a picture (three or four channels: STMap, Crop, colour transforms);
        - ``compute`` 2D, compute: the result has one or two channels (mask, depth, UV, normals, confidence) and is
                      meaningful only together with the plate.

        The stage follows the handles and the main result's type: a node with handles opens on its handles' stage,
        otherwise on the stage of its main result (``main``, or the first output); an output-settings node opens on
        what it writes (its input)."""
        from .applies import output_ports

        ports = output_ports(cls, param_defaults(cls.Params))
        ports = tuple(p for p in ports if p.name == cls.main) + tuple(p for p in ports if p.name != cls.main)
        if cls.handles:
            stage = HANDLE_KINDS[cls.handles[0].kind].stage
        else:
            first = ports
            if not first or DATA_TYPES[element_of(first[0].type.split("|")[0])].in_3d == "inputs":
                first = cls.inputs
            stage = "3d" if first and DATA_TYPES[element_of(first[0].type.split("|")[0])].in_3d == "element" else "2d"
        if stage == "3d":
            return "scene"
        # 2D: by the channel count of the node's main result. Pass-through and output-settings nodes produce no picture
        # and use 「仅原图」; source nodes that make their own picture (读取序列, 视频转序列: no pixel input) use
        # 「仅结果」, for the reason given below.
        pixels = next((n for p in ports if (n := cls._pixel_channels(p.type))), 0)
        if not pixels:
            return "plate"  # no picture: pass-through, output settings, tracking, import
        if pixels <= 2:
            return "compute"  # one or two channels (mask, depth, UV, confidence) are not a picture; viewed with the plate
        # A picture of three or four channels: its own result, whether it transforms its input (Crop, STMap) or makes
        # the picture itself (读取序列, 读取图片, 视频转序列, Constant). For a source 「仅结果」 matters: 「仅原图」 would grey
        # out the 「结果通道」 slot, which holds every layer of the file, so a multi-layer EXR would show only
        # 「原图.R/.G/.B/.A」 (as Nuke's Read, whose viewer lists every layer and channel of the file).
        return "result"

    # ==================================================================== 3. Usage
    # What cooking needs to know: delivery side effects, frame-count requirements, lens assumptions, whether the name
    # enters the fingerprint, overscan, frame sources, facts, the ops catalogue, and the questions a node answers before
    # and during a cook (facts, foresee, said_shot, derive, choices, info, cook).

    # it hands files to the user where they chose (「输出」): a side effect, so it runs on a click only, never because
    # a node is shown
    delivers: ClassVar[bool] = False
    # a model that finishes another's result (BiRefNet 边缘解混合 on a matte) rather than making the deliverable: never
    # the card's main project (engine/templates.py core_project, the year its card shows)
    finishes: ClassVar[bool] = False
    # the cards that expose its parameters all expose the same ones, named, labelled and hidden alike (a block copied
    # card to card by hand: `lab2shot check templates`, engine/templates.py shared_interfaces); a card that departs on
    # purpose declares it in its meta (`own_interface`: the node types)
    same_on_cards: ClassVar[bool] = False

    # the fewest frames of 图像 its method can work with (a temporal network's window, two frames to track between):
    # fewer is refused before cooking, saying how many it got (nodes/expects.py FrameCount); `min_frames_step`: the
    # parameter that takes every n-th frame, when the count is after it
    min_frames: ClassVar[int] = 0

    min_frames_step: ClassVar[str] = ""

    # the most frames of 图像 the method takes at once (environment light probe: 1 frame). More are refused before
    # submission instead of silently using the first; one click inserts 「FrameHold」 so the user picks the frame
    # (nodes/expects.py FrameCount(most=))
    most_frames: ClassVar[int] = 0

    # the lens model the node assumes for the picture (nodes/applies.py LENSES): declared by the family, overridable
    # per node. No message is derived from it (CG artists know whether their input was undistorted); it is a
    # declaration, sent in the catalogue
    lens: ClassVar[str] = ""

    # a node that solves the lens itself (lens "solves"): when it solves no distortion at all and is a pinhole node
    # after all, as a condition on its own parameters (COLMAP: 「镜头模型」 on a model without distortion). Checked with
    # the node's other conditions (nodes/applies.py conditions_of; check_declarations: every "solves" node has one)
    pinhole_when: ClassVar[Any] = None

    # its cooked result depends on the node's name as the artist sees it (an import node's folder is the renamed node's
    # name, io/usd.py import_group): the name is part of what it cooks (Evaluation plan), so renaming it cooks again
    named_result: ClassVar[bool] = False

    # it works on every pixel the picture keeps, its overscan too, and gives it back on the same window (data/windows.py
    # Window; the default: reading and writing go by the packet's window). A node that can only work on the plate frame
    # (one that unprojects it, renders into it, writes a PNG) declares False and says what it leaves out (N-COOK-OVERSCAN)
    keeps_overscan: ClassVar[bool] = True

    # its frames come from files (a sequence, a video): a cook's frame range selects which of them it emits, and
    # everything downstream follows (CookContext.frames)
    frame_source: ClassVar[bool] = False

    # the facts about itself it can know (NodeDef.facts before cooking, CookContext.fact while cooking) -> how a message
    # names each; a condition or a port's kinds_from can only name these
    fact_labels: ClassVar[dict[str, str]] = {}

    # the ops the node uses (ids in lab2shot/ops/ops.toml), checked against the catalogue when the class is made
    # (nodes/applies.py check_declarations). The algorithms are defined in the catalogue and run by its executor
    # (ops/run.py); the node's cook reads its inputs, runs them and writes the results.
    ops: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def facts(cls, params: dict) -> dict[str, Fact]:
        """What the node knows about itself from its parameters alone (named in fact_labels): an import node, whether
        the 模型 selected deform. Cheap and never raising (a file not there yet: not known). {} by default."""
        return {}

    @classmethod
    def foresee(cls, params: dict, info: Info) -> list[Msg]:
        """What the node can say before it is cooked, from its parameters and what its result will cover (a sequence
        with gaps in the cook's frames): shown on the node with its usage checks and said again when it is cooked
        (engine/evaluation.py, engine/cook.py). Cheap; [] by default."""
        return []

    @classmethod
    def said_shot(cls, params: dict) -> dict:
        """What the node itself says of the shot its pictures come from (data/contracts.py SHOT_KEYS), from its
        parameters alone: 「读取序列」/「读取视频」's 镜头状态 and 像素比, 「LensDistortion」's direction. An output takes it
        for the keys its Shape declares "node" for; the others come from the node's picture input (contracts.shot_of).
        Pure and cheap: the graph works out what a picture will say before anything is cooked (Evaluation.shot), and
        the node writes the same thing into its packets when it does cook. {} by default: the node says nothing of its
        own."""
        return {}

    @classmethod
    def derive(cls, params: dict) -> dict:
        """Parameters the node works out from others (P(derived_from=)): when one of those changes, the editor asks
        and sets them together, as one edit (读取序列: the layers of the chosen file)."""
        return {}

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        """The options of parameters that come from a file or from what is wired in (P(choices_from=)): {parameter:
        {"options": [values], "labels": {value: shown as}, "auto": what the empty value comes to (for a table, per
        entry name; "" when nothing was found), "none": the name of "" when it means "none" and empty is None,
        "empty": what the empty value shows when it has no "auto" (先选择相机文件, 选一台)}}. `inputs`: port -> the
        packet cooked for it (ports whose upstream has not been cooked yet are left out). The editor asks when one of
        choices_from changes (an import node: the entries of the picked file; the in-betweening nodes: a skeleton's
        joints for the joint table)."""
        return {}

    @classmethod
    def handle_data(cls, params: dict, inputs: dict) -> dict[int, dict]:
        """What the stage draws for handles that work on data the node reads from its inputs rather than on the
        picture (nodes/handles.py Poses: a skeleton, its pose before and after the correction), by the handle's index
        in `handles`. `inputs` as for `choices` (the packets standing for what is wired in); a handle it cannot give
        data for yet is left out. Asked for the displayed node only, with the status reply (server/packets.py)."""
        return {}

    @classmethod
    def source_identity(cls, params: dict) -> Any:
        """Identity of external files a node reads (e.g. size + mtime), folded into its fingerprint."""
        return None

    @classmethod
    def ready(cls, params: dict) -> None:
        """What planning this node with these parameters needs that takes a while, done ahead of it: an import reading
        what its file holds (ImportNode, through its extension's worker). The engine calls it before it plans an
        instance, outside its lock (engine/cook.py Engine._cook_all), so planning, done under the lock, finds it done
        and never waits on a worker. Its errors are not said here: planning meets them again and says them where they
        belong. Nothing for most nodes. (Not `prepare`: that name is a WorkerNode's job builder, families/base.py.)"""

    @classmethod
    def info(cls, params: dict, inputs: dict[str, list[Info]]) -> Info:
        """What the result will cover, from the parameters and what the inputs cover (port -> infos, like the
        packets cook() gets). Sources read it from their files; a frame source gives every frame it can emit (the
        engine cuts them to the cook's range). Cheap: it runs whenever the graph is planned."""
        return Info.merge([i for infos in inputs.values() for i in infos])

    @classmethod
    def params_from_input(cls, port: str, value: Any) -> dict:
        """The parameters a known value wired into `port` (one of params_inputs) sets. Must never raise."""
        return {}

    @classmethod
    def known_outputs(cls, params: dict) -> dict[str, dict]:
        """Outputs the node gives from its parameters alone, as their description (meta), known before it is cooked
        (a constant value): a parameter they drive shows and checks its value right away. Cheap."""
        return {}

    @classmethod
    def upload_channels(cls, params: dict, needed: frozenset[str]) -> dict | None:
        """The channels of the source file this node needs uploaded for this cook (`{"take": [channel names in the
        file], "write": [names in the subset]}`). None when not applicable (the node reads no file) or when the whole
        file is needed (PNG / JPG have no separate channels and need all of them). `needed`: which of its outputs are
        requested (`engine/evaluation.py needed_outputs`: those with a wire out). The status reply places this in the
        node's `channels`, and the browser decodes and uploads only those channels in its worker. Only the implementing
        node knows how ports map to channel names (「读取序列」's `_outputs_of`); the web page does not duplicate it."""
        return None

    @classmethod
    def cook(cls, ctx) -> dict:
        """Produce output packets: {port: Packet}. `ctx` is an engine CookContext."""
        raise NotImplementedError(f"{cls.id} has no cook()")

    # ==================================================================== 4. Class hooks and catalogue entry

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        # every node lists its ports by the one type order, whatever order it declared them in
        cls.inputs = in_port_order(cls.inputs)
        # A solver has no outputs its upstream project does not provide (one output per kind of upstream data). The
        # family's convenience output that computes a point cloud (`made_from` = depth + camera + confidence) is kept
        # only for nodes with a native point cloud (`native_points`); other point clouds are made by wiring 「深度转点云」,
        # a core utility whose parameters (spacing, point size, confidence threshold, colour, exclusion) are visible on
        # it. Only concrete node classes (with an id) are checked: the family base class has no `native_points`, and
        # removing the port there would remove it from every subclass, including MoGe / UniDepth / UniK3D, which do
        # have native point clouds.
        if getattr(cls, "id", "") and not getattr(cls, "native_points", ""):
            cls.outputs = tuple(p for p in cls.outputs if not (p.name == "points" and p.made_from))
        cls.outputs = in_port_order(cls.outputs)
        # The outputs are final: a parameter inherited for an output the node does not have (the 「点云」's spacing and
        # point size on a node without it) could never apply, so the node does not have it either (lost_output_params)
        from .applies import lost_output_params

        cls.Params = without_params(cls.Params, lost_output_params(cls))
        # Always-on parameter ports, resolved in one place: the node's own list plus parameters declaring
        # P(wired=True) (see the wired_ports comment). Deduplicated in order: the node's own names first
        # (「LensDistortion」's Focal Length and Filmback keep their positions), then those declared by parameters.
        # Filtered by this class's own parameters: the result computed for a family base class is inherited, but a
        # subclass may replace the parameters (families/depth_camera.py PerFrameDepthCamera.Params has 「已知 Focal
        # Length」, FaceAnything replaces it with one without a lens), and inheriting a port for a parameter the class
        # lacks would cause E-NODE-NOPARAM when building a graph. A misspelled name in the node's own list fails at
        # class creation rather than at graph build time.
        own = cls.__dict__.get("wired_ports")
        if own and (missing := [n for n in own if n not in cls.Params.model_fields]):
            raise TypeError(f"{cls.__name__}: wired_ports names {missing}, which are not parameters it has")
        if missing := [n for n in cls.strip if n not in cls.Params.model_fields]:
            raise TypeError(f"{cls.__name__}: strip names {missing}, which are not parameters it has")
        cls.wired_ports = tuple(dict.fromkeys(tuple(n for n in cls.wired_ports if n in cls.Params.model_fields)
                                              + always_wired(cls.Params)))
        from .applies import check_declarations

        from .handles import Places

        if "project" not in cls.__dict__:  # each class its own: a subclass in another extension is not its parent's
            cls.project = CORE_PROJECT if cls.runtime == "core" else unloaded_project(cls.runtime)
        if "places" in cls.__dict__:
            raise TypeError(f"{cls.__name__}: its placement is its transform handle (handles = (Places(...),)), not a separate `places`")
        placing = [h for h in cls.handles if isinstance(h, Places)]
        if len(placing) > 1:
            raise TypeError(f"{cls.__name__}: one transform handle places a node, it declares {len(placing)}")
        cls.places = placing[0] if placing else None
        # each declaration against this node's parameters and inputs (nodes/handles.py Handle.check); a family's base
        # class declares handles for parameters only its node types have, so only a node type (with an id) is checked
        if cls.__dict__.get("id"):
            for h in cls.handles:
                h.check(cls)

        check_declarations(cls)

    @classmethod
    def describe(cls) -> dict[str, Any]:
        from .applies import all_outputs, declared_cost, option_traits, resolve_params

        at_defaults = resolve_params(cls, param_defaults(cls.Params))
        # Entries of the three sections (matching the three sections of the class body), sent as one dictionary; each
        # group below is headed by the section it belongs to.
        return {
            # 1. Catalogue
            "id": cls.id,
            "label": cls.label,
            "category": cls.category,  # the tool subcategory its author suggests (a word; its place is menu/nodes.json)
            "description": cls.description,
            # 2. Ports
            "inputs": [p.describe() for p in cls.inputs],
            "outputs": [p.describe() for p in all_outputs(cls)],
            "main": cls.main_output(),
            # the input each parameter that can be driven by a wire gets once promoted
            "param_ports": {p["name"]: cls.param_port(p["name"]).describe() for p in cls.param_specs() if p["wire"]},
            # These parameters' input ports are always on the node (wired_ports): the 「提升到节点」 pin in the panel is
            # permanently on for them. Sent only when present (every browser downloads the catalogue before the editor
            # opens, so it counts against the first-load budget). The page reads `def.wired_ports?.`; absent means
            # empty.
            **({"wired_ports": list(cls.wired_ports)} if cls.wired_ports else {}),
            # The ports that belong to one of the alternative wirings (input_choice, flattened): the page does not mark
            # them optional, since not wiring any is invalid and "at least one" would be redundant. Sent only when
            # present, like wired_ports.
            **({"needs_any": list(cls.choice_inputs())} if cls.input_choice else {}),
            "ports_from": cls.ports_from,
            "ports_from_side": cls.ports_from_side,
            # 3. Usage
            "marks": standing_marks(cls),  # standing marks from its declarations (Port.shape.said, e.g. Crop's I-SHAPE-CROP)
            "ports_from_type": cls.ports_from_type,
            # the type every row's port carries, named by the server (the page never spells a type's name)
            "ports_from_type_label": type_label(cls.ports_from_type) if cls.ports_from_type else "",
            # an input table whose rows the node names itself (「切换」's ways): their names and default labels, as
            # many as it may have; sent only when present, like wired_ports (the page reads `def.ports_from_names?.`)
            **({"ports_from_names": list(cls.ports_from_names), "ports_from_labels": list(cls.ports_from_labels)}
               if cls.ports_from_names else {}),
            "ports_from_word": cls.ports_from_word if cls.ports_from and cls.ports_from_side == "inputs" else "",
            "params": cls.interface_specs(),  # the parameters, then the buttons (widget "button")
            "defaults": param_defaults(cls.Params),
            "runtime": cls.runtime,
            "handles": [h.describe() for h in cls.handles],
            "places": cls.places.placement() if cls.places else None,
            "preview": cls.default_preview(),  # the viewer's preview tag (three 2D modes or 3D), one per node
            "delivers": cls.delivers,
            # what it costs and whose licence it is at its default parameters (a node not in a graph yet: the catalogue's
            # card); a node in a graph reads its status (Evaluation.status), resolved with its own parameters
            "cost": declared_cost(cls),  # what it declares: always on a GPU, its own rating
            "at_defaults": {"cost": at_defaults.cost.describe(), "licence": at_defaults.licence.describe()},
            # the choices that change something, as a lookup table: parameter -> value -> gpu / noncommercial / rating
            "option_traits": option_traits(cls),
            "on_node": list(cls.on_node),
            **({"strip": list(cls.strip)} if cls.strip else {}),  # parameter names shown in the value strip (sent only when present; labels come with the status reply)
            "refuses": {t: why.text for t in DATA_TYPES if (why := cls.refuses(t))},
            "converts": list(cls.converts),
            # a kind of 3D data it cannot take, whatever the wire's type: kind -> why, on a wire of the kind's own type
            # and on a 场景 packed with it, and the node that converts it (Graph.scene_kinds tells what a wire carries;
            # the editor reads a wire's refusal from the status reply)
            "refuses_kinds": {k: {"wire": _text(cls.refuses(SCENE_KINDS[k.split(".")[0]].type, frozenset({k}))), "packed": why.text,
                                  "via": cls.refusal_fix(SCENE_KINDS[k.split(".")[0]].type, frozenset({k}))}
                              for k in KIND_ORDER if (why := cls.refuses("scene", frozenset({k})))},
        }


def _text(said: Msg | None) -> str:
    return said.text if said is not None else ""


_SPECS: dict[type, list[dict]] = {}


class ReadsFile:
    """Mixin for nodes that read the file the user uploaded into their "path" parameter (an upload reference, see
    lab2shot/transfer/uploads.py, handed over as PlanEnv.upload). Uploads are named by their content, so the reference is the file's identity: other
    files make another reference and re-cook the node. Without a file the node cannot be planned (the editor shows
    why, and never cooks it by itself)."""

    no_file: ClassVar[str] = "没有选择文件"

    # Whether the node can answer its questions (which ports it has, how many frames and what size, what the
    # hierarchy contains) when the server holds only the first tens of KB of the selected file. This decides whether
    # a selected file is uploaded immediately:
    #
    #   True  -> declare first, upload later (see `transfer/uploads.py`): nothing is sent on selection and bytes are
    #            uploaded only on 「计算」. A picture's layers are in the file header (the first few KB of an EXR list
    #            all layers), so selecting an image must not start an upload or show a cook in progress.
    #   False -> upload on selection (default).
    #
    # The default is deliberately False: a node that does not implement answering without the bytes uploads on
    # selection. With the opposite default, a newly added reader node would silently fail to answer: after selecting a
    # file, 「选择…」 could not list the hierarchy and would report that the upload is no longer on the server (it was
    # never uploaded), while uploading would require 「计算」, which in turn requires a chosen hierarchy to have
    # output ports, a deadlock.
    #
    # Three kinds of nodes cannot answer from the header and therefore upload on selection:
    #   - every 「导入 …」 node (`nodes/formats.py ImportNode`, one per format module): the hierarchy can be listed only
    #     from the whole file (`listing()` opens it), and without a chosen hierarchy there are no output ports; the
    #     browser does not parse scene formats, so the server needs the file (the core names no formats; the suffixes
    #     are in `webui/src/transfer/declare.ts`);
    #   - 「读取多条序列」: the sequences in a folder require the folder on disk (`_found` -> `is_dir()`);
    #   - 「读取视频」: a video's frame count requires the decoder to open the whole file (`info()` -> `_video_facts`).
    #
    # Only 「读取序列」 (and its subclass 「读取图片」) declares it: `made_ports` falls back to `_declared` and
    # `choices` falls back to guessing the colour space from the file name. The web page reads `head_enough` on the
    # file parameter (derived from this by `param_specs`, not written a second time).
    head_is_enough: ClassVar[bool] = False

    @classmethod
    def path(cls, params: dict) -> Path:
        if not params.get("path"):
            raise ValueError(cls.no_file)
        return services().plan.upload(params["path"])

    @classmethod
    def source_identity(cls, params: dict) -> Any:
        """The identity of the source this node reads (part of the node fingerprint). Uploads are content-addressed,
        so the reference itself is the identity.

        Declared but not yet uploaded is not the same as missing (see "declare first, upload later" in
        `transfer/uploads.py`): the source is on the user's machine and has not been sent yet. Without this case,
        `Evaluation.source_missing` would mark the node E-UPLOAD-GONE (the node shows an error and everything
        downstream is skipped) although the user did nothing wrong. The identity is still the reference: its id is
        derived from the file name and content only, so the cache fingerprint does not change when the bytes arrive.
        An actual cook requires all bytes and goes through `path()`."""
        try:
            cls.path(params)  # a reference that is still on this server
        except NotFound:
            if services().plan.declared_upload(params["path"]) is None:
                raise
        return params["path"]


def empty_packet(ctx, port: str):
    """An output with nothing in it this time (e.g. the depth port of a node set to give disparity, 「创建相机」 without a
    focal length): downstream it counts as not connected (a parameter it drives keeps its own value). A port this cook
    gives (wanted) must declare it may (Port.may_be_empty): the engine and the page treat such a port so — no
    N-COOK-NOTHINGIN downstream, the parameter it drives left editable — and an undeclared one would leave a value
    nobody can see or set; `lab2shot check` (empties) holds every literal port given here to the same."""
    from ..data.packet import Packet
    from .applies import output_ports

    if port in getattr(ctx, "wanted", ()) and getattr(ctx, "node_type", None) is not None:
        declared = {p.name: p for p in output_ports(ctx.node_type, ctx.params)}
        if port in declared and not declared[port].may_be_empty:
            raise TypeError(f"{ctx.node_type.__name__} gives an empty {port!r}, which does not declare may_be_empty")
    return Packet(ctx.outputs[port], ctx.output_types[port], {"empty": True})
