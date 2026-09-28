"""Format modules: how 3D data comes into Lab2Shot from DCC files and goes out to them, behind one interface.

A format module (the core's USD in lab2shot/formats/usd; the others in their extensions; a volume format later, with its
own node and data type) gives:
- its file extensions;
- a hierarchy reader: every entry of the file per kind of 3D data, as the DCC tells them apart (types.SCENE_KINDS: 相机,
  模型, 点云, 三维曲线, 骨架动画 — bones alone —, 蒙皮角色 — a mesh skinned to them) by its path in the file's hierarchy as the DCC
  shows it, and the selected entries read into our packets (centimetres, Y up,
  every parent transform composed, the file's frame numbers);
- its import node: an ImportNode below (one per format: 「导入 USD」 and the extensions' 「导入 …」);
- its output-settings node: an OutputSettings (nodes/output.py) with what it writes per kind (`writes`) and write();
- its units and axes: recorded by the file (USD metersPerUnit and upAxis) or said by the node's parameters.
A module read and written by its extension's worker hands the core scene arrays (lab2shot_shared/scene_arrays.py) and
gets them back (data/scene_arrays.py): WorkerImport below does the reading side.

The core owns the kinds, what a port carries (Graph.scene_kinds), 「输出」 and delivery, and the checks; it never
mentions a format (no format name appears in the core).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ..errors import Invalid
from ..messages import Msg
from .base import Info, NodeDef, P, Port, ReadsFile
from ..data.types import DEFORMING, SCENE_KINDS

if TYPE_CHECKING:
    from pathlib import Path

    from ..data.packet import Packet

# the kinds an import node selects, each a parameter and an output port of that name: 相机 one, the others any number
# (灯光 are ours, made by the light-probe nodes: not read from DCC files)
SELECTIONS = {"camera": "camera", "models": "model", "points": "points", "curves": "curves",
              "skeletons": "skeleton", "characters": "character"}
PORT_OF = {kind: port for port, kind in SELECTIONS.items()}


# ------------------------------------------------------------------ what a file holds


@dataclass(frozen=True)
class Entry:
    """One thing in a file: its kind, its path in the hierarchy as the DCC shows it, the frames it has, and what tells
    it apart from its neighbours (a camera's focal length, a mesh's points, a skeleton's joints)."""

    kind: str  # types.SCENE_KINDS
    path: str
    frames: tuple[int, ...]
    detail: str = ""  # "35 mm", "1,204 顶点", "52 关节 · 2 网格"
    deforming: bool = False  # a 模型 whose points change every frame
    size: tuple[int, int] | None = None  # a camera's picture size, when the file records one
    aspect: float = 0.0  # a camera's filmback height / width: its picture's proportions when the file records no size

    @property
    def said(self) -> str:
        """What tells it apart, next to its path: its frames, whether it deforms, its detail."""
        span = f"{self.frames[0]}–{self.frames[-1]}" if len(self.frames) > 1 else "静止"
        return " · ".join(x for x in (span, "变形" if self.deforming else "", self.detail) if x)

    @property
    def label(self) -> str:
        return f"{self.path} · {self.said}"


@dataclass(frozen=True)
class Listing:
    """Everything a file holds, per kind, in the file's order. A production file holds thousands:
    looked up by kind and path through an index. `options`: the values the file allows a parameter of the import node
    beyond the kinds (the animation takes of a file that holds several: the format module names the parameter, the
    core never does), `auto`: what that parameter left empty reads."""

    entries: tuple[Entry, ...]
    options: Mapping[str, tuple[str, ...]] = field(default_factory=dict, compare=False)
    auto: Mapping[str, str] = field(default_factory=dict, compare=False)

    @cached_property
    def _kinds(self) -> dict[str, list[Entry]]:
        out: dict[str, list[Entry]] = {}
        for e in self.entries:
            out.setdefault(e.kind, []).append(e)
        return out

    @cached_property
    def _paths(self) -> dict[tuple[str, str], Entry]:
        return {(e.kind, e.path): e for e in self.entries}

    def of(self, kind: str) -> list[Entry]:
        return self._kinds.get(kind, [])

    def find(self, kind: str, path: str) -> Entry | None:
        return self._paths.get((kind, path))


def count(n: int, what: str) -> str:
    return f"{n:,} {what}"


# ------------------------------------------------------------------ the import node


def selection_param(port: str, listing_from: tuple[str, ...] = ("path",)) -> Any:
    """An import node's selection of one kind: 相机 one (a path, "" none), the others any number (a list of paths).
    Widget "hierarchy": the panel shows what is chosen, and 「选择…」 opens the file's hierarchy as a tree to pick in,
    as Houdini's scene-graph tree does (a production file holds thousands of prims). Its options are the file's
    entries of that kind (NodeDef.choices, the one options mechanism), listed again when the file or `listing_from`
    changes; picking a file selects the only entry of a kind (NodeDef.derive)."""
    kind = SCENE_KINDS[SELECTIONS[port]]
    many = port != "camera"
    return P([] if many else "", label=kind.label, widget="hierarchy", group="层级", choices_from=listing_from, derived_from=("path",),
             worker=False)


def selection_ports(params: type) -> tuple[Port, ...]:
    """An import node's outputs: one per kind its Params select (selection_param), there once something of the kind is
    selected; a wire from it before waits for the selection. The 模型 port says whether the ones selected deform (the
    node's fact "models.kinds")."""
    from .applies import Param

    return tuple(Port(port, SCENE_KINDS[SELECTIONS[port]].type, SCENE_KINDS[SELECTIONS[port]].label, when=Param(port).set(),
                      waits=f"选一{'台' if port == 'camera' else '个'}{SCENE_KINDS[SELECTIONS[port]].label}",
                      kinds_from="models.kinds" if port == "models" else "")
                 for port in SELECTIONS if port in params.model_fields)


def import_file_param(suffixes: tuple[str, ...]) -> Any:
    return P("", label="文件", widget="file", group="文件", accept=list(suffixes))


class ImportNode(ReadsFile, NodeDef):
    """A format module's import node: a DCC file's entries per kind, the ones selected given on an output port of
    their kind (a port only while something of it is selected: Port.when). A format module subclasses it with its
    `suffixes`, its Params — its file (import_file_param), a selection (selection_param) for each kind its format
    holds, the settings its format does not record —, `outputs = selection_ports(Params)`, listing() and read()."""

    category = "read_scene"
    named_result = True  # its entries go under /shot/<the node's name, or the file's stem>/... (io/usd.py import_group)
    no_file = "没有选择文件"
    suffixes: ClassVar[tuple[str, ...]] = ()
    fact_labels = {"models.kinds": "选的模型"}

    @classmethod
    def path(cls, params: dict) -> Path:
        """The picked file, only when its name ends in one of the format's `suffixes`: the page's file picker offers
        only those, but a request may name any upload, and a format library picks its reader by the file (the FBX
        SDK reads .obj, .dxf, .3ds ... too). Checked here, before any reader sees the file."""
        found = super().path(params)
        if found.suffix.lower() not in cls.suffixes:
            raise Invalid(Msg("E-FORMAT-SUFFIX", name=found.name, suffixes=" ".join(cls.suffixes)))
        return found

    @classmethod
    def selections(cls) -> list[str]:
        """The selections (and output ports) of the kinds its format holds, as its Params declare them."""
        return [port for port in SELECTIONS if port in cls.Params.model_fields]

    @classmethod
    def listing(cls, params: dict) -> Listing:
        """Everything the picked file holds (cheap enough for the editor to ask: cached per file and settings)."""
        raise NotImplementedError

    @classmethod
    def read(cls, ctx, chosen: dict[str, list[Entry]]) -> dict[str, Packet]:
        """The chosen entries (port -> its entries, only ports with some) as packets, one per port into ctx.outputs."""
        raise NotImplementedError

    @classmethod
    def chosen(cls, params: dict) -> dict[str, list[Entry]]:
        """The selected entries per port. A selection the file does not have is an error listing what it has (a newly
        picked file never silently falls back to something else)."""
        listing = cls.listing(params)
        out = {}
        for port in cls.selections():
            kind = SELECTIONS[port]
            value = params.get(port) or []
            paths = [value] if isinstance(value, str) else list(value)
            missing = [p for p in paths if listing.find(kind, p) is None]
            if missing:
                label, have = SCENE_KINDS[kind].label, [e.path for e in listing.of(kind)]
                if have:
                    raise Invalid(Msg("E-FORMAT-NOTINFILE", kind=label, missing=missing, have=have))
                raise Invalid(Msg("E-FORMAT-NONEINFILE", kind=label, missing=missing))
            if paths:
                out[port] = [listing.find(kind, p) for p in paths]
        return out

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        """The file's entries per kind, for its selection: by their paths (the hierarchy picker builds its tree from
        them, every kind's together), with what tells them apart (`details`: frames, deforming, focal length, counts).
        Asked once per file: every selection of the node comes from the one answer, and so does a parameter whose values
        the file decides (Listing.options: a file's animation takes)."""
        from_file = [*cls.selections(), *(s["name"] for s in cls.param_specs()
                                          if "path" in s["choices_from"] and s["name"] not in cls.selections())]
        if not params.get("path"):
            return {name: {"options": [], "empty": "先选择文件"} for name in from_file}
        try:
            listing = cls.listing(params)
        except ValueError as exc:  # a file it can't read, or won't (a USD needing other files): said where its kinds are
            return {name: {"options": [], "empty": str(exc)} for name in from_file}
        out = {}
        for port in cls.selections():
            kind = SELECTIONS[port]
            found = listing.of(kind)
            out[port] = {"options": [e.path for e in found], "details": {e.path: e.said for e in found},
                         **({} if found else {"empty": f"文件里没有{SCENE_KINDS[kind].label}"})}
        for name in from_file[len(cls.selections()):]:
            values = list(listing.options.get(name, ()))
            out[name] = {"options": values, "auto": listing.auto.get(name, ""), **({} if values else {"empty": "文件里没有"})}
        return out

    @classmethod
    def derive(cls, params: dict) -> dict:
        """Picking a file selects the only entry of a kind (several: the user chooses); a selection made before stays
        as it is (one the new file does not have is then an error that says so)."""
        if not params.get("path"):
            return {}
        try:
            listing = cls.listing(params)
        except ValueError:  # a file it can't read: nothing follows from it (the node and its selections say why)
            return {}
        out = {}
        for port in cls.selections():
            found = listing.of(SELECTIONS[port])
            if not params.get(port) and len(found) == 1:
                out[port] = found[0].path if port == "camera" else [found[0].path]
        return out

    @classmethod
    def source_identity(cls, params):
        cls.chosen(params)  # every selection is in the file: else the node can't be planned
        return super().source_identity(params)

    @classmethod
    def info(cls, params, inputs):
        """What the node gives, known before it is cooked: the selected entries' frames and the chosen camera's
        picture size (the nodes after it plan their picture by it)."""
        chosen = cls.chosen(params)
        entries = [e for es in chosen.values() for e in es]
        camera = next(iter(chosen.get("camera", [])), None)
        size = cls.picture_size(params, camera) if camera is not None else (0, 0)
        return Info(tuple(sorted({f for e in entries for f in e.frames})), *size)

    @classmethod
    def picture_width(cls, params: dict) -> int:
        """The picture width of a camera whose file records no resolution (a format that records none says it in a
        parameter)."""
        from ..data.units import DEFAULT_WIDTH

        return DEFAULT_WIDTH

    @classmethod
    def picture_size(cls, params: dict, camera: Entry) -> tuple[int, int]:
        """A camera's picture: the size its file records, else picture_width() and its filmback's proportions."""
        if camera.size:
            return camera.size
        w = cls.picture_width(params)
        return w, int(round(w * (camera.aspect or 9 / 16)))

    @classmethod
    def facts(cls, params: dict) -> dict:
        """Whether the 模型 selected deform (a point cache), said on the port (Port.kinds_from): a format without
        per-frame vertices refuses them. Not known without a file, or with one that can't be read (the node says why)."""
        from .applies import Fact

        if not params.get("models"):  # no 模型 selected: no port to say it on, and the file is not read for it
            return {}
        try:
            chosen = cls.chosen(params).get("models", [])
        except (OSError, ValueError):
            return {}
        return {"models.kinds": Fact(frozenset({"model", *([DEFORMING] if any(e.deforming for e in chosen) else [])}), "选的模型")}

    @classmethod
    def cook(cls, ctx):
        return cls.read(ctx, cls.chosen(ctx.params))


# ------------------------------------------------------------------ a format read into scene arrays


class ArraysImport(ImportNode):
    """An import node whose format is read into scene arrays (lab2shot_shared/scene_arrays.py: every entry of the file,
    in its unit and axes): the listing and the cook are the same for every such format, only where the arrays come
    from differs (arrays(): its extension's worker, or the core reading the file itself). The module says how the file's
    lengths and axes become ours (axes())."""

    @classmethod
    def axes(cls, params: dict, top: dict):
        """The file's unit and axes: what it records (top: the arrays' unit_cm and axes) or the node's settings say."""
        from ..data.scene_arrays import Axes

        return Axes(float(top.get("unit_cm", 1.0)), np.asarray(top["axes"]) if "axes" in top else np.eye(3))

    @classmethod
    def arrays(cls, params: dict, ctx=None):
        """The file's scene arrays (a .npz path or the arrays themselves), for the listing (ctx None) or in a cook."""
        raise NotImplementedError

    @classmethod
    def listing(cls, params: dict) -> Listing:
        return _arrays_listing(cls.id, tuple(sorted((k, v) for k, v in cls.worker_params(params).items())))

    @classmethod
    def read(cls, ctx, chosen):
        from lab2shot_shared import scene_arrays as sa

        from ..data.scene_arrays import items_to_packets
        from ..io.usd import import_group

        arrays = cls.arrays(ctx.params, ctx)
        top, _ = sa.load(arrays)
        paths = {SELECTIONS[port]: [e.path for e in entries] for port, entries in chosen.items()}
        outs = {SELECTIONS[port]: ctx.outputs[port] for port in chosen}
        group = import_group(ctx.label, cls.label, cls.path(ctx.params).name)  # /shot/<the node's name or the file's stem>/...
        packets = items_to_packets(arrays, paths, cls.axes(ctx.params, top), cls.picture_width(ctx.params), outs,
                                   {"imported_from": cls.path(ctx.params).name}, group, ctx.fingerprint)
        return {PORT_OF[kind]: p for kind, p in packets.items()}


class WorkerImport(ArraysImport):
    """A format its extension's worker reads: one job per file writes every entry of it as scene arrays
    (raw/scene.npz); the listing and the cook share that job and its cached result, so the editor's list costs one
    read of the file."""

    @classmethod
    def arrays(cls, params: dict, ctx=None):
        from ..errors import CookError
        from .services import services

        if ctx is not None:
            return ctx.run_worker(None, inputs={"file": cls.path(ctx.params)}) / "scene.npz"
        try:  # the job a cook of the node sends (PlanEnv.ask_worker), so either reuses the other's result
            return services().plan.ask_worker(cls, params, {"file": cls.path(params)}) / "scene.npz"
        except CookError as exc:  # the extension is not installed, the file is not of its format ...
            raise Invalid(exc.message) from exc


@lru_cache(maxsize=32)
def _arrays_listing(node: str, params: tuple) -> Listing:
    """Everything a file holds, from its scene arrays outside any cook; once per file and settings (uploads never
    change)."""
    from lab2shot_shared import scene_arrays as sa

    from .registry import node_types

    t = node_types()[node]
    top, items = sa.load(t.arrays(dict(params)))
    entries = []
    for kind in ("camera", "model", "points", "curves", "character"):
        for item in items[kind]:
            frames = tuple(int(f) for f in np.asarray(item["frames"]).reshape(-1))
            entries.append(Entry(_kind(kind, item), sa.text(item["path"]), frames, *_detail(kind, item)))
    options = {k.removeprefix("options_"): tuple(sa.texts(v)) for k, v in top.items() if k.startswith("options_")}
    auto = {k.removeprefix("auto_"): sa.text(v) for k, v in top.items() if k.startswith("auto_")}
    return Listing(tuple(entries), options, auto)


def _kind(array_kind: str, item: dict) -> str:
    """The kind an item of the arrays is, as DCCs tell them apart: a character item without a mesh is bones alone (骨架动画)."""
    return "skeleton" if array_kind == "character" and not item.get("meshes") else array_kind


def _detail(kind: str, item: dict) -> tuple:
    """What tells an entry apart, whether a 模型 deforms, a camera's recorded picture size."""
    if kind == "camera":
        size = tuple(int(v) for v in np.asarray(item["resolution"]).reshape(-1)[:2]) if "resolution" in item else None
        aspect = float(np.median(item["v_aperture_mm"]) / np.median(item["h_aperture_mm"]))
        return f"{float(np.asarray(item['focal_mm']).reshape(-1)[0]):.4g} mm", False, size, aspect
    if kind == "model":
        pts = np.asarray(item["points"])
        return count(pts.shape[-2], "顶点"), pts.shape[0] > 1, None
    if kind == "points":
        return count(int(np.asarray(item["counts"]).max(initial=0)), "点"), False, None
    if kind == "curves":
        strands = count(int(np.asarray(item["curve_counts"]).max(initial=0)), "条")
        return f"{strands} · {count(int(np.asarray(item['counts']).max(initial=0)), '点')}", False, None
    meshes = item.get("meshes", [])
    shapes = sum(len(np.asarray(m.get("shapes", [])).reshape(-1)) for m in meshes)
    parts = [count(len(np.asarray(item["joints"]).reshape(-1)), "关节")] + ([count(len(meshes), "网格")] if meshes else []) + \
        ([count(shapes, "形变")] if shapes else [])
    return " · ".join(parts), False, None
