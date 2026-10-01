"""输出: how results leave Lab2Shot, in two kinds of node.

- An output-settings node says how one piece of data is written: its 名字 and its format's settings (「USD 输出设置」,
  「序列图输出设置」 …). Every format module gives its own, through one interface, OutputSettings below: the core's
  pictures and data files (nodes/core/image_output.py, data_output.py), the USD module (lab2shot/formats/usd), the other
  3D formats in their extensions (lab2shot/nodes/formats.py). write() writes the files, named after the 名字
  (OutputSettings.out_file); they are the node's result (type files, cached like any other), shown in the viewer as what
  they were made from. A 3D one declares per kind of 3D data what its format holds (`writes`): a kind it can't hold is
  refused on the wire, never dropped and never turned into something else on the quiet.
- 「输出」 (nodes/core/output.py) takes the files of the settings nodes wired into it and collects them for the user
  into its task's folder, one sub-folder per 名字, then packs that folder into one zip: CookContext.collector
  (nodes/services.py OutputSink) and lab2shot/transfer/outputs.py.
  The rules (a settings node in a cook is wired into an 「输出」, and the names under it differ) are
  Graph.check_delivery.

Nothing here, in 「输出」 or in the engine knows a format.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from ..errors import Invalid
from ..io.files import inside
from ..messages import Msg
from .base import NodeDef, P, Port
from .clipboard import Pasteable, clipboard_meta
from ..data.types import DATA_TYPES, DEFORMING, KIND_ORDER, kind_label, kind_of

def _either(items: list, code: str):
    """Alternatives as one message: the first, or (code: E-OUTPUT-OR / E-OUTPUT-ORELSE) the rest."""
    return items[0] if len(items) == 1 else Msg(code, first=items[0], then=_either(items[1:], code))


ML_MARK = "ML_Lab2Shot"  # in the name of every file (and output's download) a machine-learning model made


def learned_projects(provenance: dict) -> list[str]:
    """The machine-learning projects behind a result, upstream first, once each: the provenance's sources whose node
    runs a model (Evaluation.provenance "learned"), not a format module reading or writing a DCC file."""
    return list(dict.fromkeys(s["project"] for s in provenance.get("sources", []) if s.get("learned")))


def marked(name: str, projects: list[str]) -> str:
    """A delivered file's or delivery's name with the mark that a model made it (metadata goes unseen, the name is
    what a production artist sees): `名字_ML_Lab2Shot_<项目>[-<项目>…]`, the project titles in
    ASCII (「Pi3 (π³)」 → Pi3, 「SAM 3D Body」 → SAM3DBody); a name without projects stays as it is, as does one that
    already carries the mark. The one rule for every file (OutputSettings.out_file) and output's download
    (transfer/outputs.py)."""
    if not projects or ML_MARK in name:
        return name
    names = [n for n in dict.fromkeys(re.sub(r"[^A-Za-z0-9]", "", re.sub(r"\(.*?\)", "", p)) for p in projects) if n]
    return f"{name}_{ML_MARK}" + (f"_{'-'.join(names)}" if names else "")


FILES = "files"  # the data type of what a settings node gives 「输出」: the files it wrote
FILES_OUT = Port("files", FILES, "文件")


# what a file name may not hold (Windows' reserved characters, the separators, control characters)
_BAD_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')


# names Windows keeps for devices, with any extension (CON, con.txt): a folder or file of that name can't be unpacked there
_RESERVED = re.compile(r"(?i)^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$")


def name_key(name: str) -> str:
    """What two names of files or folders are compared by: one on some file system if this is the same (Unicode
    normalised, NFC, as macOS spells it differently; case folded, as Windows and macOS don't tell 「A」 from 「a」). The
    one rule: 「输出」's sub-folders per 名字 (engine/graph.py check_delivery, core/output.py), its items' sub-folders
    (transfer/outputs.py _item_dir)."""
    return unicodedata.normalize("NFC", name).casefold()


def file_name(name, label: str, error: type[Exception] | None = None) -> str:
    """An output-settings node's 名字 as the name of its files and its sub-folder: one plain file name, never a path.
    Returns it stripped and NFC-normalised; GraphError (B-DELIVER-BADNAME) when it is empty, "." / "..", holds a separator
    or a character a file name may not, ends in a dot or a space, or is a name Windows keeps for a device (CON, con.txt). The one rule, for a name the graph knows (engine/graph.py check_delivery) and for one a wire
    gives, known only when the node cooks (OutputSettings.stem; `error` Invalid there: a node's own error, not the graph's)."""
    from ..errors import GraphError

    name = unicodedata.normalize("NFC", str(name or "").strip())
    if not name or name in (".", "..") or _BAD_NAME.search(name) or name.endswith((".", " ")) or _RESERVED.match(name):
        raise (error or GraphError)(Msg("B-DELIVER-BADNAME", node=label, name=name))
    return name


def name_param(default: str) -> Any:
    """A settings node's 名字: 「输出」 puts its files in a sub-folder of that name and names them after it."""
    return P(default, label="名字", group="文件", unique=True)


# 交付的帧率（另一处要帧率的是动作模型节点，data/units.py DEFAULT_FPS 的说明）。只有文件本身按时间存的格式要它
# （时间码、或者直接按秒存）；序列图、CSV、.chan、Nuke 的关键帧都只认帧号，那些节点上没有这个参数。
def fps_param(*, applies=None) -> Any:
    """A settings node's 帧率, when its format stores time: written onto the file it delivers, nothing else."""
    from ..data.units import DEFAULT_FPS

    return P(DEFAULT_FPS, label="帧率", unit="fps", group="文件", gt=0, applies=applies)


TAKE = "core.take"  # 「按种类取出」: a scene by kind (nodes/core/scene.py Take), the fix for a wire that carries some kinds a format holds


@dataclass(frozen=True)
class Writes:
    """What a 3D settings node's format holds of one kind of 3D data (OutputSettings.writes): all of it, only still ones
    (模型: a format without per-frame vertex caches), or none. `note` says why a wire is refused (a kind it cannot
    hold, a deforming 模型 into a format that holds still ones only); `via` names a node that turns this kind into one
    the format holds: 「烘焙成模型」 for a 蒙皮角色 into a format without skeletons."""

    how: Literal["full", "static", "no"]
    note: str = ""
    via: str = ""  # a node type to put in front, as the refusal says
    # what this format cannot carry through of a kind it does take (a format that holds a 模型 but has no way to keep
    # the 分区 on its mesh): marked 部分 in 支持的数据, never a refusal — the data still gets written, part of it does
    # not. A format that drops something says so, it never drops it on the quiet
    lost: str = ""

    @classmethod
    def full(cls, lost: str = "") -> Writes:
        return cls("full", lost=lost)

    @classmethod
    def static(cls, note: str, lost: str = "") -> Writes:
        return cls("static", note, lost=lost)

    @classmethod
    def no(cls, note: str, via: str = "") -> Writes:
        return cls("no", note, via)

    def takes(self, kind: str) -> bool:
        """Whether a wire carrying `kind` (or a deforming 模型, types.DEFORMING) can come in."""
        return self.how == "full" or (self.how == "static" and kind != DEFORMING)


class OutputSettings(Pasteable, NodeDef):
    """A format module's settings node: how one piece of data is written. A subclass declares its inputs, its
    Params (with `name = name_param("默认名字")` and its format's settings), for 3D data `writes`, and write()."""

    # 每个子类自己声明它写的是哪一类（categories.py 的「输出」下面：三维 / 画面与数据），
    # 基类不替它猜——漏了就是没分类（节点菜单按 menu/nodes.json 归类，不看它）
    outputs = (FILES_OUT,)
    # a 3D one: what its format holds of each kind of 3D data (types.SCENE_KINDS), every kind declared. The editor
    # shows it (支持的数据); a wire carrying a kind it can't hold is a wiring problem (refuses),
    # and what reaches write() anyway (a scene the graph could not tell) fails the cook with the same words
    writes: ClassVar[dict[str, Writes]] = {}
    # 「复制到 Nuke」 is declared by the Pasteable mixin (nodes/clipboard.py): an output-settings node writes the
    # snippet as one of its delivered files, so `clipboard_file` says which suffix it is and files_packet records it.
    # Nothing else in the core knows the application or the format

    @classmethod
    def write(cls, ctx) -> str:
        """Write the files, each at cls.out_file(ctx, suffix) or, one per frame, cls.out_file(ctx, suffix, frame); return
        the name of the main one (a sequence as its #### pattern: 名字.####.exr)."""
        raise NotImplementedError

    @classmethod
    def out_file(cls, ctx, suffix: str, frame: int | None = None) -> Path:
        """Where the node writes a file: named after its 名字, marked when a model made what it writes (marked:
        名字_ML_Lab2Shot_ViPE.usd; a sequence: one per frame, 名字_ML_Lab2Shot_SAM3.1001.exr, frame numbers as they
        are, four digits at least) in its files packet, which 「输出」 delivers. The one place a delivered file is named."""
        name = cls.stem(ctx)
        # the graph's check keeps 名字 one plain file name (engine/graph.py check_delivery); `inside` makes sure the file
        # lands in the node's own result folder even if a name got past it
        return inside(ctx.outputs["files"], f"{name}.{frame:04d}{suffix}" if frame is not None else f"{name}{suffix}")

    @classmethod
    def stem(cls, ctx) -> str:
        """What every file the node writes is named after: its 名字, marked when a model made it. The 名字 is checked here
        by the graph's own rule (engine/graph.py file_name): a wired one is known only now."""
        return marked(file_name(ctx.params["name"], ctx.label, Invalid), learned_projects(ctx.provenance))

    @classmethod
    def sequence_main(cls, ctx, suffix: str) -> str:
        """A sequence's main file as write() returns it, its #### pattern: the name out_file gives each frame."""
        return f"{cls.stem(ctx)}.####{suffix}"

    @classmethod
    def files_packet(cls, ctx, main: str) -> dict:
        """The node's result: the files it wrote at out_file() — `main` the main one (a sequence as its #### pattern) —
        with a sidecar <名字>.lab2shot.json recording the projects behind them and their licences (CookContext.provenance,
        worked out by the engine)."""
        from ..data.packet import Packet

        prov = ctx.provenance
        out, name = ctx.outputs["files"], file_name(ctx.params["name"], ctx.label, Invalid)
        inside(out, f"{name}.lab2shot.json").write_text(json.dumps({"file": main, **prov}, ensure_ascii=False, indent=2), encoding="utf-8")
        made_from = next((p.type for ps in ctx.inputs.values() for p in ps), "")
        written = sorted(f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file())
        # the file this result can be pasted from, when the node wrote one this time (a settings node whose Nuke
        # formats are one choice among several writes it only for those)
        snippet = next((f for f in written if f.endswith(cls.clipboard_file)), "") if cls.clipboard else ""
        return {"files": Packet(out, FILES, {"name": name, "main": main, "files": written, "commercial": prov["commercial"],
                                             "learned": learned_projects(prov),
                                             "made_from": made_from if made_from in DATA_TYPES else "",
                                             **(clipboard_meta(cls.clipboard, snippet) if snippet else {})})}

    @classmethod
    def cook(cls, ctx) -> dict:
        if cls.writes:
            from ..data.scene import kinds_held, scenes

            # a 场景 input brings one scene per wire, or a list of them: `scenes` is the one place either becomes the
            # scenes to write, each with the name it goes under — a list packet holds no scene itself
            got = [named for packets in ctx.inputs.values()
                   for named in scenes([p for p in packets if p.type.startswith("scene")])]
            for _name, p in got:
                if why := cls.refusal(frozenset(kinds_held(p)), p.type == "scene"):
                    raise Invalid(why)
        return cls.files_packet(ctx, cls.write(ctx))

    @classmethod
    def writes_of(cls, kind: str) -> Writes:
        return cls.writes[kind.split(".")[0]]

    @classmethod
    def refusal(cls, kinds: frozenset[str], packed: bool = False) -> Msg | None:
        """Why this node's format can't hold what a wire carries (None it can): the first kind it refuses, with what to
        do instead (put 「烘焙成模型」 in front, wire it into a settings node that holds it, or, for a whole 场景 —
        `packed` — leave it out of the pack)."""
        from .registry import node_types

        for kind in [k for k in KIND_ORDER if k in kinds]:
            w = cls.writes_of(kind)
            if w.takes(kind):
                continue
            label = kind_label(kind)
            others = [f"「{t.label}」" for t in node_types().values() if t is not cls and issubclass(t, OutputSettings)
                      and t.writes and t.writes_of(kind).takes(kind)]
            fixes = ([Msg("E-OUTPUT-FIXVIA", node=node_types()[w.via].label)] if w.via in node_types() else []) + \
                    ([Msg("E-OUTPUT-FIXOTHER", kind=label, nodes=_either(others, "E-OUTPUT-OR"))] if others else []) + \
                    ([Msg("E-OUTPUT-FIXUNPACK", kind=label)] if packed else [])
            if not fixes:
                return Msg("E-OUTPUT-CANTHOLD", kind=label, why=w.note)
            return Msg("E-OUTPUT-CANTHOLDFIX", kind=label, why=w.note, fixes=_either(fixes, "E-OUTPUT-ORELSE"))
        return None

    @classmethod
    def refusal_fix(cls, data_type: str, kinds: frozenset[str] | None = None) -> str:
        """The node to put in front of what it refuses: 「按种类取出」 when the wire also carries a kind its format holds (that
        kind gets through: fix_kind), else the node its declaration names for the first kind it refuses (Writes.via)."""
        if not cls.writes:
            return ""
        if cls.fix_kind(data_type, kinds):
            return TAKE
        carried = kinds if kinds is not None else frozenset(k for t in data_type.split("|") if (k := kind_of(t)))
        return next((cls.writes_of(k).via for k in KIND_ORDER if k in carried and not cls.writes_of(k).takes(k)), "")

    @classmethod
    def fix_kind(cls, data_type: str, kinds: frozenset[str] | None = None) -> str:
        """The kind 「按种类取出」 lets through a wire that carries kinds its format holds and kinds it refuses: the first one it
        holds in the order it declares them ("" when the wire carries none it holds, or none it refuses; a 模型 that
        deforms is not let through as a still one)."""
        if not cls.writes:
            return ""
        carried = kinds if kinds is not None else frozenset(k for t in data_type.split("|") if (k := kind_of(t)))
        if not any(not cls.writes_of(k).takes(k) for k in carried):
            return ""
        return next((k for k in cls.writes if k in carried and cls.writes_of(k).takes(k)
                     and not (k == "model" and DEFORMING in carried)), "")

    @classmethod
    def refuses(cls, data_type: str, kinds: frozenset[str] | None = None) -> Msg | None:
        """A 3D one refuses a wire carrying a kind its format can't hold: the kinds the graph derived for the wire, or
        its type's own (a type still open between alternatives, "a|b": refused only when every one is)."""
        if not cls.writes:
            return None
        if kinds is not None:
            return cls.refusal(kinds, data_type == "scene")
        reasons = [cls.refusal(frozenset({k}) if (k := kind_of(t)) else frozenset()) for t in data_type.split("|")]
        return reasons[0] if all(reasons) else None

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {**super().describe(),
                # 「lost」 only where the format drops part of what it takes (支持的数据 marks it 部分)
                "writes": {kind: {"how": w.how, **({"lost": True} if w.lost else {})} for kind, w in cls.writes.items()}}
