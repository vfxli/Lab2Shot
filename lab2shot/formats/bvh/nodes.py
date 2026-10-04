"""The BVH module's node (the interface: lab2shot/nodes/formats.py): 「导入 BVH」 lists and reads a .bvh in the core's
environment (reader.py). Writing BVH is not offered: a skeleton animation goes out through an output setting that
holds skeletons."""

from __future__ import annotations

from ... import i18n

from typing import Literal

from ...data.units import to_cm
from ...nodes.base import NodeParams, P, Reads
from ...nodes.formats import ArraysImport, import_file_param, selection_param, selection_ports
from . import SUFFIXES


class ImportBvh(ArraysImport):
    # 10：Frame Time 正好是整数帧率时就是它（0.1 → 10，不是 12）
    version = 10
    id = "bvh.import"
    suffixes = SUFFIXES
    on_node = ("unit",)
    reads = Reads(rank=3)  # 骨架：FBX 在前

    class Params(NodeParams):
        path: str = import_file_param(SUFFIXES)
        skeletons: list[str] = selection_param("skeletons")
        # 「自动」：按骨头长度估出来的每单位多少厘米（reader.unit_guess），与 W-BVH-UNIT 提示用的同一个估计
        unit: Literal["cm", "m", "auto"] = P("cm", group="bvh", worker=False)
    outputs = selection_ports(Params, fps=True, fps_always=True)

    @classmethod
    def unit_cm(cls, params) -> float:
        """Centimetres per file unit: the chosen unit, or with 「自动」 the guess from the bones (reader.unit_guess;
        1 when the skeleton has none to measure)."""
        from lab2shot_shared import scene_arrays as sa

        from . import reader

        if params["unit"] != "auto":
            return to_cm(params["unit"])
        _, items = sa.load(cls.arrays(params))
        return (reader.unit_of(items) or (1.0, ""))[0]

    @classmethod
    def axes(cls, params, top):
        from ...data.scene_arrays import Axes

        return Axes.of(cls.unit_cm(params), "y")  # BVH is Y up

    @classmethod
    def read(cls, ctx, chosen):
        """The skeleton as its file has it, and W-BVH-UNIT when the chosen unit makes it far from a person's size (BVH
        records no unit): a retarget measures by leg length and does not mind, a direct delivery would."""
        from lab2shot_shared import scene_arrays as sa

        from . import reader

        out = super().read(ctx, chosen)  # the file parsed once: reader.parsed keeps it for the lines below
        _, items = sa.load(cls.arrays(ctx.params, ctx))
        guess, chosen_cm = reader.unit_guess(items), cls.unit_cm(ctx.params)
        if ctx.params["unit"] == "auto":
            unit = reader.unit_of(items)
            if unit and unit[1]:
                ctx.say("I-BVH-UNITAUTO", unit=i18n.Word(f"length.{unit[1]}"), guess=guess)
            elif unit:  # near no standard unit (CMU's 0.45 inch): read at the guess, and said
                ctx.say("W-BVH-UNITGUESS", guess=guess)
        elif guess and max(guess / chosen_cm, chosen_cm / guess) > reader.UNIT_OFF:
            ctx.say("W-BVH-UNIT", unit=i18n.Word("length.cm" if ctx.params["unit"] == "cm" else "length.m"),
                    reach=reader.reach(items) * chosen_cm, guess=guess)
        _, replaced, extra = reader.parsed(str(cls.path(ctx.params)))
        if replaced:
            ctx.say("W-BVH-NAMEENCODING", name=cls.path(ctx.params).name, count=len(replaced), names=replaced[:5])
        if extra:
            ctx.say("W-BVH-EXTRAFRAMES", name=cls.path(ctx.params).name, count=extra)
        return out

    @classmethod
    def arrays(cls, params, ctx=None):
        from . import reader

        return reader.arrays(cls.path(params))


NODES = (ImportBvh,)
