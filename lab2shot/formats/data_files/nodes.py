"""Output-settings nodes for data files (interface: nodes/output.py): 「人物框输出设置」, 「2D 跟踪点输出设置」 and
「曲线输出设置」. They write tables and files for other software (JSON, CSV, 3DEqualizer, .chan, USD attributes, Nuke
script text). Nuke text is produced by the Nuke format module (formats/nuke); a node that writes it declares
「复制到 Nuke」 (OutputSettings.clipboard) so that the page can copy it from the result."""

from __future__ import annotations

from typing import Literal

from ...nodes.base import NodeParams, P, Port
from ...nodes.output import Format, OutputSettings, fps_param, learned_projects, name_param
from ...nodes.applies import incoming, Param


class BoxesOutput(OutputSettings):
    id = "boxes.output"
    format = Format("boxes")
    category = "out_picture"
    inputs = (Port("boxes", "boxes"),)
    on_node = ("name", "format")
    clipboard = "nuke"  # Tracker 格式写出 .nk 片段，供「复制到 Nuke」使用
    # JSON / CSV 不生成 Nuke 节点，无可粘贴内容：按钮保留，置灰并说明原因
    clipboard_when = Param("format").one_of("tracker")

    class Params(NodeParams):
        name: str = name_param("boxes")
        format: Literal["json", "csv", "tracker"] = P(
            "json", group="file",
        )
        origin: Literal["top_left", "bottom_left"] = P("top_left", group="file",
                                                       applies=Param("format").one_of("json", "csv"))

    @classmethod
    def write(cls, ctx) -> str:
        import csv
        import json

        from ...data.payloads import read_boxes

        import numpy as np

        fmt, bottom = ctx.params["format"], ctx.params["origin"] == "bottom_left"
        src = ctx.input("boxes")
        h = src.meta["height"]
        if fmt == "tracker":
            # 每人一条 track，跟踪框的中心，复用跟踪点的写法（formats/nuke/tracks.py write_tracker）：
            # 某帧未检测到该人物时保持上一位置并禁用该帧，与 Nuke 中跟踪丢失的 track 表现一致
            from ..nuke.tracks import MANY_TRACKS, write_tracker
            from ...data.payloads import read_boxes as _read
            people = _read(src)
            frames = list(src.meta["frames"])
            xy = np.zeros((len(people), len(frames), 2), np.float64)
            seen = np.zeros((len(people), len(frames)), bool)
            for i, person in enumerate(people):
                for j, f in enumerate(frames):
                    b = person["boxes"].get(f) or person["boxes"].get(str(f))
                    if b is None:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in b)
                    xy[i, j] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    seen[i, j] = True
            out = cls.out_file(ctx, ".nk")
            if len(people) > MANY_TRACKS:
                ctx.say("N-NUKE-MANYTRACKS", count=len(people), most=MANY_TRACKS, param="format")
            note = ", ".join(learned_projects(ctx.provenance))
            out.write_text(write_tracker(xy, seen, frames, h, ctx.params["name"], note,
                                         [str(p["id"]) for p in people]), encoding="utf-8")
            return out.name
        out = cls.out_file(ctx, f".{fmt}")

        def box(b: list[float]) -> list[float]:
            x1, y1, x2, y2 = (round(float(v), 3) for v in b)
            return [x1, round(h - y2, 3), x2, round(h - y1, 3)] if bottom else [x1, y1, x2, y2]

        people = read_boxes(src)
        if fmt == "json":
            data = {"width": src.meta["width"], "height": h, "frames": src.meta["frames"],
                    "origin": ctx.params["origin"],
                    "people": [{"id": p["id"], "prominence": p.get("prominence"),
                                "boxes": {f: box(b) for f, b in sorted(p["boxes"].items(), key=lambda kv: int(kv[0]))}} for p in people]}
            out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            with out.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["frame", "id", "x1", "y1", "x2", "y2"])
                rows = [(int(f), p["id"], *box(b)) for p in people for f, b in p["boxes"].items()]
                w.writerows(sorted(rows))
        return out.name


class TracksOutput(OutputSettings):
    id = "tracks.output"
    format = Format("tracks")
    category = "out_picture"
    inputs = (Port("tracks", "tracks2d"),)
    on_node = ("name", "format")
    clipboard = "nuke"  # the Nuke formats write a .nk snippet used by 「复制到 Nuke」
    # 3DE 与 CSV 不生成 Nuke 节点，无可粘贴内容：按钮保留，置灰并说明原因
    clipboard_when = Param("format").one_of("tracker", "cornerpin")

    class Params(NodeParams):
        name: str = name_param("tracks")
        format: Literal["3de", "csv", "tracker", "cornerpin"] = P(
            "3de", group="file",
            # CornerPin 对应平面的四个角，点数必须恰好为四（formats/nuke/tracks.py corner_pin）。
            # 输入点数不为四时该选项置灰并注明当前点数，避免选择后在计算末尾才报错
            option_applies={"cornerpin": incoming("tracks", "points").eq(4)},
        )
        origin: Literal["bottom_left", "top_left"] = P(
            "bottom_left", group="file", applies=Param("format").one_of("csv"),
        )
        min_visible: int = P(5, ge=1, group="filter",
                             applies=Param("format").one_of("3de", "csv", "tracker"))

    @classmethod
    def write(cls, ctx) -> str:
        import csv

        import numpy as np

        from ...data.payloads import read_tracks
        from ..nuke.tracks import MANY_TRACKS, corner_pin, write_tracker

        src = ctx.input("tracks")
        t = read_tracks(src)
        frames, h = src.meta["frames"], src.meta["height"]
        names = src.meta["names"]
        xy, vis = t["tracks"].astype(np.float64), t["visible"]
        note = ", ".join(learned_projects(ctx.provenance))  # tracking method, written to the Nuke node's label
        if ctx.params["format"] == "cornerpin":
            out = cls.out_file(ctx, ".nk")
            out.write_text(corner_pin(xy, vis, t["query_frames"], frames, h, ctx.params["name"], note), encoding="utf-8")
            return out.name
        need = min(ctx.params["min_visible"], len(frames))  # with fewer frames in total (e.g. image-pair matches), require all
        keep = [i for i in range(len(xy)) if vis[i].sum() >= need]
        if ctx.params["format"] == "tracker":
            out = cls.out_file(ctx, ".nk")
            if len(keep) > MANY_TRACKS:
                ctx.say("N-NUKE-MANYTRACKS", count=len(keep), most=MANY_TRACKS, param="min_visible")
            out.write_text(write_tracker(xy[keep], vis[keep], frames, h, ctx.params["name"], note, [names[i] for i in keep]),
                           encoding="utf-8")
            if len(xy) > len(keep):
                ctx.say("N-OUTPUT-SHORTTRACKS", count=len(keep), short=len(xy) - len(keep), param="min_visible")
            return out.name
        bottom = ctx.params["format"] == "3de" or ctx.params["origin"] == "bottom_left"
        y = h - xy[..., 1] if bottom else xy[..., 1]
        out = cls.out_file(ctx, ".txt" if ctx.params["format"] == "3de" else ".csv")
        if ctx.params["format"] == "3de":
            lines = [str(len(keep))]
            for i in keep:
                seen = [j for j in range(len(frames)) if vis[i, j]]
                lines += [names[i], "0", str(len(seen))] + [f"{frames[j]} {xy[i, j, 0]:.4f} {y[i, j]:.4f}" for j in seen]
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            with out.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                sure = t.get("confidence")  # per-point confidence in 0..1, when the tracker provides it
                w.writerow(["frame", "point", "x", "y", "visible"] + (["confidence"] if sure is not None else []))
                for i in keep:
                    for j, f in enumerate(frames):
                        w.writerow([f, names[i], f"{xy[i, j, 0]:.4f}", f"{y[i, j]:.4f}", int(vis[i, j])]
                                   + ([f"{sure[i, j]:.3f}"] if sure is not None else []))
        if len(xy) > len(keep):
            ctx.say("N-OUTPUT-SHORTTRACKS", count=len(keep), short=len(xy) - len(keep), param="min_visible")
        else:
            ctx.say("I-OUTPUT-TRACKS", count=len(keep))
        return out.name


class CurvesOutput(OutputSettings):
    id = "curves.output"
    format = Format("curves")
    version = 2  # 2：USD 里曲线属性名按 lab2shot_shared/names.py 写，原名存在属性的 customData（io/usd.py named_attr）
    category = "out_picture"
    inputs = (Port("curves", "curves"),)
    on_node = ("name", "format")

    class Params(NodeParams):
        name: str = name_param("curves")
        format: Literal["csv", "chan", "json", "usd"] = P(
            "csv", group="file",
        )
        # 仅 USD 按时间存储（timeCodesPerSecond）；CSV / .chan / JSON 按行对应帧，不写帧率
        fps: float = fps_param(applies=Param("format").one_of("usd"))

    @classmethod
    def write(cls, ctx) -> str:
        import csv
        import json

        from ...data.payloads import read_curves

        fmt = ctx.params["format"]
        out = cls.out_file(ctx, f".{fmt}")
        src = ctx.input("curves")
        values, names, frames = read_curves(src), src.meta["names"], src.meta["frames"]
        if fmt == "csv":
            with out.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["frame", *names])
                w.writerows([f, *(f"{v:.6f}" for v in row)] for f, row in zip(frames, values))
        elif fmt == "chan":  # one line per frame without a header; channel names go to <name>_names.txt
            out.write_text("".join(" ".join(f"{v:.6f}" for v in row) + "\n" for row in values), encoding="utf-8")
            out.with_name(f"{out.stem}_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        elif fmt == "json":
            data = {"frames": frames, "curves": {n: [round(float(v), 6) for v in values[:, c]] for c, n in enumerate(names)}}
            out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        else:
            from pxr import Sdf, UsdGeom

            from ...io import usd

            stage = usd.create_stage(frames, {"curves": len(names)})
            prim = UsdGeom.Xform.Define(stage, f"{usd.ROOT_PATH}/curves").GetPrim()
            for c, n in enumerate(names):
                attr = usd.named_attr(prim, f"lab2shot:curve:{n}", Sdf.ValueTypeNames.Float)
                for f, v in zip(frames, values[:, c]):
                    attr.Set(float(v), f)
            # 帧率仅写入交付文件（nodes/output.py fps_param）：帧号不变，只设置时基
            stage.SetTimeCodesPerSecond(float(ctx.params["fps"]))
            stage.SetFramesPerSecond(float(ctx.params["fps"]))
            usd.save_stage(stage, out)
        return out.name


NODES = (BoxesOutput, TracksOutput, CurvesOutput)
