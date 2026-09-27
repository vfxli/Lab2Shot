"""Output-settings nodes for data files (interface: nodes/output.py): 「人物框输出设置」, 「2D 跟踪点输出设置」,
「曲线输出设置」 and 「Nuke 相机输出设置」. They write tables and files for other software (JSON, CSV, 3DEqualizer,
.chan, USD attributes, Nuke script text). Nuke text is produced by the Nuke format module (lab2shot/formats/nuke); a
node that writes it declares 「复制到 Nuke」 (OutputSettings.clipboard) so that the page can copy it from the result."""

from __future__ import annotations

from typing import Literal

from ..base import NodeParams, P, Port
from ..output import OutputSettings, Writes, fps_param, learned_projects, name_param
from ..applies import incoming, Param

ORIGIN_LABELS = {"bottom_left": "左下", "top_left": "左上"}
ORIGIN_HELP = "像素坐标从哪个角算起。Nuke、3DE、Houdini 用左下角；图像处理和 OpenCV 用左上角"


class BoxesOutput(OutputSettings):
    id = "core.output_boxes"
    category = "out_picture"
    inputs = (Port("boxes", "boxes", "人物框"),)
    on_node = ("name", "format")
    clipboard = "nuke"  # Tracker 格式写出 .nk 片段，供「复制到 Nuke」使用
    # JSON / CSV 不生成 Nuke 节点，无可粘贴内容：按钮保留，置灰并说明原因
    clipboard_when = Param("format").one_of("tracker")

    class Params(NodeParams):
        name: str = name_param("boxes")
        format: Literal["json", "csv", "tracker"] = P(
            "json", label="格式", group="文件", option_labels={"json": "JSON", "csv": "CSV", "tracker": "Tracker"},
            help="JSON：画面大小、帧号，和每个人的编号、显眼程度、每帧的框 [x1, y1, x2, y2]；"
                 "CSV：每行 帧号、编号、x1、y1、x2、y2；"
                 "Tracker：每个人在 Nuke 的 Tracker 节点里占一条，跟的是**框的中心**，每帧一个关键帧，"
                 "那一帧没检测到这个人就把那一帧关掉。写成 名字.nk，用节点上的「复制到 Nuke」粘贴进合成",
        )
        origin: Literal["top_left", "bottom_left"] = P("top_left", label="坐标原点", group="文件",
                                                       option_labels=ORIGIN_LABELS,
                                                       help=ORIGIN_HELP + "。左下时 y1 是框的下边。Nuke 的 Tracker 总是左下角",
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
            from ...formats.nuke.tracks import MANY_TRACKS, write_tracker
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
    id = "core.output_tracks"
    category = "out_picture"
    inputs = (Port("tracks", "tracks2d", "2D 跟踪点"),)
    on_node = ("name", "format")
    clipboard = "nuke"  # the Nuke formats write a .nk snippet used by 「复制到 Nuke」
    # 3DE 与 CSV 不生成 Nuke 节点，无可粘贴内容：按钮保留，置灰并说明原因
    clipboard_when = Param("format").one_of("tracker", "cornerpin")

    class Params(NodeParams):
        name: str = name_param("tracks")
        format: Literal["3de", "csv", "tracker", "cornerpin"] = P(
            "3de", label="格式", group="文件",
            # CornerPin 对应平面的四个角，点数必须恰好为四（formats/nuke/tracks.py corner_pin）。
            # 输入点数不为四时该选项置灰并注明当前点数，避免选择后在计算末尾才报错
            option_applies={"cornerpin": incoming("tracks", "points").eq(4)},
            option_labels={"3de": "3DE", "csv": "CSV", "tracker": "Tracker", "cornerpin": "CornerPin"},
            help="3DEqualizer：3DE 的「Import 2D Tracks」能直接读，只写可见的帧，文件是 名字.txt；CSV：每行 帧号、点名、x、y、是否可见（跟点的节点给了置信度时再加一列 0–1 的置信度），"
                 "任何软件都能读；Tracker：每个点在 Nuke 的 Tracker 节点里占一条，每帧一个关键帧，挡住的帧关掉；"
                 "CornerPin：正好四个点（平面的四个角，如 WOFTSAM 平面跟踪）写成一个 CornerPin2D 节点，from 是起始帧的四个角，to 每帧一个关键帧。"
                 "两种 Nuke 格式都写成 名字.nk，用节点上的「复制到 Nuke」粘贴进合成，或在 Nuke 里 File → Import Script",
        )
        origin: Literal["bottom_left", "top_left"] = P(
            "bottom_left", label="坐标原点", group="文件", option_labels=ORIGIN_LABELS,
            help=ORIGIN_HELP + "。3DEqualizer、Tracker、CornerPin 总是左下角", applies=Param("format").one_of("csv"),
        )
        min_visible: int = P(5, label="最少可见帧数", help="可见帧数少于这个的点不写出（太短的点对相机解算没用，还会添乱）；跟踪点一共不到这么多帧（一对图的匹配点）时，在全部帧上都可见才写出", ge=1, group="筛选",
                             applies=Param("format").one_of("3de", "csv", "tracker"))

    @classmethod
    def write(cls, ctx) -> str:
        import csv

        import numpy as np

        from ...data.payloads import read_tracks
        from ...formats.nuke.tracks import MANY_TRACKS, corner_pin, write_tracker

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
    id = "core.output_curves"
    category = "out_picture"
    inputs = (Port("curves", "curves", "曲线"),)
    on_node = ("name", "format")

    class Params(NodeParams):
        name: str = name_param("curves")
        format: Literal["csv", "chan", "json", "usd"] = P(
            "csv", label="格式", group="文件",
            option_labels={"csv": "CSV", "chan": ".chan", "json": "JSON", "usd": "USD"},
            help="CSV：第一行是曲线名，之后每行一帧，Excel、Maya 脚本都能读；.chan：Houdini 的 File CHOP、Nuke 能直接读；JSON：给脚本用；USD：/shot/curves 上每条曲线一个带动画的属性",
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
                attr = prim.CreateAttribute(f"lab2shot:curve:{usd.valid_name(n)}", Sdf.ValueTypeNames.Float)
                for f, v in zip(frames, values[:, c]):
                    attr.Set(float(v), f)
            # 帧率仅写入交付文件（nodes/output.py fps_param）：帧号不变，只设置时基
            stage.SetTimeCodesPerSecond(float(ctx.params["fps"]))
            stage.SetFramesPerSecond(float(ctx.params["fps"]))
            usd.save_stage(stage, out)
        return out.name


class NukeCameraOutput(OutputSettings):
    id = "core.output_nuke_camera"
    category = "out_scene"
    inputs = (Port("camera", "scene.camera", "相机"),)
    on_node = ("name",)
    clipboard = "nuke"
    writes = {
        "model": Writes.no("Nuke 的相机节点里只有相机，没有网格"),
        "camera": Writes.full(),
        "points": Writes.no("Nuke 的相机节点里只有相机，没有点云"),
        "curves": Writes.no("Nuke 的相机节点里只有相机，没有三维曲线"),
        "skeleton": Writes.no("Nuke 的相机节点里只有相机，没有骨架"),
        "character": Writes.no("Nuke 的相机节点里只有相机，没有角色"),
    }

    class Params(NodeParams):
        name: str = name_param("camera")

    @classmethod
    def write(cls, ctx) -> str:
        from ...data.camera import CameraSamples
        from ...formats.nuke.camera import write_camera

        src = ctx.input("camera")
        samples = CameraSamples.from_packet(src)
        projects = learned_projects(ctx.provenance)
        if (samples.lens or {}).get("distortion"):
            ctx.say("N-NUKE-CAMERADISTORTION", model=(samples.lens["distortion"] or {}).get("model", ""), port="camera")
        out = cls.out_file(ctx, ".nk")
        out.write_text(write_camera(samples, ctx.params["name"], ", ".join(projects)), encoding="utf-8")
        return out.name


NODES = (BoxesOutput, TracksOutput, CurvesOutput, NukeCameraOutput)
