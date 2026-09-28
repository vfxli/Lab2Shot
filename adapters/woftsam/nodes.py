"""Nodes provided by the WOFTSAM extension (CC BY-NC-SA 4.0: research only)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from lab2shot.sdk import (Official, measured_param, Handle, Invalid, MissingFrames, Msg, NodeDef, NodeParams, NothingToCook, P,
                          Port, RawOutput, clamped_flow_side, parse_corners, tracks, Cost, Fact, Licence, Measured)

CORNER_NAMES = ["corner_1", "corner_2", "corner_3", "corner_4"]
STATE_REFOUND, STATE_UNSURE = 2, 3


def spans(frames: list[int]) -> str:
    """Frame numbers as ranges: 1001–1012、1040."""
    out, start = [], None
    for i, f in enumerate(frames):
        if start is None:
            start = f
        if i + 1 == len(frames) or frames[i + 1] != f + 1:
            out.append(f"{start}" if start == f else f"{start}–{f}")
            start = None
    return "、".join(out[:8]) + (" 等" if len(out) > 8 else "")


def check_quad(pts: list[tuple[float, float]], width: int, height: int) -> None:
    """The four corners make a plane the tracker can start from: not crossing, not a sliver."""
    p = np.asarray(pts, float)
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    edges = np.roll(p, -1, 0) - p
    turns = np.sign(edges[:, 0] * np.roll(edges, -1, 0)[:, 1] - edges[:, 1] * np.roll(edges, -1, 0)[:, 0])
    if not (turns == turns[0]).all():
        raise Invalid(Msg("E-WOFTSAM-CROSSED"))
    if abs(area) < 64:
        raise Invalid(Msg("E-WOFTSAM-TOOSMALL", area=round(abs(area))))
    if not ((p[:, 0] >= 0) & (p[:, 0] <= width) & (p[:, 1] >= 0) & (p[:, 1] <= height)).all():
        raise Invalid(Msg("E-WOFTSAM-OUTSIDE", width=width, height=height))


class PlaneTrack(NodeDef):
    id = "woftsam.track"
    # 上游 demo.py：frames（画面）+ init_coords（起始帧上的四个角，参数）-> all_corners（每帧四个角）
    official = Official(
        cite="third_party/woftsam/repo/demo.py:65-82",
        takes={"image": "frames"},
        gives={"tracks": "all_corners"},
        note="四个角（init_coords）上游也是当参数传的（track_function(sam_predictor, conf, frames, init_coords, seq_name)，"
             "第 67 行），我们同样做成参数「四个角」，不是输入口。单应矩阵 output_H（第 71-73 行）"
             "在我们的 2D 跟踪点数据里随平面一起交付。",
    )
    # 在视图里框一块有纹理的平面，前后都跟过去；大片单色、重复纹理的墙和地面跟不住；
    # 挡住、出画的帧标成「没能确认平面」，四个角标成不可见
    runtime = "woftsam"
    # vram_gb: RTX 4090
    cost = Cost(gpu=True, vram_gb=10.1, seconds_per_frame=0.43)
    licence = Licence(note="代码和加权 RAFT 权重是 CC BY-NC-SA 4.0，只能研究用，不能商用。")
    inputs = (Port("image", "image.3", "RGB"),)
    outputs = (Port("tracks", "tracks2d", "四个角"),)
    handles = (Handle("corners", {"corners": "corners"}),)
    on_node = ("resolution",)
    # a plane is always its four corners: 「2D 跟踪点输出设置」's CornerPin asks for exactly this (applies.py Incoming)
    fact_labels = {"points": "跟踪点数"}

    @classmethod
    def facts(cls, params: dict) -> dict:
        return {"points": Fact(4, cls.fact_labels["points"])}

    class Params(NodeParams):
        corners: list[str] = P(
            [], label="四个角", widget="picks", group="平面", worker=False,
            placeholder="显示本节点，在 2D 视图里拖出平面",
        )
        resolution: Literal[960, 1920] | None = measured_param(
            "处理分辨率", {960: Measured(below=1920), 1920: Measured(gb=10.1)}, auto="原尺寸", group="平面")

    @classmethod
    def cook(cls, ctx):
        image = ctx.input("image")
        quad = parse_corners(ctx.params["corners"])
        if quad is None:  # no plane drawn yet: nothing to track, an empty result
            raise NothingToCook(Msg("N-WOFTSAM-NOPLANE"))
        frame, pts = quad
        if frame not in image.meta["frames"]:
            frames = image.meta["frames"]
            raise Invalid(Msg("E-WOFTSAM-FRAMEOUT", frame=frame, first=frames[0], last=frames[-1]))
        check_quad(pts, image.meta["width"], image.meta["height"])
        if len(image.meta["frames"]) < 2:
            raise Invalid(Msg("E-WOFTSAM-ONEFRAME"))
        resolution = clamped_flow_side(ctx, image, ctx.params["resolution"])
        raw = RawOutput(ctx.run_worker(image, extra={"corners": [list(p) for p in pts], "frame": frame, "resolution": resolution}),
                        MissingFrames.FAIL)
        state = raw.arrays("tracks.npz")["state"]
        frames = image.meta["frames"]
        refound = [f for f, s in zip(frames, state) if s == STATE_REFOUND]
        unsure = [f for f, s in zip(frames, state) if s == STATE_UNSURE]
        if unsure:
            ctx.say("W-WOFTSAM-UNSURE", frames=spans(unsure), count=len(unsure))
        if refound:
            ctx.say("N-WOFTSAM-REFOUND", frames=spans(refound))
        info = raw.result()
        # which frames the plane was found again on or not confirmed: re-acquisition after an occlusion, in the data
        plane = {"start": frame, "refound": refound, "unsure": unsure, "seconds_per_frame": info.get("seconds_per_frame")}
        return tracks(ctx, raw, image, cls.runtime, names=CORNER_NAMES, plane=plane)


NODES = (PlaneTrack,)
