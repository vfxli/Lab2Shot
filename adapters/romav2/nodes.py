"""Nodes provided by the RoMa v2 extension (MIT; its DINOv3 backbone: DINOv3 License, commercial use allowed)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from lab2shot.sdk import (rgb_port, Official, measured_param, Confidence, Job, Msg, NodeParams, NothingToCook, P, Port, Shape,
                          WorkerNode, correspondence, empty_packet, tracks_packet, Cost, Licence, Measured)


class Match(WorkerNode):
    id = "romav2.match"
    on_node = ("setting", "matches")
    # 两个输入是「RGB」和「参考图」，逐对算，没有时间上的连贯（逐帧可能抖）
    runtime = "romav2"
    # RoMaV2.match(img_like_A, img_like_B) 交出 warp_AB（A 的每个像素在 B 里的位置）、confidence_AB、
    # overlap_AB、precision_AB，反向同名的四样（romav2.py:301-370）；RoMaV2.sample(preds, num_corresp)
    # 从中采出匹配点 matches_AB（romav2.py:372-384）。
    official = Official(
        cite="third_party/romav2/repo/src/romav2/romav2.py:301-395",
        takes={"image": "img_like_A", "other": "img_like_B"},
        gives={"stmap": "warp_AB", "matches": "matches_AB", "other_matches": "matches_AB"},
        note="「ST-map」就是官方的 warp_AB（romav2.py:344）搬到画面尺寸；「画面上的匹配点」「参考图上的匹配点」两个口是同一份"
             "matches_AB（romav2.py:384）的两半：每一行是 A 的坐标加 B 的坐标，同名的点是一对。"
             "上游还出反向的 warp_BA / overlap_BA / precision（romav2.py:365-368），我们没有对应的口",
    )
    # RTX 4090，默认的「精细」设置
    cost = Cost(gpu=True, vram_gb=9.7, whole="按一对图算的（不是一段镜头逐帧），实测约 2.3 秒一对")
    licence = Licence(note="代码和权重 MIT；权重里的 DINOv3 骨干受 DINOv3 License 约束（可商用，禁军事等用途，再分发附许可证）。")
    inputs = (rgb_port(), rgb_port("参考图", name="other"))
    # 「匹配点数」0 gives no matches: both ports are empty then, as asked, and nothing downstream says so (may_be_empty)
    outputs = (Port("stmap", "image.2", "ST-map"), Port("matches", "tracks2d", "画面上的匹配点", may_be_empty=True),
               Port("other_matches", "tracks2d", "参考图上的匹配点", shape=Shape(window="input:other"), may_be_empty=True))
    confidence = Confidence("probability", help="每个像素在参考图里找不找得到、匹配准不准（RoMa v2 自己的把握，0–1）。当遮罩用先接「置信度转遮罩」")

    class Params(NodeParams):
        setting: Literal["fast", "base", "precise"] = P(
            "precise", label="精度", group="匹配",
            option_labels={"fast": "快", "base": "标准", "precise": "精细"})
        matches: Literal[0, 500, 1000, 2000] = measured_param(
            "匹配点数", {0: Measured(flat=True), 500: Measured(flat=True), 1000: Measured(flat=True), 2000: Measured(flat=True)}, default=2000,
            group="匹配")

    @classmethod
    def info(cls, params, inputs):
        """What 画面 covers: the results live on its frames and pixels (参考图 is only looked into)."""
        return inputs["image"][0]

    @classmethod
    def pairs(cls, image, other) -> list[tuple[int, int]]:
        """Which frame of 参考图 each frame of 画面 is matched with: the same frame number; a 参考图 that is one picture
        (a photo) goes with every frame."""
        if other.meta.get("still") or len(other.meta["frames"]) == 1:
            return [(f, other.meta["frames"][0]) for f in image.meta["frames"]]
        have = set(other.meta["frames"])
        pairs = [(f, f) for f in image.meta["frames"] if f in have]
        if not pairs:  # nothing to match: an empty result, not an error
            raise NothingToCook(Msg("N-ROMAV2-NOSHAREDFRAMES", first=image.meta["frames"][0], last=image.meta["frames"][-1],
                              other_first=other.meta["frames"][0], other_last=other.meta["frames"][-1]))
        return pairs

    @classmethod
    def prepare(cls, ctx):
        """Job.notes: "pairs", the (画面 frame, 参考图 frame) pairs matched."""
        image, other = ctx.input("image"), ctx.input("other")
        pairs = cls.pairs(image, other)
        if len(pairs) < len(image.meta["frames"]):
            ctx.say("N-ROMAV2-UNPAIRED", count=len(image.meta["frames"]) - len(pairs))
        return Job(image, extra={"pairs": pairs}, inputs=ctx.input_files("other"), notes={"pairs": pairs})

    @classmethod
    def convert(cls, ctx, raw, job):
        out = correspondence(ctx, raw, job.plate, cls, [a for a, _ in job.notes["pairs"]])
        out |= cls.matches_out(ctx, raw, job.plate, ctx.input("other"))
        return out

    @classmethod
    def matches_out(cls, ctx, raw, image, other) -> dict:
        """raw/matches.npz -> the matches as two point sets with the same names, one on each picture: point k on 画面 and
        point k on 参考图 are one match (each is visible on the frame of its pair only, with RoMa v2's confidence in it
        there)."""
        if not raw.path("matches.npz").exists():
            return {"matches": empty_packet(ctx, "matches"), "other_matches": empty_packet(ctx, "other_matches")}
        d = raw.arrays("matches.npz")
        names = [f"match_{fa}_{k + 1:05d}" for k, fa in enumerate(d["frame_a"])]
        out = {}
        for port, packet, xy, frame_of in (("matches", image, d["xy_a"], d["frame_a"]),
                                            ("other_matches", other, d["xy_b"], d["frame_b"])):
            frames = sorted({int(f) for f in frame_of}) if not packet.meta.get("still") else packet.meta["frames"]
            index = {f: i for i, f in enumerate(frames)}
            visible = np.zeros((len(xy), len(frames)), bool)
            visible[np.arange(len(xy)), [index[int(f)] for f in frame_of]] = True
            tracks = np.repeat(xy[:, None], len(frames), 1)  # hidden elsewhere: the position is only meaningful where visible
            sure = np.where(visible, d["confidence"][:, None], 0.0) if "confidence" in d else None
            out[port] = tracks_packet(ctx.outputs[port], frames, packet.meta["width"], packet.meta["height"],
                                      tracks, visible, np.asarray(frame_of), names, confidence=sure, extension=cls.runtime)
        return out


NODES = (Match,)
