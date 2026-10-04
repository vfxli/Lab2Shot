"""Nodes provided by the VidEoMT extension (MIT code and weights; trained on VIPSeg, which is for non-commercial research
only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (rgb_port, Official, measured_param, MissingFrames, Job, WorkerNode, NodeParams, P, Port, frame_maps, Cost,
                          Licence, Measured, class_word)

# VIPSeg's classes (upstream videomt/data_video/datasets/vps.py): (id, name, thing); their words are
# node."videomt.segment".port.classes.class.<name> (i18n/<lang>.toml). Things are countable objects, each its own
# segment (person, car ...); stuff is one segment per class (sky, road ...).
VIPSEG = (
    (0, "wall", False), (1, "ceiling", False), (2, "door", True), (3, "stair", False), (4, "ladder", True),
    (5, "escalator", False), (6, "Playground_slide", False), (7, "handrail_or_fence", False), (8, "window", True),
    (9, "rail", False), (10, "goal", True), (11, "pillar", False), (12, "pole", False), (13, "floor", False),
    (14, "ground", False), (15, "grass", False), (16, "sand", False), (17, "athletic_field", False),
    (18, "road", False), (19, "path", False), (20, "crosswalk", False), (21, "building", False), (22, "house", False),
    (23, "bridge", False), (24, "tower", False), (25, "windmill", False), (26, "well_or_well_lid", False),
    (27, "other_construction", False), (28, "sky", False), (29, "mountain", False), (30, "stone", False),
    (31, "wood", False), (32, "ice", False), (33, "snowfield", False), (34, "grandstand", False), (35, "sea", False),
    (36, "river", False), (37, "lake", False), (38, "waterfall", False), (39, "water", False),
    (40, "billboard_or_Bulletin_Board", False), (41, "sculpture", True), (42, "pipeline", False), (43, "flag", True),
    (44, "parasol_or_umbrella", True), (45, "cushion_or_carpet", False), (46, "tent", True), (47, "roadblock", True),
    (48, "car", True), (49, "bus", True), (50, "truck", True), (51, "bicycle", True), (52, "motorcycle", True),
    (53, "wheeled_machine", False), (54, "ship_or_boat", True), (55, "raft", True), (56, "airplane", True),
    (57, "tyre", False), (58, "traffic_light", False), (59, "lamp", False), (60, "person", True), (61, "cat", True),
    (62, "dog", True), (63, "horse", True), (64, "cattle", True), (65, "other_animal", True), (66, "tree", False),
    (67, "flower", False), (68, "other_plant", False), (69, "toy", False), (70, "ball_net", False),
    (71, "backboard", False), (72, "skateboard", True), (73, "bat", False), (74, "ball", True),
    (75, "cupboard_or_showcase_or_storage_rack", False), (76, "box", True),
    (77, "traveling_case_or_trolley_case", True), (78, "basket", True), (79, "bag_or_package", True),
    (80, "trash_can", False), (81, "cage", False), (82, "plate", True), (83, "tub_or_bowl_or_pot", True),
    (84, "bottle_or_cup", True), (85, "barrel", True), (86, "fishbowl", True), (87, "bed", True),
    (88, "pillow", True), (89, "table_or_desk", True), (90, "chair_or_seat", True), (91, "bench", True),
    (92, "sofa", True), (93, "shelf", False), (94, "bathtub", False), (95, "gun", True), (96, "commode", True),
    (97, "roaster", True), (98, "other_machine", False), (99, "refrigerator", True), (100, "washing_machine", True),
    (101, "Microwave_oven", True), (102, "fan", True), (103, "curtain", False), (104, "textiles", False),
    (105, "clothes", False), (106, "painting_or_poster", True), (107, "mirror", True),
    (108, "flower_pot_or_vase", True), (109, "clock", True), (110, "book", False), (111, "tool", False),
    (112, "blackboard", False), (113, "tissue", False), (114, "screen_or_television", True), (115, "computer", True),
    (116, "printer", True), (117, "Mobile_phone", True), (118, "keyboard", True),
    (119, "other_electronic_product", False), (120, "fruit", False), (121, "food", False), (122, "instrument", True),
    (123, "train", True),
)


class PanopticSegment(WorkerNode):
    missing_frames = MissingFrames.SKIP
    id = "videomt.segment"
    # 上游 videomt.py forward()：images -> outputs["pred_logits"]（每个查询的类别）、outputs["pred_masks"]
    official = Official(
        cite="third_party/videomt/repo/videomt/videomt.py:198-229",
        takes={"image": "images"},
        gives={"panoptic": "pred_masks", "classes": "pred_logits"},
    )
    version = 2  # a query's class averaged over the frames it is an object on
    # docs.md + worker.py：官方 videomt_online（窗口 1）逐帧往后看，整幅画面每个像素都有归属；
    # 边缘精度是短边约 720 像素的网络精度，要软边再接抠像
    inputs = (rgb_port(),)
    outputs = (Port("panoptic", "image.1"), Port("classes", "image.1"))
    runtime = "videomt"
    # vram_gb: RTX 4090 上测得（Tears of Steel 1280×534，默认处理尺寸 720，全新进程）：PyTorch 峰值 1.77 GB，整卡上涨 2.47 GB（含约 0.47 GB CUDA 上下文），48 帧和 192 帧一样、不随帧数涨；
    # 改为官方的 fp16 autocast 后（autocast 缓存一份半精度权重）同一 1080p 素材峰值多 0.38 GB（2.35 → 2.73 GB）
    # seconds_per_frame: 1080p 48 帧、处理尺寸 720，独占测试通道时测：网络只跑一遍 + autocast + 预读后，RTX 4090 0.069 秒/帧（原 0.36），RTX 5090 0.083 秒/帧（原 0.50）
    cost = Cost(gpu=True, vram_gb=2.4, seconds_per_frame=0.07)
    licence = Licence(note=True)

    class Params(NodeParams):
        resolution: Literal[360, 480, 720] = measured_param(
            {360: Measured(below=720), 480: Measured(below=720), 720: Measured(gb=2.2)}, default=720,
            group="segmentation")
        threshold: float = P(0.8, ge=0.3, le=0.99, group="segmentation", widget="slider")
        min_coverage: float = P(0.8, ge=0.1, le=1.0, group="segmentation")

    @classmethod
    def prepare(cls, ctx) -> Job:
        return Job(ctx.input("image"), extra={"things": [c[0] for c in VIPSEG if c[2]]})

    @classmethod
    def convert(cls, ctx, raw, job):
        import json

        import numpy as np

        image = job.plate
        segments = json.loads(raw.file("segments.json").read_text(encoding="utf-8"))  # never empty: the worker ends with nothing()
        numbered: dict[int, int] = {}
        panoptic = []
        for s in segments:
            _, name, _ = VIPSEG[s["category"]]
            if s["isthing"]:
                numbered[s["category"]] = n = numbered.get(s["category"], 0) + 1
                name = f"{name} {n}"
            panoptic.append({"index": s["id"], "name": name})
        present = sorted({s["category"] for s in segments})
        classes = [{"index": c + 1, "name": VIPSEG[c][1]} for c in present]
        # segment id -> its class number (VIPSeg id + 1: the same class has the same number in every shot)
        class_of = np.zeros(len(segments) + 1, np.float32)
        for s in segments:
            class_of[s["id"]] = s["category"] + 1
        if segments:
            names = [class_word(cls, c["name"], port="classes") or c["name"] for c in panoptic[:12]]
            ctx.say("I-VIDEOMT-SEGMENTSMORE" if len(panoptic) > 12 else "I-VIDEOMT-SEGMENTS", count=len(segments), names=names)
        return frame_maps(ctx, raw, image, {
            "panoptic": ("image.1", lambda d: d["segments"].astype(np.float32),
                         {"value_range": (0, max(len(segments), 1)), "classes": panoptic}),
            "classes": ("image.1", lambda d: class_of[d["segments"]],
                        {"value_range": (0, len(VIPSEG)), "classes": classes}),
        }, stage="write_segments")


NODES = (PanopticSegment,)
