"""Nodes provided by the VidEoMT extension (MIT code and weights; trained on VIPSeg, which is for non-commercial research
only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, MissingFrames, RawOutput, NodeDef, NodeParams, P, Port, frame_maps, Cost,
                          Licence, Measured)

# VIPSeg's classes (upstream videomt/data_video/datasets/vps.py): (id, name, Chinese name, thing). Things are countable
# objects, each its own segment (person, car ...); stuff is one segment per class (sky, road ...).
VIPSEG = (
    (0, "wall", "墙", False), (1, "ceiling", "天花板", False), (2, "door", "门", True), (3, "stair", "楼梯", False),
    (4, "ladder", "梯子", True), (5, "escalator", "自动扶梯", False), (6, "Playground_slide", "滑梯", False),
    (7, "handrail_or_fence", "扶手或栏杆", False), (8, "window", "窗", True), (9, "rail", "铁轨", False),
    (10, "goal", "球门", True), (11, "pillar", "柱子", False), (12, "pole", "杆子", False), (13, "floor", "地板", False),
    (14, "ground", "地面", False), (15, "grass", "草地", False), (16, "sand", "沙地", False),
    (17, "athletic_field", "运动场", False), (18, "road", "道路", False), (19, "path", "小路", False),
    (20, "crosswalk", "人行横道", False), (21, "building", "建筑", False), (22, "house", "房屋", False),
    (23, "bridge", "桥", False), (24, "tower", "塔", False), (25, "windmill", "风车", False),
    (26, "well_or_well_lid", "井或井盖", False), (27, "other_construction", "其他构筑物", False), (28, "sky", "天空", False),
    (29, "mountain", "山", False), (30, "stone", "石头", False), (31, "wood", "木头", False), (32, "ice", "冰", False),
    (33, "snowfield", "雪地", False), (34, "grandstand", "看台", False), (35, "sea", "海", False), (36, "river", "河", False),
    (37, "lake", "湖", False), (38, "waterfall", "瀑布", False), (39, "water", "水", False),
    (40, "billboard_or_Bulletin_Board", "广告牌或公告栏", False), (41, "sculpture", "雕塑", True),
    (42, "pipeline", "管道", False), (43, "flag", "旗子", True), (44, "parasol_or_umbrella", "伞", True),
    (45, "cushion_or_carpet", "垫子或地毯", False), (46, "tent", "帐篷", True), (47, "roadblock", "路障", True),
    (48, "car", "汽车", True), (49, "bus", "公交车", True), (50, "truck", "卡车", True), (51, "bicycle", "自行车", True),
    (52, "motorcycle", "摩托车", True), (53, "wheeled_machine", "轮式机械", False), (54, "ship_or_boat", "船", True),
    (55, "raft", "筏子", True), (56, "airplane", "飞机", True), (57, "tyre", "轮胎", False),
    (58, "traffic_light", "红绿灯", False), (59, "lamp", "灯", False), (60, "person", "人", True), (61, "cat", "猫", True),
    (62, "dog", "狗", True), (63, "horse", "马", True), (64, "cattle", "牛", True), (65, "other_animal", "其他动物", True),
    (66, "tree", "树", False), (67, "flower", "花", False), (68, "other_plant", "其他植物", False), (69, "toy", "玩具", False),
    (70, "ball_net", "球网", False), (71, "backboard", "篮板", False), (72, "skateboard", "滑板", True),
    (73, "bat", "球拍", False), (74, "ball", "球", True), (75, "cupboard_or_showcase_or_storage_rack", "柜子或货架", False),
    (76, "box", "箱子", True), (77, "traveling_case_or_trolley_case", "行李箱", True), (78, "basket", "篮子", True),
    (79, "bag_or_package", "包或包裹", True), (80, "trash_can", "垃圾桶", False), (81, "cage", "笼子", False),
    (82, "plate", "盘子", True), (83, "tub_or_bowl_or_pot", "盆碗锅", True), (84, "bottle_or_cup", "瓶子或杯子", True),
    (85, "barrel", "桶", True), (86, "fishbowl", "鱼缸", True), (87, "bed", "床", True), (88, "pillow", "枕头", True),
    (89, "table_or_desk", "桌子", True), (90, "chair_or_seat", "椅子", True), (91, "bench", "长椅", True),
    (92, "sofa", "沙发", True), (93, "shelf", "架子", False), (94, "bathtub", "浴缸", False), (95, "gun", "枪", True),
    (96, "commode", "马桶", True), (97, "roaster", "烤炉", True), (98, "other_machine", "其他机器", False),
    (99, "refrigerator", "冰箱", True), (100, "washing_machine", "洗衣机", True), (101, "Microwave_oven", "微波炉", True),
    (102, "fan", "风扇", True), (103, "curtain", "窗帘", False), (104, "textiles", "织物", False),
    (105, "clothes", "衣服", False), (106, "painting_or_poster", "画或海报", True), (107, "mirror", "镜子", True),
    (108, "flower_pot_or_vase", "花盆或花瓶", True), (109, "clock", "钟", True), (110, "book", "书", False),
    (111, "tool", "工具", False), (112, "blackboard", "黑板", False), (113, "tissue", "纸巾", False),
    (114, "screen_or_television", "屏幕或电视", True), (115, "computer", "电脑", True), (116, "printer", "打印机", True),
    (117, "Mobile_phone", "手机", True), (118, "keyboard", "键盘", True), (119, "other_electronic_product", "其他电子产品", False),
    (120, "fruit", "水果", False), (121, "food", "食物", False), (122, "instrument", "乐器", True), (123, "train", "火车", True),
)


class PanopticSegment(NodeDef):
    id = "videomt.panoptic"
    # 上游 videomt.py forward()：images -> outputs["pred_logits"]（每个查询的类别）、outputs["pred_masks"]
    official = Official(
        cite="third_party/videomt/repo/videomt/videomt.py:198-229",
        takes={"image": "images"},
        gives={"panoptic": "pred_masks", "classes": "pred_logits"},
        note="上游只吃画面；类别表是 VIPSeg 的 124 类，写死在权重里。",
    )
    version = 2  # a query's class averaged over the frames it is an object on
    # docs.md + worker.py：官方 videomt_online（窗口 1）逐帧往后看，整幅画面每个像素都有归属；
    # 边缘精度是短边约 720 像素的网络精度，要软边再接抠像
    inputs = (Port("image", "image.3", "RGB"),)
    outputs = (Port("panoptic", "image.1", "全景分割"), Port("classes", "image.1", "类别分割"))
    runtime = "videomt"
    # vram_gb: RTX 4090 上测得（Tears of Steel 1280×534，默认处理尺寸 720，全新进程）：PyTorch 峰值 1.77 GB，整卡上涨 2.47 GB（含约 0.47 GB CUDA 上下文），48 帧和 192 帧一样、不随帧数涨
    # seconds_per_frame: 同一组测量，192 帧比 48 帧多 144 帧、多用 35.7 秒
    cost = Cost(gpu=True, vram_gb=2.0, seconds_per_frame=0.25)
    licence = Licence(note="代码和权重是 MIT，但权重只在 VIPSeg 上训练，VIPSeg 只许非商业研究使用，所以按非商用对待；商用前请法务确认。")

    class Params(NodeParams):
        resolution: Literal[360, 480, 720] = measured_param(
            "处理分辨率", {360: Measured("比实测的一档省", below=720), 480: Measured("比实测的一档省", below=720), 720: Measured("作者评测尺寸：1280×534，显存 48 帧和 192 帧一样", gb=1.8)}, default=720, group="分割",
            help="画面短边缩到这个像素再分割（长边最多是它的 1.85 倍）。720 是作者评测用的尺寸，最稳；分割图始终是原画面大小")
        threshold: float = P(0.8, label="检测阈值", ge=0.3, le=0.99, group="分割", widget="slider",
                             help="一个物体整段的类别把握要超过它才输出（作者默认 0.8）。有东西没分出来就调低（如 0.6），"
                                  "分出了不存在的东西、类别乱跳就调高")
        min_coverage: float = P(0.8, label="完整度门槛", ge=0.1, le=1.0, group="分割",
                           help="一个物体自己的遮罩里，至少要有这么大比例的像素最后归它，否则整段去掉（作者默认 0.8）。"
                                "被别的物体大面积盖住的东西也想留下就调低")

    @classmethod
    def cook(cls, ctx):
        import json

        import numpy as np

        image = ctx.input("image")
        things = [c[0] for c in VIPSEG if c[3]]
        raw = RawOutput(ctx.run_worker(image, extra={"things": things}), MissingFrames.SKIP)
        segments = json.loads(raw.file("segments.json").read_text(encoding="utf-8"))  # never empty: the worker ends with nothing()
        numbered: dict[int, int] = {}
        panoptic = []
        for s in segments:
            _, name, zh, _ = VIPSEG[s["category"]]
            if s["isthing"]:
                numbered[s["category"]] = n = numbered.get(s["category"], 0) + 1
                name, zh = f"{name} {n}", f"{zh} {n}"
            panoptic.append({"index": s["id"], "name": name, "name_zh": zh})
        present = sorted({s["category"] for s in segments})
        classes = [{"index": c + 1, "name": VIPSEG[c][1], "name_zh": VIPSEG[c][2]} for c in present]
        # segment id -> its class number (VIPSeg id + 1: the same class has the same number in every shot)
        class_of = np.zeros(len(segments) + 1, np.float32)
        for s in segments:
            class_of[s["id"]] = s["category"] + 1
        if segments:
            ctx.say("I-VIDEOMT-SEGMENTS", count=len(segments), names=[c["name_zh"] for c in panoptic[:12]], more="……" if len(panoptic) > 12 else "")
        return frame_maps(ctx, raw, image, {
            "panoptic": ("image.1", lambda d: d["segments"].astype(np.float32),
                         {"value_range": (0, max(len(segments), 1)), "classes": panoptic}),
            "classes": ("image.1", lambda d: class_of[d["segments"]],
                        {"value_range": (0, len(VIPSEG)), "classes": classes}),
        }, stage="写出分割")


NODES = (PanopticSegment,)
