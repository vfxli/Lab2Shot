+++
team = "加州大学伯克利分校（UC Berkeley）+ Google DeepMind + Stability AI + 加州大学默塞德分校（UC Merced）"
people = "Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, Ming-Hsuan Yang"
paper = "MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion（ICLR 2025 Spotlight）"
paper_url = "https://arxiv.org/abs/2410.03825"
website = "https://monst3r-project.github.io/"
repo = "https://github.com/Junyi42/monst3r"
year = 2024
+++

## 这是什么

上游自己的话：MonST3R 把一段动态视频处理成随时间变化的动态点云，并附带每帧的相机位姿和内参，整个过程以前馈为主；这个表示接着能高效地完成下游任务，例如视频深度估计、动态与静态场景的分割。论文摘要里它的全名是 Motion DUSt3R：做法是把 DUSt3R 那套只用于静态场景的点图表示，通过微调迁移到动态场景，每个时刻各估一份点图。论文 ICLR 2025。

在 Lab2Shot 里，它交出每帧相机、整段一个 Focal Length、每帧深度图，外加一张官方自己算的运动物体遮罩（上游用光流判断哪些像素的运动不是相机造成的，再用 SAM 2.1 把这些区域在整段里修整干净）。尺度是相对的，按节点上的「尺度」参数换算成厘米。

## 输入输出

**官方要什么、给什么**

- 吃：一串画面（`filelist`），外加**可选的运动物体遮罩文件夹** `dynamic_mask_path`
  ——这是官方自己的一路输入（`third_party/monst3r/repo/demo.py`）。
  命令行那一层收 `--input_dir`、`--image_size`（512 或 224）、`--seq_name`、`--use_gt_davis_masks` 等。
  上游那一头**只有遮罩图这一种形式**，没有「框」这种输入。
- 给：全局优化之后一起写出——`pred_traj.txt`（每帧相机位姿）、
  `pred_intrinsics.txt`（内参）、深度图、`dynamic_masks`（它自己判出来的运动物体）、置信度、
  以及处理过的画面。

**我们怎么接的**

- 「图像」口 = `filelist`，「运动物体遮罩」输入口 = 官方的 `dynamic_mask_path`（官方的输入），
  「处理分辨率」= `--image_size`。
- 「相机」= `pred_traj.txt` 加 `pred_intrinsics.txt`，「深度图」= 深度图，
  「运动物体遮罩」输出口 = 官方 `save_dynamic_masks` 交的那一张，「置信度」= 置信度图。
- 节点上没有人物框输入口：上游只收遮罩图。要挡人就接「人物框转遮罩」，把框变成遮罩这一步放在节点图上看得见的地方。
- 节点上的「已知 Focal Length」「Filmback」对应上游的 `shared_focal` 那一路（整段一个 Focal Length）。

出处：简介来自 `third_party/monst3r/repo/README.md`（MonST3R processes a dynamic video… 那一段）；
输入输出依据 `third_party/monst3r/repo/demo.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → MonST3R 深度与相机 → 相机 / 深度图 / 点云，和 ViPE 相机解算、CUT3R 的结果互相对照；它顺带输出的运动物体遮罩（`mask`，官方导出的那一张）可以拿去做抠像或排除运动物体的参考。也可以接「运动物体遮罩」，和它自己找到的运动区域合并（节点上没有「人物框」输入口，要挡人就接「人物框转遮罩」）。
- 适合：固定机位或慢速运动的镜头——固定机位上它是这几个方法里最稳的（几乎零漂移）；画面里有人走动也可以。不适合：长焦跟拍（Focal Length 估得很离谱，见下）、变焦、镜头畸变大的素材；速度慢，长镜头要等很久。
- 尺度不是米：单位是 MonST3R 自己的要手动缩放，整段镜头内一致。
- 关键参数：`focal_px` 已知 Focal Length（输入分辨率下的像素数，例如 iPhone 长焦 5484），填了就不再猜 Focal Length；`max_frames` 每段帧数（默认按分辨率自动算，把显存控制在 18 GB 左右，竖 / 横 512 画面约 36 帧），更长的镜头自动分段、用重叠帧拼接；`step` 隔帧取样，长镜头用 2–4 可以成倍省时间。
- **手填每段最多帧数的上限是 40 帧**：24 GB 显卡上 40 帧（288×512）用 18.9 GB，且显存**随帧数线性增长**（不像 VGGT/Pi3 那样一次看完整段还留有余量）；服务器拒绝更大的数字。

## 效果和局限

RTX 4090 上，和 ViPE 相机解算对比（相机轨迹先做相似变换对齐再算误差）：

| 素材 | 结果 |
|---|---|
| iPhone 长焦跟拍，1080×1920，120 帧（真实 Focal Length 约 5484 px，ViPE 5100 px） | Focal Length 2508 px（偏小 54%），转角因此被放大（17.8°，ViPE 10.7°）；轨迹和 ViPE 的差为行程的 11%，朝向差平均 3.9°、最大 7.2°；分 4 段，共 6.5 分钟（每帧约 3 秒），显存峰值 17.0 GB |
| 同上，填入已知 Focal Length 5484 px | 转角 7.9°，朝向差平均 3.7°、最大 4.3°，轨迹差 13.6%；前十几帧相机有明显抖动 |
| 固定机位跳舞，864×480，124 帧（真实 Focal Length 约 560–590 px） | Focal Length 737 px（偏大约 28%，ViPE 1053 px）；相机几乎不动：位移只有场景深度的 0.2%，转动最大 0.1°（ViPE 约 1% 和 0.2°，CUT3R 3.7% 和 1.3°）；6 分钟，显存 16.7 GB |

- 慢：每帧约 3 秒（CUT3R 约 0.1 秒）。时间主要花在逐对预测、光流和 300 步全局优化上。
- 显存随每段帧数线性增长：一段 40 帧（288×512）峰值 18.9 GB，所以默认每段 36 帧；段与段共用 8 帧，接缝残差约为深度的 0.1–0.8%。
- 运动物体遮罩整体能框住走动的人。输出的是官方 `save_dynamic_masks` 交的那一张（开着「SAM 2.1 修整」就是 SAM 2.1 那张，关了就是光流那张）。解算时用来挡运动物体的是官方优化器内部的并集（光流 ∪ SAM 2.1 ∪ 接进来的遮罩），这一份带着光流的碎点，偶尔会把画面最下方的近处地面、灌木当成在动；它不作为输出。
- 长焦素材上 Focal Length 和转角都不可靠，这类镜头优先用 ViPE。
- 远处大面积平整区域置信度低，不会进输出点云（遮罩）。

## 团队

一作 Junyi Zhang（加州大学伯克利分校 Trevor Darrell 组），合作者来自 Google DeepMind（Deqing Sun、Forrester Cole、Charles Herrmann、Junhwa Hur——Deqing Sun 是经典光流网络 PWC-Net 的作者）、Stability AI（Varun Jampani）和加州大学默塞德分校（Ming-Hsuan Yang）。它的底子是 Naver Labs 的 DUSt3R；同一方向后来有 CUT3R、VGGT 等工作。MonST3R 是 ICLR 2025 Spotlight 论文。

## 模型下载和安装

- 运行 `uv run lab2shot ext install monst3r`：下载 MonST3R 代码（锁定版本，再补上它引用的 CroCo 子模块）、建独立 Python 环境（和 CUT3R 相同：PyTorch 2.10，约 7 GB，共用下载缓存），再从 Hugging Face 下载三份权重并校验 sha256：MonST3R 本体（2.3 GB）、SEA-RAFT 光流（79 MB）、SAM 2.1 Hiera-L（898 MB），共约 3.3 GB。
- 不需要申请权限，也不需要自行下载任何模型。

## 许可证说明

- **非商用**，只能用于研究：MonST3R 的代码和权重都是 CC BY-NC-SA 4.0（署名、禁止商用、改编后的作品必须用同样的许可证发布）。它基于 Naver DUSt3R / CroCo（同为 CC BY-NC-SA 4.0，其中两个文件含 Meta MAE 的 CC BY-NC 4.0 部分），权重是在 DUSt3R 权重上继续训练的。
- 附带的两个模型本身可以商用：SEA-RAFT 光流（普林斯顿，代码和权重 BSD-3-Clause）、SAM 2.1（Meta，代码和权重 Apache-2.0）。但整个扩展按最严的 CC BY-NC-SA 4.0 对待。
- 原仓库评测代码引用的轨迹评测工具 evo 是 GPL-3.0，推理用不到，本扩展没有安装它。没有 SMPL 人体模型、nvdiffrast 等其他依赖。

## 参考

- 论文：https://arxiv.org/abs/2410.03825
- 项目主页：https://monst3r-project.github.io/
- 代码：https://github.com/Junyi42/monst3r
- 许可证原文：https://github.com/Junyi42/monst3r/blob/main/LICENSE
- 权重模型卡：https://huggingface.co/Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt
- SEA-RAFT 光流：https://github.com/princeton-vl/SEA-RAFT （权重 https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M ）
- SAM 2.1：https://github.com/facebookresearch/sam2 （权重 https://huggingface.co/facebook/sam2.1-hiera-large ）
- 它的基础 DUSt3R（Naver）：https://github.com/naver/dust3r
