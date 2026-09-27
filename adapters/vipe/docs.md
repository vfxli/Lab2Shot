+++
team = "英伟达（NVIDIA）多伦多 AI 实验室（Spatial Intelligence Lab）"
people = "Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, …, Laura Leal-Taixé, Sanja Fidler"
paper = "ViPE: Video Pose Engine for 3D Geometric Perception（2025）"
paper_url = "https://arxiv.org/abs/2508.10934"
website = "https://research.nvidia.com/labs/toronto-ai/vipe"
repo = "https://github.com/nv-tlabs/vipe"
year = 2025
+++

## 这是什么

ViPE 从未受约束的原始视频估计相机内参、相机运动，以及稠密的、接近真实尺度的深度图；普通镜头、广角和 360° 全景素材都支持。用上游自己的一句话概括：它是一个开源的空间 AI 工具，为原始视频标注相机位姿和稠密深度图。

在 Lab2Shot 里它是**两个串联的节点**，对应上游流水线自己的两段：「ViPE 相机解算」跑到 SLAM 为止，交相机、SLAM 建出来的那些点、以及它解算时分出来的运动物体；「ViPE 深度图」接着这三样，跑官方位姿之后那一段，算出深度图。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`vipe infer YOUR_VIDEO.mp4`，一段视频或一个装视频的文件夹；
  `-p pose_only_long` 是长镜头那条流水线，关键帧滑窗，显存和内存不随视频长度增长。
- **给**（`vipe/streams/base.py`）：每帧相机位姿 `pose` 和内参 `intrinsics`；
  默认 / `dav3` 流水线另外给逐帧的近真实尺度深度图 `metric_depth`；
  它内部用 GroundingDINO + SAM + DeAOT 分出运动物体、解算时避开，并把实例遮罩 `instance`
  和「编号 → 词组」的对照表 `instance_phrases` 一起落盘（`vipe/utils/io.py` 的 `save_artifacts`）。
  `pose_only` / `pose_only_long` 两条流水线**官方自己就跳过深度图和遮罩的落盘**（`vipe/pipeline/pose_only.py`）。

**我们怎么接的**

- **两个节点，串联，照上游自己的两段分**，两个节点之间没有重复的功能：
  - 「ViPE 相机解算」跑 `pose_only` / `pose_only_long`，也就是上游跑到 SLAM 为止那一段。
    输出口就是 SLAM 那一步产生的三样：**相机** = `pose` + `intrinsics`；
    **点云** = `SLAMOutput.slam_map`（`vipe/slam/interface.py`，官方自己也把它单独存盘，
    `vipe/pipeline/pose_only.py` 的 `save_slam_map`）；
    **物体分割** = `instance`（uint8 编号图，0 是背景，名字来自 `instance_phrases`）——
    它是 SLAM 之前 init 阶段分出来的（`vipe/pipeline/default.py`）。
  - 「ViPE 深度图」跑的是 `default` / `dav3` 两条流水线里**位姿之后**那一段
    （`vipe/pipeline/default.py` 的 `_add_post_processors`）。它的输入口就是那一段真正读到的几样：
    **画面**、**相机**（`slam_output.trajectory` + `.intrinsics`）、**点云**（`slam_output.slam_map`，
    每帧投到画面上当深度图的提示，见 `vipe/pipeline/processors.py`）、
    **物体分割**（可选，上游读的是同一帧上的 `frame.mask`）。
    输出口一个：**深度图** = `metric_depth`。
- 「Focal Length」「Filmback」参数 = 给定内参时替掉它自己的 GeoCalib 估计（只有「ViPE 相机解算」有这两个参数）；
  「模式」/「深度图模型」= 上游的流水线名字。
  **「ViPE 深度图」上没有 Focal Length 和 Filmback**：它必接一台相机，内参就是那台相机的
  （上游那一段读的也是 `slam_output.intrinsics`）。
- **不一样的三点**：
  1. 上游把这两段放在一趟里（`SLAMOutput` 是内存里的对象），这里要在两个节点之间传，
     所以把 `slam_map` 写成了通用的「点云」（USD，厘米，每个关键帧一份），
     深度图节点再拼回上游的 `SLAMMap` 形状。**数据一个字节都没少**：那三样字段
     （`dense_disp_xyz` / `dense_disp_rgb` / `dense_disp_frame_inds`）原样往返。
     上游的 `backend_graph` 没有传——它只有 `secondary_keyframe=True` 时才用到，
     而上游自己从不打开那个开关（`default.py` 不传它，`processors.py` 里默认 `False`）。
  2. **深度图这一步跑在 `pose_only` 解出来的那台相机上**。上游 `default` / `dav3` 两条流水线里，
     SLAM 用的关键帧深度图先验分别是 `unidepth-l`（`configs/pipeline/default.yaml`）和
     `dav3`（`configs/pipeline/dav3.yaml`），而 `pose_only` 用的是 `moge2-l`
     （`configs/pipeline/pose_only.yaml`，上游较新的先验）。
     拆成两个节点之后，相机只解一次，深度图那一步接的就是它——**所以先验是 `moge2-l` 这一档**，
     解出来的尺度锚和上游一趟跑完 `default` / `dav3` 时不一样。
  3. **没有「遮罩」输出口**。官方落盘的是 `instance` 编号图和它的词表两样，
     「把会动的物体合成一张遮罩」是额外的用法，所以做成图上一个显式的小工具节点
     （「分割转遮罩」`core.segmentation_key`），不占解算器的口。

**出处**：简介来自 `third_party/vipe/repo/README.md`（`ViPE estimates camera intrinsics, camera motion, and dense near-metric depth maps from unconstrained raw videos, including pinhole, wide-angle, and 360-degree panorama footage.`）；
输入输出依据同一份 README、`repo/vipe/streams/base.py`、`repo/vipe/utils/io.py`、`repo/vipe/pipeline/pose_only.py`、
`repo/vipe/pipeline/default.py`、`repo/vipe/pipeline/processors.py`、`repo/vipe/slam/interface.py`、`repo/docs/usage.md`
和 `adapters/vipe/nodes.py` 的两段 `official`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 →「ViPE 相机解算」→ 相机接「USD 输出设置」再接「输出」，到 Houdini / Maya 当跟踪相机。
  要深度图就再接一个「ViPE 深度图」：把解算节点的**相机、点云、物体分割**三条线都接过去
  （模板卡「深度图 · ViPE」就是这么接好的，选素材就能算）。
- **为什么深度图那一步要接「点云」**：上游的深度图不是只看画面猜的，它每帧把 SLAM 建出来的那些点投到画面上当提示。
  只接相机、不接点云，它没有尺度参照，算不出来——所以「点云」是必接的口，不是可选的。
- **「物体分割」接不接都能算**：接上，走动的人、车就不参与深度对齐，和官方默认跑法一致；不接，
  节点上会写一句「没用上什么、为什么」，深度图照常算。
- 「ViPE 相机解算」的「物体分割」是它解算时本来就要算的那份（官方自己就把它和词表存成文件）：
  - 每个物体一个编号，编号的名字来自官方的词表（GroundingDINO 找的是 person、animal、vehicle、ball、balloon、gun、pet、car、bus，另加 sky），接「多层 EXR 输出设置」写成 Cryptomatte，Nuke 里按 person_01 这样的名字取。
  - 要一张遮罩（比如「只要会动的物体」）就接「分割转遮罩」，在参数里点类别；再接「图像合成」（运算选「相乘」）
    就能把运动物体那块遮掉送给别的解算器。
  - 边是硬边（0 / 1 的选区），不是发丝级抠像；要软边接「MatAnyone 2 精细抠像」精修。它按 GroundingDINO 的那几个词找物体，词表里没有的运动物体找不出来。
- 「ViPE 相机解算」的「点云」是 SLAM 的地图，**每个关键帧一份**（上游就是按关键帧分块存的）：
  在三维视图里能看见它解算时抓到的那些点，也是判断这次解算靠不靠谱的依据。
- 模式：
  - 「ViPE 相机解算」的「模式」——**标准（pose_only）**：几百帧以内的镜头；
    **长镜头（pose_only_long）**：上千帧，分段解算、内存有上限。
  - 「ViPE 深度图」的「深度图模型」——**标准（default）**：ViPE 官方默认流程，深度图用
    Video-Depth-Anything-Small 保证时序稳定；**Depth Anything 3（dav3）**：用 Depth-Anything-3 Giant
    按解好的相机算多视角深度图，细节更多，长焦镜头上远近会跳，显存要得多。
- 素材要求：相机要有位移（有视差）；不支持变焦镜头；不处理镜头畸变（广角素材先去畸变）；固定机位或纯摇镜头自动 Focal Length 不可靠。
- 关键参数：「已知 Focal Length」——知道实拍 Focal Length 必须填（配合 Filmback），相机更准；手机按等效 Focal Length 填、Filmback 填 36。

## 效果和局限

RTX 4090 上，时间不含启动 Python：

- iPhone 长焦跟拍（1080×1920，300 帧，标准模式）：110 秒，显存 9.6 GB（占用 12.6 GB），解出 Focal Length 5484 px，和手机长焦 105 mm 等效 Focal Length（约 5350–5600 px）一致；视图里透过相机看，人物对位正确。
- 同一镜头前 100 帧，三个模式对比：

| 模式 | 许可 | 时间（其中深度图） | 显存峰值（占用） | 内存峰值 | Focal Length | 相机轨迹和标准模式比 |
|---|---|---|---|---|---|---|
| 标准 | 可商用 | 28 s | 9.6 GB（12.6） | 9.4 GB | 5090 px | —（同一设置重跑两次，Focal Length 相差 1.6 px、位置 < 1 mm） |
| default | 非商用 | 65–78 s（32–40 s） | 9.7 GB（10.4） | 12.3 GB | 5077 px | 形状几乎相同（对齐后误差 0.5 mm、0.08°），尺度小 13% |
| dav3 | 非商用 | 56 s（31 s） | 19.1 GB（21.3） | 16.4 GB | 5087 px | 形状几乎相同（0.7 mm、0.07°），尺度大 11% |

- 这 100 帧上所有模式的自动 Focal Length 都是约 5080 px，比整段 300 帧解出的 5484 px 低 7%：镜头越短、位移越少，Focal Length 越不准；知道 Focal Length 就填上（填 5484 px 时 default 模式也正常出深度图）。
- 深度图是否靠谱：画面里的女人全身约 800–890 px 高，按 Focal Length 5080 px、身高约 1.6–1.65 m 算，她离相机约 9.5–10 米。default 给 9.7→12.7 米（整段平均算下来人高 1.98 米，约远了 20%，并且随时间缓慢变远 30%），相邻帧变化 0.37%，很稳；dav3 给 7.4→16.6 米（人高 1.7–2.6 米之间跳），相邻帧变化 1.46%，这条镜头上明显不如 default。两者的尺度都只是"接近真实尺度"，拿去做精确测量前请用已知尺寸校一次。
- 其他已知问题：固定机位镜头上自动 Focal Length 1052 px，实际约 588 px，没有视差时 Focal Length 解不准；运动物体遮罩按文字提示（人、动物、车……）检测，没列到的运动物体会干扰解算（交出来的「物体分割」同样只有这几个词里的物体）；default / dav3 要把整段镜头的画面放在内存里（1080×1920 大约每帧 90–130 MB），几百帧的镜头要注意内存，长镜头请用长镜头模式只出相机。

## 团队

英伟达多伦多 AI 实验室（Sanja Fidler 领导，现在叫 Spatial Intelligence Lab），做三维视觉和生成式三维的主力团队，代表作有 GET3D、Magic3D、NKSR 神经表面重建、Lyra 等，也参与了 NVIDIA Cosmos 世界模型。ViPE 是他们给大规模视频数据做三维标注的工具：用它标注了约 10 万条网络实拍视频、100 万条 AI 生成视频和 2 千条全景视频（约 9600 万帧）并公开。

## 模型下载和安装

- 运行 `lab2shot ext install vipe`：下载 ViPE 代码（锁定版本）、建独立 Python 环境（PyTorch 2.9 + CUDA 13，并用 pip 装的 CUDA 13.2 编译器现场编译 ViPE 的 CUDA 模块，装好约 7 GB，编译要几分钟），再下载权重，所有 Hugging Face 权重都锁定版本并校验 sha256：
  - 标准 / 长镜头用：DROID-SLAM、MoGe-2 ViT-L、GeoCalib、GroundingDINO + BERT、SAM ViT-B、DeAOT，共约 3 GB。
  - default 用：UniDepth-V2 ViT-L（1.4 GB）、Prior-Depth-Anything ViT-B + Depth-Anything-V2-Base（各 0.39 GB）、Video-Depth-Anything-Small（0.12 GB）。
  - dav3 用：Depth-Anything-3 Metric-Large（1.3 GB）、Depth-Anything-3 Giant（5.4 GB）。
  - 合计约 12 GB。不需要申请权限；已经下好的文件再次安装会跳过；运行时完全离线。

## 许可证说明

- ViPE 代码：Apache-2.0，可以商用。
- 标准 / 长镜头模式用到的模型全部可以商用：MoGe-2（MIT），GeoCalib、GroundingDINO、SAM、BERT（Apache-2.0），DROID-SLAM、DeAOT（BSD-3-Clause）。
- default 模式**非商用**：UniDepth-V2（CC-BY-NC-4.0，模型卡没写许可，按 UniDepth 仓库的 LICENSE）；Prior-Depth-Anything 的模型卡写 Apache-2.0，但它内含的 Depth-Anything-V2-Base 原模型是 CC-BY-NC-4.0；Video-Depth-Anything-Small 是 Apache-2.0。
- dav3 模式**非商用**：Depth-Anything-3 Giant（CC-BY-NC-4.0）；Depth-Anything-3 Metric-Large 是 Apache-2.0。
- 选 default / dav3 算出的相机和深度图都只能用于研究，不能用于商业项目，使用时注明出处。
- 注意：运动物体遮罩代码来自 Segment-and-Track-Anything，是 AGPL-3.0：内部使用没问题，如果修改后通过网络对外提供服务，需要公开源码。ViPE 仓库里的 UniK3D 部分（全景模式用）是非商用许可，本扩展不下载也不调用。

## 参考

- 论文：https://arxiv.org/abs/2508.10934
- 项目主页：https://research.nvidia.com/labs/toronto-ai/vipe
- 代码：https://github.com/nv-tlabs/vipe
- 文档和公开数据集：https://nv-tlabs.github.io/vipe/
- 许可证原文：https://github.com/nv-tlabs/vipe/blob/main/LICENSE
- UniDepth：https://github.com/lpiccinelli-eth/UniDepth （许可证 https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE ，模型卡 https://huggingface.co/lpiccinelli/unidepth-v2-vitl14 ）
- Prior-Depth-Anything：https://github.com/SpatialVision/Prior-Depth-Anything （模型 https://huggingface.co/Rain729/Prior-Depth-Anything ，内含的 https://huggingface.co/depth-anything/Depth-Anything-V2-Base ）
- Depth-Anything-3：https://github.com/ByteDance-Seed/Depth-Anything-3 （模型卡 https://huggingface.co/depth-anything/DA3-GIANT 、https://huggingface.co/depth-anything/DA3METRIC-LARGE ）
- 其他组件：MoGe https://github.com/microsoft/MoGe 、DROID-SLAM https://github.com/princeton-vl/DROID-SLAM 、GeoCalib https://github.com/cvg/GeoCalib 、Segment-and-Track-Anything https://github.com/z-x-yang/Segment-and-Track-Anything
