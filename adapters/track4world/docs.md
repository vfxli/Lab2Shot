+++
team = "香港科技大学、腾讯 PCG ARC Lab"
people = "Jiahao Lu, Jiayi Xu, Wenbo Hu, Ruijie Zhu, Chengfeng Zhao, Sai-Kit Yeung, Ying Shan, Yuan Liu"
paper = "Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels（ECCV 2026）"
paper_url = "https://arxiv.org/abs/2603.02573"
website = "https://jiah-cloud.github.io/Track4World.github.io/"
repo = "https://github.com/TencentARC/Track4World"
year = 2026
+++

## 这是什么

Track4World 以全局前馈的方式，从单目视频估计任意两帧之间每个像素的稠密三维场景流，从而在以世界为中心的坐标系里高效、稠密地跟住每一个像素。

在 Lab2Shot 里，它做**整帧稠密的 3D 点跟踪**：参考帧上的每一个像素（按「跟踪点间隔」取）在整段镜头里的三维位置，连同每帧的相机、点云和有效区域，一次前馈算出来。输出是带编号、速度的**稠密动画点云**，在 Houdini 里可以从运动的表面发射粒子、灰尘，或者做运动物体的代理几何（阴影、交互）。

**仅限研究，许可证禁止用于生产**：Track4World 的许可证写明只能用于学术目的，在任何情况下都不得用于商业或生产用途，范围包括推理代码和权重；Depth Anything 3 骨干权重是 CC BY-NC 4.0。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`demo.py --mp4_path`，一段 .mp4。`--mode 3d_ff` 是从第一帧起的三维跟点，
  `--query_frame` 起始帧，`--image_size` 缩放上限，`--max_frames` 一次最多帧数，`--metric_scale` 打开真实尺度（要 DA3 骨干），
  `--coordinate` 选世界坐标由 Depth Anything 3 还是 Pi3 来解。
  推理那一行 `model.infer(rgbs_selected, iters, sw, is_training, tracking3d)` **只吃画面**：
  没有相机、没有遮罩、没有查询点。
- **给**：每个像素在世界坐标里的三维轨迹和二维轨迹，它自己解出的每帧相机（`camera_poses`），
  以及每帧的 `world_points` 和 `mask`。

**我们怎么接的**

- 「RGB」= 上游那段素材；「参考帧」= `--query_frame`，「处理分辨率」= `--image_size`，「每段最多帧数」= `--max_frames`；真实尺度一直开着。
- 「3D 跟踪点」「2D 跟踪点」= 上游自己的两份轨迹；「点云」= 每帧 `world_points`，「有效区域」= 每帧 `mask`；「相机」= 上游自己解的那几台。
- **节点上没有「相机」和「遮罩」输入口**：上游都不收这两样。
- **不一样的一点**：上游一次算一段 clip，这里把镜头切成 clip，从参考帧往前往后各跟一遍，
  clip 之间在共享帧上用相似变换对齐（见 `adapters/track4world/worker.py`）——它每个 clip 自己解自己的世界和尺度。

**出处**：简介来自 `third_party/track4world/repo/README.md`（「Track4World estimates dense 3D scene flow of every pixel between arbitrary frame pairs from a monocular video in a global feedforward manner…」）；
输入输出依据同一份 README、`repo/demo.py` 和 `adapters/track4world/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → Track4World 稠密 3D 跟踪点 →「合成场景」（点和相机）→「USD 输出设置」（单位选米给 Houdini）→「输出」。
  模板「稠密 3D 跟踪点 · Track4World」就是这样。
- 「参考帧」：从哪一帧的像素开始跟；前后都会跟过去。
- 「跟踪点间隔」：跟踪点在处理分辨率的画面上每隔几个像素一个（8：约 4 千个点；4：约 1.6 万个）。天空和模型没把握的像素不要。
- 「手动点」：在参考帧上点的点也一起跟，排在最前面（user_01…）。
- 世界和相机都是 Track4World 自己解出来的（真实尺度），从「相机」输出。上游不收相机，
  节点上也就没有「相机」输入口：要把结果摆进自己那台相机的世界，用图上显式的小工具节点，看得见。
- 只想跟画面里的一块（比如只要运动的人）：上游没有遮罩这种输入，节点上也就没有「遮罩」口。
  在图上接「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ Track4World 的「RGB」口，遮挡发生在送进模型之前，一眼看得见。
- 输出「点云」和「有效区域」：官方同一次计算顺带出的每帧稠密世界坐标点（上游 `world_points`）和它的有效遮罩（上游 `masks`）。
  「点云间隔」决定每隔几个像素取一个点，颜色取自这一帧的画面。它和「3D 跟踪点」不是一回事：这一份每帧各算各的，
  点和点之间没有对应关系；「3D 跟踪点」才是同一个点跟过整段。
- 「每段最多帧数」：一次送进模型的帧数（默认 64）。更长的镜头分段跟：相邻两段共用 8 帧，每个点在下一段的第一帧交接，
  每段自己的世界对齐到第一段；在交接帧被挡住的点不再往后跟（保持最后的位置，标成看不见）。显存不够时停下来，列出更小的安全选项让你选，不会自己调低。

## 效果和局限

RTX 4090 上：

- **显存放得下 24 GB**：处理分辨率 512（竖幅 288×512）、约 2 千 3 个点：每段 64 帧时峰值 14.0 GB，一段 120 帧时 18.7 GB；
  24 帧固定机位镜头 10.6 GB。模型加载约 12–39 秒（第一次要从硬盘读 12 GB 权重），跟点每帧 0.13–0.17 秒。
  超过「每段最多帧数」的镜头分段接起来（120 帧按 64 帧一段，显存 14 GB）。
- 「每段最多帧数」上限是 120；处理分辨率 640 只在每段 64 帧上实测过（两个都调到各自的上限没有验证过，显存可能不够）。
- 处理分辨率 640（官方默认）：1280×534 的手持镜头（缩成 640×256）、每段 64 帧、60 帧：显卡峰值约 14 GB
  （和 512 档同一个量级——显存主要看「每段最多帧数」和模型本身），节点用时约 **52 秒 / 60 帧**（含一次模型加载）。
- **缩完之后短边至少要 256**（worker.py 的 `MIN_SIDE`）：相关金字塔 5 层，每层把 1/8 特征图减半，
  最后一层还要剩 2 → 1/8 分辨率的特征图 ≥ 32 → 短边 ≥ 256。不够时**算之前**就报
  `E-TRACK4WORLD-SMALLSIDE`，不会等到模型内部报「output (H: 0, W: 2)」。
  同一档处理分辨率在不同宽高比上结果不同：竖幅 288×512 的短边够用，1280×534 这种扁画面在 512 档缩成 512×192 就不够，
  所以按短边判，不按档位判。
- 稳定：24 帧固定机位镜头，每个点每帧的移动中位数 0.55 厘米（同一段用 MoGe 逐帧深度图的 TAPIP3D、「跟踪点转 3D」是 4 厘米左右：
  单帧深度图的闪烁）。

已知局限：
- 很重：模型权重约 7 GB（显存里 fp32），加载要把 5.5 GB 的检查点和 6.8 GB 的骨干读进内存（队列按 20 GB 内存等）。
- 分段交接会丢掉在交接帧被挡住的点；段和段之间的世界靠相似变换对齐，长镜头会有小的累积误差。
- 单目估计的深度图：远处、玻璃、天空不可靠；真实尺度来自 Depth Anything 3 的真实尺度分支，只是估计。

## 团队

香港科技大学 Yuan Liu 组（Jiahao Lu 一作）和腾讯 PCG ARC Lab（Wenbo Hu、Ying Shan）。ECCV 2026。

## 模型下载和安装

- 自动安装：`lab2shot ext install track4world`。锁定 GitHub 仓库的一个版本和作者的 utils3d 分支，独立的 PyTorch 环境（不用编译）。
- 下载：Hugging Face 上的 track4world_da3.pth（5.5 GB，锁定版本）和 Depth Anything 3 骨干 DA3NESTED-GIANT-LARGE-1.1（6.8 GB；
  和 Depth Anything 3 扩展包是同一份文件，只存一份）。不下载 Pi3、MoGe 骨干和 SAM 2 物体选择。
- 运行时离线：骨干从安装好的文件建；上游建模型时会从 torchvision 下载 ImageNet 的 ConvNeXt，这里跳过（检查点里有它的全部权重）。

## 许可证说明

Track4World License：只许学术研究，禁止商用和生产。骨干 Depth Anything 3（DA3NESTED-GIANT-LARGE-1.1）权重 CC BY-NC 4.0；
仓库里 AllTracker 部分 MIT。节点和写出的文件都标「非商用」，节点说明写着「仅限研究，许可证禁止用于生产」。

## 参考

- 论文：https://arxiv.org/abs/2603.02573
- 代码：https://github.com/TencentARC/Track4World
- 权重：https://huggingface.co/TencentARC/Track4World

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
