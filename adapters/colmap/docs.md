+++
team = "苏黎世联邦理工学院计算机视觉与几何组（ETH Zurich CVG）+ 北卡罗来纳大学教堂山分校（UNC Chapel Hill）"
people = "Johannes L. Schönberger, Jan-Michael Frahm"
paper = "Structure-from-Motion Revisited（CVPR 2016）"
paper_url = "https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html"
website = "https://colmap.github.io/"
repo = "https://github.com/colmap/colmap"
year = 2016
+++

## 这是什么

上游 README 的「About」一节这样介绍它：COLMAP 是一套通用的运动恢复结构（Structure-from-Motion，
SfM）与多视图立体（Multi-View Stereo，MVS）流程，带图形界面和命令行，功能覆盖从有序和无序的
多张画面还原三维结构，按新版 BSD 许可发布。

在 Lab2Shot 里，只接它的 SfM 那一半，做成「COLMAP 相机解算」：在每帧里找几千个特征点，
跨帧匹配，三角化出三维点，同时解出每一帧的相机位置、朝向和 Focal Length，交出每帧相机加一片稀疏点云。
它全是几何计算，不靠深度学习去估，所以在有视差的镜头上结果硬，可以和 ViPE 的结果互相校验。
装的是 COLMAP 官方发布的 4.2.0。

## 输入输出

**官方要什么、给什么**

- 吃：一个画面文件夹。命令行是 `colmap automatic_reconstructor --image_path IMAGES --workspace_path WORKSPACE`，
  或者拆开的 `feature_extractor --image_path IMAGES --database_path DATABASE` 再 `exhaustive_matcher`、`mapper`。
  可选的遮罩是**官方的输入**：`--ImageReader.mask_path` 指一个文件夹，同名加 `.png`，
  黑色（灰度 0）的地方不提特征。镜头模型由 `--ImageReader.camera_model` 定（默认 `SIMPLE_RADIAL`）。
- 给：一份稀疏模型，三个文件——
  `cameras.txt`：每台相机一行 `CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]`，Focal Length 是**像素**；
  `images.txt`：每张画面两行，四元数 `QW QX QY QZ` 加位移 `TX TY TZ`，以及这一张上的 2D 点；
  `points3D.txt`：稀疏三维点。

**我们怎么接的**

- 「RGB」口就是 `--image_path` 那个文件夹，「运动物体遮罩」口就是 `--ImageReader.mask_path`，
  节点上的**「镜头模型」**（参数 `fit_model`）就是 `--ImageReader.camera_model`。
- 「相机」口是 `images.txt` 的四元数加位移（转成 Lab2Shot 的厘米、Y 轴向上），「点云」口是 `points3D.txt`，
  「Focal Length」「镜头内参」两个口都来自 `cameras.txt` 那一行里的 `MODEL` 和 `PARAMS[]`：Focal Length 单独交出
  （按「Filmback」换成毫米），其余（模型、畸变系数、主点、fx≠fy 折算的像素比）打成一份「镜头内参」。
- **「镜头模型」是要求，「镜头内参」是结果**：前者是让 COLMAP 按哪种模型去解（`--ImageReader.camera_model`），
  后者是它解完之后 `cameras.txt` 里写的那一行。
- **「镜头内参」口只在带畸变的那几档下用得上**：「镜头模型」选 SIMPLE_PINHOLE、PINHOLE 时，
  上游根本没解畸变，这个口没有值——它在节点上**变灰、鼠标停上去写清原因、线也接不出去**（口不消失，位置不跳）。
- **不一样的两处**：①「Filmback」输出口给的是**节点上那个参数的原值**——`cameras.txt` 里没有 Filmback 这一项，
  那个口只是把参数带给下游；② 上游只有遮罩图这一种形式，没有「框」这种输入，所以节点上没有人物框口：
  要挡人就在节点图上接「人物框转遮罩」，那一步看得见。

出处：简介来自 `third_party/colmap/repo/README.md`（About 那一段）；输入输出依据 `third_party/colmap/repo/doc/cli.rst`、
`doc/faq.rst`、`doc/format.rst`，以及 `adapters/colmap/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法（模板「相机解算 · COLMAP」）：读取序列 → ViTDet 人物框 →「人物框转遮罩」→ **COLMAP 相机解算**的「运动物体遮罩」口（人身上的特征点不参与解算）→ 合成场景 → USD 输出设置，进 Houdini / Maya 验证相机、摆场景。节点上没有「人物框」输入口：上游只有一张 `mask_path` 遮罩图这一种形式，框转遮罩这一步在节点图上做。
- 其他输入：「运动物体遮罩」可以接 SAM 3 等节点出的遮罩（车、人群、晃动的树都遮掉）；知道 Focal Length 就填「已知 Focal Length」；要用一台相机（比如 ViPE 解出的或读取的 3DE 相机）的 Focal Length，把它接到「拆分相机」（关掉「逐帧」），再把「Focal Length」接到这里的「已知 Focal Length」上（点参数旁边的接入按钮），Focal Length 固定、只解位置。
- **必须有视差**：相机要真的在移动（推轨、手持走动、航拍）。固定机位、三脚架上的纯摇镜头解不出来——节点会报错并建议改用 ViPE。
- 不支持变焦镜头（整段只有一个 Focal Length）；画面里大部分被运动物体占满、纯白墙、强烈运动模糊时容易失败或断成几段（只保留最大的一段，其余帧用插值补）。
- 关键参数：
  - 尺度（厘米 / 单位）：COLMAP 的结果**没有真实尺度**。默认按 1 个单位 = 1 米（100）放；和 ViPE 或现场实测的距离（比如两个标记点间距）对比后再改。
  - 镜头模型：画面已经去畸变（模板里「LensDistortion」之后）用默认 SIMPLE_PINHOLE；手机和长焦也用无畸变（长焦时 Focal Length 和畸变会互相抵消，把 Focal Length 估大）；原始画面上让 COLMAP 自己解畸变选 SIMPLE_RADIAL、RADIAL 或 OPENCV；鱼眼用三档鱼眼模型（见下面「镜头模型」）。
  - 隔帧：长镜头设 2–3 能快很多，跳过的帧由前后帧插值；快速运动的镜头保持 1。
  - 解算方式：默认「全局」快、比增量稳、整条轨迹不漂；全局解不出来时再试「增量」（慢，两次计算结果可能差很多，见下）。匹配方式视频用「相邻帧」；帧数很少或镜头走回原处时用「所有帧两两」（帧多会非常慢）。
  - 「处理分辨率」可选 1000 / 1500 / 2000（默认 2000），更大的没有验证过（COLMAP 只在打开「显卡提取特征」时才用显卡，经典 SIFT 本身开销不大）。
  - 「所有帧两两」的开销是帧数的平方：帧数（隔帧之后）超过 300 时提交前拒绝，提示改用「相邻帧」——这是「匹配方式」乘「帧数」的组合检查，不是单个参数的上限。

## 效果和局限

RTX 4090 机器上（特征点用 CPU 提取，全局解算，相邻帧匹配，遮罩遮掉约 12.5% 的画面）：

- iPhone 长焦手持跟拍（1080×1920，300 帧，隔 2 帧共解 150 帧）：
  - 填了已知 Focal Length（ViPE 解出的 5484 像素）+ 无畸变镜头：150 帧全部解出，用时 68 秒，平均重投影误差 0.77 像素，17,863 个点。
  - Focal Length 让它自己解（径向 k1）：同样 150 帧全部解出，89 秒，重投影误差 0.68 像素，但 Focal Length 解成 6593 像素，比 ViPE 和拍摄记录（约 5350–5600 像素）长了约 20%。节点会提示"长焦镜头 Focal Length 很难只靠画面解准"——**知道 Focal Length 必须填。**
- 固定机位：按预期报错"相机没有位移"，提示改用 ViPE。
- **全局 vs 增量解算方式**（同一段跟拍 300 帧全部参与，其余默认；和 ViPE 的相机用「相机对比」按比例+旋转+位置对齐后比较，ViPE 行程 13.3 米）：

  | 设置 | 注册帧 | 重投影误差 | 点数 | 建图用时（提取 + 匹配另约 100 秒） | 和 ViPE 比：位置 RMS / 角度差中位 / 最大 |
  |---|---|---|---|---|---|
  | 全局，Focal Length 自己解 | 299 / 300，一段 | 0.73 px | 20,122 | 118 秒 | 7.7 cm / 2.51° / 8.3°（Focal Length 比 ViPE 长 21%） |
  | 增量，Focal Length 自己解（第一次） | 299 / 300，两段 | 0.80 px | 16,127 | 约 330 秒 | 7.4 cm / 2.43° / 8.0° |
  | 增量，Focal Length 自己解（第二次） | 154 / 300，断成 6 段 | 0.96 px | 9,141 | 442 秒 | 144 cm / 4.41° / 9.7° |
  | 全局，已知 Focal Length（ViPE 的）+ 无畸变 | 300 / 300 | 0.80 px | 25,528 | 168 秒 | 2.8 cm / 0.37° / 0.69° |
  | 增量，已知 Focal Length + 无畸变 | 299 / 300 | 0.81 px | 19,658 | 197 秒 | 2.8 cm / 0.50° / 0.83° |

  - 全局解算在这个镜头上建图快 1.2–3.8 倍（不是 GLOMAP 论文里大规模无序照片集上的一两个数量级：视频按相邻帧匹配时增量解算本来就不慢），两次计算几乎一样（Focal Length 6645 / 6626 像素，已知 Focal Length 时点数 25,530 / 25,528）；增量解算两次计算差别很大（多线程的顺序不同），第二次只接上一半的帧。所以默认用全局。
  - 视差很小时全局解算也不稳：跟拍开头 40 帧（朝远处的墙走，长焦，人物用 MonST3R 的运动遮罩遮掉），141 对画面里 107 对几乎只有旋转，两次计算分别解出 72 个点、Focal Length 4257 像素和 53 个点、Focal Length 1901 像素；改成「所有帧两两」只剩 8 个点。这种镜头要么填已知 Focal Length，要么用 ViPE。同一个镜头走进花园后的 40 帧（近处有灌木，视差大）两次计算：4,895 / 4,999 个点，Focal Length 5800 / 5772 像素（ViPE 5479），很稳。
  - 真正决定精度的是 Focal Length：已知 Focal Length 时两种方式都和 ViPE 对得很好（位置差 0.2%、角度差不到 0.5°）；Focal Length 让它自己解时长焦镜头的 Focal Length 偏长约 20%，朝向也跟着差 2–3°。
  - 固定机位两种方式都按预期报错（741 对画面全是纯旋转 / 平面，没有视差）。
- 局限：
  - 尺度任意、世界坐标以第一帧相机为原点，地面不一定水平：把 GeoCalib 的重力方向接到「自动落地」上，它先转正，再按点云放到 y = 0；比例和 ViPE 或实测距离对比后改「尺度」。
  - 这是纯几何方法，场景要"静止"；动的物体必须用「运动物体遮罩」口排除（「ViTDet 人物框」→「人物框转遮罩」，或 SAM 3 等节点出的遮罩）。
  - 完全离线，所以关掉了"回环检测"（需要额外下载词汇树），镜头绕一圈回到原处时，用「所有帧两两」匹配来闭合。

### 镜头模型

两段有真值的公开素材，各取 45 帧（隔 3 帧）、所有帧两两匹配、处理分辨率 2000：

- 桶形畸变：TUM RGB-D fr1_desk，640×480，真值 OpenCV k1 0.262 k2 -0.953 p1 -0.005 p2 0.003 k3 1.163，fx 517.3；
- 鱼眼：TUM-VI room1，512×512，真值 Kannala-Brandt k1 0.0035 k2 0.0007 k3 -0.0021 k4 0.0002，fx 191.0，视场 195°。

| 镜头模型 | 素材 | 焦距未知 · 全局 | 焦距未知 · 增量 | 焦距已知 · 全局 | 焦距已知 · 增量 |
|---|---|---|---|---|---|
| SIMPLE_PINHOLE | 桶形 | 45 帧，0.71 px | 45，0.76 | 45，0.76 | 45，0.79 |
| SIMPLE_RADIAL | 桶形 | 45，0.70 | 45，0.76 | 45，0.76 | 45，0.79 |
| RADIAL | 桶形 | 45，0.69（k1 0.100 k2 -0.183） | 45，0.75 | 45，0.74（k1 0.101 k2 -0.191） | 45，0.78 |
| OPENCV | 桶形 | 45，0.69（fx 512） | 45，0.73 | 45，0.75 | 45，0.77 |
| FULL_OPENCV | 桶形 | **0 帧** | 13 帧，系数 ±100 | **0 帧** | 2 帧 |
| SIMPLE_RADIAL_FISHEYE | 鱼眼 | 一段式 **0 对匹配** → 两段式 45，0.18，fx 192 | 两段式 0 帧 | 45，0.18 | 45，0.42 |
| RADIAL_FISHEYE | 鱼眼 | 一段式 0 对 → 两段式 45，0.18，fx 191 | 两段式 0 帧 | 45，0.18 | 45，0.45 |
| OPENCV_FISHEYE | 鱼眼 | 一段式 0 对 → 两段式 45，0.17，fx 191，k1 0.0002 k2 0.0013 k3 -0.0012 | 两段式 0 帧 | 45，0.18 | 45，0.39 |

由此定下的行为：

- **鱼眼 + 焦距未知**时一段式流程一对匹配都验不过（几何验证要用相机去畸变，畸变模型下 COLMAP 只有一个猜的默认焦距，
  鱼眼投影全错，用户看到的就是「画面之间找不到匹配的点」）。worker 因此分**两段**：先按 SIMPLE_PINHOLE 提特征、匹配、
  验证（基础矩阵和焦距无关），再把库里的相机换成目标模型去建图。全局建图能从默认焦距收敛到真值；增量建图起不来，
  这个组合提交前拦下（B-COLMAP-FISHEYEINCREMENTAL）。
- **FULL_OPENCV 不提供**（12 个系数，四种组合都散）。
- **没有验证、暂不提供**：FOV、THIN_PRISM_FISHEYE、RAD_TAN_THIN_PRISM_FISHEYE、SIMPLE_DIVISION、DIVISION、SIMPLE_FISHEYE、
  FISHEYE、EUCM。核心公式表里有它们，「LensDistortion」手填仍能用。
- 整条链路（页面提交，两段式）在 TUM-VI room1 相机走动的一段上（原始帧 800–1160 隔 6 帧，60 帧，转成三通道 PNG），
  OPENCV_FISHEYE、全局、所有帧两两、隔 2 帧：60/60 帧注册，重投影 0.26 px，Focal Length 191 px（真值 191），
  按像素半径比 80° 以内最大 0.44 px、中位 0.19 px。同一段素材开头相机几乎不动的 120 帧：一段式 40/40 但焦距 477 px
  （W-COLMAP-ROTATION 警告没有视差），所有帧两两时 E-COLMAP-NOPARALLAX 拒绝——焦距要有视差才解得出。

## 团队

COLMAP 由 Johannes L. Schönberger 在苏黎世联邦理工学院计算机视觉与几何组（ETH Zurich CVG）读博期间开发，导师是 Jan-Michael Frahm（北卡罗来纳大学教堂山分校）和 Marc Pollefeys（ETH Zurich）；版权方就是 ETH Zurich 和 UNC Chapel Hill。它是学术界和工业界用得最多的开源重建管线之一，Schönberger 因 COLMAP 获得 2020 年 PAMI Mark Everingham 奖。现在由 Schönberger、Paul-Edouard Sarlin、Shaohui Liu、Linfei Pan 等人维护；Lab2Shot 默认用的"全局解算"来自同一团队的 GLOMAP（ECCV 2024，Pan、Baráth、Pollefeys、Schönberger），在视频上建图比传统增量解算快 1–4 倍、结果更稳定。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install colmap`。会下载锁定版本（4.2.0）的官方源码（约 30 MB，用来对照许可证和文档）和 COLMAP 官方发布的 `pycolmap-cuda12` 4.2.0 预编译包，连同 CUDA 12 运行库一起装进独立环境（约 450 MB）。不需要编译，几分钟装完。
- **没有模型权重**：COLMAP 是传统算法，不下载任何神经网络模型；不需要申请权限，不需要额外手动下载，离线运行。
- 默认设置下特征提取和匹配都在 CPU 上做；用到显卡的部分（例如「显卡提取特征」）需要显卡驱动支持 CUDA 12.9 或更新。

## 许可证说明

**默认设置可以商用。** COLMAP 本身是 BSD-3-Clause（版权 ETH Zurich 和 UNC Chapel Hill），只要求再分发时保留版权声明、不能用两所学校的名义做宣传。默认用 CPU 提取 SIFT 特征（VLFeat 库，BSD），整条流程可商用；SIFT 专利已在 2020 年到期。

**「显卡提取特征」这个选项是非商用的**：它用到的 SiftGPU 版权属于北卡罗来纳大学，只允许教育、研究和非营利用途。打开它时节点会标"非商用"。关掉时结果一样，只是慢一点。

其他说明：官方预编译包里关掉了 GPL/AGPL 的组件（LSD、CGAL）；但它静态链接了 SuiteSparse 的 CHOLMOD（部分模块是 GPL-2.0+），自己用不受影响，只有把这个包再打包分发给别人时才需要遵守 GPL。

## 参考

- 论文 Structure-from-Motion Revisited（CVPR 2016）：https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html
- 全局解算 GLOMAP 论文（ECCV 2024）：https://arxiv.org/abs/2407.20219
- 官方文档：https://colmap.github.io/
- 代码：https://github.com/colmap/colmap （本扩展锁定的 4.2.0：https://github.com/colmap/colmap/tree/4.2.0 ）
- 许可证原文：https://github.com/colmap/colmap/blob/main/COPYING.txt ，文档里的说明 https://colmap.github.io/license.html
- SiftGPU 许可证：https://github.com/colmap/colmap/blob/main/src/thirdparty/SiftGPU/LICENSE
- 预编译包 pycolmap-cuda12：https://pypi.org/project/pycolmap-cuda12/
- 作者主页：https://demuc.de/

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
