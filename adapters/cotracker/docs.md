+++
team = "Meta AI（FAIR）与牛津大学视觉几何组（VGG）"
people = "Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht"
paper = "CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos（2024）"
paper_url = "https://arxiv.org/abs/2410.11831"
website = "https://cotracker3.github.io/"
repo = "https://github.com/facebookresearch/co-tracker"
year = 2024
+++

## 这是什么

上游 README 这样介绍它：CoTracker 是一个基于 Transformer 的快速模型，视频里任意一个点都跟得住，
并把光流的一些好处带到了点跟踪上。README 列出它能跟的三种：视频里的任意一个像素；一批准稠密的
像素一起跟；点可以在任意一帧上手动选，也可以按网格撒。这一代 CoTracker3 的论文题目是
《Simpler and Better Point Tracking by Pseudo-Labelling Real Videos》。

在 Lab2Shot 里，它是「CoTracker3 2D 跟踪点」：每个点每帧给出位置和「看得见 / 被挡住」。
很多点是一起算的，点与点之间互相参考，所以一整片表面（墙面、地面、衣服）会一起动，单个点不容易
跑飞。结果可以用来辅助匹配移动（给 3DE / SynthEyes 提供 2D 特征点）、平面跟踪（CornerPin、稳像）
和 roto 跟形。代码和权重都是 CC-BY-NC 4.0，**非商用**。

## 输入输出

**官方要什么、给什么**

- 吃：一段视频加一张可选的分割遮罩。官方 `demo.py` 收 `--video_path`、`--mask_path`、`--grid_size`、
  `--grid_query_frame`、`--backward_tracking`、`--offline`。
  预测器这一层是 `CoTrackerPredictor(video, queries, segm_mask, grid_size, grid_query_frame, backward_tracking)`：
  `video` 是 `(B, T, 3, H, W)`，`segm_mask` 是 `(B, 1, H, W)` 的分割遮罩，给了它，网格点就只撒在遮罩里；
  不给遮罩时也可以另给 `queries`（自己选的点）。
- 给：`tracks`（每个点每帧的位置）和 `visibilities`（每帧这个点可不可见）。

**我们怎么接的**

- 「RGB」口就是 `video`，「遮罩」口就是官方的 `segm_mask`，
  「网格点数」就是 `--grid_size`，「参考帧」就是 `--grid_query_frame`，「方式」对应 `--offline`。
- 「2D 跟踪点」口里带着 `tracks` 和 `visibilities` 两样：可见性是这份数据自带的一列，不另开一个口
  （「可见门槛」参数按它筛）。
- 「手动点」参数走的是官方 `queries` 那一路：在视图里点几下，就按点选的位置起跟；「网格点数」不是 0 时网格点也一起跟。

出处：简介来自 `third_party/cotracker/repo/README.md`；输入输出依据 `third_party/cotracker/repo/demo.py`、
`cotracker/predictor.py` 和 `adapters/cotracker/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → CoTracker3 2D 跟踪点 → 2D 跟踪点输出设置（3DEqualizer 2D 点文件或 CSV），
  导进 3DE 相机解算，或者在 Nuke 里做 Tracker / CornerPin。模板「2D 跟踪点 · CoTracker3」就是这条链。
- 手动加点：显示这个节点，在 2D 视图里点要跟的位置，每点一次加一个点（user_01、user_02…）。
- 只想跟一个物体或一块表面：把遮罩接到遮罩输入上，网格就只撒在遮罩里（网格铺满遮罩的外框）。
  自己点的点可以在任意一帧，之前的帧会倒着往回跟。
- 关键参数：`方式`。整段（默认，官方叫 offline）：一次看一大段画面、前后双向一起算，质量最好、速度最快；
  逐段滑动（online）：16 帧一个小窗口往前推，显存最省，适合非常多的点。
  `可见门槛`：整段模式判断遮挡很严（官方阈值 0.9），衣服、头发上的点常被标成「看不见」，
  位置其实是对的——调到 0.5 左右就好；改它不用重新跑模型。
  `处理分辨率`：留空用官方的 512x384（画面会被挤成这个比例，和官方一致）；填数字则保持画面比例、
  长边为这个像素数（竖拍素材可以试 768）。
- 长镜头：离线模式按显存自动分段（RTX 4090 上 400 个点一段最多 240 帧，超出后等分成几段），段与段重叠 1 帧：
  上一段在这一帧上算出的位置就是下一段的起跟位置，段间不做混合。在线模式显存固定，多长都行。

## 效果和局限

RTX 4090 上（grid 20 = 400 个点，处理分辨率 512x384）：

- sh020（iPhone 长焦手持跟拍，1080x1920，1001–1120 共 120 帧）：离线 12 ms/帧、显存峰值 5.6 GB；
  在线 19 ms/帧、1.5 GB。可见比例 57%（跟拍镜头很多点出画，最后一帧还剩 31%）。
  楼面上的点跟着手持晃动一起走；逐帧拟合单应矩阵后的残差中位数 0.52 px（离线）/ 0.50 px（在线），
  比 TAPNext++（0.66 px）稳。保持比例、长边 768 处理：14 ms/帧、6.7 GB，残差 0.56 px，没有更好。
- sh010（固定机位跳舞，864x480，124 帧）：背景点偏离起始位置中位数 0.43 px（离线 / 在线一样），
  每个背景点整段最大偏移的中位数约 1.0 px；整体可见比例 88%（离线）/ 94%（在线）。
- sh020 300 帧：离线自动分成两段（162 帧，重叠 24 帧），12 ms/帧、7.4 GB；和整段一次算（13.7 GB）
  相比位置相差中位数 0.9 px；在线 18 ms/帧、1.6 GB。
- 已知问题：离线模式的"可见"判断很严（沿用官方阈值 0.9），跳舞人物衣服上的点大多被判成"看不见"，
  但位置其实跟得住，可以用输出里的 `confidence` 自己定阈值；分段处理时，出画很久的点在后一段里
  更容易被误判为"看得见"；竖拍素材被挤成 512x384 横幅后竖向分辨率偏低。

## 团队

Meta AI（FAIR）和牛津大学视觉几何组（VGG，Andrea Vedaldi、Christian Rupprecht）的合作。
CoTracker 系列是最常用的开源点跟踪器之一（CoTracker、CoTracker2、CoTracker3）；
同一批作者还做了 VGGT、Dynamic Replica 等三维视觉工作。CoTracker3 用大量真实视频做"伪标注"训练，
模型比前代小、效果更好。

## 模型下载和安装

- 自动安装：`lab2shot ext install cotracker`。会下载 co-tracker 仓库（锁定版本）、独立的 PyTorch 环境
  （约 5 GB，和其他扩展共用下载缓存），以及 Hugging Face 上的两个权重：
  `scaled_offline.pth` 和 `scaled_online.pth`（各约 100 MB），几分钟完成，下完会自动校验。
- 不需要申请权限，不需要自行下载任何东西。

## 许可证说明

**非商用。** 代码和 CoTracker3 权重都是 CC-BY-NC 4.0：可以用于研究、学习等非商业用途，
可以修改，使用和分享时要署名 Meta；不能用于商业项目（包括商业片的制作）。
仓库里少量代码来自 PIPs（MIT）、TAP-Vid 和 LocoTrack（Apache-2.0），不改变整体的非商用限制。
需要商用的点跟踪请用 TAPNext++（Apache-2.0）。

## 参考

- 论文：https://arxiv.org/abs/2410.11831
- 项目主页：https://cotracker3.github.io/
- 代码：https://github.com/facebookresearch/co-tracker
- 模型卡（权重和许可证）：https://huggingface.co/facebook/cotracker3

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
